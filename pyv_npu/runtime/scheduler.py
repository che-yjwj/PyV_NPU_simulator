from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, Set
import heapq
from ..config import SimConfig
from ..isa.npu_ir import Program, NPUOp, Tensor, DTYPE_MAP
from ..isa.opcode import Opcode
from .resources import BankTracker, DramBankTracker, IssueQueueTracker, L0SPMTracker, IOBufferTracker, L2CacheTracker
from .bus import SystemBusTracker
from .te import calculate_systolic_array_cycles


@dataclass
class BookingInfo:
    dram_transfers: List[Dict[str, Any]] = field(default_factory=list)
    spm_slots: List[Dict[str, Any]] = field(default_factory=list)
    issue_slots: List[Dict[str, Any]] = field(default_factory=list)
    l0_spm_loads: List[Dict[str, Any]] = field(default_factory=list)
    io_buffer_pops: List[Dict[str, Any]] = field(default_factory=list)
    io_buffer_pushes: List[Dict[str, Any]] = field(default_factory=list)
    system_bus_transfers: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class ScheduleItem:
    op: NPUOp
    start_cycle: int
    end_cycle: int
    engine: str
    stall_cycles: int = 0
    stall_breakdown: Dict[str, int] = field(default_factory=dict)
    cycle_breakdown: Dict[str, int] = field(default_factory=dict)


@dataclass(order=True)
class Event:
    time: int
    type: str = field(compare=False)
    item: Any = field(compare=False, default=None)


def event_driven_schedule(p: Program, config: SimConfig) -> Tuple[List[ScheduleItem], Dict[str, Any]]:
    time = 0
    schedule: List[ScheduleItem] = []
    event_queue: List[Event] = []

    # --- Resource Trackers ---
    dram_bank_tracker = DramBankTracker(config)
    spm_banks = BankTracker(config)
    system_bus = SystemBusTracker(config)
    issue_queue = IssueQueueTracker(config)
    l0_spm_trackers = [L0SPMTracker(config) for _ in range(config.tc)]
    io_buffer = IOBufferTracker(config, name="io_buffer")
    l2_cache = L2CacheTracker(config)
    
    # --- Engine Free Time Pools ---
    tc_free_time = [0] * config.tc
    vc_free_time = [0] * config.vc
    dma_free_time = [0] * config.dma_channels
    cpu_free_time = [0]

    # --- State Tracking ---
    tensor_ready_time: Dict[str, int] = {t.name: 0 for t in p.inputs}
    tensor_ready_time.update({t.name: 0 for t in p.initializers})
    completed_tickets: Set[int] = set()
    op_pending_time: Dict[str, int] = {}

    # --- Operation Queues ---
    cpu_op_queue = p.ops.copy()
    npu_work_queue: List[NPUOp] = []

    heapq.heappush(event_queue, Event(time=0, type='CHECK_SCHED'))

    while event_queue:
        event = heapq.heappop(event_queue)
        time = event.time

        if event.type == 'OP_COMPLETE':
            op, end_cycle, ticket, engine_idx, booking_info = event.item
            if op.outputs:
                for out_tensor in op.outputs:
                    tensor_ready_time[out_tensor.name] = end_cycle
            
            if ticket is not None:
                completed_tickets.add(ticket)
            
            if get_engine_for_op(op, config.mode) == "TC" and engine_idx is not None:
                l0_spm_trackers[engine_idx].commit_load(op.inputs)
            
            heapq.heappush(event_queue, Event(time=end_cycle, type='CHECK_SCHED'))
        
        elif event.type == 'ENQCMD_COMPLETE':
            npu_op, ticket = event.item
            npu_op.args['_ticket'] = ticket
            npu_work_queue.append(npu_op)
            heapq.heappush(event_queue, Event(time, type='CHECK_SCHED'))

        elif event.type == 'CHECK_SCHED':
            run_scheduler_pass(
                cpu_op_queue=cpu_op_queue, npu_work_queue=npu_work_queue,
                time=time, config=config, schedule=schedule, event_queue=event_queue,
                tensor_ready_time=tensor_ready_time, completed_tickets=completed_tickets,
                engine_pools={'TC': tc_free_time, 'VC': vc_free_time, 'DMA': dma_free_time, 'CPU': cpu_free_time},
                resource_trackers={'dram_banks': dram_bank_tracker, 'spm_banks': spm_banks, 'issue_queue': issue_queue, 'l0_spms': l0_spm_trackers, 'io_buffer': io_buffer, 'system_bus': system_bus, 'l2_cache': l2_cache},
                op_pending_time=op_pending_time
            )

    schedule.sort(key=lambda x: x.start_cycle)
    
    # Calculate total_cycles
    total_cycles = max(item.end_cycle for item in schedule) if schedule else 0

    stats = {"dram_collisions": dram_bank_tracker.collisions}
    
    # Add total_cycles to stats
    stats["total_cycles"] = total_cycles

    # Add L2 cache stats if enabled
    if l2_cache.enabled:
        l2_stats = l2_cache.get_stats()
        stats["l2_cache_stats"] = l2_stats

    return schedule, stats


def _find_best_candidate_op(op_queues, time, config, tensor_ready_time, completed_tickets, op_pending_time, engine_pools, resource_trackers):
    """Iterates through ready ops and finds the one that can start earliest."""
    best_op_details = None
    earliest_start_time = float('inf')

    for op_queue in op_queues:
        for op in list(op_queue):
            # Check if the operation is ready to be scheduled
            if op.opcode == Opcode.TWAIT:
                if op.args.get('twait') and op.args['twait'].ticket not in completed_tickets:
                    continue
            elif not all(t.name in tensor_ready_time for t in op.inputs):
                if op.name not in op_pending_time:
                    op_pending_time[op.name] = time
                continue

            ideal_start_time = max((tensor_ready_time.get(t.name, 0) for t in op.inputs), default=time)
            engine_type = get_engine_for_op(op, config.mode)
            engine_pool = engine_pools.get(engine_type)
            if not engine_pool:
                continue

            # Find the best engine for this operation
            for i, engine_free_time in enumerate(engine_pool):
                tentative_start_time = max(ideal_start_time, engine_free_time)
                
                actual_start, actual_end, reason, breakdown, booking = calculate_op_timing(op, tentative_start_time, config, resource_trackers, engine_idx=i)
                
                if actual_start < earliest_start_time:
                    earliest_start_time = actual_start
                    first_pending_time = op_pending_time.get(op.name, time)
                    
                    # Calculate stall breakdown
                    stall_breakdown = {}
                    dep_stall = ideal_start_time - first_pending_time
                    if dep_stall > 0:
                        stall_breakdown["DEP"] = dep_stall
                    
                    resource_stall = actual_start - ideal_start_time
                    if resource_stall > 0:
                        if engine_free_time > ideal_start_time:
                            stall_breakdown["RESOURCE_ENGINE"] = resource_stall
                        elif reason != "NONE":
                            stall_breakdown[reason] = resource_stall
                    
                    best_op_details = {
                        'op': op, 'op_queue': op_queue, 'start': actual_start, 'end': actual_end,
                        'engine': engine_type, 'engine_idx': i, 'stall_cycles': sum(stall_breakdown.values()),
                        'stall_breakdown': stall_breakdown, 'cycle_breakdown': breakdown, 'booking_info': booking
                    }
    return best_op_details


def run_scheduler_pass(cpu_op_queue, npu_work_queue, time, config, schedule, event_queue, tensor_ready_time, completed_tickets, op_pending_time, engine_pools, resource_trackers):
    """
    Finds the best operation to schedule next and commits it.
    """
    op_queues = [cpu_op_queue]
    if config.mode == 'tight':
        op_queues.append(npu_work_queue)

    best_op_details = _find_best_candidate_op(
        op_queues, time, config, tensor_ready_time, completed_tickets,
        op_pending_time, engine_pools, resource_trackers
    )

    if not best_op_details:
        return

    # --- Commit the chosen operation ---
    op = best_op_details['op']
    op_queue = best_op_details['op_queue']
    engine_type = best_op_details['engine']
    engine_idx = best_op_details['engine_idx']
    booking_info = best_op_details['booking_info']
    end_cycle = best_op_details['end']

    # Commit resource bookings
    dram_banks = resource_trackers['dram_banks']
    spm = resource_trackers['spm_banks']
    issue_queue = resource_trackers['issue_queue']
    system_bus = resource_trackers['system_bus']
    io_buffer = resource_trackers['io_buffer']

    if best_op_details['stall_breakdown'].get("RESOURCE_DRAM_BANK"):
        dram_banks.collisions += 1
    for booking in booking_info.dram_transfers:
        dram_banks.commit_transfer(**booking)
    for booking in booking_info.spm_slots:
        spm.commit_slot(**booking)
    for booking in booking_info.issue_slots:
        issue_queue.commit_issue(**booking)
    for booking in booking_info.system_bus_transfers:
        system_bus.commit_transfer(**booking)
    for _ in booking_info.io_buffer_pops:
        io_buffer.pop()
    for push_info in booking_info.io_buffer_pushes:
        io_buffer.push(**push_info)
    
    # Update engine availability
    engine_pools[engine_type][engine_idx] = end_cycle

    # Add to schedule
    schedule.append(ScheduleItem(
        op=op, start_cycle=best_op_details['start'], end_cycle=end_cycle,
        engine=f"{engine_type}{engine_idx}", stall_cycles=best_op_details['stall_cycles'],
        stall_breakdown=best_op_details['stall_breakdown'], cycle_breakdown=best_op_details['cycle_breakdown']
    ))

    # Remove from queue and push completion event
    op_queue.remove(op)

    if config.mode == 'tight' and op.opcode == Opcode.ENQCMD_T:
        npu_op_desc = op.args['enqcmd'].npu_op_desc
        ticket = op.args['enqcmd'].ticket
        heapq.heappush(event_queue, Event(time=end_cycle, type='ENQCMD_COMPLETE', item=(npu_op_desc, ticket)))
    else:
        ticket = op.args.get('_ticket')
        heapq.heappush(event_queue, Event(time=end_cycle, type='OP_COMPLETE', item=(op, end_cycle, ticket, engine_idx, booking_info)))
    
    # Trigger next scheduling check
    heapq.heappush(event_queue, Event(time, 'CHECK_SCHED'))


# Opcode-specific timing calculation helpers
def _calculate_control_op_timing(op: NPUOp, start_cycle: int, config: SimConfig, resources: Dict[str, Any], **kwargs) -> Tuple[int, int, str, Dict[str, int], BookingInfo]:
    issue_queue = resources['issue_queue']
    booking_info = BookingInfo()

    if op.opcode == Opcode.ENQCMD_T:
        issue_start = issue_queue.probe_issue_time(start_cycle)
        duration = config.tight_mode_doorbell_latency
        booking_info.issue_slots.append({'issue_cycle': issue_start})
        return issue_start, issue_start + duration, "NONE", {'control': duration}, booking_info

    if op.opcode == Opcode.TWAIT:
        duration = config.tight_mode_csr_latency
        return start_cycle, start_cycle + duration, "NONE", {'control': duration}, booking_info


def _calculate_dma_op_timing(op: NPUOp, start_cycle: int, config: SimConfig, resources: Dict[str, Any], **kwargs) -> Tuple[int, int, str, Dict[str, int], BookingInfo]:
    dram_banks, io_buffer, system_bus, l2_cache = resources['dram_banks'], resources['io_buffer'], resources['system_bus'], resources['l2_cache']
    booking_info = BookingInfo()
    breakdown = {}

    if op.opcode == Opcode.LOAD:
        if not op.outputs:
            return start_cycle, start_cycle + 1, "NONE", breakdown, booking_info
        tensor = op.outputs[0]
        num_bytes = tensor.num_elements * DTYPE_MAP.get(tensor.dtype, 1)
        address = op.inputs[0].address

        l2_latency = 0
        is_hit = False
        l2_cache_enabled_for_op = False

        if address is not None:
            l2_cache_enabled_for_op = l2_cache.enabled
            if l2_cache_enabled_for_op:
                is_hit, l2_latency = l2_cache.access(address)
                breakdown['l2_access'] = l2_latency # Add L2 access latency to breakdown

        if l2_cache_enabled_for_op and is_hit:
            # L2 Hit: Data comes from L2 cache. No DRAM or System Bus access needed.
            # The operation starts at start_cycle and takes l2_latency.
            actual_start = start_cycle
            total_duration = l2_latency
            stall_reason = "NONE"
            
            # Still need to push to IO buffer
            if not io_buffer.can_push(num_bytes):
                return start_cycle + 1, start_cycle + 2, "RESOURCE_IO_BUFFER_FULL", breakdown, booking_info
            booking_info.io_buffer_pushes.append({'num_bytes': num_bytes, 'tensor_name': tensor.name})
            
            return actual_start, actual_start + total_duration, stall_reason, breakdown, booking_info
        else:
            # L2 Miss or L2 Disabled: Incur L2 miss latency (if enabled), then proceed to DRAM and System Bus.
            dram_bus_start_cycle = start_cycle + l2_latency

        # Common path for L2 Miss or L2 Disabled: Access DRAM and System Bus
        if not io_buffer.can_push(num_bytes):
            return start_cycle + 1, start_cycle + 2, "RESOURCE_IO_BUFFER_FULL", breakdown, booking_info

        # Probe DRAM banks
        dram_start, dram_duration, dram_stall, ch_id, b_id = dram_banks.probe_transfer(dram_bus_start_cycle, address, num_bytes)
        
        # Probe system bus, starting from when the DRAM is available
        bus_start, bus_duration, bus_stall = system_bus.probe_transfer(dram_start, num_bytes)

        actual_start = start_cycle # The operation is considered to start at the original start_cycle
        # Total duration includes L2 miss latency (if applicable) + DRAM + Bus
        total_duration = (bus_start + bus_duration) - actual_start
        
        stall_reason = "NONE"
        if dram_start > dram_bus_start_cycle:
            stall_reason = dram_stall
        elif bus_start > dram_start:
            stall_reason = bus_stall

        breakdown['dram_access'] = dram_duration
        breakdown['bus_access'] = bus_duration

        booking_info.dram_transfers.append({'channel_id': ch_id, 'bank_id': b_id, 'start_cycle': dram_start, 'duration': dram_duration})
        booking_info.system_bus_transfers.append({'start_cycle': bus_start, 'duration': bus_duration, 'num_bytes': num_bytes})
        booking_info.io_buffer_pushes.append({'num_bytes': num_bytes, 'tensor_name': tensor.name})
        
        return actual_start, actual_start + total_duration, stall_reason, breakdown, booking_info

    if op.opcode == Opcode.STORE:
        tensor = op.inputs[0]
        # Check if the required tensor is at the head of the FIFO.
        next_item = io_buffer.peek()
        if not next_item or next_item[1] != tensor.name:
            return start_cycle + 1, start_cycle + 2, "RESOURCE_IO_BUFFER_EMPTY", breakdown, booking_info
        
        num_bytes = next_item[0]
        address = op.outputs[0].address

        l2_latency = 0
        is_hit = False # Not a real hit, just bypassing L2 if address is None
        l2_cache_enabled_for_op = False

        if address is not None:
            l2_cache_enabled_for_op = l2_cache.enabled
            if l2_cache_enabled_for_op:
                is_hit, l2_latency = l2_cache.access(address)
                breakdown['l2_access'] = l2_latency # Add L2 access latency to breakdown

        # For STORE, we always proceed to DRAM (write-through policy for simplicity)
        # The L2 latency is incurred first, then DRAM/Bus access starts.
        dram_bus_start_cycle = start_cycle + l2_latency

        # Pop from IO buffer regardless of L2 hit/miss, as data is consumed
        booking_info.io_buffer_pops.append({})

        # Probe DRAM banks
        dram_start, dram_duration, dram_stall, ch_id, b_id = dram_banks.probe_transfer(dram_bus_start_cycle, address, num_bytes)

        # Probe system bus, starting from when the DRAM is available
        bus_start, bus_duration, bus_stall = system_bus.probe_transfer(dram_start, num_bytes)

        actual_start = start_cycle # The operation is considered to start at the original start_cycle
        # Total duration includes L2 latency (if applicable) + DRAM + Bus
        total_duration = (bus_start + bus_duration) - actual_start
        
        stall_reason = "NONE"
        if dram_start > dram_bus_start_cycle:
            stall_reason = dram_stall
        elif bus_start > dram_start:
            stall_reason = bus_stall

        breakdown['dram_access'] = dram_duration
        breakdown['bus_access'] = bus_duration

        booking_info.dram_transfers.append({'channel_id': ch_id, 'bank_id': b_id, 'start_cycle': dram_start, 'duration': dram_duration})
        booking_info.system_bus_transfers.append({'start_cycle': bus_start, 'duration': bus_duration, 'num_bytes': num_bytes})
        
        return actual_start, actual_start + total_duration, stall_reason, breakdown, booking_info


def _calculate_tc_op_timing(op: NPUOp, start_cycle: int, config: SimConfig, resources: Dict[str, Any], engine_idx: int, **kwargs) -> Tuple[int, int, str, Dict[str, int], BookingInfo]:
    spm, l0_spms, io_buffer = resources['spm_banks'], resources['l0_spms'], resources['io_buffer']
    booking_info = BookingInfo()
    breakdown = {}

    # Check if all required tensors are at the head of the FIFO in the correct order.
    if len(io_buffer.queue) < len(op.inputs):
        return start_cycle + 1, start_cycle + 2, "RESOURCE_IO_BUFFER_EMPTY", breakdown, booking_info
    
    for i, t in enumerate(op.inputs):
        if io_buffer.queue[i][1] != t.name:
            # Return a new stall reason for mismatch
            return start_cycle + 1, start_cycle + 2, "RESOURCE_IO_BUFFER_MISMATCH", breakdown, booking_info
    
    for t in op.inputs:
        booking_info.io_buffer_pops.append({})

    l0_spm = l0_spms[engine_idx]
    is_hit = l0_spm.probe_hit(op.inputs)
    
    load_cycles = 0
    if not is_hit:
        bytes_to_load = l0_spm.get_required_load_bytes(op.inputs)
        load_cycles = bytes_to_load // 128 
    
    breakdown = calculate_systolic_array_cycles(
        op.args.get('tile_m', 128), op.args.get('tile_n', 128), op.args.get('tile_k', 128), 
        config.systolic_array_height, config.systolic_array_width
    )
    compute_cycles = breakdown['total']
    
    total_op_duration = compute_cycles + (l0_spm.latency if is_hit else load_cycles)
    breakdown['l0_access'] = l0_spm.latency if is_hit else load_cycles
    breakdown['total'] = total_op_duration

    num_banks_needed = config.spm_banks // 2
    op_start_cycle, chosen_banks = spm.probe_earliest_free_slot(start_cycle, total_op_duration, num_banks_needed)
    
    stall_reason = "NONE"
    if op_start_cycle > start_cycle:
        stall_reason = "RESOURCE_SPM"
        
    op_end_cycle = op_start_cycle + total_op_duration
    booking_info.spm_slots.append({'cycle': op_start_cycle, 'duration': total_op_duration, 'chosen_banks': chosen_banks})
    
    out_bytes = sum(t.num_elements * DTYPE_MAP.get(t.dtype, 1) for t in op.outputs)
    if op.outputs:
        if not io_buffer.can_push(out_bytes):
            return start_cycle + 1, start_cycle + 2, "RESOURCE_IO_BUFFER_FULL", breakdown, booking_info
        booking_info.io_buffer_pushes.append({'num_bytes': out_bytes, 'tensor_name': op.outputs[0].name})

    return op_start_cycle, op_end_cycle, stall_reason, breakdown, booking_info


def _calculate_vc_op_timing(op: NPUOp, start_cycle: int, config: SimConfig, resources: Dict[str, Any], **kwargs) -> Tuple[int, int, str, Dict[str, int], BookingInfo]:
    spm, io_buffer = resources['spm_banks'], resources['io_buffer']
    booking_info = BookingInfo()
    breakdown = {}

    # Check if all required tensors are at the head of the FIFO in the correct order.
    if len(io_buffer.queue) < len(op.inputs):
        return start_cycle + 1, start_cycle + 2, "RESOURCE_IO_BUFFER_EMPTY", breakdown, booking_info
    
    for i, t in enumerate(op.inputs):
        if io_buffer.queue[i][1] != t.name:
            # Return a new stall reason for mismatch
            return start_cycle + 1, start_cycle + 2, "RESOURCE_IO_BUFFER_MISMATCH", breakdown, booking_info
    
    for t in op.inputs:
        booking_info.io_buffer_pops.append({})

    num_elements = op.args.get('num_elements', 2048)
    compute_cycles = max(5, num_elements // 16)
    num_banks_needed = 2
    compute_start_cycle, chosen_banks = spm.probe_earliest_free_slot(start_cycle, compute_cycles, num_banks_needed)
    stall_reason = "NONE"
    if compute_start_cycle > start_cycle: stall_reason = "RESOURCE_SPM"
    breakdown = {'total': compute_cycles, 'compute': compute_cycles}
    booking_info.spm_slots.append({'cycle': compute_start_cycle, 'duration': compute_cycles, 'chosen_banks': chosen_banks})
    
    out_bytes = sum(t.num_elements * DTYPE_MAP.get(t.dtype, 1) for t in op.outputs)
    if op.outputs:
        if not io_buffer.can_push(out_bytes):
            return start_cycle + 1, start_cycle + 2, "RESOURCE_IO_BUFFER_FULL", breakdown, booking_info
        booking_info.io_buffer_pushes.append({'num_bytes': out_bytes, 'tensor_name': op.outputs[0].name})

    return compute_start_cycle, compute_start_cycle + compute_cycles, stall_reason, breakdown, booking_info


def calculate_op_timing(op: NPUOp, start_cycle: int, config: SimConfig, resources: Dict[str, Any], engine_idx: int) -> Tuple[int, int, str, Dict[str, int], BookingInfo]:
    """
    Calculates the start and end cycle for an NPU operation, considering resource availability.
    This function is a dispatcher that calls the appropriate helper based on the opcode.
    """
    opcode_to_handler = {
        Opcode.ENQCMD_T: _calculate_control_op_timing,
        Opcode.TWAIT: _calculate_control_op_timing,
        Opcode.LOAD: _calculate_dma_op_timing,
        Opcode.STORE: _calculate_dma_op_timing,
        Opcode.MATMUL: _calculate_tc_op_timing,
        Opcode.CONV: _calculate_tc_op_timing,
        Opcode.GELU: _calculate_vc_op_timing,
        Opcode.SOFTMAX: _calculate_vc_op_timing,
        Opcode.ADD: _calculate_vc_op_timing,
        Opcode.MUL: _calculate_vc_op_timing,
        Opcode.LAYERNORM: _calculate_vc_op_timing,
        Opcode.ERF: _calculate_vc_op_timing,
    }

    handler = opcode_to_handler.get(op.opcode)

    if handler:
        return handler(op=op, start_cycle=start_cycle, config=config, resources=resources, engine_idx=engine_idx)
    else:
        # Default case for unknown or simple ops
        return start_cycle, start_cycle + 1, "NONE", {}, BookingInfo()


def get_engine_for_op(op: NPUOp, mode: str = 'loose') -> str:
    if mode == 'tight' and op.opcode in (Opcode.ENQCMD_T, Opcode.TWAIT):
        return "CPU"
    elif op.opcode in (Opcode.MATMUL, Opcode.CONV):
        return "TC"
    elif op.opcode in (Opcode.GELU, Opcode.SOFTMAX, Opcode.ADD, Opcode.MUL, Opcode.LAYERNORM, Opcode.ERF):
        return "VC"
    else:
        return "DMA"


def simple_greedy_schedule(p: Program) -> List[ScheduleItem]:
    # ... (rest of the function is unchanged)
    return []
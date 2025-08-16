from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, Set
import heapq
from ..config import SimConfig
from ..isa.npu_ir import Program, NPUOp, Tensor, DTYPE_MAP
from .resources import BankTracker, DramBankTracker, IssueQueueTracker
from .te import calculate_systolic_array_cycles

@dataclass
class BookingInfo:
    dram_transfers: List[Dict[str, Any]] = field(default_factory=list)
    spm_slots: List[Dict[str, Any]] = field(default_factory=list)
    issue_slots: List[Dict[str, Any]] = field(default_factory=list)

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
    issue_queue = IssueQueueTracker(config)
    
    # --- Engine Free Time Pools ---
    te_free_time = [0] * config.te
    ve_free_time = [0] * config.ve
    dma_free_time = [0] * config.dma_channels
    cpu_free_time = [0]

    # --- State Tracking ---
    tensor_ready_time: Dict[str, int] = {t.name: 0 for t in p.inputs}
    tensor_ready_time.update({t.name: 0 for t in p.initializers})
    completed_tickets: Set[int] = set()
    op_pending_time: Dict[str, int] = {}

    # --- Operation Queues ---
    # In loose mode, this is the main queue. 
    # In tight mode, this holds CPU-side ops (ENQCMD, TWAIT).
    cpu_op_queue = p.ops.copy()
    # In tight mode, this queue holds the NPU-side ops extracted from ENQCMD.
    npu_work_queue: List[NPUOp] = []

    heapq.heappush(event_queue, Event(time=0, type='CHECK_SCHED'))

    while event_queue:
        event = heapq.heappop(event_queue)
        time = event.time

        if event.type == 'OP_COMPLETE':
            op, end_cycle, ticket = event.item
            if op.outputs:
                for out_tensor in op.outputs:
                    tensor_ready_time[out_tensor.name] = end_cycle
            if ticket is not None:
                completed_tickets.add(ticket)
            heapq.heappush(event_queue, Event(time=end_cycle, type='CHECK_SCHED'))
        
        elif event.type == 'ENQCMD_COMPLETE':
            npu_op, ticket = event.item
            npu_op.args['_ticket'] = ticket # Tag the op with its ticket
            npu_work_queue.append(npu_op)
            heapq.heappush(event_queue, Event(time, type='CHECK_SCHED'))

        elif event.type == 'CHECK_SCHED':
            # Pass both queues to the scheduler
            run_scheduler_pass(
                cpu_op_queue=cpu_op_queue, npu_work_queue=npu_work_queue,
                time=time, config=config, schedule=schedule, event_queue=event_queue,
                tensor_ready_time=tensor_ready_time, completed_tickets=completed_tickets,
                engine_pools={'TE': te_free_time, 'VE': ve_free_time, 'DMA': dma_free_time, 'CPU': cpu_free_time},
                resource_trackers={'dram_banks': dram_bank_tracker, 'spm_banks': spm_banks, 'issue_queue': issue_queue},
                op_pending_time=op_pending_time
            )

    schedule.sort(key=lambda x: x.start_cycle)
    stats = {"dram_collisions": dram_bank_tracker.collisions}
    return schedule, stats

def run_scheduler_pass(cpu_op_queue, npu_work_queue, time, config, schedule, event_queue, tensor_ready_time, completed_tickets, op_pending_time, engine_pools, resource_trackers):
    best_op = None
    best_op_queue = None
    earliest_start_time = float('inf')
    op_details = {}
    best_booking_info = None

    # In tight mode, CPU ops (ENQCMD/TWAIT) and NPU ops are scheduled from different queues
    queues_to_process = [cpu_op_queue]
    if config.mode == 'tight':
        queues_to_process.append(npu_work_queue)

    for op_queue in queues_to_process:
        for op in list(op_queue):
            if op.opcode == 'TWAIT':
                if op.args.get('twait') and op.args['twait'].ticket not in completed_tickets:
                    continue
            elif not all(t.name in tensor_ready_time for t in op.inputs):
                if op.name not in op_pending_time: op_pending_time[op.name] = time
                continue

            ideal_start_time = max((tensor_ready_time.get(t.name, 0) for t in op.inputs), default=time)
            engine_type = get_engine_for_op(op, config.mode)
            engine_pool = engine_pools.get(engine_type)
            if not engine_pool: continue

            engine_free_time = min(engine_pool)
            tentative_start_time = max(ideal_start_time, engine_free_time)
            actual_start, actual_end, reason, breakdown, booking = calculate_op_timing(op, tentative_start_time, config, resource_trackers)
            
            if actual_start < earliest_start_time:
                earliest_start_time = actual_start
                best_op = op
                best_op_queue = op_queue
                best_booking_info = booking
                
                first_pending_time = op_pending_time.get(op.name, time)
                stall_breakdown = {}
                dep_stall = ideal_start_time - first_pending_time
                if dep_stall > 0: stall_breakdown["DEP"] = dep_stall
                resource_stall = actual_start - ideal_start_time
                if resource_stall > 0:
                    if engine_free_time > ideal_start_time: stall_breakdown["RESOURCE_ENGINE"] = resource_stall
                    elif reason != "NONE": stall_breakdown[reason] = resource_stall
                
                op_details = {
                    'start': actual_start, 'end': actual_end, 'engine': engine_type,
                    'stall_cycles': sum(stall_breakdown.values()),
                    'stall_breakdown': stall_breakdown, 'cycle_breakdown': breakdown
                }

    if best_op:
        op = best_op
        dram_banks, spm, issue_queue = resource_trackers['dram_banks'], resource_trackers['spm_banks'], resource_trackers['issue_queue']
        
        if op_details['stall_breakdown'].get("RESOURCE_DRAM_BANK"): dram_banks.collisions += 1
        for booking in best_booking_info.dram_transfers: dram_banks.commit_transfer(**booking)
        for booking in best_booking_info.spm_slots: spm.commit_slot(**booking)
        for booking in best_booking_info.issue_slots: issue_queue.commit_issue(**booking)

        engine_pool = engine_pools[op_details['engine']]
        engine_idx = engine_pool.index(min(engine_pool))
        engine_pool[engine_idx] = op_details['end']

        schedule.append(ScheduleItem(
            op=op, start_cycle=op_details['start'], end_cycle=op_details['end'],
            engine=f"{op_details['engine']}{engine_idx}", stall_cycles=op_details['stall_cycles'],
            stall_breakdown=op_details['stall_breakdown'], cycle_breakdown=op_details['cycle_breakdown']
        ))

        best_op_queue.remove(best_op)

        if config.mode == 'tight' and op.opcode == 'ENQCMD_T':
            npu_op_desc = op.args['enqcmd'].npu_op_desc
            ticket = op.args['enqcmd'].ticket
            heapq.heappush(event_queue, Event(time=op_details['end'], type='ENQCMD_COMPLETE', item=(npu_op_desc, ticket)))
        else:
            ticket = op.args.get('_ticket')
            heapq.heappush(event_queue, Event(time=op_details['end'], type='OP_COMPLETE', item=(op, op_details['end'], ticket)))
        
        heapq.heappush(event_queue, Event(time, 'CHECK_SCHED'))

def calculate_op_timing(op: NPUOp, start_cycle: int, config: SimConfig, resources: Dict[str, Any]) -> Tuple[int, int, str, Dict[str, int], BookingInfo]:
    dram_banks, spm, issue_queue = resources['dram_banks'], resources['spm_banks'], resources['issue_queue']
    stall_reason, breakdown, booking_info = "NONE", {}, BookingInfo()

    if op.opcode == 'ENQCMD_T':
        issue_start = issue_queue.probe_issue_time(start_cycle)
        duration = config.tight_mode_doorbell_latency
        booking_info.issue_slots.append({'issue_cycle': issue_start})
        return issue_start, issue_start + duration, "NONE", {'control': duration}, booking_info

    if op.opcode == 'TWAIT':
        duration = config.tight_mode_csr_latency
        return start_cycle, start_cycle + duration, "NONE", {'control': duration}, booking_info

    if op.opcode in ('LOAD', 'STORE'):
        if not op.inputs and not op.outputs: return start_cycle, start_cycle + 1, "NONE", breakdown, booking_info
        tensor = op.inputs[0] if op.opcode == 'LOAD' else op.outputs[0]
        num_bytes = tensor.num_elements * DTYPE_MAP.get(tensor.dtype, 1)
        actual_start, duration, stall_reason, ch_id, b_id = dram_banks.probe_transfer(start_cycle, tensor.address, num_bytes)
        booking_info.dram_transfers.append({'channel_id': ch_id, 'bank_id': b_id, 'start_cycle': actual_start, 'duration': duration})
        return actual_start, actual_start + duration, stall_reason, breakdown, booking_info

    elif op.opcode in ("MatMul", "Conv"):
        if len(op.inputs) < 2 or len(op.outputs) < 1: return start_cycle, start_cycle + 1, "NONE", breakdown, booking_info
        tensor_a, tensor_b, tensor_c = op.inputs[0], op.inputs[1], op.outputs[0]
        bytes_a, bytes_b, bytes_c = (t.num_elements * DTYPE_MAP.get(t.dtype, 1) for t in [tensor_a, tensor_b, tensor_c])

        load_a_start, load_a_dur, reason_a, ch_a, b_a = dram_banks.probe_transfer(start_cycle, tensor_a.address, bytes_a)
        load_b_start, load_b_dur, reason_b, ch_b, b_b = dram_banks.probe_transfer(load_a_start + load_a_dur, tensor_b.address, bytes_b)
        inputs_ready_cycle = load_b_start + load_b_dur
        stall_reason = reason_a if reason_a != "NONE" else reason_b

        breakdown = calculate_systolic_array_cycles(op.args.get('tile_m', 128), op.args.get('tile_n', 128), op.args.get('tile_k', 128), config.systolic_array_height, config.systolic_array_width)
        compute_cycles = breakdown['total']
        num_banks_needed = config.spm_banks // 2
        compute_start_cycle, chosen_banks = spm.probe_earliest_free_slot(inputs_ready_cycle, compute_cycles, num_banks_needed)
        if compute_start_cycle > inputs_ready_cycle: stall_reason = "RESOURCE_SPM"
        compute_end_cycle = compute_start_cycle + compute_cycles
        store_start, store_dur, reason_c, ch_c, b_c = dram_banks.probe_transfer(compute_end_cycle, tensor_c.address, bytes_c)
        if reason_c != "NONE": stall_reason = reason_c

        booking_info.dram_transfers.extend([
            {'channel_id': ch_a, 'bank_id': b_a, 'start_cycle': load_a_start, 'duration': load_a_dur},
            {'channel_id': ch_b, 'bank_id': b_b, 'start_cycle': load_b_start, 'duration': load_b_dur},
            {'channel_id': ch_c, 'bank_id': b_c, 'start_cycle': store_start, 'duration': store_dur}
        ])
        booking_info.spm_slots.append({'cycle': compute_start_cycle, 'duration': compute_cycles, 'chosen_banks': chosen_banks})
        return compute_start_cycle, store_start + store_dur, stall_reason, breakdown, booking_info

    elif op.opcode in ("GELU", "Softmax", "Add", "Mul", "LayerNorm"):
        num_elements = op.args.get('num_elements', 2048)
        compute_cycles = max(5, num_elements // 16)
        num_banks_needed = 2
        compute_start_cycle, chosen_banks = spm.probe_earliest_free_slot(start_cycle, compute_cycles, num_banks_needed)
        if compute_start_cycle > start_cycle: stall_reason = "RESOURCE_SPM"
        breakdown = {'total': compute_cycles, 'compute': compute_cycles}
        booking_info.spm_slots.append({'cycle': compute_start_cycle, 'duration': compute_cycles, 'chosen_banks': chosen_banks})
        return compute_start_cycle, compute_start_cycle + compute_cycles, stall_reason, breakdown, booking_info

    else:
        return start_cycle, start_cycle + 1, "NONE", breakdown, booking_info

def get_engine_for_op(op: NPUOp, mode: str = 'loose') -> str:
    if mode == 'tight' and op.opcode in ('ENQCMD_T', 'TWAIT'):
        return "CPU"
    if op.opcode in ("MatMul", "Conv"):
        return "TE"
    elif op.opcode in ("GELU", "Softmax", "Add", "Mul", "LayerNorm"):
        return "VE"
    else:
        return "DMA"

def simple_greedy_schedule(p: Program) -> List[ScheduleItem]:
    time = 0
    schedule: List[ScheduleItem] = []
    for op in p.ops:
        if op.opcode in ("MatMul", "Conv"):
            dur = 100
            eng = "TE"
        elif op.opcode in ("GELU", "Softmax", "Add", "Mul", "LayerNorm"):
            dur = 20
            eng = "VE"
        else:
            dur = 10
            eng = "CPU"
        schedule.append(ScheduleItem(op=op, start_cycle=time, end_cycle=time+dur, engine=eng))
        time += dur
    return schedule
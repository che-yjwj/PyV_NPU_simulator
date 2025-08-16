from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, Set
import heapq
from ..config import SimConfig
from ..isa.npu_ir import Program, NPUOp, Tensor, DTYPE_MAP
from .resources import BankTracker, DramBankTracker
from .te import calculate_systolic_array_cycles

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

    dram_bank_tracker = DramBankTracker(config)
    spm_banks = BankTracker(config)
    
    te_free_time = [0] * config.te
    ve_free_time = [0] * config.ve
    dma_free_time = [0] * config.dma_channels

    tensor_ready_time: Dict[str, int] = {t.name: 0 for t in p.inputs}
    tensor_ready_time.update({t.name: 0 for t in p.initializers})
    completed_tickets: Set[int] = set()
    op_pending_time: Dict[str, int] = {}

    cpu_op_queue = p.ops.copy()
    npu_op_queue: List[Tuple[NPUOp, int]] = []
    
    heapq.heappush(event_queue, Event(time=0, type='CHECK_SCHED'))

    while event_queue:
        event = heapq.heappop(event_queue)
        time = event.time

        if event.type == 'OP_COMPLETE':
            op, end_cycle, ticket = event.item
            for out_tensor in op.outputs:
                tensor_ready_time[out_tensor.name] = end_cycle
            if ticket is not None:
                completed_tickets.add(ticket)
            heapq.heappush(event_queue, Event(time=end_cycle, type='CHECK_SCHED'))

        elif event.type == 'CHECK_SCHED':
            run_scheduler_pass(
                cpu_op_queue=cpu_op_queue,
                npu_op_queue=npu_op_queue,
                time=time,
                config=config,
                schedule=schedule,
                event_queue=event_queue,
                tensor_ready_time=tensor_ready_time,
                completed_tickets=completed_tickets,
                engine_pools={'TE': te_free_time, 'VE': ve_free_time, 'DMA': dma_free_time},
                resource_trackers={'dram_banks': dram_bank_tracker, 'spm_banks': spm_banks},
                op_pending_time=op_pending_time
            )

    schedule.sort(key=lambda x: x.start_cycle)
    stats = {
        "dram_collisions": dram_bank_tracker.collisions
    }
    return schedule, stats

def run_scheduler_pass(cpu_op_queue, npu_op_queue, time, config, schedule, event_queue, tensor_ready_time, completed_tickets, op_pending_time, engine_pools, resource_trackers):
    q_to_process = None
    is_tight = False
    if config.mode == 'tight':
        is_tight = True
        q_to_process = npu_op_queue
    else: # loose mode
        q_to_process = cpu_op_queue

    best_op_item = None
    earliest_start_time = float('inf')
    op_details = {}

    for op_item in list(q_to_process):
        op = op_item[0] if is_tight else op_item

        if not all(t.name in tensor_ready_time for t in op.inputs):
            if op.name not in op_pending_time:
                op_pending_time[op.name] = time
            continue

        ideal_start_time = max((tensor_ready_time.get(t.name, 0) for t in op.inputs), default=time)
        engine_type = get_engine_for_op(op)
        engine_pool = engine_pools.get(engine_type)
        if not engine_pool: continue

        engine_free_time = min(engine_pool)
        
        tentative_start_time = max(ideal_start_time, engine_free_time)
        actual_start, actual_end, resource_reason, breakdown = calculate_op_timing(op, tentative_start_time, config, resource_trackers)
        
        first_pending_time = op_pending_time.get(op.name, time)
        stall_breakdown = {}
        
        # 1. Dependency Stall
        dep_stall = ideal_start_time - first_pending_time
        if dep_stall > 0:
            stall_breakdown["DEP"] = dep_stall

        # 2. Resource Stall
        # The time spent waiting for a resource after dependencies are met
        resource_stall = actual_start - ideal_start_time
        if resource_stall > 0:
            # Determine the primary resource bottleneck
            if engine_free_time > ideal_start_time:
                stall_breakdown["RESOURCE_ENGINE"] = resource_stall
            else:
                # resource_reason from calculate_op_timing is the key
                stall_breakdown[resource_reason] = resource_stall
        
        total_stall = sum(stall_breakdown.values())

        if actual_start < earliest_start_time:
            earliest_start_time = actual_start
            best_op_item = op_item
            op_details = {
                'start': actual_start, 'end': actual_end, 'engine': engine_type,
                'stall_cycles': total_stall,
                'stall_breakdown': stall_breakdown,
                'cycle_breakdown': breakdown
            }

    if best_op_item:
        op = best_op_item[0] if is_tight else best_op_item
        ticket = best_op_item[1] if is_tight else None

        engine_pool = engine_pools[op_details['engine']]
        engine_idx = engine_pool.index(min(engine_pool))
        engine_pool[engine_idx] = op_details['end']

        schedule.append(ScheduleItem(
            op=op,
            start_cycle=op_details['start'],
            end_cycle=op_details['end'],
            engine=f"{op_details['engine']}{engine_idx}",
            stall_cycles=op_details['stall_cycles'],
            stall_breakdown=op_details['stall_breakdown'],
            cycle_breakdown=op_details['cycle_breakdown']
        ))

        q_to_process.remove(best_op_item)

        heapq.heappush(event_queue, Event(time=op_details['end'], type='OP_COMPLETE', item=(op, op_details['end'], ticket)))
        heapq.heappush(event_queue, Event(time, 'CHECK_SCHED'))

def calculate_op_timing(op: NPUOp, start_cycle: int, config: SimConfig, resources: Dict[str, Any]) -> Tuple[int, int, str, Dict[str, int]]:
    dram_banks: DramBankTracker = resources['dram_banks']
    spm: BankTracker = resources['spm_banks']
    stall_reason = "NONE"
    breakdown = {}
    
    if op.opcode in ('LOAD', 'STORE'):
        if not op.inputs and not op.outputs:
            return start_cycle, start_cycle + 1, "NONE", breakdown
        # Assuming the first tensor is the one being transferred
        tensor = op.inputs[0] if op.opcode == 'LOAD' else op.outputs[0]
        byte_size = DTYPE_MAP.get(tensor.dtype, 1)
        num_bytes = tensor.num_elements * byte_size
        dram_end_cycle, stall_reason = dram_banks.book_transfer(start_cycle, tensor.address, num_bytes)
        actual_start = start_cycle 
        return actual_start, dram_end_cycle, stall_reason, breakdown
    elif op.opcode in ("MatMul", "Conv"):
        if len(op.inputs) < 2 or len(op.outputs) < 1:
            return start_cycle, start_cycle + 1, "NONE", breakdown
        # Simplified sequential model: Load A, Load B, Compute, Store C
        tensor_a, tensor_b = op.inputs[0], op.inputs[1]
        tensor_c = op.outputs[0]

        bytes_in_a = tensor_a.num_elements * DTYPE_MAP.get(tensor_a.dtype, 1)
        bytes_in_b = tensor_b.num_elements * DTYPE_MAP.get(tensor_b.dtype, 1)
        
        load_a_end, reason_a = dram_banks.book_transfer(start_cycle, tensor_a.address, bytes_in_a)
        load_b_end, reason_b = dram_banks.book_transfer(load_a_end, tensor_b.address, bytes_in_b)
        
        inputs_ready_cycle = load_b_end
        stall_reason = reason_a if reason_a != "NONE" else reason_b

        breakdown = calculate_systolic_array_cycles(
            tile_m=op.args.get('tile_m', 128),
            tile_n=op.args.get('tile_n', 128),
            tile_k=op.args.get('tile_k', 128),
            array_height=config.systolic_array_height, 
            array_width=config.systolic_array_width
        )
        compute_cycles = breakdown['total']

        num_banks_needed = config.spm_banks // 2
        compute_start_cycle = spm.find_earliest_free_slot(inputs_ready_cycle, compute_cycles, num_banks_needed)
        if compute_start_cycle > inputs_ready_cycle:
            stall_reason = "RESOURCE_SPM"
        
        compute_end_cycle = compute_start_cycle + compute_cycles
        bytes_out_c = tensor_c.num_elements * DTYPE_MAP.get(tensor_c.dtype, 1)
        store_end_cycle, reason_c = dram_banks.book_transfer(compute_end_cycle, tensor_c.address, bytes_out_c)
        
        # The most recent stall reason is the most relevant for the resource stall breakdown
        if reason_c != "NONE":
            stall_reason = reason_c

        return compute_start_cycle, store_end_cycle, stall_reason, breakdown
    elif op.opcode in ("GELU", "Softmax", "Add", "Mul", "LayerNorm"):
        num_elements = op.args.get('num_elements', 2048)
        compute_cycles = max(5, num_elements // 16)
        num_banks_needed = 2
        compute_start_cycle = spm.find_earliest_free_slot(start_cycle, compute_cycles, num_banks_needed)
        compute_end_cycle = compute_start_cycle + compute_cycles
        if compute_start_cycle > start_cycle:
            stall_reason = "RESOURCE_SPM"
        breakdown = {'total': compute_cycles, 'compute': compute_cycles}
        return compute_start_cycle, compute_end_cycle, stall_reason, breakdown
    else:
        return start_cycle, start_cycle + 1, "NONE", breakdown

def get_engine_for_op(op: NPUOp) -> str:
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

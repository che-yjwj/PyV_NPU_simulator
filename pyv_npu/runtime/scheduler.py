
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, Set
import heapq
from ..config import SimConfig
from ..isa.npu_ir import Program, NPUOp
from .resources import BandwidthTracker, BankTracker

@dataclass
class ScheduleItem:
    op: NPUOp
    start_cycle: int
    end_cycle: int
    engine: str  # "TE", "VE", "DMA"

# --- L2 Event-Driven Scheduler ---

@dataclass(order=True)
class Event:
    time: int
    type: str = field(compare=False)
    item: Any = field(compare=False, default=None)

def event_driven_schedule(p: Program, config: SimConfig) -> List[ScheduleItem]:
    """
    L2 Event-Driven Scheduler with Resource Contention.
    Models SPM bank conflicts and DRAM bandwidth limitations.
    """
    # 1. Initialization
    time = 0
    schedule: List[ScheduleItem] = []
    event_queue: List[Event] = []

    # Resource trackers
    dram = BandwidthTracker("dram", config)
    spm_banks = BankTracker(config)
    
    # Engine availability (next free time)
    te_free_time = [0] * config.te
    ve_free_time = [0] * config.ve
    dma_free_time = [0] * config.dma_channels

    # Dependency tracking
    tensor_ready_time: Dict[str, int] = {t.name: 0 for t in p.inputs}
    tensor_ready_time.update({t.name: 0 for t in p.initializers})
    completed_tickets: Set[int] = set()

    # Queues for ops
    cpu_op_queue = p.ops.copy()
    npu_op_queue: List[NPUOp] = []
    
    # Initial event to start scheduling
    heapq.heappush(event_queue, Event(time=0, type='CHECK_SCHED'))

    # 2. Main Simulation Loop
    while event_queue:
        event = heapq.heappop(event_queue)
        time = event.time

        if event.type == 'OP_COMPLETE':
            op, end_cycle, ticket = event.item
            for out_tensor in op.outputs:
                tensor_ready_time[out_tensor.name] = end_cycle
            if ticket is not None:
                completed_tickets.add(ticket)
            # After an op completes, check for new scheduling opportunities
            heapq.heappush(event_queue, Event(time=end_cycle, type='CHECK_SCHED'))

        elif event.type == 'CHECK_SCHED':
            # Try to schedule any ready ops
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
                resource_trackers={'dram': dram, 'spm_banks': spm_banks}
            )

    # Sort final schedule by start time for reporting
    schedule.sort(key=lambda x: x.start_cycle)
    return schedule

def run_scheduler_pass(cpu_op_queue, npu_op_queue, time, config, schedule, event_queue, tensor_ready_time, completed_tickets, engine_pools, resource_trackers):
    """A single pass of the scheduler to find and schedule the next available op."""
    
    op_queue_to_schedule = None

    if config.mode == 'tight':
        # --- Tight-Coupled Mode Scheduling ---
        # 1. Process CPU-side instructions
        if cpu_op_queue:
            cpu_op = cpu_op_queue[0]
            if cpu_op.opcode == 'ENQCMD_T':
                enqcmd = cpu_op.args['enqcmd']
                npu_op_queue.append((enqcmd.npu_op_desc, enqcmd.ticket))
                cpu_op_queue.pop(0)
                heapq.heappush(event_queue, Event(time, 'CHECK_SCHED'))
                # Don't schedule NPU ops in the same pass as a CPU op
                return 
            elif cpu_op.opcode == 'TWAIT':
                twait = cpu_op.args['twait']
                if twait.ticket in completed_tickets:
                    cpu_op_queue.pop(0)
                    heapq.heappush(event_queue, Event(time, 'CHECK_SCHED'))
                    return # Move to next CPU op
                # If ticket is not ready, stall CPU but continue to schedule NPU ops.

        # 2. Process NPU-side instructions (from the npu_op_queue)
        op_queue_to_schedule = npu_op_queue
        is_tight_mode_npu_pass = True
    else:
        # --- Loose-Coupled Mode Scheduling ---
        op_queue_to_schedule = cpu_op_queue
        is_tight_mode_npu_pass = False

    # Find the next op that is ready to run from the selected queue
    best_op_to_schedule = None
    earliest_op_start_time = float('inf')
    op_timing_details = {}

    # Use a consistent name for the item from the queue
    for op_item in list(op_queue_to_schedule):
        if is_tight_mode_npu_pass:
            op, ticket = op_item
        else:
            op, ticket = op_item, None

        dep_ready = all(t.name in tensor_ready_time for t in op.inputs)
        if not dep_ready:
            continue

        dep_ready_time = max((tensor_ready_time.get(t.name, 0) for t in op.inputs), default=time)
        engine_type = get_engine_for_op(op)
        engine_pool = engine_pools.get(engine_type)
        if not engine_pool: continue

        engine_free_time = min(engine_pool)
        tentative_start_time = max(dep_ready_time, engine_free_time)
        actual_start, actual_end = calculate_op_timing(op, tentative_start_time, config, resource_trackers)

        if actual_start < earliest_op_start_time:
            earliest_op_start_time = actual_start
            best_op_to_schedule = op_item
            op_timing_details = {'start': actual_start, 'end': actual_end, 'engine_type': engine_type}

    if best_op_to_schedule:
        if is_tight_mode_npu_pass:
            op, ticket = best_op_to_schedule
        else:
            op, ticket = best_op_to_schedule, None

        start_cycle = op_timing_details['start']
        end_cycle = op_timing_details['end']
        engine_type = op_timing_details['engine_type']
        engine_pool = engine_pools[engine_type]

        engine_idx = engine_pool.index(min(engine_pool))
        engine_pool[engine_idx] = end_cycle

        op_queue_to_schedule.remove(best_op_to_schedule)

        schedule.append(ScheduleItem(op, start_cycle, end_cycle, f"{engine_type}{engine_idx}"))
        heapq.heappush(event_queue, Event(time=end_cycle, type='OP_COMPLETE', item=(op, end_cycle, ticket)))
        heapq.heappush(event_queue, Event(time, 'CHECK_SCHED'))


def calculate_op_timing(op: NPUOp, start_cycle: int, config: SimConfig, resources: Dict[str, Any]) -> Tuple[int, int]:
    """
    Calculates the true start and end cycle for an op, considering resource contention.
    Returns (actual_start_cycle, actual_end_cycle).
    """
    dram: BandwidthTracker = resources['dram']
    spm: BankTracker = resources['spm_banks']
    
    # --- Data Movement Timing ---
    if op.opcode in ('LOAD', 'STORE'):
        # For now, assume LOAD/STORE ops are implicitly handled by compute ops.
        # A more detailed model would have explicit DMA ops.
        # This is a placeholder to show where the logic would go.
        num_bytes = op.args.get('num_bytes', 0)
        dram_end_cycle = dram.book_transfer(start_cycle, num_bytes)
        return start_cycle, dram_end_cycle

    # --- Compute Timing (including implicit data movement) ---
    elif op.opcode in ("MatMul", "Conv"):
        # Simplified model: Load A, Load B, Compute, Store C
        m, n, k = op.args.get('tile_m', 128), op.args.get('tile_n', 128), op.args.get('tile_k', 128)
        
        # 1. Load Inputs
        bytes_in_a = (m * k * 2) # float16
        bytes_in_b = (k * n * 2) # float16
        
        # Assume DMA can fetch in parallel if channels > 1
        load_a_end = dram.book_transfer(start_cycle, bytes_in_a)
        load_b_end = dram.book_transfer(start_cycle, bytes_in_b)
        
        # Inputs must be fully loaded before compute can start
        inputs_ready_cycle = max(load_a_end, load_b_end)

        # 2. Compute
        # Placeholder for actual compute cycle estimation
        total_macs = m * n * k
        compute_cycles = total_macs // (config.ve * 1) # Highly simplified
        compute_cycles = max(1, compute_cycles)

        # SPM banks are needed during compute
        # Assume a compute op needs half the banks
        num_banks_needed = config.spm_banks // 2
        compute_start_cycle = spm.find_earliest_free_slot(inputs_ready_cycle, compute_cycles, num_banks_needed)
        compute_end_cycle = compute_start_cycle + compute_cycles

        # 3. Store Output
        bytes_out_c = (m * n * 2) # float16
        store_end_cycle = dram.book_transfer(compute_end_cycle, bytes_out_c)

        return start_cycle, store_end_cycle

    elif op.opcode in ("GELU", "Softmax", "Add", "Mul", "LayerNorm"):
        # Simplified model for element-wise ops, assuming in-SPM operation
        num_elements = op.args.get('num_elements', 2048)
        # Assume 16 elements per cycle on a VE
        compute_cycles = max(1, num_elements // 16)
        
        # Assume element-wise ops need fewer banks
        num_banks_needed = 2
        compute_start_cycle = spm.find_earliest_free_slot(start_cycle, compute_cycles, num_banks_needed)
        compute_end_cycle = compute_start_cycle + compute_cycles
        return compute_start_cycle, compute_end_cycle

    else: # Default for unknown ops
        return start_cycle, start_cycle + 1

def get_engine_for_op(op: NPUOp) -> str:
    """Determines the engine type for an NPUOp."""
    if op.opcode in ("MatMul", "Conv"):
        return "TE"
    elif op.opcode in ("GELU", "Softmax", "Add", "Mul", "LayerNorm"):
        return "VE"
    else: # LOAD, STORE, etc.
        return "DMA"

def simple_greedy_schedule(p: Program) -> List[ScheduleItem]:
    """Original L0/L1 scheduler for comparison."""
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

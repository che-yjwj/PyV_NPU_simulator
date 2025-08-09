
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple
import heapq

from ..config import SimConfig
from ..isa.npu_ir import Program, NPUOp, Tensor

@dataclass
class ScheduleItem:
    op: NPUOp
    start_cycle: int
    end_cycle: int
    engine: str  # "TE", "VE", "DMA", "CPU"

# --- L2 Event-Driven Scheduler ---

@dataclass(order=True)
class Event:
    time: int
    # Use a field for the item since NPUOp is not comparable
    item: Any = field(compare=False)

def event_driven_schedule(p: Program, config: SimConfig) -> List[ScheduleItem]:
    """
    An event-driven scheduler that models engine contention and data dependencies.
    - Manages TE, VE, and DMA engines.
    - Ops can only run when their dependencies are met and an engine is free.
    - Time advances based on the next available event.
    """
    # 1. Initialization
    schedule: List[ScheduleItem] = []
    event_queue: List[Event] = []
    
    # Engine availability (time when the engine becomes free)
    te_free_time = [0] * config.te
    ve_free_time = [0] * config.ve
    dma_free_time = [0] * config.dma_channels

    # Data availability (time when a tensor is ready)
    tensor_ready_time: Dict[str, int] = {}

    # Op readiness (ops waiting for dependencies)
    op_queue = p.ops.copy()
    
    # 2. Main Simulation Loop
    while op_queue or event_queue:
        # Find the next op that can be scheduled
        next_op_to_schedule = -1
        best_start_time = float('inf')

        for i, op in enumerate(op_queue):
            # Check data dependencies
            dep_ready_time = 0
            for inp in op.inputs:
                dep_ready_time = max(dep_ready_time, tensor_ready_time.get(inp.name, 0))

            # Check engine availability and find the earliest start time
            start_time = dep_ready_time
            eng_type, eng_idx = get_engine_for_op(op)

            if eng_type == "TE":
                free_times = te_free_time
            elif eng_type == "VE":
                free_times = ve_free_time
            else: # DMA/CPU for now
                free_times = dma_free_time

            # Find the earliest available engine of the required type
            earliest_engine_free_time = min(free_times)
            start_time = max(start_time, earliest_engine_free_time)

            if start_time < best_start_time:
                best_start_time = start_time
                next_op_to_schedule = i

        # If no op is ready, advance time based on the event queue
        if next_op_to_schedule == -1:
            if not event_queue: break # Should not happen if op_queue is not empty
            event = heapq.heappop(event_queue)
            # An engine has become free, re-evaluate op readiness in the next loop
            continue

        # 3. Schedule the selected op
        op = op_queue.pop(next_op_to_schedule)
        start_time = best_start_time
        
        eng_type, eng_idx = get_engine_for_op(op)
        duration = estimate_op_duration(op, config)
        end_time = start_time + duration

        # Find the specific engine to use
        if eng_type == "TE":
            engine_pool = te_free_time
        elif eng_type == "VE":
            engine_pool = ve_free_time
        else: # DMA/CPU
            engine_pool = dma_free_time
        
        # Find first available engine and assign the op to it
        for i in range(len(engine_pool)):
            if engine_pool[i] <= start_time:
                engine_pool[i] = end_time
                eng_idx = i
                break
        
        full_eng_name = f"{eng_type}{eng_idx}"
        schedule.append(ScheduleItem(op, start_time, end_time, full_eng_name))

        # Update tensor readiness and add an event for when the op completes
        for out in op.outputs:
            tensor_ready_time[out.name] = end_time
        
        heapq.heappush(event_queue, Event(time=end_time, item=op.name))

    return schedule

def get_engine_for_op(op: NPUOp) -> Tuple[str, int]:
    """Determines the engine type and an index for an NPUOp."""
    if op.opcode in ("MatMul", "Conv"):
        return "TE", 0
    elif op.opcode in ("GELU", "Softmax", "Add", "Mul", "LayerNorm"):
        return "VE", 0
    else: # Data movement or other ops
        return "DMA", 0

def estimate_op_duration(op: NPUOp, config: SimConfig) -> int:
    """Estimates the execution duration of an op in clock cycles."""
    # L2: Basic placeholder estimates, could be a detailed model
    if op.opcode in ("MatMul", "Conv"):
        # Simplified: proportional to MACs, divided by throughput
        # This is a huge simplification. A real model would be complex.
        m, n, k = 128, 128, 128 # Placeholder dimensions
        if op.opcode == "MatMul":
            m, k = op.inputs[0].shape
            k, n = op.inputs[1].shape
        
        total_macs = m * n * k
        # Assuming 1 MAC per cycle per VE, and TEs feed VEs
        # This is where a real performance model would go.
        cycles = total_macs // (config.ve * 1) 
        return cycles if cycles > 0 else 100 # return 100 cycles minimum
        
    elif op.opcode in ("GELU", "Softmax", "Add", "Mul", "LayerNorm"):
        # Proportional to number of elements
        num_elements = op.inputs[0].num_elements()
        # Simplified throughput: elements per cycle
        cycles = num_elements // 16 
        return cycles if cycles > 0 else 20 # return 20 cycles minimum
    else:
        # For DMA ops, duration depends on data size and bandwidth
        # Placeholder:
        return 50

def simple_greedy_schedule(p: Program) -> List[ScheduleItem]:
    """Assign MatMul to TE; elementwise to VE; sequential, no overlap."""
    time = 0
    schedule: List[ScheduleItem] = []
    for op in p.ops:
        if op.opcode in ("MatMul", "Conv"):
            dur = 100  # placeholder cycles
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

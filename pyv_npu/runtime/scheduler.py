
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
    ticket: int = -1

# --- L2 Event-Driven Scheduler ---

@dataclass(order=True)
class Event:
    time: int
    # Event type, e.g., 'CPU_DISPATCH', 'NPU_OP_COMPLETE'
    type: str = field(compare=False)
    # Payload associated with the event
    item: Any = field(compare=False, default=None)

def event_driven_schedule(p: Program, config: SimConfig) -> List[ScheduleItem]:
    """A more accurate event-driven scheduler.
    
    The simulation proceeds by jumping from one event to the next.
    This allows for proper modeling of parallel execution on multiple engines.
    """
    # 1. Initialization
    time = 0
    schedule: List[ScheduleItem] = []
    event_queue: List[Event] = []

    # Engine availability is tracked by their next free time
    te_free_time = [0] * config.te
    ve_free_time = [0] * config.ve
    dma_free_time = [0] * config.dma_channels
    cpu_free_time = [0] # Single CPU core

    # Resource and dependency tracking
    tensor_ready_time: Dict[str, int] = {}
    # Initialize tensor_ready_time with graph inputs and initializers
    for inp_tensor in p.inputs:
        tensor_ready_time[inp_tensor.name] = 0
    for init_tensor in p.initializers:
        tensor_ready_time[init_tensor.name] = 0

    ticket_completed_time: Dict[int, int] = {}
    
    # Queues for ops waiting to be processed or scheduled
    # In tight mode, this is the stream of instructions from the CPU
    # In loose mode, this queue is empty, and npu_op_queue is pre-populated
    cpu_op_queue = p.ops.copy()
    npu_op_queue: List[NPUOp] = []

    if config.mode == 'loose':
        npu_op_queue = cpu_op_queue
        cpu_op_queue = []
        # For loose mode, all ops can be scheduled immediately
        for op in npu_op_queue:
            heapq.heappush(event_queue, Event(time=0, type='NPU_OP_READY', item=op))
    else: # tight mode
        # Start with a single event to kick off the CPU dispatch
        heapq.heappush(event_queue, Event(time=0, type='CPU_DISPATCH'))

    # 2. Main Simulation Loop
    while event_queue:
        event = heapq.heappop(event_queue)
        time = event.time

        # --- Event: CPU is ready to dispatch next instruction ---
        if event.type == 'CPU_DISPATCH':
            if not cpu_op_queue: continue

            op = cpu_op_queue[0]
            is_ready = False
            
            # Check dependencies for the CPU instruction
            if op.opcode == 'TWAIT':
                ticket = op.args['inst'].ticket
                if ticket in ticket_completed_time:
                    is_ready = True
            else: # ENQCMD_T and other future CPU ops are always ready
                is_ready = True

            if is_ready:
                cpu_op_queue.pop(0)
                start_time = max(time, cpu_free_time[0])
                duration = estimate_op_duration(op, config)
                end_time = start_time + duration
                cpu_free_time[0] = end_time
                schedule.append(ScheduleItem(op, start_time, end_time, "CPU0"))

                if op.opcode == 'ENQCMD_T':
                    inst = op.args['inst']
                    # Add the NPU op to the queue of ops waiting for an engine
                    npu_op_queue.append(inst.npu_op_desc)
                    # Tag the op with its ticket
                    inst.npu_op_desc.args['ticket'] = inst.ticket
                    # Trigger an event to check NPU scheduling
                    heapq.heappush(event_queue, Event(time=end_time, type='CHECK_NPU_SCHED'))
                
                # Schedule the next CPU dispatch event
                heapq.heappush(event_queue, Event(time=end_time, type='CPU_DISPATCH'))

        # --- Event: An NPU op has finished ---
        elif event.type == 'NPU_OP_COMPLETE':
            op, engine_type, engine_idx = event.item
            # Update tensor and ticket readiness
            for out in op.outputs:
                tensor_ready_time[out.name] = time
            ticket = op.args.get('ticket', -1)
            if ticket != -1:
                ticket_completed_time[ticket] = time
                # If a TWAIT was waiting, the CPU might be unblocked
                heapq.heappush(event_queue, Event(time=time, type='CPU_DISPATCH'))
            
            # Trigger a check for more NPU work
            heapq.heappush(event_queue, Event(time=time, type='CHECK_NPU_SCHED'))

        # --- Event: Check if any queued NPU ops can be scheduled ---
        elif event.type in ('CHECK_NPU_SCHED', 'NPU_OP_READY'):
            # Iterate over a copy as we may modify the queue
            for op in list(npu_op_queue):
                # Check data dependencies
                dep_ready = all(t.name in tensor_ready_time for t in op.inputs)
                if not dep_ready: continue
                
                dep_time = 0
                for t in op.inputs:
                    dep_time = max(dep_time, tensor_ready_time.get(t.name, 0))

                # Check engine availability
                eng_type, _ = get_engine_for_op(op)
                if eng_type == "TE": engine_pool = te_free_time
                elif eng_type == "VE": engine_pool = ve_free_time
                else: engine_pool = dma_free_time

                # Find the best engine to schedule this op on
                best_engine_idx = -1
                earliest_start_time = float('inf')
                
                for i, free_time in enumerate(engine_pool):
                    current_engine_start_time = max(time, dep_time, free_time)
                    if current_engine_start_time < earliest_start_time:
                        earliest_start_time = current_engine_start_time
                        best_engine_idx = i
                
                if best_engine_idx != -1: # If a suitable engine was found
                    # Found a free slot, schedule the op
                    npu_op_queue.remove(op)
                    duration = estimate_op_duration(op, config)
                    end_time = earliest_start_time + duration
                    engine_pool[best_engine_idx] = end_time
                    
                    ticket = op.args.get('ticket', -1)
                    schedule.append(ScheduleItem(op, earliest_start_time, end_time, f"{eng_type}{best_engine_idx}", ticket))
                    
                    # Schedule the completion event for this op
                    item = (op, eng_type, best_engine_idx)
                    heapq.heappush(event_queue, Event(time=end_time, type='NPU_OP_COMPLETE', item=item))
                    # After scheduling, re-check if more NPU ops can be scheduled
                    heapq.heappush(event_queue, Event(time=end_time, type='CHECK_NPU_SCHED'))
                    break # Break from inner loop to re-evaluate the npu_op_queue after a schedule

    return schedule

def get_engine_for_op(op: NPUOp) -> Tuple[str, int]:
    """Determines the engine type and an index for an NPUOp."""
    if op.opcode in ("MatMul", "Conv"):
        return "TE", 0
    elif op.opcode in ("GELU", "Softmax", "Add", "Mul", "LayerNorm"):
        return "VE", 0
    elif op.opcode in ("ENQCMD_T", "TWAIT", "TBAR", "TSTAT"):
        return "CPU", 0
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
    elif op.opcode in ("ENQCMD_T", "TWAIT", "TBAR", "TSTAT"):
        return 1 # Assume CPU custom instructions take 1 cycle
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


from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple
import heapq
from ..bridge.mem import NPUControlMemory
from ..config import SimConfig
from ..isa.npu_ir import Program, NPUOp, Tensor
from pyv.log import logger


@dataclass
class NpuDescriptor:
    """Represents the 64-byte descriptor read from memory."""
    opcode: int = 0
    flags: int = 0
    in0_addr: int = 0
    in1_addr: int = 0
    out_addr: int = 0
    tile_m: int = 0
    tile_n: int = 0
    tile_k: int = 0
    ticket: int = 0


class Scheduler:
    """Manages the NPU simulation, receiving jobs from the RISC-V core."""
    def __init__(self, config: SimConfig, memory: NPUControlMemory):
        self.config = config
        self.memory = memory
        self.running = False
        logger.info("Scheduler initialized.")

    def _parse_descriptor(self, addr: int) -> NpuDescriptor:
        """Reads 64 bytes from memory and parses it into a descriptor object."""
        desc = NpuDescriptor()
        desc.opcode    = self.memory._read(addr + 0, 4)
        desc.flags     = self.memory._read(addr + 4, 4)
        in0_lo         = self.memory._read(addr + 8, 4)
        in0_hi         = self.memory._read(addr + 12, 4)
        desc.in0_addr  = (in0_hi << 32) | in0_lo
        in1_lo         = self.memory._read(addr + 16, 4)
        in1_hi         = self.memory._read(addr + 20, 4)
        desc.in1_addr  = (in1_hi << 32) | in1_lo
        out_lo         = self.memory._read(addr + 24, 4)
        out_hi         = self.memory._read(addr + 28, 4)
        desc.out_addr  = (out_hi << 32) | out_lo
        desc.tile_m    = self.memory._read(addr + 32, 4)
        desc.tile_n    = self.memory._read(addr + 36, 4)
        desc.tile_k    = self.memory._read(addr + 40, 4)
        desc.ticket    = self.memory._read(addr + 60, 4)
        logger.info(f"Parsed Descriptor at {addr:08X}: {desc}")
        return desc

    def handle_doorbell(self, new_tail: int):
        """Entry point for jobs submitted from the RISC-V core via MMIO."""
        logger.info(f"Scheduler: Doorbell handled with new tail index {new_tail}.")

        head = self.memory.queue_head
        base = self.memory.queue_base
        size = self.config.queue_size
        
        if new_tail == head:
            logger.warning("Doorbell rung, but no new jobs (head == tail).")
            return

        logger.info(f"Processing ring buffer from head={head} to tail={new_tail}")

        ops_to_schedule = []
        
        current = head
        while current != new_tail:
            # Assume 64-byte descriptors
            desc_addr = base + (current * 64)
            logger.info(f"Reading descriptor at index {current}, address {desc_addr:08X}")
            
            desc = self._parse_descriptor(desc_addr)
            
            # Create an NPU op based on the descriptor.
            if desc.opcode == 0:
                op_name = "MatMul"
            elif desc.opcode == 1:
                op_name = "GELU"
            else:
                logger.warning(f"Skipping unknown Opcode: {desc.opcode}")
                current = (current + 1) % size
                continue

            # Create tensor and op objects
            input_tensor0 = Tensor(name=f"t_in0_{desc.ticket}", shape=[desc.tile_m, desc.tile_k], dtype="float16", addr=desc.in0_addr)
            input_tensor1 = Tensor(name=f"t_in1_{desc.ticket}", shape=[desc.tile_k, desc.tile_n], dtype="float16", addr=desc.in1_addr)
            output_tensor = Tensor(name=f"t_out_{desc.ticket}", shape=[desc.tile_m, desc.tile_n], dtype="float16", addr=desc.out_addr)
            op = NPUOp(opcode=op_name, inputs=[input_tensor0, input_tensor1], outputs=[output_tensor], args=vars(desc))
            ops_to_schedule.append(op)

            current = (current + 1) % size

        if not ops_to_schedule:
            logger.warning("No valid operations found in the given range.")
            return

        # Create a program for the collected ops.
        program = Program(ops=ops_to_schedule, inputs=[], outputs=[], initializers=[])

        # Run the event-driven scheduler.
        logger.info(f"Running event-driven scheduler for {len(ops_to_schedule)} new operations...")
        schedule = event_driven_schedule(program, self.config)

        if schedule:
            logger.info(f"Scheduling complete. Total {len(schedule)} ops scheduled.")
            logger.info(f"Estimated end cycle: {schedule[-1].end_cycle}")
            # Update head pointer and set interrupt status
            self.memory.queue_head = new_tail
            logger.info(f"Updated queue head to {new_tail}")
            self.memory.irq_status |= 1 # Set bit 0 to indicate completion
            logger.info(f"Set IRQ_STATUS bit 0. New status: {self.memory.irq_status:08X}")
        else:
            logger.warning("No operations were scheduled.")

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
        npu_op_queue = p.ops.copy()
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
        if op.opcode == "MatMul" and op.inputs:
            m, k = op.inputs[0].shape
            if len(op.inputs) > 1:
                k, n = op.inputs[1].shape
        
        total_macs = m * n * k
        # Assuming 1 MAC per cycle per VE, and TEs feed VEs
        # This is where a real performance model would go.
        cycles = total_macs // (config.ve * 1) 
        return cycles if cycles > 0 else 100 # return 100 cycles minimum
        
    elif op.opcode in ("GELU", "Softmax", "Add", "Mul", "LayerNorm"):
        # Proportional to number of elements
        num_elements = op.inputs[0].num_elements() if op.inputs else 16*20
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

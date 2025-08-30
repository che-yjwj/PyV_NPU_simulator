from __future__ import annotations
from pyv_npu.runtime.scheduler import event_driven_schedule
from pyv_npu.isa.npu_ir import Program, NPUOp, Tensor
from pyv_npu.config import SimConfig

def test_io_buffer_backpressure():
    # Create a program that will cause backpressure
    # 1. Load a tensor that almost fills the buffer
    # 2. A compute op that depends on the loaded tensor
    # 3. A second load that would overflow the buffer
    # 4. A store op that empties the buffer

    # Config with a small IO buffer (2KB)
    config = SimConfig()
    config.io_buffer_size_kb = 2

    # Tensors (1.5KB each)
    t_in1 = Tensor(name="input1", shape=[1, 768], dtype="float16", address=0x1000)
    t_loaded1 = Tensor(name="loaded1", shape=[1, 768], dtype="float16")
    t_in2 = Tensor(name="input2", shape=[1, 768], dtype="float16", address=0x2000)
    t_loaded2 = Tensor(name="loaded2", shape=[1, 768], dtype="float16")
    t_compute_out = Tensor(name="compute_out", shape=[1, 1], dtype="float16")
    t_stored = Tensor(name="stored", shape=[1, 768], dtype="float16", address=0x3000)

    # Ops
    load_op1 = NPUOp(name="load1", opcode="LOAD", inputs=[t_in1], outputs=[t_loaded1])
    load_op2 = NPUOp(name="load2", opcode="LOAD", inputs=[t_in2], outputs=[t_loaded2])
    compute_op = NPUOp(name="compute", opcode="MatMul", inputs=[t_loaded1], outputs=[t_compute_out])
    store_op = NPUOp(name="store", opcode="STORE", inputs=[t_loaded1], outputs=[t_stored])

    program = Program(
        ops=[load_op1, load_op2, compute_op, store_op],
        inputs=[t_in1, t_in2],
        outputs=[t_stored]
    )

    schedule, stats = event_driven_schedule(program, config)

    # Find the ops in the schedule
    load1_item = next(s for s in schedule if s.op.name == "load1")
    load2_item = next(s for s in schedule if s.op.name == "load2")
    compute_item = next(s for s in schedule if s.op.name == "compute")
    store_item = next(s for s in schedule if s.op.name == "store")

    # 1. The compute op must start after the first load is complete.
    assert compute_item.start_cycle >= load1_item.end_cycle

    # 2. The store op must also start after the first load is complete.
    assert store_item.start_cycle >= load1_item.end_cycle

    # 3. The second load should be stalled because the first load filled the buffer.
    # The store should happen before the second load can start.
    assert store_item.end_cycle < load2_item.start_cycle

    # 4. The compute op should have been stalled waiting for the buffer.
    assert "RESOURCE_IO_BUFFER_EMPTY" in compute_item.stall_breakdown

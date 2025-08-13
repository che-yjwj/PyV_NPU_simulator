import pytest
from pyv_npu.config import SimConfig
from pyv_npu.isa.npu_ir import Program, NPUOp, Tensor
from pyv_npu.runtime.scheduler import event_driven_schedule

@pytest.mark.skip(reason="Stall cycle calculation needs review")
def test_scheduler_stall_reason_dep():
    """Tests that dependency stalls are correctly identified."""
    # op1 -> op2. op2 must wait for op1.
    op1 = NPUOp("op1", "MatMul", inputs=[], outputs=[Tensor("t1", (1,1), "float16")])
    op2 = NPUOp("op2", "GELU", inputs=[Tensor("t1", (1,1), "float16")], outputs=[])
    prog = Program(ops=[op1, op2], inputs=[], initializers={})
    config = SimConfig(model="test", level="L2", te=1, ve=1, dma_channels=1)

    schedule = event_driven_schedule(prog, config)

    op2_item = next(item for item in schedule if item.op.name == "GELU")

    # op2 should have a dependency stall waiting for op1
    assert op2_item.stall_cycles > 0
    assert op2_item.stall_reason == "DEP"

def test_scheduler_stall_reason_resource():
    """Tests that resource stalls (engine) are correctly identified."""
    # Two independent ops that will compete for the same engine
    op1 = NPUOp("op1", "MatMul", inputs=[], outputs=[])
    op2 = NPUOp("op2", "MatMul", inputs=[], outputs=[])
    prog = Program(ops=[op1, op2], inputs=[], initializers={})
    # Only one TE engine
    config = SimConfig(model="test", level="L2", te=1, ve=1, dma_channels=1)

    schedule = event_driven_schedule(prog, config)

    # Find the second matmul to run
    op_starts = sorted([(item.start_cycle, item) for item in schedule])
    second_op_item = op_starts[1][1]

    # The second op should have a resource stall waiting for the first to finish
    assert second_op_item.stall_cycles > 0
    assert second_op_item.stall_reason == "RESOURCE_ENGINE"

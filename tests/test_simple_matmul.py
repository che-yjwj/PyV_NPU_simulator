
from pyv_npu.config import SimConfig
from pyv_npu.isa.npu_ir import Program, NPUOp, Tensor
from pyv_npu.runtime.simulator import run as run_sim

def test_simple_matmul():
    t_in1 = Tensor("t_in1", (1, 128), "float16", address=0)
    t_in2 = Tensor("t_in2", (128, 128), "float16", address=4096)
    t_out = Tensor("t_out", (1, 128), "float16", address=8192)

    op1 = NPUOp("LOAD", "load1", inputs=[t_in1], outputs=[t_in1])
    op2 = NPUOp("LOAD", "load2", inputs=[t_in2], outputs=[t_in2])
    op3 = NPUOp("MatMul", "matmul1", inputs=[t_in1, t_in2], outputs=[t_out])
    op4 = NPUOp("STORE", "store1", inputs=[t_out], outputs=[t_out])

    prog = Program(ops=[op1, op2, op3, op4], inputs=[t_in1, t_in2], outputs=[t_out], initializers={})
    config = SimConfig(model="test", sim_level='CA_HYBRID', tc=1, vc=1, dma_channels=1)

    schedule, stats = run_sim(prog, config)

    print(schedule)

    engines_used = {it.engine for it in schedule}
    engine_types_used = {e.rstrip('0123456789') for e in engines_used}
    assert "TC" in engine_types_used

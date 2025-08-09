
from pyv_npu.ir.onnx_importer import load_onnx_as_model_ir
from pyv_npu.compiler.mapper import map_model_ir_to_npu_program
from pyv_npu.runtime.simulator import run

def test_smoke():
    g = load_onnx_as_model_ir("examples/tinyllama.onnx")
    p = map_model_ir_to_npu_program(g)
    rep = run(p)
    assert rep.total_cycles > 0
    assert set(rep.engine_util.keys()) == {"TE","VE","CPU"}

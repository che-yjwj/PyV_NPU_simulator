import pytest
from unittest.mock import MagicMock

# Mock the onnx library to avoid dependency on a real model file for the smoke test.
try:
    import onnx
    from onnx import helper
    from onnx import TensorProto
except ImportError:
    onnx = MagicMock()

from pyv_npu.config import SimConfig
from pyv_npu.ir.onnx_importer import load_onnx_as_model_ir
from pyv_npu.compiler.mapper import map_model_ir_to_npu_program
from pyv_npu.compiler.allocator import Allocator
from pyv_npu.runtime.simulator import run

@pytest.fixture
def mock_onnx_model():
    """Creates a mock ONNX model object with a single MatMul op."""
    nodes = [
        helper.make_node('MatMul', ['X', 'W'], ['Y'], name='matmul1'),
    ]
    graph_def = helper.make_graph(
        nodes,
        'test-graph-smoke',
        [helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, 128])],
        [helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, 128])],
        [helper.make_tensor('W', TensorProto.FLOAT, [128, 128], [1.0]*128*128)],
    )
    return helper.make_model(graph_def, producer_name='pytest-smoke')

def test_smoke(monkeypatch, mock_onnx_model):
    # Mock onnx.load to return our fake model
    monkeypatch.setattr(onnx, "load", lambda path: mock_onnx_model)

    onnx_model_path = "smoke_model.onnx"
    g = load_onnx_as_model_ir(onnx_model_path)
    p = map_model_ir_to_npu_program(g)
    
    config = SimConfig(model=onnx_model_path)

    schedule, stats = run(p, config)

    total_cycles = max((it.end_cycle for it in schedule), default=0)
    assert total_cycles > 0
    engines_used = {it.engine for it in schedule}
    engine_types_used = {e.rstrip('0123456789') for e in engines_used}
    assert "TC" in engine_types_used
    assert "DMA" in engine_types_used

def test_l2_simulator_smoke(monkeypatch, mock_onnx_model):
    monkeypatch.setattr(onnx, "load", lambda path: mock_onnx_model)

    onnx_model_path = "smoke_model.onnx"
    g = load_onnx_as_model_ir(onnx_model_path)
    p = map_model_ir_to_npu_program(g)
    
    config = SimConfig(model=onnx_model_path, sim_level='CA_HYBRID')
    
    allocator = Allocator(config.dram_base_address, config.dram_page_size)
    allocator.allocate(p)

    schedule, stats = run(p, config)
    total_cycles = max((it.end_cycle for it in schedule), default=0)
    assert total_cycles > 0
    engines_used = {it.engine for it in schedule}
    assert any(k.startswith("TC") for k in engines_used)
    assert "dram_collisions" in stats

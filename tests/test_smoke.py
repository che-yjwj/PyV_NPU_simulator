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
    """Creates a mock ONNX model object with multiple ops for testing."""
    # Create a graph with MatMul -> Erf -> MatMul -> Softmax to exercise different engines
    nodes = [
        helper.make_node('MatMul', ['X', 'W1'], ['Y1'], name='matmul1'),
        helper.make_node('Erf', ['Y1'], ['Y2'], name='erf1'),  # Using Erf as a stand-in for a VE-like op
        helper.make_node('MatMul', ['Y2', 'W2'], ['Y3'], name='matmul2'),
        helper.make_node('Softmax', ['Y3'], ['Y'], name='softmax1')
    ]
    # Define all intermediate tensors for the graph to be valid
    graph_def = helper.make_graph(
        nodes,
        'test-graph-smoke',
        [helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, 128])],
        [helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, 128])],
        [
            helper.make_tensor('W1', TensorProto.FLOAT, [128, 128], [1.0]*128*128),
            helper.make_tensor('W2', TensorProto.FLOAT, [128, 128], [1.0]*128*128)
        ],
        value_info=[
            helper.make_tensor_value_info('Y1', TensorProto.FLOAT, [1, 128]),
            helper.make_tensor_value_info('Y2', TensorProto.FLOAT, [1, 128]),
            helper.make_tensor_value_info('Y3', TensorProto.FLOAT, [1, 128]),
        ]
    )
    return helper.make_model(graph_def, producer_name='pytest-smoke')

def test_smoke(monkeypatch, mock_onnx_model):
    # Mock onnx.load to return our fake model
    monkeypatch.setattr(onnx, "load", lambda path: mock_onnx_model)

    onnx_model_path = "examples/tinyllama.onnx"
    g = load_onnx_as_model_ir(onnx_model_path)
    p = map_model_ir_to_npu_program(g)
    
    # Create a default SimConfig for the test
    config = SimConfig(model=onnx_model_path)

    schedule, stats = run(p, config)

    total_cycles = max((it.end_cycle for it in schedule), default=0)
    assert total_cycles > 0
    # The event-driven scheduler (default for IA_TIMING) names engines with an index.
    # We check that the set of engine types used matches our expectations.
    engines_used = {it.engine for it in schedule}
    engine_types_used = {e.rstrip('0123456789') for e in engines_used}
    assert engine_types_used == {"TC", "VC", "DMA"}

def test_l2_simulator_smoke(monkeypatch, mock_onnx_model):
    # Mock onnx.load to return our fake model
    monkeypatch.setattr(onnx, "load", lambda path: mock_onnx_model)

    onnx_model_path = "examples/tinyllama.onnx"
    g = load_onnx_as_model_ir(onnx_model_path)
    p = map_model_ir_to_npu_program(g)
    
    # Create a SimConfig for L2 simulation
    config = SimConfig(model=onnx_model_path, sim_level='CA_HYBRID')
    
    # Manually run allocator since we are not using the CLI
    allocator = Allocator(config.dram_base_address, config.dram_page_size)
    allocator.allocate(p)

    schedule, stats = run(p, config)
    total_cycles = max((it.end_cycle for it in schedule), default=0)
    assert total_cycles > 0
    engines_used = {it.engine for it in schedule}
    assert any(k.startswith("TC") for k in engines_used)
    assert any(k.startswith("VC") for k in engines_used)
    assert "dram_collisions" in stats

@pytest.mark.skip(reason="Tight mode scheduler is not yet implemented")
def test_tight_mode_l2_simulator_smoke(monkeypatch, mock_onnx_model):
    # Mock onnx.load to return our fake model
    monkeypatch.setattr(onnx, "load", lambda path: mock_onnx_model)

    onnx_model_path = "examples/tinyllama.onnx"
    g = load_onnx_as_model_ir(onnx_model_path)
    p = map_model_ir_to_npu_program(g, mode='tight')
    
    # Create a SimConfig for L2 simulation in tight mode
    config = SimConfig(model=onnx_model_path, sim_level='CA_HYBRID', mode='tight')

    schedule, stats = run(p, config)
    assert stats['total_cycles'] > 0
    assert any(k.startswith("TC") for k in stats['engine_utilization'].keys())
    assert any(k.startswith("VC") for k in stats['engine_utilization'].keys())
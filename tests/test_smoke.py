
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

    rep = run(p, config)
    assert rep.total_cycles > 0
    assert set(rep.engine_util.keys()) == {"TE", "VE", "CPU"}


import pytest
import os
from unittest.mock import MagicMock

# Mock the onnx library at the top level for all tests in this file
# This is a simple way to avoid the DecodeError if a real onnx model is not present.
# In a real-world scenario, we would use a small, valid ONNX file for testing.
try:
    import onnx
    from onnx import helper
    from onnx import TensorProto
except ImportError:
    onnx = MagicMock()


from pyv_npu.ir.onnx_importer import load_onnx_as_model_ir
from pyv_npu.ir.model_ir import Graph

@pytest.fixture
def mock_onnx_model():
    """Creates a mock ONNX model object for testing."""
    # Create a dummy graph with one node
    node_def = helper.make_node(
        'MatMul',
        ['X', 'W'],
        ['Y'],
        name='test_node'
    )
    # Create dummy tensors for inputs, outputs, and initializers
    graph_def = helper.make_graph(
        [node_def],
        'test-graph',
        [helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, 2])],
        [helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, 4])],
        [helper.make_tensor('W', TensorProto.FLOAT, [2, 4], [1.0]*8)]
    )
    # Create a dummy model
    model_def = helper.make_model(graph_def, producer_name='pytest')
    return model_def

def test_load_onnx_model(monkeypatch, mock_onnx_model):
    """
    Tests that an ONNX model can be successfully loaded into a Graph representation,
    using a mocked ONNX model to avoid dependency on a real model file.
    """
    onnx_model_path = "examples/tinyllama.onnx" # Path is still used, but its content is ignored

    # Mock onnx.load to return our fake model instead of reading the file
    mock_load = lambda path: mock_onnx_model
    monkeypatch.setattr(onnx, "load", mock_load)

    # Load the model using the importer
    graph = load_onnx_as_model_ir(onnx_model_path)

    # Assert that the loaded object is an instance of Graph
    assert isinstance(graph, Graph)

    # Assert that the graph contains the data from our mock model
    assert len(graph.nodes) == 1
    assert graph.nodes[0].name == 'test_node'
    assert graph.nodes[0].op_type == 'MatMul'
    assert 'X' in graph.inputs
    assert 'Y' in graph.outputs
    assert 'W' in graph.tensors
    assert graph.tensors['X'].shape == (1, 2)
    assert graph.tensors['W'].shape == (2, 4)

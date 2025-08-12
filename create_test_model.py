import onnx
from onnx import helper
from onnx import TensorProto

def create_model():
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
    model = helper.make_model(graph_def, producer_name='pytest-smoke')
    onnx.save(model, 'test_model.onnx')

if __name__ == '__main__':
    create_model()

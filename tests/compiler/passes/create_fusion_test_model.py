import onnx
from onnx import helper, TensorProto
import numpy as np

def create_fusion_test_model(path: str):
    """Creates an ONNX model with a MatMul -> Add -> GELU pattern."""
    # Define the input/output tensors
    A = helper.make_tensor_value_info('A', TensorProto.FLOAT, [1, 128, 64])
    Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, 128, 128])

    # Define the initializers (weights)
    B_init = helper.make_tensor('B', TensorProto.FLOAT, [1, 64, 128], np.random.randn(1, 64, 128).astype(np.float32).tobytes(), raw=True)
    Bias_init = helper.make_tensor('Bias', TensorProto.FLOAT, [1, 1, 128], np.random.randn(1, 1, 128).astype(np.float32).tobytes(), raw=True)

    # Define intermediate tensors for value_info
    matmul_out_vi = helper.make_tensor_value_info('matmul_out', TensorProto.FLOAT, [1, 128, 128])
    add_out_vi = helper.make_tensor_value_info('add_out', TensorProto.FLOAT, [1, 128, 128])

    # Define the nodes
    matmul_node = helper.make_node(
        'MatMul',
        inputs=['A', 'B'],
        outputs=['matmul_out'],
        name='MatMul_0'
    )

    add_node = helper.make_node(
        'Add',
        inputs=['matmul_out', 'Bias'],
        outputs=['add_out'],
        name='Add_0'
    )

    # Using Erf for GELU approximation as ONNX GELU can be complex
    gelu_node = helper.make_node(
        'Erf', # Using Erf as a stand-in for GELU for simplicity
        inputs=['add_out'],
        outputs=['Y'],
        name='Gelu_0'
    )

    # Create the graph
    graph_def = helper.make_graph(
        nodes=[matmul_node, add_node, gelu_node],
        name='FusionTestGraph',
        inputs=[A],
        outputs=[Y],
        initializer=[B_init, Bias_init],
        value_info=[matmul_out_vi, add_out_vi] # Explicitly add intermediate tensor info
    )

    # Create the model
    model_def = helper.make_model(graph_def, producer_name='fusion-test-model-generator')
    model_def.opset_import[0].version = 14

    onnx.save(model_def, path)
    print(f"Saved fusion test model to {path}")

if __name__ == "__main__":
    create_fusion_test_model("fusion_test_model.onnx")
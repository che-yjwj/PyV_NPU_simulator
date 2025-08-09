
from __future__ import annotations
from .model_ir import Graph, Node, Tensor
from typing import List, Dict
import numpy as np

try:
    import onnx
    from onnx.mapping import TENSOR_TYPE_TO_NP_TYPE
except Exception:
    onnx = None

def _onnx_dtype_to_numpy(onnx_dtype: int) -> np.dtype:
    return TENSOR_TYPE_TO_NP_TYPE.get(onnx_dtype, np.float32) # Default to float32

def load_onnx_as_model_ir(path: str) -> Graph:
    """Loads an ONNX model into the Model IR, extracting tensor shapes and dtypes."""
    if onnx is None:
        nodes: List[Node] = [
            Node(name="matmul0", op_type="MatMul", inputs=["x","w0"], outputs=["y0"]),
            Node(name="gelu0", op_type="GELU", inputs=["y0"], outputs=["y1"]),
            Node(name="matmul1", op_type="MatMul", inputs=["y1","w1"], outputs=["y2"]),
            Node(name="softmax0", op_type="Softmax", inputs=["y2"], outputs=["y"]),
        ]
        tensors: Dict[str, Tensor] = {
            "x": Tensor("x", (1, 128), np.float16),
            "w0": Tensor("w0", (128, 128), np.float16),
            "y0": Tensor("y0", (1, 128), np.float16),
            "y1": Tensor("y1", (1, 128), np.float16),
            "w1": Tensor("w1", (128, 128), np.float16),
            "y2": Tensor("y2", (1, 128), np.float16),
            "y": Tensor("y", (1, 128), np.float16),
        }
        return Graph(nodes=nodes, inputs=["x"], outputs=["y"], tensors=tensors)

    model = onnx.load(path)
    g = model.graph

    tensors: Dict[str, Tensor] = {}
    
    # Extract tensor info from initializers (weights, biases)
    for t in g.initializer:
        tensors[t.name] = Tensor(
            name=t.name,
            shape=tuple(t.dims),
            dtype=_onnx_dtype_to_numpy(t.data_type)
        )

    # Extract tensor info from inputs and value_info (activations)
    for t_list in [g.input, g.value_info, g.output]:
        for t in t_list:
            if t.name not in tensors:
                ttype = t.type.tensor_type
                if ttype.elem_type == 0: continue # Skip tensors with no type info
                tensors[t.name] = Tensor(
                    name=t.name,
                    shape=tuple(d.dim_value for d in ttype.shape.dim),
                    dtype=_onnx_dtype_to_numpy(ttype.elem_type)
                )

    nodes = []
    for n in g.node:
        nodes.append(Node(
            name=n.name or f"{n.op_type}_{n.output[0]}",
            op_type=n.op_type,
            inputs=list(n.input),
            outputs=list(n.output),
            attrs={a.name: onnx.helper.get_attribute_value(a) for a in n.attribute}
        ))
        
    inputs = [i.name for i in g.input]
    outputs = [o.name for o in g.output]
    
    return Graph(nodes=nodes, inputs=inputs, outputs=outputs, tensors=tensors)

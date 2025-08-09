
from __future__ import annotations
from .model_ir import Graph, Node
from typing import List
try:
    import onnx
except Exception:
    onnx = None

def load_onnx_as_model_ir(path: str) -> Graph:
    """Lightweight placeholder: if onnx is unavailable, synthesize a tiny graph."""
    if onnx is None:
        nodes: List[Node] = [
            Node(name="matmul0", op_type="MatMul", inputs=["x","w0"], outputs=["y0"]),
            Node(name="gelu0", op_type="GELU", inputs=["y0"], outputs=["y1"]),
            Node(name="matmul1", op_type="MatMul", inputs=["y1","w1"], outputs=["y2"]),
            Node(name="softmax0", op_type="Softmax", inputs=["y2"], outputs=["y"]),
        ]
        return Graph(nodes=nodes, inputs=["x"], outputs=["y"])
    # Minimal real import path
    model = onnx.load(path)
    nodes = []
    for n in model.graph.node:
        nodes.append(Node(
            name=n.name or n.op_type,
            op_type=n.op_type,
            inputs=list(n.input),
            outputs=list(n.output),
            attrs={a.name: onnx.helper.get_attribute_value(a) for a in n.attribute}
        ))
    inputs = [i.name for i in model.graph.input]
    outputs = [o.name for o in model.graph.output]
    return Graph(nodes=nodes, inputs=inputs, outputs=outputs)

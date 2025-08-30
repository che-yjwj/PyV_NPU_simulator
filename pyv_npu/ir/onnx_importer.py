
from __future__ import annotations
from .model_ir import Graph, Node, Tensor
from typing import List, Dict
import numpy as np

import onnx


def _onnx_dtype_to_numpy(onnx_dtype: int) -> np.dtype:
    # In newer ONNX versions, the mapping is done via a helper function.
    try:
        return onnx.helper.tensor_dtype_to_np_dtype(onnx_dtype)
    except KeyError:
        # Provide a fallback for unknown types, similar to the old .get()
        return np.float32 # Default to float32

def load_onnx_as_model_ir(path: str) -> Graph:
    """Loads an ONNX model into the Model IR, extracting tensor shapes and dtypes."""
    

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
        
    initializer_names = {t.name for t in g.initializer}
    inputs = [i.name for i in g.input if i.name not in initializer_names]
    outputs = [o.name for o in g.output]
    
    return Graph(nodes=nodes, inputs=inputs, outputs=outputs, tensors=tensors, initializers=list(initializer_names))


from __future__ import annotations
from typing import List
from ..ir.model_ir import Graph
from ..isa.npu_ir import Program, NPUOp, Tensor as NpuTensor

def map_model_ir_to_npu_program(g: Graph) -> Program:
    """Maps the Model IR Graph to an NPU Program, creating NPU Tensors."""
    ops: List[NPUOp] = []
    # Ensure all tensors are available in the graph's tensor map
    if not g.tensors:
        raise ValueError("Graph has no tensor information. Cannot map to NPU program.")

    for n in g.nodes:
        # Look up input and output tensors from the graph's tensor map
        try:
            inputs = [NpuTensor(
                name=t_name,
                shape=g.tensors[t_name].shape,
                dtype=g.tensors[t_name].dtype
            ) for t_name in n.inputs]
            
            outputs = [NpuTensor(
                name=t_name,
                shape=g.tensors[t_name].shape,
                dtype=g.tensors[t_name].dtype
            ) for t_name in n.outputs]
        except KeyError as e:
            print(f"Error mapping node {n.name}: Tensor {e} not found in graph tensor map.")
            # Optionally, decide how to handle this: skip node, use placeholder, or raise error
            raise

        ops.append(NPUOp(
            opcode=n.op_type,
            name=n.name,
            args=n.attrs, # Pass attributes through
            inputs=inputs,
            outputs=outputs
        ))
    return Program(ops=ops)

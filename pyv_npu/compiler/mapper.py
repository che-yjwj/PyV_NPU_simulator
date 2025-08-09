
from __future__ import annotations
from typing import List
from ..ir.model_ir import Graph, Node
from ..isa.npu_ir import Program, NPUOp

def map_model_ir_to_npu_program(g: Graph) -> Program:
    """Very thin mapper: op_type -> NPUOp(opcode=op_type)."""
    ops: List[NPUOp] = []
    for n in g.nodes:
        ops.append(NPUOp(opcode=n.op_type, args={"name": n.name, "inputs": n.inputs, "outputs": n.outputs}))
    return Program(ops=ops)

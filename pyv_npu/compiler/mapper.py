
from __future__ import annotations
from typing import List, Dict
from ..ir.model_ir import Graph
from ..isa.npu_ir import Program, NPUOp, Tensor as NpuTensor
from ..isa.riscv_ext import ENQCMD_T, TWAIT

def map_model_ir_to_npu_program(g: Graph, mode: str = 'loose') -> Program:
    """Maps the Model IR Graph to an NPU Program.

    In 'loose' mode, it creates a simple list of NPU compute operations.
    In 'tight' mode, it creates a mixed stream of CPU (custom ISA) and NPU ops
    that waits for results only when data dependencies require it.
    """
    if mode == 'loose':
        return _map_to_loose_program(g)

    # --- Tight-Coupled Mode Logic ---
    ops: List[NPUOp] = []
    if not g.tensors:
        raise ValueError("Graph has no tensor information. Cannot map to NPU program.")

    ticket_counter = 0
    # Tracks which ticket will produce a given tensor name
    tensor_to_ticket: Dict[str, int] = {}
    # Tracks which tickets have already been waited for
    waited_tickets = set()

    for n in g.nodes:
        # 1. Check dependencies: Insert TWAITs if needed
        for t_name in n.inputs:
            if t_name in tensor_to_ticket:
                ticket_to_wait_on = tensor_to_ticket[t_name]
                if ticket_to_wait_on not in waited_tickets:
                    wait_op = NPUOp(
                        opcode='TWAIT',
                        name=f"wait_{ticket_to_wait_on}",
                        args={'inst': TWAIT(ticket=ticket_to_wait_on)}
                    )
                    ops.append(wait_op)
                    waited_tickets.add(ticket_to_wait_on)

        # 2. Create the core NPU operation
        try:
            inputs = [NpuTensor(name=t, shape=g.tensors[t].shape, dtype=g.tensors[t].dtype) for t in n.inputs]
            outputs = [NpuTensor(name=t, shape=g.tensors[t].shape, dtype=g.tensors[t].dtype) for t in n.outputs]
        except KeyError as e:
            raise ValueError(f"Error mapping node {n.name}: Tensor {e} not found in graph tensor map.") from e

        npu_op = NPUOp(opcode=n.op_type, name=n.name, args=n.attrs, inputs=inputs, outputs=outputs)

        # 3. Create and add the ENQCMD_T operation
        ticket = ticket_counter
        enq_op = NPUOp(
            opcode='ENQCMD_T',
            name=f"enq_{n.name}",
            args={'inst': ENQCMD_T(npu_op_desc=npu_op, ticket=ticket)}
        )
        ops.append(enq_op)

        # 4. Update tracking for outputs
        for t in n.outputs:
            tensor_to_ticket[t] = ticket
        ticket_counter += 1

    # 5. Final TWAITs for any remaining graph outputs
    for t_name, ticket in tensor_to_ticket.items():
        if t_name in g.outputs and ticket not in waited_tickets:
            wait_op = NPUOp(
                opcode='TWAIT',
                name=f"wait_final_{ticket}",
                args={'inst': TWAIT(ticket=ticket)}
            )
            ops.append(wait_op)
            waited_tickets.add(ticket)
            
    # Map graph inputs/outputs/initializers to NpuTensors
    npu_inputs = [NpuTensor(name=t_name, shape=g.tensors[t_name].shape, dtype=g.tensors[t_name].dtype) for t_name in g.inputs]
    npu_outputs = [NpuTensor(name=t_name, shape=g.tensors[t_name].shape, dtype=g.tensors[t_name].dtype) for t_name in g.outputs]
    npu_initializers = [NpuTensor(name=t_name, shape=g.tensors[t_name].shape, dtype=g.tensors[t_name].dtype) for t_name, tensor in g.tensors.items() if t_name not in g.inputs and t_name not in g.outputs]

    return Program(ops=ops, inputs=npu_inputs, outputs=npu_outputs, initializers=npu_initializers)

def _map_to_loose_program(g: Graph) -> Program:
    """Helper for original loose mode mapping."""
    ops: List[NPUOp] = []
    if not g.tensors:
        raise ValueError("Graph has no tensor information.")

    for n in g.nodes:
        try:
            inputs = [NpuTensor(name=t, shape=g.tensors[t].shape, dtype=g.tensors[t].dtype) for t in n.inputs]
            outputs = [NpuTensor(name=t, shape=g.tensors[t].shape, dtype=g.tensors[t].dtype) for t in n.outputs]
        except KeyError as e:
            raise ValueError(f"Error mapping node {n.name}: Tensor {e} not found in graph tensor map.") from e

        ops.append(NPUOp(opcode=n.op_type, name=n.name, args=n.attrs, inputs=inputs, outputs=outputs))
    
    # Map graph inputs/outputs/initializers to NpuTensors
    npu_inputs = [NpuTensor(name=t_name, shape=g.tensors[t_name].shape, dtype=g.tensors[t_name].dtype) for t_name in g.inputs]
    npu_outputs = [NpuTensor(name=t_name, shape=g.tensors[t_name].shape, dtype=g.tensors[t_name].dtype) for t_name in g.outputs]
    npu_initializers = [NpuTensor(name=t_name, shape=g.tensors[t_name].shape, dtype=g.tensors[t_name].dtype) for t_name, tensor in g.tensors.items() if t_name not in g.inputs and t_name not in g.outputs]

    return Program(ops=ops, inputs=npu_inputs, outputs=npu_outputs, initializers=npu_initializers)

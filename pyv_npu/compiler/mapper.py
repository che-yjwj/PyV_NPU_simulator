from __future__ import annotations
from typing import List, Dict
from ..ir.model_ir import Graph, Node
from ..isa.npu_ir import Program, NPUOp, Tensor as NpuTensor
from ..isa.riscv_ext import ENQCMD_T, TWAIT
from ..isa.opcode import Opcode


def _add_op_args(n: Node, npu_op: NPUOp):
    """Adds necessary args for scheduling from the node's tensors."""
    if npu_op.opcode in (Opcode.MATMUL, Opcode.CONV, Opcode.MATMULADDGELU):
        if len(npu_op.inputs) >= 2:
            shape_A = npu_op.inputs[0].shape
            shape_B = npu_op.inputs[1].shape
            m = shape_A[-2]
            k = shape_A[-1]
            n = shape_B[-1]
            npu_op.args['tile_m'] = m
            npu_op.args['tile_n'] = n
            npu_op.args['tile_k'] = k
    elif npu_op.opcode in (Opcode.GELU, Opcode.SOFTMAX, Opcode.ADD, Opcode.MUL, Opcode.LAYERNORM):
        if npu_op.inputs:
            npu_op.args['num_elements'] = npu_op.inputs[0].num_elements


def map_model_ir_to_npu_program(g: Graph, mode: str = 'loose') -> Program:
    """Maps the Model IR Graph to an NPU Program."""
    if mode == 'loose':
        return _map_to_loose_program(g)
    elif mode == 'tight':
        return _map_to_tight_program(g)

    # Fallback or error
    raise ValueError(f"Unknown mapping mode: {mode}")


def _map_to_loose_program(g: Graph) -> Program:
    """Helper for loose mode mapping, generating explicit LOAD/STORE ops."""
    ops: List[NPUOp] = []
    if not g.tensors:
        raise ValueError("Graph has no tensor information.")

    spm_tensor_map: Dict[str, NpuTensor] = {}

    # Load initializers first
    for init_name in g.initializers:
        try:
            dram_t = NpuTensor.from_model_ir_tensor(g.tensors[init_name])
            spm_t = dram_t.clone_with_suffix("_spm")
            ops.append(NPUOp(opcode=Opcode.LOAD, name=f"load_init_{dram_t.name}", inputs=[dram_t], outputs=[spm_t]))
            spm_tensor_map[init_name] = spm_t
        except KeyError as e:
            raise ValueError(f"Error mapping initializer: Tensor {e} not found in graph tensor map.") from e

    for n in g.nodes:
        try:
            input_tensors_spm = []
            for t_name in n.inputs:
                if t_name in spm_tensor_map:
                    input_tensors_spm.append(spm_tensor_map[t_name])
                else:
                    dram_t = NpuTensor.from_model_ir_tensor(g.tensors[t_name])
                    spm_t = dram_t.clone_with_suffix("_spm")
                    ops.append(NPUOp(opcode=Opcode.LOAD, name=f"load_input_{dram_t.name}", inputs=[dram_t], outputs=[spm_t]))
                    input_tensors_spm.append(spm_t)
                    spm_tensor_map[t_name] = spm_t

            output_tensors_spm = []
            for t_name in n.outputs:
                 dram_t = NpuTensor.from_model_ir_tensor(g.tensors[t_name])
                 spm_t = dram_t.clone_with_suffix("_spm")
                 output_tensors_spm.append(spm_t)
                 spm_tensor_map[t_name] = spm_t
            
            try:
                compute_op = NPUOp(opcode=Opcode(n.op_type), name=n.name, args=n.attrs.copy(), inputs=input_tensors_spm, outputs=output_tensors_spm)
            except ValueError as e:
                raise ValueError(f"Unsupported ONNX op_type '{n.op_type}' in node '{n.name}'.") from e

            _add_op_args(n, compute_op)
            ops.append(compute_op)

        except KeyError as e:
            raise ValueError(f"Error mapping node {n.name}: Tensor {e} not found in graph tensor map.") from e

    # Final STORE operations for graph outputs
    for out_name in g.outputs:
        if out_name in spm_tensor_map:
            try:
                spm_t = spm_tensor_map[out_name]
                dram_t = NpuTensor.from_model_ir_tensor(g.tensors[out_name])
                ops.append(NPUOp(opcode=Opcode.STORE, name=f"store_output_{dram_t.name}", inputs=[spm_t], outputs=[dram_t]))
            except KeyError as e:
                raise ValueError(f"Error mapping output: Tensor {e} not found in graph tensor map.") from e

    try:
        npu_inputs = [NpuTensor.from_model_ir_tensor(g.tensors[t_name]) for t_name in g.inputs]
        npu_outputs = [NpuTensor.from_model_ir_tensor(g.tensors[t_name]) for t_name in g.outputs]
        npu_initializers = [NpuTensor.from_model_ir_tensor(g.tensors[t_name]) for t_name in g.initializers]
    except KeyError as e:
        raise ValueError(f"Error mapping final program tensors: Tensor {e} not found in graph tensor map.") from e
    
    return Program(ops=ops, inputs=npu_inputs, outputs=npu_outputs, initializers=npu_initializers)


def _map_to_tight_program(g: Graph) -> Program:
    """Helper for tight mode mapping."""
    ops: List[NPUOp] = []
    if not g.tensors:
        raise ValueError("Graph has no tensor information.")

    ticket_counter = 0
    for n in g.nodes:
        try:
            inputs = [NpuTensor.from_model_ir_tensor(g.tensors[t]) for t in n.inputs]
            outputs = [NpuTensor.from_model_ir_tensor(g.tensors[t]) for t in n.outputs]
        except KeyError as e:
            raise ValueError(f"Error mapping node {n.name}: Tensor {e} not found in graph tensor map.") from e

        # 1. Create the actual NPU operation descriptor
        try:
            npu_op_desc = NPUOp(opcode=Opcode(n.op_type), name=n.name, args=n.attrs.copy(), inputs=inputs, outputs=outputs)
        except ValueError as e:
            raise ValueError(f"Unsupported ONNX op_type '{n.op_type}' in node '{n.name}'.") from e
        _add_op_args(n, npu_op_desc)

        # 2. Create the ENQCMD_T instruction
        enq_op = NPUOp(
            opcode=Opcode.ENQCMD_T,
            name=f"enq_{n.name}",
            args={'enqcmd': ENQCMD_T(npu_op_desc=npu_op_desc, ticket=ticket_counter)}
        )
        ops.append(enq_op)

        # 3. Create a corresponding TWAIT instruction
        twait_op = NPUOp(
            opcode=Opcode.TWAIT,
            name=f"twait_{ticket_counter}",
            args={'twait': TWAIT(ticket=ticket_counter)}
        )
        ops.append(twait_op)

        ticket_counter += 1

    npu_inputs = [NpuTensor.from_model_ir_tensor(g.tensors[t_name]) for t_name in g.inputs]
    npu_outputs = [NpuTensor.from_model_ir_tensor(g.tensors[t_name]) for t_name in g.outputs]
    npu_initializers = [NpuTensor.from_model_ir_tensor(g.tensors[t_name]) for t_name in g.initializers]

    return Program(ops=ops, inputs=npu_inputs, outputs=npu_outputs, initializers=npu_initializers)

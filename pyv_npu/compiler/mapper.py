
from __future__ import annotations
from typing import List, Dict
from ..ir.model_ir import Graph, Node
from ..isa.npu_ir import Program, NPUOp, Tensor as NpuTensor
from ..isa.riscv_ext import ENQCMD_T, TWAIT

def _add_op_args(n: Node, npu_op: NPUOp):
    """Adds necessary args for scheduling from the node's tensors."""
    if npu_op.opcode in ("MatMul", "Conv"):
        if len(npu_op.inputs) >= 2:
            m, k = npu_op.inputs[0].shape
            k, n = npu_op.inputs[1].shape
            npu_op.args['tile_m'] = m
            npu_op.args['tile_n'] = n
            npu_op.args['tile_k'] = k
    elif npu_op.opcode in ("GELU", "Softmax", "Add", "Mul", "LayerNorm"):
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

    # This map tracks the SPM version of a DRAM tensor name
    spm_tensor_map: Dict[str, NpuTensor] = {}

    for n in g.nodes:
        # --- Decompose MatMul into LOAD-COMPUTE-STORE ---
        if n.op_type == "MatMul":
            # For MatMul, we assume inputs are in DRAM and need to be loaded to SPM,
            # and the output needs to be stored back to DRAM.

            # Get original tensor definitions from the graph
            try:
                dram_input_a = NpuTensor.from_model_ir_tensor(g.tensors[n.inputs[0]])
                dram_input_b = NpuTensor.from_model_ir_tensor(g.tensors[n.inputs[1]])
                dram_output_c = NpuTensor.from_model_ir_tensor(g.tensors[n.outputs[0]])
            except KeyError as e:
                raise ValueError(f"Tensor {e} not found in graph tensor map for node {n.name}") from e

            # Create new tensor objects representing their location in SPM
            spm_input_a = dram_input_a.clone_with_suffix("_spm")
            spm_input_b = dram_input_b.clone_with_suffix("_spm")
            spm_output_c = dram_output_c.clone_with_suffix("_spm")

            # 1. Create LOAD ops for inputs
            ops.append(NPUOp(opcode='LOAD', name=f"load_{n.name}_a", inputs=[dram_input_a], outputs=[spm_input_a]))
            ops.append(NPUOp(opcode='LOAD', name=f"load_{n.name}_b", inputs=[dram_input_b], outputs=[spm_input_b]))

            # 2. Create the actual COMPUTE operation
            # Its inputs and outputs are now the tensors in SPM
            compute_op = NPUOp(opcode='MatMul', name=n.name, args=n.attrs.copy(), inputs=[spm_input_a, spm_input_b], outputs=[spm_output_c])
            _add_op_args(n, compute_op)
            ops.append(compute_op)

            # 3. Create STORE op for the output
            ops.append(NPUOp(opcode='STORE', name=f"store_{n.name}_c", inputs=[spm_output_c], outputs=[dram_output_c]))
            
            # For subsequent ops, the original output tensor name now points to the SPM version
            spm_tensor_map[n.outputs[0]] = spm_output_c
        
        else:
            # --- Logic for other ops (e.g., GELU, Add) ---
            # Decompose into LOAD-COMPUTE-STORE as well for now.
            input_tensors_dram = [NpuTensor.from_model_ir_tensor(g.tensors[t_name]) for t_name in n.inputs]
            output_tensors_dram = [NpuTensor.from_model_ir_tensor(g.tensors[t_name]) for t_name in n.outputs]

            input_tensors_spm = [t.clone_with_suffix("_spm") for t in input_tensors_dram]
            output_tensors_spm = [t.clone_with_suffix("_spm") for t in output_tensors_dram]

            for dram_t, spm_t in zip(input_tensors_dram, input_tensors_spm):
                ops.append(NPUOp(opcode='LOAD', name=f"load_{n.name}_{dram_t.name}", inputs=[dram_t], outputs=[spm_t]))

            compute_op = NPUOp(opcode=n.op_type, name=n.name, args=n.attrs.copy(), inputs=input_tensors_spm, outputs=output_tensors_spm)
            _add_op_args(n, compute_op)
            ops.append(compute_op)

            for spm_t, dram_t in zip(output_tensors_spm, output_tensors_dram):
                ops.append(NPUOp(opcode='STORE', name=f"store_{n.name}_{dram_t.name}", inputs=[spm_t], outputs=[dram_t]))

    npu_inputs = [NpuTensor.from_model_ir_tensor(g.tensors[t_name]) for t_name in g.inputs]
    npu_outputs = [NpuTensor.from_model_ir_tensor(g.tensors[t_name]) for t_name in g.outputs]
    npu_initializers = [NpuTensor.from_model_ir_tensor(g.tensors[t_name]) for t_name in g.initializers]
    
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
        npu_op_desc = NPUOp(opcode=n.op_type, name=n.name, args=n.attrs.copy(), inputs=inputs, outputs=outputs)
        _add_op_args(n, npu_op_desc)

        # 2. Create the ENQCMD_T instruction
        enq_op = NPUOp(
            opcode='ENQCMD_T',
            name=f"enq_{n.name}",
            args={'enqcmd': ENQCMD_T(npu_op_desc=npu_op_desc, ticket=ticket_counter)}
        )
        ops.append(enq_op)

        # 3. Create a corresponding TWAIT instruction
        twait_op = NPUOp(
            opcode='TWAIT',
            name=f"twait_{ticket_counter}",
            args={'twait': TWAIT(ticket=ticket_counter)}
        )
        ops.append(twait_op)

        ticket_counter += 1

    npu_inputs = [NpuTensor.from_model_ir_tensor(g.tensors[t_name]) for t_name in g.inputs]
    npu_outputs = [NpuTensor.from_model_ir_tensor(g.tensors[t_name]) for t_name in g.outputs]
    npu_initializers = [NpuTensor.from_model_ir_tensor(g.tensors[t_name]) for t_name in g.initializers]

    return Program(ops=ops, inputs=npu_inputs, outputs=npu_outputs, initializers=npu_initializers)

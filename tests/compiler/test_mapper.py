import pytest
from pyv_npu.compiler.mapper import map_model_ir_to_npu_program
from pyv_npu.ir.model_ir import Graph, Node, Tensor
from pyv_npu.isa.opcode import Opcode

# The new mapper requires a more realistic Graph object, including a tensor map.
def helper_create_model_ir(nodes):
    """Helper to create a valid Graph for testing the new mapper."""
    tensors = {
        "inp": Tensor(name="inp", shape=(1, 128), dtype="float32"),
        "w1": Tensor(name="w1", shape=(128, 128), dtype="float32"),
        "b1": Tensor(name="b1", shape=(128,), dtype="float32"),
        "out1": Tensor(name="out1", shape=(1, 128), dtype="float32"),
        "out2": Tensor(name="out2", shape=(1, 128), dtype="float32"),
        "final_out": Tensor(name="final_out", shape=(1, 128), dtype="float32"),
    }
    return Graph(
        nodes=nodes,
        inputs=["inp"],
        outputs=["final_out"],
        initializers=["w1", "b1"],
        tensors=tensors,
    )


def test_map_ir_to_loose_mode_single_op():
    """Tests mapping a single compute node in loose mode."""
    # given
    ir = helper_create_model_ir(
        [Node(name="node1", op_type="MatMul", inputs=["inp", "w1"], outputs=["out1"])]
    )
    # when
    program = map_model_ir_to_npu_program(ir, mode='loose')
    # then
    # The mapper should generate LOADs for inputs and the MatMul compute op.
    compute_ops = [op for op in program.ops if op.opcode == Opcode.MATMUL]
    assert len(compute_ops) == 1
    assert compute_ops[0].name == "node1"
    assert Opcode.LOAD in [op.opcode for op in program.ops]


def test_map_ir_to_tight_mode_single_op():
    """Tests mapping a single compute node in tight mode."""
    # given
    ir = helper_create_model_ir(
        [Node(name="node1", op_type="MatMul", inputs=["inp", "w1"], outputs=["out1"])]
    )
    # when
    program = map_model_ir_to_npu_program(ir, mode='tight')
    # then
    assert len(program.ops) == 2
    assert program.ops[0].opcode == Opcode.ENQCMD_T
    assert program.ops[1].opcode == Opcode.TWAIT
    # Check that the descriptor inside ENQCMD_T contains the correct compute op
    enq_cmd_args = program.ops[0].args['enqcmd']
    assert enq_cmd_args.npu_op_desc.opcode == Opcode.MATMUL


def test_map_ir_to_loose_mode_multiple_ops():
    """Tests mapping a sequence of nodes in loose mode."""
    # given
    ir = helper_create_model_ir(
        [
            Node(name="node1", op_type="MatMul", inputs=["inp", "w1"], outputs=["out1"]),
            Node(name="node2", op_type="Add", inputs=["out1", "b1"], outputs=["out2"]),
        ]
    )
    # when
    program = map_model_ir_to_npu_program(ir, mode='loose')
    # then
    op_names = [op.name for op in program.ops]
    assert "node1" in op_names
    assert "node2" in op_names
    opcodes = [op.opcode for op in program.ops]
    assert Opcode.MATMUL in opcodes
    assert Opcode.ADD in opcodes


def test_map_ir_to_tight_mode_multiple_ops():
    """Tests mapping a sequence of nodes in tight mode."""
    # given
    ir = helper_create_model_ir(
        [
            Node(name="node1", op_type="MatMul", inputs=["inp", "w1"], outputs=["out1"]),
            Node(name="node2", op_type="Add", inputs=["out1", "b1"], outputs=["out2"]),
        ]
    )
    # when
    program = map_model_ir_to_npu_program(ir, mode='tight')
    # then
    assert len(program.ops) == 4  # enq, twait, enq, twait
    assert program.ops[0].opcode == Opcode.ENQCMD_T
    assert program.ops[0].args['enqcmd'].npu_op_desc.name == "node1"
    assert program.ops[1].opcode == Opcode.TWAIT
    assert program.ops[2].opcode == Opcode.ENQCMD_T
    assert program.ops[2].args['enqcmd'].npu_op_desc.name == "node2"
    assert program.ops[3].opcode == Opcode.TWAIT


def test_map_ir_empty_nodes():
    """Tests mapping a graph with no compute nodes."""
    # given
    ir = helper_create_model_ir([])
    # when
    loose_program = map_model_ir_to_npu_program(ir, mode='loose')
    tight_program = map_model_ir_to_npu_program(ir, mode='tight')
    # then
    # The new mapper creates LOADs for initializers and STOREs for outputs.
    # We should check that no *compute* ops are created.
    loose_compute_ops = [op for op in loose_program.ops if op.opcode not in (Opcode.LOAD, Opcode.STORE)]
    assert len(loose_compute_ops) == 0
    assert len(tight_program.ops) == 0


@pytest.mark.parametrize("mode", ["loose", "tight"])
def test_map_ir_unsupported_opcode(mode):
    """Tests that an unsupported op_type raises a ValueError."""
    # given
    ir = helper_create_model_ir([Node(name="node1", op_type="UnsupportedOp", inputs=[], outputs=[])])
    # when/then
    with pytest.raises(ValueError, match="Unsupported ONNX op_type"):
        map_model_ir_to_npu_program(ir, mode=mode)


@pytest.mark.parametrize("mode", ["loose", "tight"])
def test_map_ir_missing_tensor_info(mode):
    """Tests that a missing tensor in the graph's tensor map raises an error."""
    # given
    ir = helper_create_model_ir([Node(name="node1", op_type="MatMul", inputs=["inp", "w1"], outputs=["out1"])])
    # Intentionally remove a required tensor from the map
    del ir.tensors["w1"]
    # when/then
    with pytest.raises(ValueError, match="Tensor 'w1' not found"):
        map_model_ir_to_npu_program(ir, mode=mode)

def test_map_ir_unknown_mode():
    """Tests that an unknown mapping mode raises a ValueError."""
    # given
    ir = helper_create_model_ir([])
    # when/then
    with pytest.raises(ValueError, match="Unknown mapping mode: invalid_mode"):
        map_model_ir_to_npu_program(ir, mode='invalid_mode')


def test_map_ir_loose_mode_graph_split():
    """Tests mapping a graph with a 1-to-many fan-out in loose mode."""
    # given
    tensors = {
        "inp": Tensor(name="inp", shape=(1, 128), dtype="float32"),
        "w1": Tensor(name="w1", shape=(128, 128), dtype="float32"),
        "w2": Tensor(name="w2", shape=(128, 128), dtype="float32"),
        "out1": Tensor(name="out1", shape=(1, 128), dtype="float32"),
        "out2": Tensor(name="out2", shape=(1, 128), dtype="float32"),
        "final_out": Tensor(name="final_out", shape=(1, 128), dtype="float32"),
    }
    nodes = [
        Node(name="node1", op_type="MatMul", inputs=["inp", "w1"], outputs=["out1"]),
        # out1 is used by both node2 and node3
        Node(name="node2", op_type="MatMul", inputs=["out1", "w2"], outputs=["out2"]),
        Node(name="node3", op_type="GELU", inputs=["out1"], outputs=["final_out"]),
    ]
    ir = Graph(
        nodes=nodes,
        inputs=["inp"],
        outputs=["out2", "final_out"],
        initializers=["w1", "w2"],
        tensors=tensors,
    )

    # when
    program = map_model_ir_to_npu_program(ir, mode='loose')

    # then
    # Check that the intermediate tensor 'out1' is created once in SPM.
    out1_spm_tensor = None
    for op in program.ops:
        if op.name == "node1":
            out1_spm_tensor = op.outputs[0]
            break
    
    assert out1_spm_tensor is not None, "SPM tensor for out1 not found"

    # Ensure that the subsequent nodes (node2, node3) use this same SPM tensor
    # and don't trigger another LOAD operation for 'out1'.
    node2_input = next(op.inputs[0] for op in program.ops if op.name == "node2")
    node3_input = next(op.inputs[0] for op in program.ops if op.name == "node3")

    assert node2_input is out1_spm_tensor
    assert node3_input is out1_spm_tensor

    # Verify no redundant LOAD for 'out1'
    load_ops = [op for op in program.ops if op.opcode == Opcode.LOAD]
    load_op_dram_names = [op.inputs[0].name for op in load_ops]
    assert "out1" not in load_op_dram_names


def test_loose_mode_creates_store_for_graph_output():
    """Tests that a STORE op is created for a graph output from a node."""
    # given
    nodes = [Node(name="node1", op_type="MatMul", inputs=["inp", "w1"], outputs=["final_out"])]
    tensors = {
        "inp": Tensor(name="inp", shape=(1, 128), dtype="float32"),
        "w1": Tensor(name="w1", shape=(128, 128), dtype="float32"),
        "final_out": Tensor(name="final_out", shape=(1, 128), dtype="float32"),
    }
    ir = Graph(
        nodes=nodes,
        inputs=["inp"],
        outputs=["final_out"],
        initializers=["w1"],
        tensors=tensors,
    )

    # when
    program = map_model_ir_to_npu_program(ir, mode='loose')

    # then
    store_ops = [op for op in program.ops if op.opcode == Opcode.STORE]
    assert len(store_ops) == 1
    assert store_ops[0].name == "store_output_final_out"
    assert store_ops[0].inputs[0].name == "final_out_spm"
    assert store_ops[0].outputs[0].name == "final_out"


def test_loose_mode_raises_error_for_missing_output_tensor():
    """Tests ValueError when a graph output is not in the tensor map."""
    # given
    ir = helper_create_model_ir([])
    # Intentionally create a malformed graph
    ir.outputs = ["missing_tensor"]

    # when/then
    with pytest.raises(ValueError, match="Error mapping final program tensors: Tensor .* not found"):
        map_model_ir_to_npu_program(ir, mode='loose')

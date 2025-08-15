import pytest
from pyv_npu.compiler.mapper import map_model_ir_to_npu_program, _add_op_args
from pyv_npu.ir.model_ir import Graph, Node, Tensor
from pyv_npu.isa.npu_ir import NPUOp, Tensor as NpuTensor

@pytest.fixture
def sample_graph():
    """Provides a sample Model IR graph for testing."""
    t_in = Tensor(name='input', shape=(128, 128), dtype='float16')
    t_w = Tensor(name='weights', shape=(128, 128), dtype='float16')
    t_out = Tensor(name='output', shape=(128, 128), dtype='float16')
    
    node = Node(name='matmul', op_type='MatMul', inputs=['input', 'weights'], outputs=['output'])
    
    tensors = {'input': t_in, 'weights': t_w, 'output': t_out}
    graph = Graph(
        nodes=[node],
        inputs=['input'],
        outputs=['output'],
        initializers=['weights'],
        tensors=tensors
    )
    return graph

def test_map_loose_mode(sample_graph: Graph):
    """Tests basic mapping in loose mode."""
    program = map_model_ir_to_npu_program(sample_graph, mode='loose')
    assert len(program.ops) == 1
    assert program.ops[0].opcode == 'MatMul'
    assert 'tile_m' in program.ops[0].args
    assert program.ops[0].args['tile_m'] == 128

def test_map_tight_mode(sample_graph: Graph):
    """Tests basic mapping in tight mode."""
    program = map_model_ir_to_npu_program(sample_graph, mode='tight')
    # Each node should produce an ENQCMD_T and a TWAIT op
    assert len(program.ops) == 2
    assert program.ops[0].opcode == 'ENQCMD_T'
    assert program.ops[1].opcode == 'TWAIT'

    # Check that the descriptor inside ENQCMD_T is correct
    enq_cmd = program.ops[0].args['enqcmd']
    assert enq_cmd.npu_op_desc.opcode == 'MatMul'
    assert 'tile_k' in enq_cmd.npu_op_desc.args
    assert enq_cmd.ticket == 0
    assert program.ops[1].args['twait'].ticket == 0

def test_mapper_unknown_mode(sample_graph: Graph):
    """Tests that an unknown mode raises a ValueError."""
    with pytest.raises(ValueError, match="Unknown mapping mode: invalid_mode"):
        map_model_ir_to_npu_program(sample_graph, mode='invalid_mode')

def test_mapper_tensor_not_found(sample_graph: Graph):
    """Tests that a missing tensor in the graph's tensor map raises an error."""
    # Invalidate the graph by removing a tensor that a node needs
    del sample_graph.tensors['weights']
    with pytest.raises(ValueError, match="Tensor 'weights' not found"):
        map_model_ir_to_npu_program(sample_graph, mode='loose')

def test_add_op_args_matmul():
    """Tests the _add_op_args helper for MatMul."""
    node = Node(name='n', op_type='MatMul', inputs=[], outputs=[])
    npu_op = NPUOp(
        opcode='MatMul', 
        name='n', 
        inputs=[
            NpuTensor('in1', (64, 32), 'fp16'),
            NpuTensor('in2', (32, 128), 'fp16')
        ]
    )
    _add_op_args(node, npu_op)
    assert npu_op.args['tile_m'] == 64
    assert npu_op.args['tile_k'] == 32
    assert npu_op.args['tile_n'] == 128

def test_add_op_args_elementwise():
    """Tests the _add_op_args helper for element-wise ops."""
    node = Node(name='n', op_type='GELU', inputs=[], outputs=[])
    npu_op = NPUOp(
        opcode='GELU', 
        name='n', 
        inputs=[NpuTensor('in1', (1, 2048), 'fp16')]
    )
    _add_op_args(node, npu_op)
    assert npu_op.args['num_elements'] == 2048

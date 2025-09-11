import pytest
import copy
from pyv_npu.ir.onnx_importer import load_onnx_as_model_ir
from pyv_npu.compiler.passes.fusion import apply_fusion_pass
from pyv_npu.runtime.simulator import run as run_sim
from pyv_npu.compiler.mapper import map_model_ir_to_npu_program
from pyv_npu.compiler.allocator import Allocator
from pyv_npu.config import SimConfig


import os

# Get the directory of the current test file
TEST_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(TEST_DIR, "fusion_test_model.onnx")

@pytest.fixture
def fusion_test_model_graph():
    """Loads the test model for fusion."""
    # Ensure the model exists. It's created by a separate script.
    return load_onnx_as_model_ir(MODEL_PATH)

def test_fusion_pass_reduces_nodes(fusion_test_model_graph):
    """Tests that the fusion pass correctly reduces the number of nodes."""
    original_graph = fusion_test_model_graph
    assert len(original_graph.nodes) == 3, "Test model should have 3 nodes initially"

    # Apply the fusion pass
    fused_graph = apply_fusion_pass(original_graph)

    assert len(fused_graph.nodes) == 1, "Graph should have exactly one node after fusion"
    assert fused_graph.nodes[0].op_type == "MatMulAddGelu", "The fused node should have the correct op_type"

def test_fusion_pass_improves_performance(fusion_test_model_graph):
    """Tests that the fusion pass results in fewer cycles."""
    config = SimConfig(model=MODEL_PATH)
    allocator = Allocator(config.dram_base_address, config.dram_page_size)

    # --- Run without fusion ---
    graph_no_fusion = copy.deepcopy(fusion_test_model_graph)
    prog_no_fusion = map_model_ir_to_npu_program(graph_no_fusion, mode='loose')
    allocator.allocate(prog_no_fusion)
    schedule_no_fusion, stats_no_fusion = run_sim(prog_no_fusion, config)
    cycles_no_fusion = stats_no_fusion['total_cycles']

    # --- Run with fusion ---
    fused_graph = apply_fusion_pass(fusion_test_model_graph)
    prog_with_fusion = map_model_ir_to_npu_program(fused_graph, mode='loose')
    allocator.allocate(prog_with_fusion)
    schedule_with_fusion, stats_with_fusion = run_sim(prog_with_fusion, config)
    cycles_with_fusion = stats_with_fusion['total_cycles']

    print(f"Cycles without fusion: {cycles_no_fusion}")
    print(f"Cycles with fusion: {cycles_with_fusion}")

    assert cycles_with_fusion < cycles_no_fusion, "Fusion should result in fewer execution cycles"

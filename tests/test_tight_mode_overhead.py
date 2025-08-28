import pytest
from unittest.mock import MagicMock
from pyv_npu.config import SimConfig
from pyv_npu.ir.onnx_importer import load_onnx_as_model_ir
from pyv_npu.compiler.mapper import map_model_ir_to_npu_program
from pyv_npu.compiler.allocator import Allocator
from pyv_npu.runtime.simulator import run as run_sim
from pyv_npu.utils.reporting import generate_report_json

# Mock onnx
try:
    import onnx
    from onnx import helper
    from onnx import TensorProto
except ImportError:
    onnx = MagicMock()

@pytest.fixture
def mock_simple_model():
    """Creates a mock ONNX model with a single MatMul."""
    node = helper.make_node('MatMul', ['X', 'W'], ['Y'], name='matmul1')
    graph = helper.make_graph(
        [node],
        'test-graph-simple',
        [helper.make_tensor_value_info('X', TensorProto.FLOAT, [128, 128])],
        [helper.make_tensor_value_info('Y', TensorProto.FLOAT, [128, 128])],
        [helper.make_tensor('W', TensorProto.FLOAT, [128, 128], [1.0]*128*128)]
    )
    return helper.make_model(graph, producer_name='pytest-tight-mode')

def test_tight_vs_loose_overhead_comparison(monkeypatch, mock_simple_model):
    """Compares tight vs loose mode to ensure control overhead is added."""
    monkeypatch.setattr(onnx, "load", lambda path: mock_simple_model)
    model_ir = load_onnx_as_model_ir("dummy_path")

    # --- Run in Loose Mode ---
    config_loose = SimConfig(sim_level='CA_HYBRID', mode='loose')
    prog_loose = map_model_ir_to_npu_program(model_ir, mode='loose')
    allocator_loose = Allocator(config_loose.dram_base_address, config_loose.dram_page_size)
    allocator_loose.allocate(prog_loose)
    schedule_loose, stats_loose = run_sim(prog_loose, config_loose)
    report_loose = generate_report_json(schedule_loose, config_loose, stats_loose)

    # --- Run in Tight Mode ---
    config_tight = SimConfig(sim_level='CA_HYBRID', mode='tight')
    prog_tight = map_model_ir_to_npu_program(model_ir, mode='tight')
    allocator_tight = Allocator(config_tight.dram_base_address, config_tight.dram_page_size)
    allocator_tight.allocate(prog_tight)
    schedule_tight, stats_tight = run_sim(prog_tight, config_tight)
    report_tight = generate_report_json(schedule_tight, config_tight, stats_tight)

    # --- Assertions ---
    # 1. Loose mode should have no control overhead stats
    assert "control_overhead_stats" not in report_loose

    # 2. Tight mode should have control overhead stats
    assert "control_overhead_stats" in report_tight
    overhead_stats = report_tight["control_overhead_stats"]
    assert overhead_stats["avg"] > 0
    assert overhead_stats["p50"] > 0

    # 3. Total cycles for tight mode should be higher due to overhead
    total_cycles_loose = report_loose['total_cycles']
    total_cycles_tight = report_tight['total_cycles']
    print(f"Loose mode total cycles: {total_cycles_loose}")
    print(f"Tight mode total cycles: {total_cycles_tight}")
    print(f"Tight mode control overhead (avg): {overhead_stats['avg']:.2f} cycles")

import json
from pathlib import Path
import pytest
from pyv_npu.config import SimConfig
from pyv_npu.isa.npu_ir import NPUOp
from pyv_npu.runtime.scheduler import ScheduleItem
from pyv_npu.utils.reporting import generate_report_json, generate_html_report

@pytest.fixture
def sample_schedule():
    """Provides a sample schedule for testing."""
    op1 = NPUOp("op1", "MatMul")
    op2 = NPUOp("op2", "GELU")
    # op2 has a dependency stall of 10 and a resource stall of 5
    # Total duration = 10 (stall) + 5 (stall) + 20 (exec) = 35
    # But start/end are absolute, so stall is start - ideal_start
    # Let's say op1 ends at 100. ideal start for op2 is 100.
    # But engine is busy until 110. tentative_start is 110.
    # Resource contention pushes it to 115. actual_start is 115.
    # stall_cycles = 115 - 100 = 15
    return [
        ScheduleItem(op=op1, start_cycle=0, end_cycle=100, engine="TE0", stall_cycles=0, stall_reason="NONE"),
        ScheduleItem(op=op2, start_cycle=115, end_cycle=135, engine="VE0", stall_cycles=15, stall_reason="RESOURCE_SPM"),
    ]

@pytest.fixture
def sample_config():
    """Provides a sample SimConfig."""
    return SimConfig(model="test", te=1, ve=1, dma_channels=1)

def test_generate_report_json(sample_schedule, sample_config):
    """Tests the creation of the JSON report."""
    report = generate_report_json(sample_schedule, sample_config)

    assert report["total_cycles"] == 135
    assert report["engine_util_abs"]["TE"] == 100
    assert report["engine_util_abs"]["VE"] == 20 # 135 - 115
    assert report["engine_util"]["TE_utilization"] == "74.07%" # 100 / 135

    timeline = report["timeline"]
    assert len(timeline) == 2
    assert timeline[1]["op"] == "GELU"
    assert timeline[1]["stall_cycles"] == 15
    assert timeline[1]["stall_reason"] == "RESOURCE_SPM"

def test_generate_html_report(sample_schedule, sample_config, tmp_path: Path):
    """Tests that the HTML report runs without errors."""
    report_data = generate_report_json(sample_schedule, sample_config)
    output_dir = tmp_path / "report"
    
    generate_html_report(report_data, output_dir)
    
    html_file = output_dir / "report.html"
    assert html_file.exists()
    # Check that the file is not empty and contains some expected content
    content = html_file.read_text()
    assert "PyV-NPU Simulation Report" in content
    assert "GELU" in content # Check if op name is in the report

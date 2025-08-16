import json
from pathlib import Path
import pytest
from pyv_npu.config import SimConfig
from pyv_npu.isa.npu_ir import NPUOp
from pyv_npu.runtime.scheduler import ScheduleItem
from pyv_npu.utils.reporting import generate_report_json, generate_html_report

@pytest.fixture
def sample_schedule():
    """Provides a sample schedule for testing, including cycle breakdown."""
    op1 = NPUOp(opcode="MatMul", name="op1")
    op2 = NPUOp(opcode="GELU", name="op2")
    
    op1_breakdown = {'total': 100, 'compute': 80, 'fill_drain': 20}
    op2_stall_breakdown = {"RESOURCE_SPM": 15}
    
    return [
        ScheduleItem(op=op1, start_cycle=0, end_cycle=100, engine="TE0", stall_cycles=0, stall_breakdown={}, cycle_breakdown=op1_breakdown),
        ScheduleItem(op=op2, start_cycle=115, end_cycle=135, engine="VE0", stall_cycles=15, stall_breakdown=op2_stall_breakdown),
    ]

@pytest.fixture
def sample_config():
    """Provides a sample SimConfig."""
    return SimConfig(model="test", te=1, ve=1, dma_channels=1)

def test_generate_report_json(sample_schedule, sample_config):
    """Tests the creation of the JSON report, including cycle breakdown."""
    report = generate_report_json(sample_schedule, sample_config, stats={})

    assert report["total_cycles"] == 135
    timeline = report["timeline"]
    assert len(timeline) == 2

    op1_report = timeline[0]
    assert op1_report['op'] == 'MatMul'
    assert op1_report['compute'] == 80
    assert op1_report['fill_drain'] == 20

    op2_report = timeline[1]
    assert op2_report['op'] == 'GELU'
    assert op2_report.get('compute', 0) == 0

def test_generate_html_report(sample_schedule, sample_config, tmp_path: Path):
    """Tests that the HTML report runs and includes breakdown data."""
    report_data = generate_report_json(sample_schedule, sample_config, stats={})
    output_dir = tmp_path / "report"
    
    generate_html_report(report_data, output_dir)
    
    html_file = output_dir / "report.html"
    assert html_file.exists()
    
    content = html_file.read_text(encoding='utf-8')
    # Check for data that must be in the report, rather than a specific title string
    assert "GELU" in content # Check if op type is in the report data
    assert "op1" in content  # Check if op name is in the report data
    assert "fill_drain" in content

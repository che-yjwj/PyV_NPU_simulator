import json
from pathlib import Path
import pytest
from pyv_npu.config import SimConfig
from pyv_npu.isa.npu_ir import NPUOp
from pyv_npu.runtime.scheduler import ScheduleItem
from pyv_npu.utils.reporting import generate_report_json, generate_report
from pyv_npu.utils import reporting

@pytest.fixture
def sample_schedule():
    """Provides a sample schedule for testing, including cycle breakdown."""
    op1 = NPUOp(opcode="MatMul", name="op1")
    op2 = NPUOp(opcode="GELU", name="op2")
    
    op1_breakdown = {'total': 100, 'compute': 80, 'fill_drain': 20}
    op2_stall_breakdown = {"RESOURCE_SPM": 15}
    
    return [
        ScheduleItem(op=op1, start_cycle=0, end_cycle=100, engine="TC0", stall_cycles=0, stall_breakdown={}, cycle_breakdown=op1_breakdown),
        ScheduleItem(op=op2, start_cycle=115, end_cycle=135, engine="VC0", stall_cycles=15, stall_breakdown=op2_stall_breakdown),
    ]

@pytest.fixture
def sample_config():
    """Provides a sample SimConfig."""
    return SimConfig(model="test", tc=1, vc=1, dma_channels=1)

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

def test_aggregate_statistics_calculation():
    """Tests the calculation of aggregate stats like utilization and stalls."""
    # 1. Define a schedule with the correct ScheduleItem structure
    op_cpu1 = NPUOp(opcode="ENQCMD_T", name="enq_gemm")
    op_tc = NPUOp(opcode="MatMul", name="gemm0")
    op_vc = NPUOp(opcode="Add", name="add0")

    schedule = [
        ScheduleItem(op=op_cpu1, start_cycle=0, end_cycle=3, engine="CPU0", stall_cycles=0, stall_breakdown={}),
        ScheduleItem(op=op_tc, start_cycle=10, end_cycle=100, engine="TC0", stall_cycles=7, stall_breakdown={"CONTROL": 7}),
        ScheduleItem(op=op_vc, start_cycle=105, end_cycle=115, engine="VC0", stall_cycles=5, stall_breakdown={"DEP": 5}),
    ]
    
    stats = {} # The function calculates its own control overhead stats now

    # 2. Generate report
    config = SimConfig(model="test", mode='tight', tc=1, vc=1, dma_channels=1)
    report = generate_report_json(schedule, config, stats)

    # 3. Assert correctness
    assert report["total_cycles"] == 115

    # Utilization
    util = report["engine_utilization"]
    # Note: The function aggregates TC0, TC1... into a single TC_utilization key
    # Note: The values are formatted strings
    assert util["TC_utilization"] == f"{(90 / 115):.2%}" # 90 cycles for 1 TC
    assert util["VC_utilization"] == f"{(10 / 115):.2%}" # 10 cycles for 1 VC
    assert util["CPU_utilization"] == f"{(3 / 115):.2%}" # 3 cycles for 1 CPU
    assert util["DMA_utilization"] == "0.00%" # DMA was not used

    # Stalls are now part of the timeline, not a separate section.
    # Let's check the timeline entry for the TC op
    tc_op_report = next(item for item in report["timeline"] if item["name"] == "gemm0")
    assert tc_op_report["stall_cycles"] == 7
    assert tc_op_report["stall_breakdown"]["CONTROL"] == 7
    
    # Control overhead is calculated internally now
    overhead = report["control_overhead_stats"]
    assert overhead["avg"] == 3.0
    assert overhead["min"] == 3.0
    assert overhead["max"] == 3.0
    assert overhead["p50"] == 3.0

def test_percentiles_empty_data():
    """Tests the _calculate_percentiles helper with empty data."""
    assert reporting._calculate_percentiles([]) == {}

def test_generate_report_json_empty_schedule():
    """Tests generate_report_json with an empty schedule."""
    report = generate_report_json([], SimConfig(), {})
    assert report["total_cycles"] == 0
    assert report["timeline"] == []
    assert report["engine_utilization"] == {}

def test_generate_report_full(sample_schedule, sample_config, tmp_path: Path, capsys):
    """Tests the main generate_report function that writes all artifacts."""
    sample_config.report_dir = str(tmp_path)

    generate_report(sample_schedule, sample_config, {})

    # Check for JSON report
    assert (tmp_path / "report.json").exists()

    # Check for HTML report
    html_file = tmp_path / "report.html"
    assert html_file.exists()
    content = html_file.read_text(encoding='utf-8')
    assert "GELU" in content
    assert "op1" in content
    assert "stall_cycles" in content

    # Check for ASCII Gantt in stdout
    captured = capsys.readouterr()
    assert "ASCII Gantt Chart" in captured.out
    assert "TC0" in captured.out
    assert "VC0" in captured.out
    assert "M" in captured.out # Check for op name in ASCII chart (via op char)


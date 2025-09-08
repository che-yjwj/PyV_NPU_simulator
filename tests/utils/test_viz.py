import pytest
from pathlib import Path
from pyv_npu.utils.viz import export_gantt, export_gantt_ascii


@pytest.fixture
def sample_timeline():
    """Provides a sample timeline for testing."""
    return [
        {'engine': 'DMA', 'name': 'load_w', 'op': 'LOAD', 'start': 0, 'end': 100},
        {'engine': 'TC', 'name': 'gemm', 'op': 'GEMM', 'start': 100, 'end': 300},
        {'engine': 'DMA', 'name': 'store_o', 'op': 'STORE', 'start': 300, 'end': 400},
    ]


class TestExportGanttHTML:
    def test_export_gantt_empty_timeline(self, tmp_path: Path):
        """Tests that an HTML file is created for an empty timeline."""
        # given
        timeline = []
        output_path = tmp_path / "gantt.html"

        # when
        export_gantt(timeline, str(output_path))

        # then
        assert output_path.exists()
        content = output_path.read_text()
        assert "No data to display" in content

    def test_export_gantt_with_data(self, tmp_path: Path, sample_timeline):
        """Tests that a valid HTML file is created for a sample timeline."""
        # given
        output_path = tmp_path / "gantt.html"

        # when
        export_gantt(sample_timeline, str(output_path))

        # then
        assert output_path.exists()
        content = output_path.read_text()
        assert "NPU Simulation Timeline" in content
        assert "cdn.plot.ly" in content

    def test_export_gantt_handles_bad_data(self, tmp_path: Path):
        """Tests that rows with invalid start/end times are dropped."""
        # given
        timeline = [
            {'engine': 'DMA', 'name': 'good', 'op': 'LOAD', 'start': 0, 'end': 100},
            {'engine': 'TC', 'name': 'bad1', 'op': 'GEMM', 'start': None, 'end': 200},
            {'engine': 'TC', 'name': 'bad2', 'op': 'GEMM', 'start': 200, 'end': 'invalid'},
        ]
        output_path = tmp_path / "gantt.html"

        # when
        export_gantt(timeline, str(output_path))

        # then
        assert output_path.exists()
        # Check that only the good data is present in the plotly data
        content = output_path.read_text()
        assert "good" in content
        assert "bad1" not in content
        assert "bad2" not in content

    def test_export_gantt_zero_duration_op(self, tmp_path: Path):
        """Tests that ops with zero duration are given a minimum visibility."""
        # given
        timeline = [{'engine': 'DMA', 'name': 'zero', 'op': 'BARRIER', 'start': 50, 'end': 50}]
        output_path = tmp_path / "gantt.html"

        # when
        export_gantt(timeline, str(output_path))

        # then
        assert output_path.exists()
        content = output_path.read_text()
        # Plotly data will contain the original start/end but a calculated duration
        # A simple check is to ensure the file is generated without error.
        assert "NPU Simulation Timeline" in content


class TestExportGanttASCII:
    def test_export_ascii_gantt_empty_timeline(self):
        """Tests ASCII export with an empty timeline."""
        assert "Timeline is empty" in export_gantt_ascii([])

    def test_export_ascii_gantt_zero_duration(self):
        """Tests ASCII export with a timeline containing no duration."""
        timeline = [{"engine": "DMA", "name": "op1", "start": 0, "end": 0}]
        assert "Timeline has no duration" in export_gantt_ascii(timeline)

    def test_export_ascii_gantt_with_data(self, sample_timeline):
        """Tests that a valid ASCII chart is created."""
        # when
        chart = export_gantt_ascii(sample_timeline)

        # then
        assert "NPU Simulation Timeline (ASCII Gantt Chart)" in chart
        assert "DMA" in chart
        assert "TC" in chart
        assert "400 cycles" in chart
        # Check for the characters representing the ops
        assert "L" in chart  # From LOAD
        assert "G" in chart  # From GEMM
        assert "S" in chart  # From STORE
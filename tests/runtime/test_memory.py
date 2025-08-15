import pytest
from pyv_npu.runtime.memory import MemoryStats

def test_memory_stats_initialization():
    """Tests that the MemoryStats dataclass initializes correctly."""
    # Test default initialization
    stats_default = MemoryStats()
    assert stats_default.dram_bytes_read == 0
    assert stats_default.dram_bytes_written == 0
    assert stats_default.sram_spills == 0

    # Test initialization with values
    stats_custom = MemoryStats(
        dram_bytes_read=1024,
        dram_bytes_written=512,
        sram_spills=4
    )
    assert stats_custom.dram_bytes_read == 1024
    assert stats_custom.dram_bytes_written == 512
    assert stats_custom.sram_spills == 4

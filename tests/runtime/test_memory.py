import pytest
from pyv_npu.config import SimConfig
from pyv_npu.runtime.memory import DramAddressMapper

def test_dram_address_mapper_interleave():
    """Tests the DRAM address mapper with a simple interleave policy."""
    config = SimConfig(
        dram_channels=4,
        dram_banks_per_channel=8,
        dram_mapping_policy="interleave",
        dram_page_size=2048  # Explicitly set for this test
    )
    
    mapper = DramAddressMapper(config)

    # Address 0 should map to channel 0, bank 0
    # row_number = 0 // 2048 = 0
    # channel_id = 0 % 4 = 0
    # bank_id = (0 // 4) % 8 = 0
    assert mapper.map(0) == (0, 0)

    # Address that falls into the next row
    # row_number = 2048 // 2048 = 1
    # channel_id = 1 % 4 = 1
    # bank_id = (1 // 4) % 8 = 0
    assert mapper.map(2048) == (1, 0)

    # Address that falls into the 4th row (should wrap channel)
    # row_number = (2048 * 4) // 2048 = 4
    # channel_id = 4 % 4 = 0
    # bank_id = (4 // 4) % 8 = 1
    assert mapper.map(2048 * 4) == (0, 1)

    # A large address
    # row_number = 12345678 // 2048 = 6028
    # channel_id = 6028 % 4 = 0
    # bank_id = (6028 // 4) % 8 = 1507 % 8 = 3
    assert mapper.map(12345678) == (0, 3)
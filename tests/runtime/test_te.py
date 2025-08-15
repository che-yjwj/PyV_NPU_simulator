import pytest
from pyv_npu.runtime.te import calculate_systolic_array_cycles

# Test cases for the V1 systolic array cost model

def test_basic_calculation():
    """Tests a basic scenario with simple numbers."""
    # On a 16x16 array, a 128x128x128 matmul
    breakdown = calculate_systolic_array_cycles(
        tile_m=128, tile_n=128, tile_k=128,
        array_height=16, array_width=16
    )
    # fill/drain = 16 + 16 - 2 = 30
    # compute = (128*128*128) / (16*16) = 2097152 / 256 = 8192
    # total = 30 + 8192 = 8222
    assert breakdown['fill_drain'] == 30
    assert breakdown['compute'] == 8192
    assert breakdown['total'] == 8222

def test_perfectly_tiled():
    """Tests a scenario where the tile fits the array perfectly."""
    breakdown = calculate_systolic_array_cycles(
        tile_m=16, tile_n=16, tile_k=16,
        array_height=16, array_width=16
    )
    # fill/drain = 16 + 16 - 2 = 30
    # compute = (16*16*16) / (16*16) = 16
    # total = 30 + 16 = 46
    assert breakdown['total'] == 46

def test_tall_matrix():
    """Tests with a tall and narrow matrix (more M, less N)."""
    breakdown = calculate_systolic_array_cycles(
        tile_m=256, tile_n=64, tile_k=128,
        array_height=16, array_width=16
    )
    # fill/drain = 30
    # compute = (256*64*128) / 256 = 8192
    # total = 30 + 8192 = 8222
    assert breakdown['total'] == 8222

def test_large_k_dimension():
    """Tests with a large K dimension, increasing compute time."""
    breakdown = calculate_systolic_array_cycles(
        tile_m=128, tile_n=128, tile_k=256,
        array_height=16, array_width=16
    )
    # fill/drain = 30
    # compute = (128*128*256) / 256 = 16384
    # total = 30 + 16384 = 16414
    assert breakdown['total'] == 16414

def test_edge_case_zero_dimension():
    """Tests the guard against zero-sized arrays."""
    breakdown = calculate_systolic_array_cycles(
        tile_m=128, tile_n=128, tile_k=128,
        array_height=0, array_width=16
    )
    assert breakdown['total'] == 1

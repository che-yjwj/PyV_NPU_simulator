import pytest
import math
from pyv_npu.runtime.te import calculate_systolic_array_cycles

# Test cases format: 
# (tile_m, tile_n, tile_k, array_h, array_w)

SYSTOLIC_ARRAY_TEST_CASES = [
    # --- Original Test Cases ---
    (64, 64, 64, 16, 16),
    (16, 16, 16, 16, 16),
    (32, 32, 16, 16, 16),

    # --- Boundary Value Tests ---
    (1, 1, 1, 1, 1),
    (16, 16, 16, 1, 1),
    (1, 16, 16, 16, 16),
    (16, 1, 16, 16, 16),
    (16, 16, 1, 16, 16),
    (16, 16, 16, 1, 16),

    # --- Unaligned Tile Tests ---
    (17, 16, 16, 16, 16),
    (16, 17, 16, 16, 16),
    (16, 16, 17, 16, 16),
    (17, 19, 23, 16, 16),

    # --- Different Array Shapes ---
    (32, 16, 32, 32, 8),
    (16, 32, 32, 8, 32),

    # --- Large Number Tests ---
    (256, 256, 256, 16, 16),
    (256, 256, 256, 128, 128),

    # --- Zero dimension tests ---
    (16, 16, 0, 16, 16),
    (0, 16, 16, 16, 16),
    (16, 0, 16, 16, 16),

    # --- Additional Realistic Scenarios ---
    (128, 128, 128, 16, 16),
    (256, 128, 64, 32, 32),
]

@pytest.mark.parametrize(
    "tile_m, tile_n, tile_k, array_h, array_w",
    SYSTOLIC_ARRAY_TEST_CASES
)
def test_systolic_array_parametrized(tile_m, tile_n, tile_k, array_h, array_w):
    """Tests systolic array cycle calculation with a wide range of parameters."""
    result = calculate_systolic_array_cycles(tile_m, tile_n, tile_k, array_h, array_w)

    # Calculate expected values using the same logic as the implementation
    if array_h <= 0 or array_w <= 0:
        expected_total = 1
    else:
        fill_drain = array_h + array_w - 2
        num_pes = array_h * array_w
        if num_pes == 0:
            compute = 0
        else:
            compute = math.ceil((tile_m * tile_n * tile_k) / num_pes)
        expected_total = fill_drain + compute

    # Using pytest.approx for floating point comparisons that might occur
    assert result['total'] == pytest.approx(expected_total)

    # Also check if components add up
    assert result['total'] == pytest.approx(result['fill_drain'] + result['compute'])

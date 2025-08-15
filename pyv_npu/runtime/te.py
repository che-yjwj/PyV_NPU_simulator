from __future__ import annotations
from typing import Dict

def calculate_systolic_array_cycles(tile_m: int, tile_n: int, tile_k: int, array_height: int, array_width: int) -> Dict[str, int]:
    """
    Calculates the execution cycles for a MatMul operation on a systolic array.

    This is a V1 cost model that includes basic fill/drain overhead.

    Returns:
        A dictionary containing total, compute, and fill/drain cycles.
    """
    if array_height <= 0 or array_width <= 0:
        return {'total': 1, 'compute': 1, 'fill_drain': 0}

    # Time to fill the array pipeline and drain the last result
    fill_drain_cycles = array_height + array_width - 2

    # Time to perform the actual computation, assuming perfect pipelining
    # This is a simplified throughput calculation.
    # It assumes 1 MAC operation per cycle per Processing Element (PE).
    num_pes = array_height * array_width
    compute_cycles = (tile_m * tile_n * tile_k) // num_pes

    total_cycles = compute_cycles + fill_drain_cycles

    return {
        'total': total_cycles,
        'compute': compute_cycles,
        'fill_drain': fill_drain_cycles
    }
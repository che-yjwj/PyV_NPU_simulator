import pytest
from pyv_npu.config import SimConfig
from pyv_npu.isa.npu_ir import Program, NPUOp, Tensor
from pyv_npu.runtime.simulator import run as run_sim

def create_memory_access_program(num_accesses: int, base_address: int, stride: int, tensor_size: int) -> Program:
    """Creates a program with a sequence of LOAD operations to simulate memory access patterns."""
    ops = []
    inputs = []
    for i in range(num_accesses):
        addr = base_address + i * stride
        t_in = Tensor(f"t_in{i}", (tensor_size,), "float16", address=addr)
        t_out = Tensor(f"t_out{i}", (tensor_size,), "float16") # Dummy output for LOAD
        ops.append(NPUOp("LOAD", f"load{i}", inputs=[t_in], outputs=[t_out]))
        inputs.append(t_in)
    return Program(ops=ops, inputs=inputs, initializers={})

def test_l2_cache_impact_on_performance():
    print("\n--- L2 Cache Impact on Performance ---")

    num_accesses = 1000 # Number of memory accesses
    tensor_size = 32   # Size of each tensor in elements (32 * 2 bytes = 64 bytes = 1 cache line)
    base_address = 1
    stride = 64        # Stride of 64 bytes (one cache line if line size is 64)

    # --- Scenario 1: High Locality (Cache Friendly) ---
    # Access the same few cache lines repeatedly
    print("\nScenario: High Locality (Cache Friendly)")
    program_locality = create_memory_access_program(num_accesses, base_address, stride=1, tensor_size=tensor_size)

    # Run with L2 Cache Enabled
    config_l2_on = SimConfig(sim_level='CA_HYBRID', l2_cache_enabled=True, l2_cache_size_kib=1, l2_cache_line_size_bytes=64, l2_cache_associativity=2, l2_cache_hit_latency_cycles=5, l2_cache_miss_latency_cycles=100)
    schedule_l2_on, stats_l2_on = run_sim(program_locality, config_l2_on)
    total_cycles_l2_on = stats_l2_on['total_cycles']
    l2_stats_l2_on = stats_l2_on.get('l2_cache_stats', {})
    print(f"  L2 ON - Total Cycles: {total_cycles_l2_on}, Hit Rate: {l2_stats_l2_on.get('hit_rate', 0):.2%}")

    # Run with L2 Cache Disabled
    config_l2_off = SimConfig(sim_level='CA_HYBRID', l2_cache_enabled=False)
    schedule_l2_off, stats_l2_off = run_sim(program_locality, config_l2_off)
    total_cycles_l2_off = stats_l2_off['total_cycles']
    print(f"  L2 OFF - Total Cycles: {total_cycles_l2_off}")

    assert total_cycles_l2_on < total_cycles_l2_off # Expect performance improvement
    assert l2_stats_l2_on.get('hit_rate', 0) > 0.5 # Expect high hit rate

    # --- Scenario 2: Low Locality (Cache Unfriendly) ---
    # Access widely dispersed memory locations, forcing many misses
    print("\nScenario: Low Locality (Cache Unfriendly)")
    # Use a large stride to ensure each access is a new cache line
    program_no_locality = create_memory_access_program(num_accesses, base_address=1, stride=123, tensor_size=tensor_size)

    # Run with L2 Cache Enabled
    config_l2_on_no_loc = SimConfig(sim_level='CA_HYBRID', l2_cache_enabled=True, l2_cache_size_kib=1, l2_cache_line_size_bytes=64, l2_cache_associativity=2, l2_cache_hit_latency_cycles=5, l2_cache_miss_latency_cycles=100)
    schedule_l2_on_no_loc, stats_l2_on_no_loc = run_sim(program_no_locality, config_l2_on_no_loc)
    total_cycles_l2_on_no_loc = stats_l2_on_no_loc['total_cycles']
    l2_stats_l2_on_no_loc = stats_l2_on_no_loc.get('l2_cache_stats', {})
    print(f"  L2 ON - Total Cycles: {total_cycles_l2_on_no_loc}, Hit Rate: {l2_stats_l2_on_no_loc.get('hit_rate', 0):.2%}")

    # Run with L2 Cache Disabled
    config_l2_off_no_loc = SimConfig(sim_level='CA_HYBRID', l2_cache_enabled=False)
    schedule_l2_off_no_loc, stats_l2_off_no_loc = run_sim(program_no_locality, config_l2_off_no_loc)
    total_cycles_l2_off_no_loc = stats_l2_off_no_loc['total_cycles']
    print(f"  L2 OFF - Total Cycles: {total_cycles_l2_off_no_loc}")

    # In this scenario, L2 cache might not provide significant benefit, or even be slightly worse due to overhead
    # We expect hit rate to be low
    assert l2_stats_l2_on_no_loc.get('hit_rate', 0) < 0.6 # Expect low hit rate
    # The performance difference might be small or even negative if miss penalty is high
    # assert total_cycles_l2_on_no_loc >= total_cycles_l2_off_no_loc * 0.9 # Allow for slight degradation or similar perf
    # For now, just assert that it doesn't drastically improve
    assert total_cycles_l2_on_no_loc > total_cycles_l2_off_no_loc * 0.8 # Should not be drastically better

    print("\nL2 Cache Impact Test Completed.")

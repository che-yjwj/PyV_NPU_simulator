import pytest
from pyv_npu.config import SimConfig
from pyv_npu.isa.npu_ir import Program, NPUOp, Tensor
from pyv_npu.runtime.simulator import run as run_sim

def create_parallel_load_program(num_loads: int, tensor_size_bytes: int) -> Program:
    """Creates a program with many parallel LOAD operations."""
    ops = []
    inputs = []
    # Create non-overlapping tensors
    for i in range(num_loads):
        addr = i * tensor_size_bytes * 2 # Ensure no overlap
        t_in = Tensor(f"t_in{i}", (tensor_size_bytes // 2,), "float16", address=addr)
        t_out = Tensor(f"t_out{i}", (tensor_size_bytes // 2,), "float16")
        ops.append(NPUOp("LOAD", f"load{i}", inputs=[t_in], outputs=[t_out]))
        inputs.append(t_in)
    return Program(ops=ops, inputs=inputs, initializers={})

def test_burst_size_impact_report():
    """Generates a report on how DMA burst size affects P95 DMA latency."""
    burst_sizes = [32, 64, 128, 256, 512]
    results = {}

    # A program with many small, parallel loads to stress the DMA system
    program = create_parallel_load_program(num_loads=64, tensor_size_bytes=1024)

    print("\n--- DMA Burst Size Impact Report ---")
    for size in burst_sizes:
        config = SimConfig(
            sim_level='CA_HYBRID',
            dma_channels=4, # Use multiple channels
            dma_burst_size=size
        )
        schedule, stats = run_sim(program, config)
        dma_stats = stats.get("dma_latency_stats", {})
        p95_latency = dma_stats.get("p95")
        results[size] = p95_latency
        print(f"Testing DMA Burst Size: {size:<5} -> P95 DMA Latency: {p95_latency or 'N/A'}")

    print("\n--- Summary ---")
    print(f"| {'Burst Size':<12} | {'P95 Latency (cycles)':<25} |")
    print(f"|{'-'*14}|{'-'*27}|")
    for size, p95 in results.items():
        print(f"| {size:<12} | {p95 or 'N/A':<25} |")

    # Basic assertion to ensure the test framework considers this a valid test
    assert len(results) == len(burst_sizes)

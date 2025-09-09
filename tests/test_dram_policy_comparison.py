import pytest
from pyv_npu.config import SimConfig
from pyv_npu.isa.npu_ir import Program, NPUOp, Tensor
from pyv_npu.runtime.simulator import run as run_sim

def create_program_with_parallel_loads(addr1, addr2):
    """Creates a program with two parallel LOAD operations to specific addresses."""
    t_in1 = Tensor("t_in1", (1024,), "float16", address=addr1)
    t_in2 = Tensor("t_in2", (1024,), "float16", address=addr2)
    t_out1 = Tensor("t_in1_spm", (1024,), "float16")
    t_out2 = Tensor("t_in2_spm", (1024,), "float16")
    op1 = NPUOp("LOAD", "load1", inputs=[t_in1], outputs=[t_out1])
    op2 = NPUOp("LOAD", "load2", inputs=[t_in2], outputs=[t_out2])
    prog = Program(ops=[op1, op2], inputs=[t_in1, t_in2], initializers={})
    return prog

def test_dram_policy_collision_comparison():
    print("\nDEBUG: Testing test_dram_policy_collision_comparison...")
    """
    Tests that different DRAM mapping policies result in different collision counts.
    """
    # --- Run with settings designed to cause collisions ---
    # Use a single channel and bank, so any parallel access will collide.
    config_collide = SimConfig(sim_level='CA_HYBRID', dram_channels=1, dram_banks_per_channel=1, dma_channels=2)
    prog_collide = create_program_with_parallel_loads(addr1=0, addr2=4096)
    schedule_collide, stats_collide = run_sim(prog_collide, config_collide)
    collisions_collide = stats_collide.get("dram_collisions", 0)
    print(f"Collisions with single bank: {collisions_collide}")
    assert collisions_collide > 0

    # --- Run with settings designed to avoid collisions ---
    # Use multiple channels. Addresses are chosen to map to different channels.
    config_no_collide = SimConfig(
        sim_level='CA_HYBRID',
        dram_channels=2,
        dram_banks_per_channel=2,
        dram_mapping_policy='interleave',
        dram_page_size=2048  # Set page size so addr 2048 maps to a new channel
    )
    # addr1 (0) maps to channel 0. addr2 (2048) maps to channel 1.
    prog_no_collide = create_program_with_parallel_loads(addr1=0, addr2=2048)
    schedule_no_collide, stats_no_collide = run_sim(prog_no_collide, config_no_collide)
    collisions_no_collide = stats_no_collide.get("dram_collisions", 0)
    print(f"Collisions with interleaved addresses: {collisions_no_collide}")

    # The main point is that the policy and hardware config has an effect.
    # The colliding case should have more collisions than the non-colliding case.
    assert collisions_collide > collisions_no_collide

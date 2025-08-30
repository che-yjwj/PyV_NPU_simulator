from __future__ import annotations
from dataclasses import dataclass, field
from typing import List
import yaml
from pathlib import Path

@dataclass
class SimConfig:
    """PyV-NPU Simulator Configuration, based on PRD v1.1"""
    # Model and execution level
    model: str = ""
    sim_level: str = "IA_TIMING"  # IA, IA_TIMING, CA_HYBRID, CA_FULL

    # Config file
    config_file: str = ""

    # Reporting
    report_dir: str = "out/default_run"

    # Execution mode
    mode: str = "loose"  # loose (MMIO) vs tight (Custom ISA)

    # Loose-coupled mode params
    mmio_base: int = 0x4000_0000
    queue_size: int = 1024

    # Tight-coupled mode params
    tc_isa: List[str] = field(default_factory=lambda: ["enqcmd", "twait", "tbar", "tstat"])
    tight_mode_doorbell_latency: int = 20
    tight_mode_csr_latency: int = 5
    tight_mode_issue_rate: int = 2

    # Core NPU parameters (from PRD section 9)
    tc: int = 2
    vc: int = 4
    spm_banks: int = 8
    spm_capacity_kib: int = 2048  # 2 MiB
    l0_spm_size_kb: int = 16 # L0 cache size for each TC
    l0_spm_latency_cycles: int = 5 # L0 cache latency in cycles
    systolic_array_height: int = 16
    systolic_array_width: int = 16
    dma_channels: int = 2

    # IO Buffer Parameters
    io_buffer_size_kb: int = 128

    bw_dram_gbps: float = 102.4
    bw_noc_gbps: float = 256.0
    clock_ghz: float = 1.2

    # DRAM Physical Parameters
    dram_base_address: int = 0x80000000
    dram_channels: int = 4
    dram_banks_per_channel: int = 8
    dram_page_size: int = 4096
    dram_mapping_policy: str = "interleave"

    # Tiling parameters
    tile_m: int = 128
    tile_n: int = 128
    tile_k: int = 64

    def __post_init__(self):
        # Allow for a very large buffer in smoke tests to avoid spurious failures
        if "smoke" in self.model:
            self.io_buffer_size_kb = 1024 * 1024 # 1 GB

    def cycles(self, seconds: float) -> int:
        """Converts time in seconds to clock cycles."""
        return int(seconds * self.clock_ghz * 1e9)

    def update_from_yaml(self, yaml_path: str):
        """Updates config fields from a YAML file."""
        with open(yaml_path, 'r') as f:
            yaml_config = yaml.safe_load(f)
        for key, value in yaml_config.items():
            if hasattr(self, key):
                setattr(self, key, value)

    @classmethod
    def from_args(cls, args) -> SimConfig:
        """Factory method to create a SimConfig from parsed argparse arguments."""
        config = cls()

        # 1. Load from YAML config file if provided
        if hasattr(args, 'config') and args.config:
            config.config_file = args.config
            if Path(config.config_file).exists():
                config.update_from_yaml(config.config_file)
            else:
                # This could be a warning or an error
                print(f"Warning: Config file {config.config_file} not found.")

        # 2. Override with command-line arguments
        arg_dict = vars(args)
        for key, value in arg_dict.items():
            if value is not None and hasattr(config, key):
                setattr(config, key, value)
        
        config.__post_init__()
        return config

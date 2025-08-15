from __future__ import annotations
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
    level: str = "L1"  # L0, L1, L2, L3

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
    te_isa: List[str] = field(default_factory=lambda: ["enqcmd", "twait", "tbar", "tstat"])

    # Core NPU parameters (from PRD section 9)
    te: int = 2
    ve: int = 4
    spm_banks: int = 8
    spm_capacity_kib: int = 2048  # 2 MiB
    systolic_array_height: int = 16
    systolic_array_width: int = 16
    dma_channels: int = 2
    bw_dram_gbps: float = 102.4
    bw_noc_gbps: float = 256.0
    clock_ghz: float = 1.2

    # Tiling parameters
    tile_m: int = 128
    tile_n: int = 128
    tile_k: int = 64

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
        # We check if the argument was explicitly provided by the user, 
        # to avoid overwriting YAML values with argparse defaults.
        arg_dict = vars(args)
        for key, value in arg_dict.items():
            # A simple way to check if an arg was set by the user is to see if it's not None
            # or not the default. This can be tricky. A common pattern is to check
            # against the parser's default values.
            if value is not None and hasattr(config, key):
                # For this to work well, argparse defaults should be None
                setattr(config, key, value)

        return config
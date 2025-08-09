from __future__ import annotations
from dataclasses import dataclass, field
from typing import List

@dataclass
class SimConfig:
    """PyV-NPU Simulator Configuration, based on PRD v1.1"""
    # Model and execution level
    model: str
    level: str = "L1"  # L0, L1, L2, L3

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

    @classmethod
    def from_args(cls, args) -> SimConfig:
        """Factory method to create a SimConfig from parsed argparse arguments."""
        # Core config from args
        config = cls(
            model=args.model,
            level=args.level,
            report_dir=args.report,
            mode=args.mode,
        )

        # Mode-specific params
        if config.mode == 'loose':
            config.mmio_base = args.mmio_base
            config.queue_size = args.queue_size
        elif config.mode == 'tight':
            if isinstance(args.isa, str):
                config.te_isa = args.isa.split(',')
            else:
                config.te_isa = args.isa

        # In a real scenario, one might override hw params from a yaml file here
        # e.g. if args.hw_config: config.update_from_yaml(args.hw_config)

        return config
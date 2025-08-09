
from dataclasses import dataclass

@dataclass
class SimulatorConfig:
    level: int = 1              # 1=functional+timing-lite, 2=cycle, 3=cycle+contention
    te_count: int = 1
    ve_count: int = 1
    sram_kib: int = 512         # local SRAM per engine (KiB)
    dram_bandwidth_gbps: float = 25.6
    clock_mhz: float = 800.0

    def cycles(self, seconds: float) -> int:
        return int(seconds * self.clock_mhz * 1e6)

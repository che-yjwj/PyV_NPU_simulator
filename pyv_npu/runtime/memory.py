
from dataclasses import dataclass

@dataclass
class MemoryStats:
    dram_bytes_read: int = 0
    dram_bytes_written: int = 0
    sram_spills: int = 0

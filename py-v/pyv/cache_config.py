
from dataclasses import dataclass
from typing import Literal

WritePolicy = Literal["Write-Back", "Write-Through"]

@dataclass
class CacheConfig:
    """Configuration for a cache memory system."""
    size_kb: int = 64
    line_size_bytes: int = 64
    associativity: int = 8
    hit_latency_cycles: int = 1
    miss_latency_cycles: int = 100  # Penalty to access next level of memory
    write_policy: WritePolicy = "Write-Back"

    def __post_init__(self):
        if not self.size_kb > 0:
            raise ValueError("Cache size must be positive.")
        if not self.line_size_bytes > 0:
            raise ValueError("Line size must be positive.")
        if not self.associativity > 0:
            raise ValueError("Associativity must be positive.")
        
        self.size_bytes = self.size_kb * 1024
        if self.size_bytes % self.line_size_bytes != 0:
            raise ValueError("Cache size must be a multiple of line size.")
        
        self.num_lines = self.size_bytes // self.line_size_bytes
        if self.num_lines % self.associativity != 0:
            raise ValueError("Number of lines must be a multiple of associativity.")
            
        self.num_sets = self.num_lines // self.associativity


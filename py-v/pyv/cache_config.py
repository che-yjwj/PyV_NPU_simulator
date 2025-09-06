
from dataclasses import dataclass, field
from typing import Literal

WritePolicy = Literal["Write-Back", "Write-Through"]

@dataclass
class CacheConfig:
    """Configuration for a cache memory system."""
    name: str = "Cache"
    size_kb: int = 64
    line_size_bytes: int = 64
    associativity: int = 8
    hit_latency_cycles: int = 1
    miss_latency_cycles: int = 100  # Penalty to access next level of memory
    write_policy: WritePolicy = "Write-Back"

    # Derived properties
    size_bytes: int = field(init=False)
    num_lines: int = field(init=False)
    num_sets: int = field(init=False)

    def __post_init__(self):
        if not self.size_kb > 0:
            raise ValueError("Cache size must be positive.")
        if not self.line_size_bytes > 0:
            raise ValueError("Line size must be positive.")
        if not self.associativity > 0:
            raise ValueError("Associativity must be positive.")

        def is_power_of_two(n):
            return (n > 0) and (n & (n - 1) == 0)

        if not is_power_of_two(self.line_size_bytes):
            raise ValueError("Line size must be a power of two for bitwise address decomposition.")
        
        self.size_bytes = self.size_kb * 1024
        if self.size_bytes % self.line_size_bytes != 0:
            raise ValueError("Cache size must be a multiple of line size.")
        
        self.num_lines = self.size_bytes // self.line_size_bytes
        if self.num_lines % self.associativity != 0:
            raise ValueError("Number of lines must be a multiple of associativity.")
            
        self.num_sets = self.num_lines // self.associativity
        if self.num_sets > 0 and not is_power_of_two(self.num_sets):
            raise ValueError("Number of sets must be a power of two for bitwise address decomposition.")


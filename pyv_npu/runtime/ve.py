
from dataclasses import dataclass

@dataclass
class VectorEngine:
    ops_per_cycle: int = 1024

    def estimate_cycles(self, length:int) -> int:
        return max(1, length // max(1, self.ops_per_cycle))

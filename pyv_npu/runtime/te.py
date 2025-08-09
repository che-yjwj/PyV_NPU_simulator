
from dataclasses import dataclass

@dataclass
class TensorEngine:
    macs_per_cycle: int = 1024   # placeholder capability

    def estimate_cycles(self, m:int, n:int, k:int) -> int:
        # simplistic GEMM estimate
        macs = m * n * k
        return max(1, macs // max(1, self.macs_per_cycle))

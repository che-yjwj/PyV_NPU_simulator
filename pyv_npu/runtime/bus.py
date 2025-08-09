
from dataclasses import dataclass

@dataclass
class SimulatedBus:
    bandwidth_gbps: float

    def transfer_ns(self, bytes_: int) -> float:
        # bytes -> bits / (Gbps) = seconds ; return ns
        if self.bandwidth_gbps <= 0: return 0.0
        seconds = (bytes_ * 8) / (self.bandwidth_gbps * 1e9)
        return seconds * 1e9

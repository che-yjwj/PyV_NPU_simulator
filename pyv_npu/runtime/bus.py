from dataclasses import dataclass
from typing import List, Tuple
from ..config import SimConfig

@dataclass
class SimulatedBus:
    bandwidth_gbps: float

    def transfer_ns(self, bytes_: int) -> float:
        if self.bandwidth_gbps <= 0: return 0.0
        seconds = (bytes_ * 8) / (self.bandwidth_gbps * 1e9)
        return seconds * 1e9

class SystemBusTracker:
    """Models a shared system bus with a fixed bandwidth."""
    def __init__(self, config: SimConfig):
        self.name = "system_bus"
        self.config = config
        self.bw_gbps = config.bw_bus_gbps
        self.timeline: List[Tuple[int, int]] = [] # Stores (end_cycle, num_bytes)

    def get_transfer_cycles(self, num_bytes: int) -> int:
        """Calculates the number of cycles required to transfer data over the bus."""
        if self.bw_gbps == 0: return 0
        seconds = (num_bytes * 8) / (self.bw_gbps * 1e9)
        return self.config.cycles(seconds)

    def probe_transfer(self, start_cycle: int, num_bytes: int) -> Tuple[int, int, str]:
        """Probes for the earliest time a transfer can complete and identifies stalls."""
        duration = self.get_transfer_cycles(num_bytes)
        if duration == 0:
            return start_cycle, 0, "NONE"

        last_end_cycle = self.timeline[-1][0] if self.timeline else 0
        actual_start = max(start_cycle, last_end_cycle)
        
        stall_reason = "RESOURCE_BUS" if actual_start > start_cycle else "NONE"

        return actual_start, duration, stall_reason

    def commit_transfer(self, start_cycle: int, duration: int, num_bytes: int):
        """Commits the transfer to the bus timeline."""
        end_cycle = start_cycle + duration
        self.timeline.append((end_cycle, num_bytes))
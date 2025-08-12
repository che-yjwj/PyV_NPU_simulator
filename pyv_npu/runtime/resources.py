from __future__ import annotations
from ..config import SimConfig
from typing import List, Tuple

class BandwidthTracker:
    """Models a resource with a fixed bandwidth (e.g., DRAM, NoC)."""
    def __init__(self, name: str, config: SimConfig):
        self.name = name
        self.config = config
        # Timeline of (end_cycle, bytes_transferred)
        self.timeline: List[Tuple[int, int]] = []
        if name == "dram":
            self.bw_gbps = config.bw_dram_gbps
        elif name == "noc":
            self.bw_gbps = config.bw_noc_gbps
        else:
            raise ValueError(f"Unknown bandwidth tracker: {name}")

    def get_transfer_cycles(self, num_bytes: int) -> int:
        """Calculates cycles to transfer data based on bandwidth."""
        if self.bw_gbps == 0: return 0
        seconds = num_bytes / (self.bw_gbps * 1e9)
        return self.config.cycles(seconds)

    def book_transfer(self, start_cycle: int, num_bytes: int) -> int:
        """
        Books a data transfer, returning the end cycle.
        This is a simple model that assumes sequential use of the bus.
        A more complex model would handle overlapping requests.
        """
        # Find the latest end time of any previous transfer
        last_end_cycle = 0
        if self.timeline:
            last_end_cycle = self.timeline[-1][0]
        
        # The transfer can only start after the bus is free
        actual_start_cycle = max(start_cycle, last_end_cycle)
        
        duration = self.get_transfer_cycles(num_bytes)
        end_cycle = actual_start_cycle + duration
        
        self.timeline.append((end_cycle, num_bytes))
        return end_cycle

class BankTracker:
    """Models SPM bank contention."""
    def __init__(self, config: SimConfig):
        self.num_banks = config.spm_banks
        # List of timelines, one for each bank
        # Each timeline is a list of (start_cycle, end_cycle) tuples
        self.bank_timelines: List[List[Tuple[int, int]]] = [[] for _ in range(self.num_banks)]

    def find_earliest_free_slot(self, start_cycle: int, duration: int, num_banks_needed: int) -> int:
        """
        Finds the earliest time a given number of banks are free for a specific duration.
        This is a greedy approach.
        """
        if num_banks_needed > self.num_banks:
            raise ValueError(f"Requesting {num_banks_needed} banks, but only {self.num_banks} exist.")

        current_try_cycle = start_cycle
        
        while True:
            free_banks = []
            for i in range(self.num_banks):
                is_free = True
                # Check for overlaps in this bank's timeline
                for busy_start, busy_end in self.bank_timelines[i]:
                    if not (current_try_cycle + duration <= busy_start or current_try_cycle >= busy_end):
                        is_free = False
                        break
                if is_free:
                    free_banks.append(i)
            
            if len(free_banks) >= num_banks_needed:
                # Found a slot, book it and return
                chosen_banks = free_banks[:num_banks_needed]
                for bank_idx in chosen_banks:
                    self.bank_timelines[bank_idx].append((current_try_cycle, current_try_cycle + duration))
                    # Keep timelines sorted
                    self.bank_timelines[bank_idx].sort()
                return current_try_cycle
            
            # If no slot found, advance time and retry.
            # A smarter implementation would jump to the next available slot time.
            current_try_cycle += 1

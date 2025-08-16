from __future__ import annotations
from ..config import SimConfig
from typing import List, Tuple
from .memory import DramAddressMapper

class BandwidthTracker:
    """Models a resource with a fixed bandwidth (e.g., DRAM, NoC)."""
    def __init__(self, name: str, config: SimConfig):
        self.name = name
        self.config = config
        self.timeline: List[Tuple[int, int]] = []
        if name == "dram":
            self.bw_gbps = config.bw_dram_gbps
        elif name == "noc":
            self.bw_gbps = config.bw_noc_gbps
        else:
            raise ValueError(f"Unknown bandwidth tracker: {name}")

    def get_transfer_cycles(self, num_bytes: int) -> int:
        if self.bw_gbps == 0: return 0
        seconds = num_bytes / (self.bw_gbps * 1e9)
        return self.config.cycles(seconds)

    def book_transfer(self, start_cycle: int, num_bytes: int) -> int:
        last_end_cycle = self.timeline[-1][0] if self.timeline else 0
        actual_start_cycle = max(start_cycle, last_end_cycle)
        duration = self.get_transfer_cycles(num_bytes)
        end_cycle = actual_start_cycle + duration
        self.timeline.append((end_cycle, num_bytes))
        return end_cycle

class BankTracker:
    """Models SPM bank contention. Refactored to separate probe and commit."""
    def __init__(self, config: SimConfig):
        self.num_banks = config.spm_banks
        self.bank_timelines: List[List[Tuple[int, int]]] = [[] for _ in range(self.num_banks)]

    def probe_earliest_free_slot(self, start_cycle: int, duration: int, num_banks_needed: int) -> Tuple[int, List[int]]:
        if num_banks_needed > self.num_banks:
            raise ValueError(f"Requesting {num_banks_needed} banks, but only {self.num_banks} exist.")
        
        current_try_cycle = start_cycle
        while True:
            free_banks = []
            for i in range(self.num_banks):
                is_free = True
                for busy_start, busy_end in self.bank_timelines[i]:
                    if not (current_try_cycle + duration <= busy_start or current_try_cycle >= busy_end):
                        is_free = False
                        break
                if is_free:
                    free_banks.append(i)
            
            if len(free_banks) >= num_banks_needed:
                return current_try_cycle, free_banks[:num_banks_needed]
            
            current_try_cycle += 1

    def commit_slot(self, cycle: int, duration: int, chosen_banks: List[int]):
        for bank_idx in chosen_banks:
            self.bank_timelines[bank_idx].append((cycle, cycle + duration))
            self.bank_timelines[bank_idx].sort()

class DramBankTracker:
    """Models DRAM channel and bank contention. Refactored for probe/commit."""
    def __init__(self, config: SimConfig):
        self.config = config
        self.mapper = DramAddressMapper(config)
        self.num_channels = config.dram_channels
        self.num_banks_per_channel = config.dram_banks_per_channel
        self.bank_free_time: List[List[int]] = [
            [0] * self.num_banks_per_channel for _ in range(self.num_channels)
        ]
        self.collisions = 0

    def get_transfer_cycles(self, num_bytes: int) -> int:
        if self.config.bw_dram_gbps == 0 or self.num_channels == 0:
            return 0
        channel_bw_gbps = self.config.bw_dram_gbps / self.num_channels
        seconds = num_bytes / (channel_bw_gbps * 1e9)
        return self.config.cycles(seconds)

    def probe_transfer(self, start_cycle: int, address: int, num_bytes: int) -> Tuple[int, int, str, int, int]:
        if address is None:
            duration = self.get_transfer_cycles(num_bytes)
            return start_cycle, duration, "NONE", -1, -1

        channel_id, bank_id = self.mapper.map(address)
        duration = self.get_transfer_cycles(num_bytes)
        if duration == 0:
            return start_cycle, 0, "NONE", channel_id, bank_id

        bank_available_cycle = self.bank_free_time[channel_id][bank_id]
        actual_start = max(start_cycle, bank_available_cycle)
        stall_reason = "RESOURCE_DRAM_BANK" if actual_start > start_cycle else "NONE"
        
        return actual_start, duration, stall_reason, channel_id, bank_id

    def commit_transfer(self, channel_id: int, bank_id: int, start_cycle: int, duration: int):
        if channel_id < 0:
            return # Address was None, nothing to commit
        self.bank_free_time[channel_id][bank_id] = start_cycle + duration

class IssueQueueTracker:
    """Models the NPU's internal command issue queue."""
    def __init__(self, config: SimConfig):
        self.issue_rate = config.tight_mode_issue_rate
        self.next_issue_cycle = 0

    def probe_issue_time(self, start_cycle: int) -> int:
        """Calculates when the next command can be issued."""
        return max(start_cycle, self.next_issue_cycle)

    def commit_issue(self, issue_cycle: int):
        """Commits the command issue, updating the next available time."""
        self.next_issue_cycle = issue_cycle + self.issue_rate

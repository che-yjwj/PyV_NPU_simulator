from __future__ import annotations
from ..config import SimConfig
from typing import List, Tuple, Dict
from collections import deque
from ..isa.npu_ir import Tensor, DTYPE_MAP
from .memory import DramAddressMapper

class IOBufferTracker:
    """Models a simple FIFO buffer for data between DMA and Compute."""
    def __init__(self, config: SimConfig, name: str):
        self.name = name
        self.capacity_bytes = config.io_buffer_size_kb * 1024
        self.current_fill_bytes = 0
        self.queue = deque()
        self.resident_tensors = set()

    def can_push(self, num_bytes: int) -> bool:
        """Check if there is enough space to push data."""
        return self.current_fill_bytes + num_bytes <= self.capacity_bytes

    def can_pop(self, num_bytes: int) -> bool:
        """Check if there is enough data to pop."""
        return self.current_fill_bytes >= num_bytes

    def can_pop_tensor(self, tensor_name: str) -> bool:
        """Check if a specific tensor is available in the buffer."""
        return tensor_name in self.resident_tensors

    def push(self, num_bytes: int, tensor_name: str):
        """Push a tensor's data into the buffer."""
        if not self.can_push(num_bytes):
            raise ValueError(f"[{self.name}] Buffer overflow! Cannot push {num_bytes} bytes.")
        self.current_fill_bytes += num_bytes
        self.queue.append((num_bytes, tensor_name))
        self.resident_tensors.add(tensor_name)

    def pop(self, num_bytes: int, tensor_name: str):
        """Pop a specific tensor's data from the buffer."""
        if not self.can_pop_tensor(tensor_name):
            raise ValueError(f"[{self.name}] Tensor {tensor_name} not in buffer for popping.")
        if not self.can_pop(num_bytes):
            raise ValueError(f"[{self.name}] Buffer underflow! Cannot pop {num_bytes} bytes.")
        
        # This is a simplified model. A real FIFO would pop from the front.
        # Here we assume any resident tensor can be popped.
        self.current_fill_bytes -= num_bytes
        self.resident_tensors.remove(tensor_name)
        # In a more complex model, we would need to manage the deque properly


class L0SPMTracker:
    """Models the L0 SPM cache for a single Tensor Core."""
    def __init__(self, config: SimConfig):
        self.size_bytes = config.l0_spm_size_kb * 1024
        self.latency = config.l0_spm_latency_cycles
        self.resident_tensors: Dict[str, Tensor] = {}
        self.timeline: List[Tuple[int, int]] = []

    def probe_hit(self, tensors: List[Tensor]) -> bool:
        """Checks if all required tensors are resident in the L0 SPM."""
        return all(t.name in self.resident_tensors for t in tensors)

    def get_required_load_bytes(self, tensors: List[Tensor]) -> int:
        """Calculates the total bytes that need to be loaded into L0."""
        bytes_to_load = 0
        for t in tensors:
            if t.name not in self.resident_tensors:
                bytes_to_load += t.num_elements * DTYPE_MAP.get(t.dtype, 1)
        return bytes_to_load

    def commit_load(self, tensors: List[Tensor]):
        """Evicts old tensors and loads new ones (simplified LRU)."""
        self.resident_tensors = {t.name: t for t in tensors}


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
        while current_try_cycle < start_cycle + 1000000: # Add a timeout
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
        return -1, [] # Indicate failure

    def commit_slot(self, cycle: int, duration: int, chosen_banks: List[int]):
        for bank_idx in chosen_banks:
            self.bank_timelines[bank_idx].append((cycle, cycle + duration))
            self.bank_timelines[bank_idx].sort()

class DramBankTracker:
    """Models DRAM channel and bank contention using timelines."""
    def __init__(self, config: SimConfig):
        self.config = config
        self.mapper = DramAddressMapper(config)
        self.num_channels = config.dram_channels
        self.channel_timelines: List[List[Tuple[int, int]]] = [[] for _ in range(self.num_channels)]
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

        last_end_cycle = self.channel_timelines[channel_id][-1][0] if self.channel_timelines[channel_id] else 0
        actual_start = max(start_cycle, last_end_cycle)
        
        stall_reason = "RESOURCE_DRAM_BANK" if actual_start > start_cycle else "NONE"
        if stall_reason == "RESOURCE_DRAM_BANK":
            self.collisions += 1

        return actual_start, duration, stall_reason, channel_id, bank_id

    def commit_transfer(self, channel_id: int, bank_id: int, start_cycle: int, duration: int):
        if channel_id < 0:
            return
        end_cycle = start_cycle + duration
        self.channel_timelines[channel_id].append((end_cycle, duration))
        self.channel_timelines[channel_id].sort()

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

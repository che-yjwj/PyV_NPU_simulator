from __future__ import annotations
import math # Moved import to top
from ..config import SimConfig
from typing import List, Tuple, Dict, Any # Added Any
from collections import deque
from ..isa.npu_ir import Tensor, DTYPE_MAP
from .memory import DramAddressMapper

class IOBufferTracker:
    """Models a buffer for data between DMA and Compute.
    This is not a strict FIFO, allowing out-of-order consumption,
    but uses a deque to be able to transition to a strict FIFO later.
    """
    def __init__(self, config: SimConfig, name: str):
        self.name = name
        self.capacity_bytes = config.io_buffer_size_kb * 1024
        self.current_fill_bytes = 0
        self.queue = deque() # Stores (num_bytes, tensor_name)

    def can_push(self, num_bytes: int) -> bool:
        """Check if there is enough space to push data."""
        return self.current_fill_bytes + num_bytes <= self.capacity_bytes

    def push(self, num_bytes: int, tensor_name: str):
        """Push a tensor's data into the buffer."""
        if not self.can_push(num_bytes):
            raise ValueError(f"[{self.name}] Buffer overflow! Cannot push {num_bytes} bytes.")
        self.current_fill_bytes += num_bytes
        self.queue.append((num_bytes, tensor_name))

    def can_pop(self, num_bytes: int | None = None) -> bool:
        """Check if the buffer is not empty and optionally if the next item has the expected size."""
        if not self.queue:
            return False
        if num_bytes is not None:
            return self.queue[0][0] == num_bytes
        return True

    def pop(self) -> tuple[int, str]:
        """Pop the next tensor from the buffer in FIFO order."""
        if not self.queue:
            raise ValueError(f"[{self.name}] Buffer underflow! Cannot pop from an empty buffer.")
        
        num_bytes, tensor_name = self.queue.popleft()
        self.current_fill_bytes -= num_bytes
        return num_bytes, tensor_name

    def peek(self) -> tuple[int, str] | None:
        """Peek at the next item in the buffer without removing it."""
        if not self.queue:
            return None
        return self.queue[0]


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

        # Factor in DMA burst size for DRAM transfers
        if self.name == "dram" and self.config.dma_burst_size > 0:
            burst_size = self.config.dma_burst_size
            num_bytes = (num_bytes + burst_size - 1) // burst_size * burst_size

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
        self.num_ports = config.spm_bank_ports
        self.bank_timelines: List[List[Tuple[int, int]]] = [[] for _ in range(self.num_banks)]

    def probe_earliest_free_slot(self, start_cycle: int, duration: int, num_banks_needed: int) -> Tuple[int, List[int]]:
        if num_banks_needed > self.num_banks:
            raise ValueError(f"Requesting {num_banks_needed} banks, but only {self.num_banks} exist.")
        
        current_try_cycle = start_cycle
        while current_try_cycle < start_cycle + 1000000: # Add a timeout
            # Check for port availability at the current cycle
            num_busy_at_start = 0
            for i in range(self.num_banks):
                for busy_start, busy_end in self.bank_timelines[i]:
                    if current_try_cycle >= busy_start and current_try_cycle < busy_end:
                        num_busy_at_start += 1
                        break # Check next bank
            
            if self.num_ports - num_busy_at_start < num_banks_needed:
                current_try_cycle += 1
                continue

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

class L2CacheTracker:
    """Models an L2 cache with set-associativity and LRU replacement."""
    def __init__(self, config: SimConfig):
        self.hits = 0
        self.misses = 0

        if not config.l2_cache_enabled:
            self.enabled = False
            return

        self.enabled = True
        self.size_bytes = config.l2_cache_size_kib * 1024
        self.line_size = config.l2_cache_line_size_bytes
        self.associativity = config.l2_cache_associativity
        self.hit_latency = config.l2_cache_hit_latency_cycles
        self.miss_latency = config.l2_cache_miss_latency_cycles
        
        if self.size_bytes == 0 or self.line_size == 0 or self.associativity == 0:
            raise ValueError("L2 cache parameters must be non-zero.")

        self.num_lines = int(self.size_bytes // self.line_size)
        self.num_sets = int(self.num_lines // self.associativity)
        
        if self.num_sets == 0:
            raise ValueError("L2 cache size is too small for the given associativity.")

        # Cache storage: list of deques, where each deque is a set
        # Each element in the deque is the tag
        self.sets: List[deque] = [deque(maxlen=self.associativity) for _ in range(self.num_sets)]

        # Address bit calculation
        # Ensure line_size and num_sets are positive for log2
        if self.line_size <= 0 or self.num_sets <= 0:
            raise ValueError("line_size and num_sets must be positive for log2 calculation.")

        self.offset_bits = int(math.log2(self.line_size))
        self.index_bits = int(math.log2(self.num_sets))
        self.index_mask = (1 << self.index_bits) - 1

        self.hits = 0
        self.misses = 0

    def _get_address_parts(self, address: int) -> Tuple[int, int]:
        """Extracts tag and index from a memory address."""
        index = (address >> self.offset_bits) & self.index_mask
        tag = address >> (self.offset_bits + self.index_bits)
        return tag, index

    def access(self, address: int) -> Tuple[bool, int]:
        """
        Accesses the cache with a given address.
        Returns a tuple: (is_hit, latency_in_cycles).
        """
        if not self.enabled:
            return False, 0 # No latency contribution if disabled

        tag, index = self._get_address_parts(address)
        target_set = self.sets[index]

        # Check for hit
        if tag in target_set:
            self.hits += 1
            # Move the accessed tag to the end (most recently used)
            target_set.remove(tag)
            target_set.append(tag)
            return True, self.hit_latency

        # Handle miss
        self.misses += 1
        # The deque will automatically handle eviction of the least recently used
        # item if it's full when we append.
        target_set.append(tag)
        return False, self.miss_latency

    def get_stats(self) -> Dict[str, Any]:
        """Returns a dictionary of cache statistics."""
        total_accesses = self.hits + self.misses
        if total_accesses == 0:
            return {"hit_rate": 0, "miss_rate": 0, "hits": 0, "misses": 0}
        
        hit_rate = self.hits / total_accesses
        miss_rate = self.misses / total_accesses
        return {"hit_rate": hit_rate, "miss_rate": miss_rate, "hits": self.hits, "misses": self.misses}

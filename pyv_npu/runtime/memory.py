from __future__ import annotations
from ..config import SimConfig

class DramAddressMapper:
    """Maps a physical DRAM address to a channel and bank."""

    def __init__(self, config: SimConfig):
        self.policy = config.dram_mapping_policy
        self.channels = config.dram_channels
        self.banks_per_channel = config.dram_banks_per_channel
        
        # 1. Validate config parameter
        self.row_size = config.dram_page_size
        if self.row_size <= 0:
            raise ValueError("DRAM page size (row_size) must be a positive integer.")

        # 2. Refactor to a dispatch dictionary for extensibility
        self._map_policies = {
            "interleave": self._map_interleave,
        }
        if self.policy not in self._map_policies:
            raise ValueError(f"Unknown or unsupported DRAM mapping policy: {self.policy}")
        self._map_func = self._map_policies[self.policy]

    def map(self, address: int) -> tuple[int, int]:
        """Maps a physical address to (channel_id, bank_id) using the configured policy."""
        return self._map_func(address)

    def _map_interleave(self, address: int) -> tuple[int, int]:
        """A simple policy: interleave channels based on low-order bits of the row number."""
        row_number = address // self.row_size
        channel_id = row_number % self.channels
        
        # Interleave banks based on the next bits
        bank_id = (row_number // self.channels) % self.banks_per_channel
        
        return channel_id, bank_id
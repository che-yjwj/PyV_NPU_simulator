from __future__ import annotations
from ..config import SimConfig

class DramAddressMapper:
    """Maps a physical DRAM address to a channel and bank."""

    def __init__(self, config: SimConfig):
        self.policy = config.dram_mapping_policy
        self.channels = config.dram_channels
        self.banks_per_channel = config.dram_banks_per_channel
        # Use page size from config instead of hardcoded value
        self.row_size = config.dram_page_size

    def map(self, address: int) -> tuple[int, int]:
        """
        Maps a physical address to (channel_id, bank_id).
        Supports different mapping policies based on config.
        """
        if self.policy == "interleave":
            # A simple policy: interleave channels based on low-order bits of the row number
            row_number = address // self.row_size
            channel_id = row_number % self.channels
            
            # Interleave banks based on the next bits
            bank_id = (row_number // self.channels) % self.banks_per_channel
            
            return channel_id, bank_id
        else:
            raise ValueError(f"Unknown or unsupported DRAM mapping policy: {self.policy}")
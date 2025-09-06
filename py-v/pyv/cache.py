
from collections import OrderedDict
from .cache_config import CacheConfig

class CacheLine:
    """Represents a single line in a cache set."""
    def __init__(self, line_size_bytes: int):
        self.valid = False
        self.dirty = False
        self.tag = -1
        self.data = bytearray(line_size_bytes)

class CacheSet:
    """Represents a set of cache lines, implementing LRU replacement."""
    def __init__(self, associativity: int, line_size_bytes: int):
        self.lines = OrderedDict()
        for _ in range(associativity):
            line = CacheLine(line_size_bytes)
            self.lines[id(line)] = line # Use id as a unique key for the OrderedDict

    def find_line(self, tag: int) -> CacheLine | None:
        for line in self.lines.values():
            if line.valid and line.tag == tag:
                # Move the accessed line to the end to mark it as most recently used
                self.lines.move_to_end(id(line))
                return line
        return None

    def get_lru_line(self) -> CacheLine:
        # The first item in the OrderedDict is the least recently used
        lru_line_id = next(iter(self.lines))
        lru_line = self.lines[lru_line_id]
        self.lines.move_to_end(lru_line_id)
        return lru_line

class L1Cache:
    """A configurable L1 Cache with LRU replacement and Write-Back policy."""
    def __init__(self, config: CacheConfig, lower_memory):
        self.config = config
        self.lower_memory = lower_memory # Next level of memory (e.g., L2 cache or main memory)
        self.sets = [CacheSet(config.associativity, config.line_size_bytes) for _ in range(config.num_sets)]
        
        # Calculate bit shifts and masks for address decomposition
        self.offset_bits = self.config.line_size_bytes.bit_length() - 1
        self.index_bits = self.config.num_sets.bit_length() - 1
        self.offset_mask = (1 << self.offset_bits) - 1
        self.index_mask = ((1 << self.index_bits) - 1) << self.offset_bits

    def _decompose_address(self, address: int) -> tuple[int, int, int]:
        """Decomposes an address into tag, index, and offset."""
        offset = address & self.offset_mask
        index = (address & self.index_mask) >> self.offset_bits
        tag = address >> (self.offset_bits + self.index_bits)
        return tag, index, offset

    def read(self, address: int) -> tuple[bytearray, bool]:
        """Read data from the cache. Returns (data, hit)."""
        tag, index, offset = self._decompose_address(address)
        cache_set = self.sets[index]
        line = cache_set.find_line(tag)

        if line:
            # Cache Hit
            return line.data[offset], True
        else:
            # Cache Miss
            self._handle_miss(address, index)
            # After miss handling, the line is in the cache, so we read again.
            line = cache_set.find_line(tag)
            return line.data[offset], False

    def write(self, address: int, value: int):
        """Write data to the cache."""
        tag, index, offset = self._decompose_address(address)
        cache_set = self.sets[index]
        line = cache_set.find_line(tag)

        if line:
            # Cache Hit
            line.data[offset] = value
            if self.config.write_policy == "Write-Back":
                line.dirty = True
            # In a Write-Through policy, we would write to lower_memory here.
        else:
            # Cache Miss
            self._handle_miss(address, index)
            line = cache_set.find_line(tag)
            line.data[offset] = value
            if self.config.write_policy == "Write-Back":
                line.dirty = True

    def _handle_miss(self, address: int, index: int):
        """Handles a cache miss by fetching data from lower memory."""
        cache_set = self.sets[index]
        victim_line = cache_set.get_lru_line()

        if victim_line.valid and victim_line.dirty:
            # Write-Back the victim line to lower memory
            old_address = self._reconstruct_address(victim_line.tag, index)
            self.lower_memory.write_block(old_address, victim_line.data)

        # Fetch the new block from lower memory
        block_start_address = address & ~self.offset_mask
        new_data = self.lower_memory.read_block(block_start_address, self.config.line_size_bytes)
        
        # Update the victim line with the new data
        new_tag, _, _ = self._decompose_address(address)
        victim_line.valid = True
        victim_line.dirty = False
        victim_line.tag = new_tag
        victim_line.data[:] = new_data

    def _reconstruct_address(self, tag: int, index: int) -> int:
        """Reconstructs the block start address from tag and index."""
        return (tag << (self.index_bits + self.offset_bits)) | (index << self.offset_bits)



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
        # Use OrderedDict to maintain LRU order. The first item is the LRU.
        self.lines = OrderedDict([(i, CacheLine(line_size_bytes)) for i in range(associativity)])

    def find_line(self, tag: int) -> tuple[int, CacheLine] | tuple[None, None]:
        """Finds a line with a given tag. If found, moves it to the end (MRU)."""
        for key, line in self.lines.items():
            if line.valid and line.tag == tag:
                self.lines.move_to_end(key)
                return key, line
        return None, None

    def get_lru_line(self) -> tuple[int, CacheLine]:
        """Gets the least recently used line and moves it to the end (making it MRU)."""
        # The first item in the OrderedDict is the least recently used
        lru_key = next(iter(self.lines))
        lru_line = self.lines[lru_key]
        self.lines.move_to_end(lru_key)
        return lru_key, lru_line

class L1Cache:
    """
    A configurable L1 Cache.
    This class is responsible for cache logic (hit/miss, LRU) but not for timing.
    The MemorySystem is responsible for handling the timing of miss penalties.
    """
    def __init__(self, config: CacheConfig, memory):
        self.config = config
        self.memory = memory
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

    def read(self, address: int) -> tuple[int, bool]:
        """
        Reads a byte from the cache. Handles misses and write-backs.
        Returns (value, hit_status).
        """
        hit, line, tag, index, offset = self.probe(address)

        if hit:
            return self.read_from_line(line, offset), True

        # Handle Miss
        victim_line = self.get_victim_line(index)

        # Write-back if victim is dirty
        if victim_line.valid and victim_line.dirty:
            victim_address = self.reconstruct_address(victim_line.tag, index)
            self.memory.write_block(victim_address, victim_line.data)

        # Fetch new block from memory
        block_start_address = address & ~self.offset_mask
        new_data = self.memory.read_block(block_start_address, self.config.line_size_bytes)

        # Fill the cache line
        self.fill_line(victim_line, tag, new_data)

        return self.read_from_line(victim_line, offset), False

    def write(self, address: int, value: int) -> bool:
        """
        Writes a byte to the cache. Handles misses (write-allocate) and write-backs.
        Returns hit_status.
        """
        hit, line, tag, index, offset = self.probe(address)

        if hit:
            self.write_to_line(line, offset, value)
            return True

        # Handle Miss (Write-Allocate)
        victim_line = self.get_victim_line(index)

        # Write-back if victim is dirty
        if victim_line.valid and victim_line.dirty:
            victim_address = self.reconstruct_address(victim_line.tag, index)
            self.memory.write_block(victim_address, victim_line.data)

        # Fetch new block from memory
        block_start_address = address & ~self.offset_mask
        new_data = self.memory.read_block(block_start_address, self.config.line_size_bytes)

        # Fill the cache line
        self.fill_line(victim_line, tag, new_data)

        # Perform the write on the new line
        self.write_to_line(victim_line, offset, value)

        return False

    def probe(self, address: int) -> tuple[bool, CacheLine | None, int, int, int]:
        """
        Probes the cache for a given address.
        Returns (hit, line, tag, index, offset).
        """
        tag, index, offset = self._decompose_address(address)
        cache_set = self.sets[index]
        _, line = cache_set.find_line(tag)
        hit = line is not None
        return hit, line, tag, index, offset

    def read_from_line(self, line: CacheLine, offset: int) -> int:
        """Reads a byte from a given cache line at a specific offset."""
        return line.data[offset]

    def write_to_line(self, line: CacheLine, offset: int, value: int):
        """Writes a byte to a given cache line at a specific offset."""
        line.data[offset] = value
        if self.config.write_policy == "Write-Back":
            line.dirty = True

    def get_victim_line(self, index: int) -> CacheLine:
        """Gets the victim line for a given set index using LRU policy."""
        cache_set = self.sets[index]
        _, victim_line = cache_set.get_lru_line()
        return victim_line

    def fill_line(self, line: CacheLine, tag: int, block_data: bytearray):
        """Fills a cache line with new data from memory."""
        line.valid = True
        line.dirty = False
        line.tag = tag
        line.data[:] = block_data

    def reconstruct_address(self, tag: int, index: int) -> int:
        """Reconstructs the block start address from tag and index."""
        return (tag << (self.index_bits + self.offset_bits)) | (index << self.offset_bits)


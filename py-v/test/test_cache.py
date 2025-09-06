
import unittest
import sys
import os

# Add the parent directory of 'pyv' to the python path
# This is a common pattern for running tests in a sub-directory
# that is also a package.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pyv.cache_config import CacheConfig
from pyv.cache import L1Cache

class MockMemory:
    """A mock lower memory to test the cache against."""
    def __init__(self):
        self.mem = bytearray(2**16) # 64KB of memory

    def read_block(self, address: int, size: int) -> bytearray:
        return self.mem[address:address+size]

    def write_block(self, address: int, data: bytearray):
        self.mem[address:address+len(data)] = data

class TestL1Cache(unittest.TestCase):

    def setUp(self):
        """Set up a default cache configuration and mock memory for each test."""
        self.config = CacheConfig(
            size_kb=1,
            line_size_bytes=64,
            associativity=2
        )
        self.mock_memory = MockMemory()
        self.cache = L1Cache(self.config, self.mock_memory)

    def test_address_decomposition(self):
        """Verify that addresses are correctly decomposed into tag, index, and offset."""
        # 1KB cache, 64B lines -> 16 lines. 2-way assoc -> 8 sets.
        # 6 bits for offset (2^6=64), 3 bits for index (2^3=8)
        # Address: 0b_TAG_INDEX_OFFSET
        # Address: 0b_..._101_101010 (Tag, Index=5, Offset=42)
        address = 0b1111_101_101010
        tag, index, offset = self.cache._decompose_address(address)
        self.assertEqual(tag, 0b1111)
        self.assertEqual(index, 5)
        self.assertEqual(offset, 42)

    def test_read_miss_and_hit(self):
        """Test a read miss followed by a read hit."""
        address = 0x1234
        self.mock_memory.mem[address] = 0xAB

        # 1. First read: Miss
        value, hit = self.cache.read(address)
        self.assertFalse(hit, "First read should be a miss")
        self.assertEqual(value, 0xAB, "Value read on miss should be correct")

        # 2. Second read: Hit
        # Modify memory to ensure the cache is being used
        self.mock_memory.mem[address] = 0xFF 
        value, hit = self.cache.read(address)
        self.assertTrue(hit, "Second read should be a hit")
        self.assertEqual(value, 0xAB, "Value read on hit should come from cache")

    def test_write_back_on_eviction(self):
        """Test that a dirty line is written back to memory upon eviction."""
        # Address 1 and 2 map to the same set
        address1 = 0x1000 # index 0
        address2 = 0x2000 # index 0 (since there are 8 sets, 0x1000 wraps around)
        address3 = 0x3000 # index 0

        # Write to address1, making its line dirty
        self.cache.write(address1, 0xCC)
        tag1, index1, _ = self.cache._decompose_address(address1)
        _, line1 = self.cache.sets[index1].find_line(tag1)
        self.assertTrue(line1.dirty, "Line should be dirty after write")

        # Access address2, should not evict address1 yet (2-way associativity)
        self.cache.read(address2)

        # Access address3, this should evict the line for address1 (LRU)
        self.cache.read(address3)

        # Verify that the value 0xCC was written back to the mock memory
        block_start_addr1 = address1 & ~self.cache.offset_mask
        offset1 = address1 & self.cache.offset_mask
        self.assertEqual(self.mock_memory.mem[block_start_addr1 + offset1], 0xCC)

if __name__ == '__main__':
    unittest.main()

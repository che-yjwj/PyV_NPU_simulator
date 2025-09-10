
import pytest
from pyv_npu.config import SimConfig
from pyv_npu.runtime.resources import BankTracker, IOBufferTracker, L2CacheTracker


@pytest.fixture
def config():
    """Default config for resource tests."""
    return SimConfig()


def test_io_buffer_fifo_behavior(config: SimConfig):
    """Tests that the IOBufferTracker behaves like a FIFO queue."""
    config.io_buffer_size_kb = 1024  # 1MB buffer
    tracker = IOBufferTracker(config, "test_buffer")

    # Push three items
    tracker.push(100, "tensor_A")
    tracker.push(200, "tensor_B")
    tracker.push(50, "tensor_C")

    # Check current fill level
    assert tracker.current_fill_bytes == 350

    # Pop them and check order
    size1, name1 = tracker.pop()
    assert size1 == 100
    assert name1 == "tensor_A"
    assert tracker.current_fill_bytes == 250

    size2, name2 = tracker.pop()
    assert size2 == 200
    assert name2 == "tensor_B"
    assert tracker.current_fill_bytes == 50

    size3, name3 = tracker.pop()
    assert size3 == 50
    assert name3 == "tensor_C"
    assert tracker.current_fill_bytes == 0

    # Test underflow
    with pytest.raises(ValueError, match="Buffer underflow"):
        tracker.pop()


def test_bank_tracker_no_contention(config: SimConfig):
    """Tests that two ops needing different banks can run in parallel."""
    config.spm_banks = 4
    config.spm_bank_ports = 4  # Ports are not a constraint
    tracker = BankTracker(config)

    # Op1 needs 2 banks for 100 cycles
    start_cycle_1, chosen_banks_1 = tracker.probe_earliest_free_slot(0, 100, 2)
    assert start_cycle_1 == 0
    assert len(chosen_banks_1) == 2
    tracker.commit_slot(start_cycle_1, 100, chosen_banks_1)

    # Op2 needs 2 different banks for 100 cycles
    start_cycle_2, chosen_banks_2 = tracker.probe_earliest_free_slot(0, 100, 2)

    assert start_cycle_2 == 0  # Should be able to start in parallel
    assert len(chosen_banks_2) == 2
    assert not set(chosen_banks_1) & set(chosen_banks_2)  # Must be different banks


def test_bank_tracker_bank_contention(config: SimConfig):
    """Tests that two ops needing the same banks are serialized."""
    config.spm_banks = 2
    config.spm_bank_ports = 2  # Ports are not a constraint
    tracker = BankTracker(config)

    # Op1 takes both banks for 100 cycles
    start_cycle_1, chosen_banks_1 = tracker.probe_earliest_free_slot(0, 100, 2)
    assert start_cycle_1 == 0
    tracker.commit_slot(start_cycle_1, 100, chosen_banks_1)

    # Op2 also needs 2 banks
    start_cycle_2, chosen_banks_2 = tracker.probe_earliest_free_slot(0, 100, 2)

    # Since all banks were busy, op2 must wait
    assert start_cycle_2 == 100


def test_bank_tracker_port_contention(config: SimConfig):
    """Tests that port limits serialize operations even with free banks."""
    config.spm_banks = 8  # Plenty of banks
    config.spm_bank_ports = 2  # But only 2 ports
    tracker = BankTracker(config)

    # Op1 uses 2 ports/banks for 100 cycles
    start_cycle_1, chosen_banks_1 = tracker.probe_earliest_free_slot(0, 100, 2)
    assert start_cycle_1 == 0
    tracker.commit_slot(start_cycle_1, 100, chosen_banks_1)

    # Op2 needs 1 port/bank. Even though banks are free, ports are not.
    start_cycle_2, chosen_banks_2 = tracker.probe_earliest_free_slot(0, 100, 1)

    # All ports are busy until cycle 100
    assert start_cycle_2 == 100


def test_bank_tracker_partial_port_contention(config: SimConfig):
    """Tests that a free port can be used while others are busy."""
    config.spm_banks = 8
    config.spm_bank_ports = 4
    tracker = BankTracker(config)

    # Op1 uses 3 ports
    start_cycle_1, chosen_banks_1 = tracker.probe_earliest_free_slot(0, 100, 3)
    assert start_cycle_1 == 0
    tracker.commit_slot(start_cycle_1, 100, chosen_banks_1)

    # Op2 needs 1 port. There is one port free.
    start_cycle_2, chosen_banks_2 = tracker.probe_earliest_free_slot(0, 50, 1)

    # Should be able to start immediately
    assert start_cycle_2 == 0
    tracker.commit_slot(start_cycle_2, 50, chosen_banks_2)

    # Op3 needs 1 port. All 4 are now busy at the start.
    # Op2 finishes at cycle 50, freeing up its port.
    start_cycle_3, chosen_banks_3 = tracker.probe_earliest_free_slot(0, 10, 1)
    assert start_cycle_3 == 50


@pytest.fixture
def l2_config():
    """Config for L2 cache tests."""
    config = SimConfig()
    config.l2_cache_enabled = True
    config.l2_cache_size_kib = 1 # 1 KiB for easy testing
    config.l2_cache_line_size_bytes = 64
    config.l2_cache_associativity = 2 # 2-way set associative
    config.l2_cache_hit_latency_cycles = 5
    config.l2_cache_miss_latency_cycles = 50
    return config

def test_l2_cache_initialization(l2_config: SimConfig):
    """Tests L2CacheTracker initialization."""
    tracker = L2CacheTracker(l2_config)
    assert tracker.enabled is True
    assert tracker.size_bytes == 1024 # 1 KiB
    assert tracker.line_size == 64
    assert tracker.associativity == 2
    assert tracker.hit_latency == 5
    assert tracker.miss_latency == 50
    assert tracker.num_lines == 1024 // 64 # 16 lines
    assert tracker.num_sets == 16 // 2 # 8 sets
    assert tracker.hits == 0
    assert tracker.misses == 0

def test_l2_cache_disabled(config: SimConfig):
    """Tests L2CacheTracker when disabled."""
    config.l2_cache_enabled = False
    tracker = L2CacheTracker(config)
    assert tracker.enabled is False
    is_hit, latency = tracker.access(0x1000)
    assert is_hit is False
    assert latency == 0
    assert tracker.hits == 0
    assert tracker.misses == 0 # No misses counted if disabled

def test_l2_cache_hit_and_miss(l2_config: SimConfig):
    """Tests basic hit and miss behavior."""
    tracker = L2CacheTracker(l2_config)

    # First access: Miss
    is_hit, latency = tracker.access(0x1000) # Address 0x1000
    assert is_hit is False
    assert latency == l2_config.l2_cache_miss_latency_cycles
    assert tracker.hits == 0
    assert tracker.misses == 1
    assert tracker.get_stats()['miss_rate'] == 1.0

    # Second access to same address: Hit
    is_hit, latency = tracker.access(0x1000)
    assert is_hit is True
    assert latency == l2_config.l2_cache_hit_latency_cycles
    assert tracker.hits == 1
    assert tracker.misses == 1
    assert tracker.get_stats()['hit_rate'] == 0.5

    # Access another address (miss)
    is_hit, latency = tracker.access(0x2000) # Different address, likely different set or new line
    assert is_hit is False
    assert latency == l2_config.l2_cache_miss_latency_cycles
    assert tracker.hits == 1
    assert tracker.misses == 2
    assert tracker.get_stats()['miss_rate'] == 2/3

def test_l2_cache_lru_replacement(l2_config: SimConfig):
    """Tests LRU replacement policy."""
    # Configure a small cache to easily test replacement
    l2_config.l2_cache_size_kib = 1 # 1 KiB
    l2_config.l2_cache_line_size_bytes = 64
    l2_config.l2_cache_associativity = 2 # 2-way
    # num_lines = 1024 / 64 = 16
    # num_sets = 16 / 2 = 8
    
    tracker = L2CacheTracker(l2_config)

    # Addresses mapping to Set 0 (index 0)
    addr_A = 0x0000 # Tag A, Index 0
    addr_B = 0x0200 # Tag B, Index 0 (assuming 8 sets, 0x200 / 64 = 8, so index 0)
    addr_C = 0x0400 # Tag C, Index 0

    # Fill Set 0
    is_hit, _ = tracker.access(addr_A) # Miss, A in cache
    assert not is_hit
    is_hit, _ = tracker.access(addr_B) # Miss, B in cache
    assert not is_hit
    assert tracker.misses == 2

    # Access A again (makes A most recently used)
    is_hit, _ = tracker.access(addr_A) # Hit
    assert is_hit
    assert tracker.hits == 1

    # Access C (should evict B, as B is LRU in Set 0)
    is_hit, _ = tracker.access(addr_C) # Miss, C in cache, B evicted
    assert not is_hit
    assert tracker.misses == 3

    # Access B (should be a miss now, as it was evicted)
    is_hit, _ = tracker.access(addr_B) # Miss
    assert not is_hit
    assert tracker.misses == 4

    # Access A again (should be a miss, as A was evicted by B)
    is_hit, _ = tracker.access(addr_A) # Miss
    assert not is_hit
    assert tracker.misses == 5

    # Access C (should be a miss, as C was evicted by A)
    is_hit, _ = tracker.access(addr_C) # Miss
    assert not is_hit
    assert tracker.misses == 6

    assert tracker.get_stats()['hits'] == 1
    assert tracker.get_stats()['misses'] == 6
    assert abs(tracker.get_stats()['hit_rate'] - (1/7)) < 0.001

def test_l2_cache_stats(l2_config: SimConfig):
    """Tests statistics collection."""
    tracker = L2CacheTracker(l2_config)

    # No accesses
    stats = tracker.get_stats()
    assert stats['hits'] == 0
    assert stats['misses'] == 0
    assert stats['hit_rate'] == 0
    assert stats['miss_rate'] == 0

    # Some accesses
    tracker.access(0x1000) # Miss
    tracker.access(0x1000) # Hit
    tracker.access(0x2000) # Miss
    tracker.access(0x3000) # Miss

    stats = tracker.get_stats()
    assert stats['hits'] == 1
    assert stats['misses'] == 3
    assert abs(stats['hit_rate'] - 0.25) < 0.001
    assert abs(stats['miss_rate'] - 0.75) < 0.001

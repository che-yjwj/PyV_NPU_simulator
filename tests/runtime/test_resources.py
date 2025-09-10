
import pytest
from pyv_npu.config import SimConfig
from pyv_npu.runtime.resources import BankTracker, IOBufferTracker


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

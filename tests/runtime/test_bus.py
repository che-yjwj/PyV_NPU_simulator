import pytest
from pyv_npu.runtime.bus import SimulatedBus


def test_bus_initialization():
    """Tests that the SimulatedBus dataclass initializes correctly."""
    bus = SimulatedBus(bandwidth_gbps=10.0)
    assert bus.bandwidth_gbps == 10.0


def test_bus_transfer_ns():
    """Tests the transfer_ns calculation."""
    # 10 GB/s bandwidth
    bus = SimulatedBus(bandwidth_gbps=10.0)

    # Transfer 10 bytes
    # 10 bytes * 8 bits/byte = 80 bits
    # 80 bits / (10 * 1e9 bits/sec) = 8 * 1e-9 seconds = 8 ns
    assert bus.transfer_ns(10) == pytest.approx(8.0)

    # Transfer 1 GB (1e9 bytes)
    # 1e9 bytes * 8 bits/byte = 8e9 bits
    # 8e9 bits / (10 * 1e9 bits/sec) = 0.8 seconds = 800,000,000 ns
    assert bus.transfer_ns(1e9) == pytest.approx(800000000.0)


def test_bus_zero_bandwidth():
    """Tests that zero or negative bandwidth results in zero transfer time."""
    bus_zero = SimulatedBus(bandwidth_gbps=0.0)
    assert bus_zero.transfer_ns(1024) == 0.0

    bus_neg = SimulatedBus(bandwidth_gbps=-10.0)
    assert bus_neg.transfer_ns(1024) == 0.0

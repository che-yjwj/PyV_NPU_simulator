import pytest
from pyv_npu.config import SimConfig
from pyv_npu.runtime.resources import IOBufferTracker

@pytest.fixture
def buffer_config():
    """Provides a default SimConfig for the IO buffer tests."""
    # Set a small buffer size for easier testing of capacity limits
    return SimConfig(io_buffer_size_kb=1) # 1 KB buffer

@pytest.fixture
def io_buffer(buffer_config):
    """Provides an empty IOBufferTracker instance."""
    return IOBufferTracker(config=buffer_config, name="test_buffer")

def test_buffer_initialization(io_buffer):
    """Tests that the buffer is initialized correctly."""
    assert io_buffer.current_fill_bytes == 0
    assert io_buffer.capacity_bytes == 1024
    assert len(io_buffer.queue) == 0

def test_push_simple(io_buffer):
    """Tests a single push operation."""
    io_buffer.push(num_bytes=100, tensor_name="tensor_a")
    assert io_buffer.current_fill_bytes == 100
    assert len(io_buffer.queue) == 1
    assert io_buffer.queue[0] == (100, "tensor_a")

def test_push_multiple(io_buffer):
    """Tests pushing multiple items."""
    io_buffer.push(100, "tensor_a")
    io_buffer.push(200, "tensor_b")
    assert io_buffer.current_fill_bytes == 300
    assert len(io_buffer.queue) == 2
    assert io_buffer.queue[1] == (200, "tensor_b")

def test_push_exceed_capacity(io_buffer):
    """Tests that pushing beyond capacity raises a ValueError."""
    io_buffer.push(1000, "tensor_a")
    with pytest.raises(ValueError, match="Buffer overflow"):
        io_buffer.push(100, "tensor_b")

def test_can_push(io_buffer):
    """Tests the can_push method."""
    assert io_buffer.can_push(1024)
    assert not io_buffer.can_push(1025)
    io_buffer.push(512, "tensor_a")
    assert io_buffer.can_push(512)
    assert not io_buffer.can_push(513)

def test_can_pop_tensor(io_buffer):
    """Tests the can_pop_tensor method."""
    assert not io_buffer.can_pop_tensor("tensor_a")
    io_buffer.push(100, "tensor_a")
    assert io_buffer.can_pop_tensor("tensor_a")
    assert not io_buffer.can_pop_tensor("tensor_b")
    io_buffer.push(200, "tensor_b")
    assert io_buffer.can_pop_tensor("tensor_b")

def test_pop_simple(io_buffer):
    """Tests a single pop operation."""
    io_buffer.push(100, "tensor_a")
    io_buffer.pop(100, "tensor_a")
    assert io_buffer.current_fill_bytes == 0
    assert len(io_buffer.queue) == 0
    assert not io_buffer.can_pop_tensor("tensor_a")

def test_pop_out_of_order(io_buffer):
    """Tests that the correct tensor is popped even if it's not at the front."""
    io_buffer.push(100, "tensor_a")
    io_buffer.push(200, "tensor_b")
    io_buffer.push(300, "tensor_c")

    # Pop the middle element
    io_buffer.pop(200, "tensor_b")
    assert io_buffer.current_fill_bytes == 400 # 100 + 300
    assert len(io_buffer.queue) == 2
    assert not io_buffer.can_pop_tensor("tensor_b")
    assert io_buffer.can_pop_tensor("tensor_a")
    assert io_buffer.can_pop_tensor("tensor_c")
    assert io_buffer.queue[0] == (100, "tensor_a")
    assert io_buffer.queue[1] == (300, "tensor_c")

def test_pop_non_existent(io_buffer):
    """Tests that popping a non-existent tensor raises a ValueError."""
    io_buffer.push(100, "tensor_a")
    with pytest.raises(ValueError, match="not in buffer for popping"):
        io_buffer.pop(100, "tensor_b")
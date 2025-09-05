import pytest
from pyv_npu.config import SimConfig
from pyv_npu.runtime.resources import IOBufferTracker

@pytest.fixture
def buffer_config():
    """Provides a default SimConfig for the IO buffer tests."""
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

def test_pop_fifo_order(io_buffer):
    """Tests that pop returns items in FIFO order."""
    io_buffer.push(100, "tensor_a")
    io_buffer.push(200, "tensor_b")

    item1 = io_buffer.pop()
    assert item1 == (100, "tensor_a")
    assert io_buffer.current_fill_bytes == 200
    assert len(io_buffer.queue) == 1

    item2 = io_buffer.pop()
    assert item2 == (200, "tensor_b")
    assert io_buffer.current_fill_bytes == 0
    assert len(io_buffer.queue) == 0

def test_pop_from_empty_buffer(io_buffer):
    """Tests that popping from an empty buffer raises a ValueError."""
    with pytest.raises(ValueError, match="Buffer underflow"):
        io_buffer.pop()

def test_peek(io_buffer):
    """Tests that peek returns the next item without removing it."""
    assert io_buffer.peek() is None
    io_buffer.push(100, "tensor_a")
    io_buffer.push(200, "tensor_b")

    peeked_item = io_buffer.peek()
    assert peeked_item == (100, "tensor_a")
    # Verify that the buffer state is unchanged
    assert io_buffer.current_fill_bytes == 300
    assert len(io_buffer.queue) == 2

    io_buffer.pop()
    peeked_item_after_pop = io_buffer.peek()
    assert peeked_item_after_pop == (200, "tensor_b")

def test_can_pop(io_buffer):
    """Tests the can_pop method."""
    assert not io_buffer.can_pop()
    io_buffer.push(100, "tensor_a")
    assert io_buffer.can_pop()
    assert io_buffer.can_pop(num_bytes=100)
    assert not io_buffer.can_pop(num_bytes=99)
    io_buffer.pop()
    assert not io_buffer.can_pop()
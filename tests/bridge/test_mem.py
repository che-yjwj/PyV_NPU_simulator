import pytest
from unittest.mock import MagicMock

# Import the simulator and clear the global state before tests
from pyv.simulator import Simulator
Simulator.clear()

from pyv_npu.bridge.mem import NPUControlMemory, REG_QUEUE_BASE_LO, REG_DOORBELL

@pytest.fixture
def mem_module():
    """Provides a NPUControlMemory instance with a simulator environment."""
    # Ensure a clean slate for each test
    Simulator.clear()
    # Set up the global simulator instance that the pyv framework expects
    sim = Simulator()

    # Mock the scheduler to check for interactions
    scheduler = MagicMock()
    # Use a small size for testing
    mem = NPUControlMemory(size=256, scheduler=scheduler)
    
    # Add the DUT to the simulator
    sim.addObj(mem)
    # Initialize the simulator and all its objects
    sim.init()

    yield mem # Provide the memory module to the test


def test_memory_initialization(mem_module: NPUControlMemory):
    """Tests that the memory module initializes correctly."""
    assert len(mem_module.mem) == 256
    assert mem_module.scheduler is not None

def test_mmio_read_write(mem_module: NPUControlMemory):
    """Tests writing to and reading from an MMIO register."""
    test_addr = REG_QUEUE_BASE_LO
    test_value = 0xDEADBEEF

    # Simulate a write cycle
    mem_module.write_port.we_i.write(True)
    mem_module.read_port0.addr_i.write(test_addr)
    mem_module.write_port.wdata_i.write(test_value)
    mem_module.read_port0.width_i.write(4) # 4 bytes for a word
    
    # Run the simulator for one cycle to process the write
    Simulator.globalSim.run_comb_logic()
    Simulator.globalSim.tick()

    # Simulate a read
    read_value = mem_module._read(addr=test_addr, w=4)

    assert read_value == test_value
    assert mem_module.queue_base == test_value

def test_doorbell_trigger(mem_module: NPUControlMemory):
    """Tests that writing to the doorbell register calls the scheduler."""
    doorbell_addr = REG_DOORBELL
    ticket_value = 42

    # Simulate a write to the doorbell
    mem_module.write_port.we_i.write(True)
    mem_module.read_port0.addr_i.write(doorbell_addr)
    mem_module.write_port.wdata_i.write(ticket_value)
    mem_module.read_port0.width_i.write(4)

    # Run the simulator for one cycle
    Simulator.globalSim.run_comb_logic()
    Simulator.globalSim.tick()

    # Check if the scheduler's handle_doorbell method was called with the correct value
    mem_module.scheduler.handle_doorbell.assert_called_once_with(ticket_value)

def test_general_memory_read_write(mem_module: NPUControlMemory):
    """Tests basic byte, half-word, and word R/W to general memory."""
    test_addr = 0x10 # A non-MMIO address
    test_word_value = 0x12345678

    # --- Test Word Write ---
    mem_module.write_port.we_i.write(True)
    mem_module.read_port0.addr_i.write(test_addr)
    mem_module.write_port.wdata_i.write(test_word_value)
    mem_module.read_port0.width_i.write(4)
    
    # Run the simulator for one cycle
    Simulator.globalSim.run_comb_logic()
    Simulator.globalSim.tick()

    # Verify memory content (little-endian)
    assert mem_module.mem[test_addr] == 0x78
    assert mem_module.mem[test_addr + 1] == 0x56
    assert mem_module.mem[test_addr + 2] == 0x34
    assert mem_module.mem[test_addr + 3] == 0x12

    # --- Test Word Read ---
    read_value = mem_module._read(addr=test_addr, w=4)
    assert read_value == test_word_value

    # --- Test Byte Read ---
    read_byte = mem_module._read(addr=test_addr, w=1)
    assert read_byte == 0x78

    # --- Test Half-word Read ---
    read_half = mem_module._read(addr=test_addr, w=2)
    assert read_half == 0x5678

def test_out_of_bounds_read(mem_module: NPUControlMemory):
    """Tests that reading from an out-of-bounds address is handled gracefully."""
    # The current implementation returns 0 for out-of-bounds reads
    read_value = mem_module._read(addr=0x1000, w=4) # size is 256
    assert read_value == 0
from enum import Enum, auto
from .module import Module
from .clocked import Clocked
from .mem import Memory, ReadPort, WritePort
from .cache import L1Cache
from .cache_config import CacheConfig
from .port import Input, Output

class MemState(Enum):
    IDLE = auto()
    WRITE_BACK = auto()
    READ_ALLOCATE = auto()

class MemorySystem(Module, Clocked):
    """
    A memory system with a multi-state FSM to handle cache misses and stalls.
    It orchestrates the L1 caches and the main memory.
    """
    def __init__(self, mem_size_bytes: int = 8 * 1024):
        super().__init__(name='MemorySystem')

        self.icache_config = CacheConfig(name="ICache")
        self.dcache_config = CacheConfig(name="DCache")
        
        # The lower memory is shared. The MemorySystem orchestrates access.
        self.main_memory = Memory(mem_size_bytes)
        
        self.l1_icache = L1Cache(self.icache_config, self.main_memory)
        self.l1_dcache = L1Cache(self.dcache_config, self.main_memory)

        # --- Ports ---
        # Port 0 is for data, Port 1 is for instructions
        self.read_port0 = ReadPort(re_i=Input(bool, [self.process]), width_i=Input(int), addr_i=Input(int, [self.process]), rdata_o=Output(int))
        self.read_port1 = ReadPort(re_i=Input(bool, [self.process]), width_i=Input(int), addr_i=Input(int, [self.process]), rdata_o=Output(int))
        self.write_port = WritePort(we_i=Input(bool, [self.process]), wdata_i=Input(int))
        self.stall_o = Output(bool)

        # --- State ---
        self.state = MemState.IDLE
        self.latency_counter = 0
        self.pending_request = None

    def process(self):
        """Combinational logic: Probe cache, handle hits, and initiate misses."""
        # The system is stalled if it's not idle.
        if self.state != MemState.IDLE:
            self.stall_o.write(True)
            # While busy, don't accept new requests. Data outputs are not driven.
            return

        # --- Request Selection (Prioritize Data over Instruction) ---
        is_d_read = self.read_port0.re_i.read()
        is_d_write = self.write_port.we_i.read()
        is_i_read = self.read_port1.re_i.read()

        if is_d_read or is_d_write:
            active_port, active_cache = self.read_port0, self.l1_dcache
            addr = active_port.addr_i.read()
            is_write = is_d_write
        elif is_i_read:
            active_port, active_cache = self.read_port1, self.l1_icache
            addr = active_port.addr_i.read()
            is_write = False
        else:
            # No request
            self.stall_o.write(False)
            return

        # --- Cache Probe ---
        hit, line, tag, index, offset = active_cache.probe(addr)

        if hit:
            self.stall_o.write(False)
            if is_write:
                # Write hits are handled in the next tick (1 cycle latency)
                # No stall is needed, but the write is not instantaneous.
                wdata = self.write_port.wdata_i.read()
                self.pending_request = ('write_hit', line, offset, wdata)
            else:
                # Read hits are combinational
                rdata = active_cache.read_from_line(line, offset)
                active_port.rdata_o.write(rdata)
        else: # Cache Miss
            self.stall_o.write(True)
            # Store all necessary info for the state machine to handle the miss
            self.pending_request = ('miss', active_cache, is_write, addr, tag, index, offset)

    def _tick(self):
        """Synchronous logic: Update state machine for misses and write hits."""
        
        # --- Handle completed write hits from the previous cycle ---
        if self.pending_request and self.pending_request[0] == 'write_hit':
            _, line, offset, wdata = self.pending_request
            self.l1_dcache.write_to_line(line, offset, wdata)
            self.pending_request = None

        # --- State Machine Logic ---
        if self.state == MemState.IDLE:
            if self.pending_request and self.pending_request[0] == 'miss':
                _, cache, is_write, addr, tag, index, offset = self.pending_request
                
                victim_line = cache.get_victim_line(index)
                
                # Store victim line and other details for subsequent states
                self.pending_request = ('miss_inprogress', cache, is_write, addr, tag, index, offset, victim_line)

                if victim_line.valid and victim_line.dirty:
                    self.state = MemState.WRITE_BACK
                    self.latency_counter = cache.config.miss_latency_cycles
                else:
                    self.state = MemState.READ_ALLOCATE
                    self.latency_counter = cache.config.miss_latency_cycles

        elif self.state == MemState.WRITE_BACK:
            self.latency_counter -= 1
            if self.latency_counter <= 0:
                _, cache, _, _, _, index, _, victim_line = self.pending_request
                
                # Reconstruct victim address and write back to main memory
                victim_addr = cache.reconstruct_address(victim_line.tag, index)
                self.main_memory.write_block(victim_addr, victim_line.data)
                
                # Transition to next state
                self.state = MemState.READ_ALLOCATE
                self.latency_counter = cache.config.miss_latency_cycles

        elif self.state == MemState.READ_ALLOCATE:
            self.latency_counter -= 1
            if self.latency_counter <= 0:
                _, cache, is_write, addr, tag, _, offset, victim_line = self.pending_request
                
                # Fetch new block from main memory
                block_start_addr = addr & ~cache.offset_mask
                new_data = self.main_memory.read_block(block_start_addr, cache.config.line_size_bytes)
                
                # Fill the cache line with the new data
                cache.fill_line(victim_line, tag, new_data)
                
                # If it was a write miss, perform the write now
                if is_write:
                    wdata = self.write_port.wdata_i.read()
                    cache.write_to_line(victim_line, offset, wdata)
                
                # Miss handling is complete
                self.state = MemState.IDLE
                self.pending_request = None

    def _prepare_next_val(self): pass
    def _reset(self): 
        self.state = MemState.IDLE
        self.latency_counter = 0
        self.pending_request = None

    def load_instructions(self, instructions: list[int]):
        self.main_memory.mem[:len(instructions)] = instructions

    def debug_read_mem(self, address: int, num_bytes: int) -> list[int]:
        return self.main_memory.mem[address:address+num_bytes]
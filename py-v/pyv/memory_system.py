from enum import Enum, auto
from .module import Module
from .clocked import Clocked
from .mem import Memory, ReadPort, WritePort
from .cache import L1Cache
from .cache_config import CacheConfig
from .port import Output

class MemState(Enum):
    IDLE = auto()
    CACHE_LOOKUP = auto()
    WAIT_FOR_MEMORY = auto()

class MemorySystem(Module, Clocked):
    """A memory system including separate I/D caches and a main memory."""
    def __init__(self, mem_size_bytes: int = 8 * 1024):
        super().__init__(name='MemorySystem')

        # Separate configs can be used for I-Cache and D-Cache if needed
        self.icache_config = CacheConfig(size_kb=4, associativity=4)
        self.dcache_config = CacheConfig(size_kb=4, associativity=4)

        self.main_memory = Memory(mem_size_bytes)
        self.l1_icache = L1Cache(self.icache_config, self.main_memory)
        self.l1_dcache = L1Cache(self.dcache_config, self.main_memory)

        # Ports for Instruction Fetch (IF) -> Connect to I-Cache
        self.read_port1 = ReadPort(
            re_i=self.main_memory.read_port1.re_i, # Pass-through re signal
            width_i=self.main_memory.read_port1.width_i,
            addr_i=self.main_memory.read_port1.addr_i,
            rdata_o=self.main_memory.read_port1.rdata_o
        )

        # Ports for Load/Store (MEM) -> Connect to D-Cache
        self.read_port0 = ReadPort(
            re_i=self.main_memory.read_port0.re_i,
            width_i=self.main_memory.read_port0.width_i,
            addr_i=self.main_memory.read_port0.addr_i,
            rdata_o=self.main_memory.read_port0.rdata_o
        )
        self.write_port = WritePort(
            we_i=self.main_memory.write_port.we_i,
            wdata_i=self.main_memory.write_port.wdata_i
        )

        self.stall_o = Output(bool)

        self.state = MemState.IDLE
        self.miss_counter = 0
        self.active_port = None
        self.active_cache = None

    def _prepare_next_val(self):
        pass

    def _tick(self):
        self.stall_o.write(False)

        if self.state == MemState.IDLE:
            if self.write_port.we_i.read() or self.read_port0.re_i.read():
                self.active_port = self.read_port0
                self.active_cache = self.l1_dcache
                self.state = MemState.CACHE_LOOKUP
            elif self.read_port1.re_i.read():
                self.active_port = self.read_port1
                self.active_cache = self.l1_icache
                self.state = MemState.CACHE_LOOKUP
            else:
                return

        if self.state == MemState.CACHE_LOOKUP:
            addr = self.active_port.addr_i.read()
            is_write = (self.active_cache == self.l1_dcache and self.write_port.we_i.read())

            if is_write:
                wdata = self.write_port.wdata_i.read()
                self.active_cache.write(addr, wdata)
                self.state = MemState.IDLE
            else: # Is a read
                data, hit = self.active_cache.read(addr)
                if hit:
                    self.active_port.rdata_o.write(data)
                    self.state = MemState.IDLE
                else:
                    self.stall_o.write(True)
                    self.state = MemState.WAIT_FOR_MEMORY
                    self.miss_counter = self.active_cache.config.miss_latency_cycles

        elif self.state == MemState.WAIT_FOR_MEMORY:
            self.stall_o.write(True)
            self.miss_counter -= 1
            if self.miss_counter <= 0:
                self.stall_o.write(False)
                self.state = MemState.IDLE
                addr = self.active_port.addr_i.read()
                data, _ = self.active_cache.read(addr)
                self.active_port.rdata_o.write(data)

    def _reset(self):
        self.main_memory._reset()
        self.state = MemState.IDLE
        self.miss_counter = 0

    def load_instructions(self, instructions: list[int]):
        self.main_memory.mem[:len(instructions)] = instructions
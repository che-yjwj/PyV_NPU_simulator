from pyv.module import Module
from pyv.port import Input, Output
from pyv.util import MASK_32, PyVObj
from pyv.log import logger
from pyv.clocked import Clocked, MemList


NPU_MMIO_BASE = 0x40000000
NPU_MMIO_SIZE = 0x10000

# PRD-defined MMIO register offsets
REG_QUEUE_BASE_LO = NPU_MMIO_BASE + 0x00
REG_QUEUE_BASE_HI = NPU_MMIO_BASE + 0x04
REG_QUEUE_HEAD_LO = NPU_MMIO_BASE + 0x10
REG_QUEUE_HEAD_HI = NPU_MMIO_BASE + 0x14
REG_QUEUE_TAIL_LO = NPU_MMIO_BASE + 0x18
REG_QUEUE_TAIL_HI = NPU_MMIO_BASE + 0x1C
REG_DOORBELL = NPU_MMIO_BASE + 0x20
REG_IRQ_STATUS = NPU_MMIO_BASE + 0x24
REG_IRQ_MASK = NPU_MMIO_BASE + 0x28


class ReadPort(PyVObj):
    """Read port"""
    def __init__(
        self,
        re_i: Input[bool],
        width_i: Input[int],
        addr_i: Input[int],
        rdata_o: Output[int]
    ):
        super().__init__(name='UnnamedReadPort')

        self.re_i = re_i
        """Read-enable input"""
        self.width_i = width_i
        """Data width input (1, 2, or 4)"""
        self.addr_i = addr_i
        """Address input (also for write)"""
        self.rdata_o = rdata_o
        """Read port 0 Read data output"""


class WritePort(PyVObj):
    """Write port"""
    def __init__(
        self,
        we_i: Input[bool],
        wdata_i: Input[int]
    ):
        super().__init__(name='UnnamedWritePort')

        self.we_i = we_i
        """Write-enable input"""
        self.wdata_i = wdata_i
        """Write data input"""

# TODO: Check if addr is valid


class NPUControlMemory(Module, Clocked):
    """Simple memory module with 2 read ports and 1 write port

    A memory is represented by a simple list of bytes.

    Byte-ordering: Little-endian
    """

    def __init__(self, size: int = 32, scheduler=None):
        """Memory constructor.

        Args:
            size: Size of memory in bytes.
            scheduler: The NPU scheduler instance.
        """
        super().__init__(name='NPUControlMemory')
        MemList.add_to_mem_list(self)
        self.mem = [0 for i in range(0, size)]
        """Memory array. List of length `size`."""
        self.scheduler = scheduler

        # MMIO Registers State
        self.queue_base = 0
        self.queue_head = 0
        self.queue_tail = 0
        self.irq_status = 0
        self.irq_mask = 0

        # Read port 0
        self.read_port0 = ReadPort(
            re_i=Input(bool, [self.process_read0]),
            width_i=Input(int, [self.process_read0]),
            addr_i=Input(int, [self.process_read0]),
            rdata_o=Output(int)
        )

        # Read port 1
        self.read_port1 = ReadPort(
            re_i=Input(bool, [self.process_read1]),
            width_i=Input(int, [self.process_read1]),
            addr_i=Input(int, [self.process_read1]),
            rdata_o=Output(int)
        )

        # Write port (uses addr, width from read port 0)
        self.write_port = WritePort(
            we_i=Input(bool, [None]),
            wdata_i=Input(int, [None])
        )

    def _read(self, addr, w):
        # Handle MMIO reads
        if NPU_MMIO_BASE <= addr < NPU_MMIO_BASE + NPU_MMIO_SIZE:
            val = 0
            if addr == REG_QUEUE_BASE_LO:
                val = self.queue_base & 0xFFFFFFFF
            elif addr == REG_QUEUE_BASE_HI:
                val = (self.queue_base >> 32) & 0xFFFFFFFF
            elif addr == REG_QUEUE_HEAD_LO:
                val = self.queue_head & 0xFFFFFFFF
            elif addr == REG_QUEUE_HEAD_HI:
                val = (self.queue_head >> 32) & 0xFFFFFFFF
            elif addr == REG_QUEUE_TAIL_LO:
                val = self.queue_tail & 0xFFFFFFFF
            elif addr == REG_QUEUE_TAIL_HI:
                val = (self.queue_tail >> 32) & 0xFFFFFFFF
            elif addr == REG_IRQ_STATUS:
                val = self.irq_status
            elif addr == REG_IRQ_MASK:
                val = self.irq_mask
            else:
                logger.warning(
                    f"NPUControlMemory: Read from unhandled MMIO address {addr:08X}"
                )
            
            logger.debug(
                f"MEM ({self.name}): read value {val:08X} from MMIO address {addr:08X}"
            )
            return val

        # During the processing of the current cycle, it might occur that
        # an unstable port value is used as the address. However, the port
        # will eventually become stable, so we should "allow" that access
        # by just returning a dummy value, e.g., 0.
        #
        # Note: An actual illegal address exception caused by a running
        # program should be handled synchronously, i.e. with the next
        # active clock edge (tick).
        try:
            if w == 1:  # byte
                val = MASK_32 & self.mem[addr]
            elif w == 2:  # half word
                val = MASK_32 & (self.mem[addr + 1] << 8 | self.mem[addr])
            elif w == 4:  # word
                val = MASK_32 & (
                    self.mem[addr + 3] << 24
                    | self.mem[addr + 2] << 16
                    | self.mem[addr + 1] << 8
                    | self.mem[addr])
            else:
                raise Exception(
                    f'ERROR (Memory ({self.name}), read): Invalid width {w}')

            logger.debug(
                f"MEM ({self.name}): read value {val:08X} from address {addr:08X}"
            )
        except IndexError:
            val = 0

        return val

    def _process_read(self, read_port):
        re = read_port.re_i.read()
        addr = read_port.addr_i.read()
        w = read_port.width_i.read()

        if re:
            val = self._read(addr, w)
        else:
            val = 0

        read_port.rdata_o.write(val)

    def process_read0(self):
        self._process_read(self.read_port0)

    def process_read1(self):
        self._process_read(self.read_port1)

    def _prepare_next_val(self):
        # We need this also in Memory, because it could happen that these pins
        # are driven by registers, so we save the values first before the
        # registers tick.
        self.we_next = self.write_port.we_i.read()
        self.addr_next = self.read_port0.addr_i.read()
        self.wdata_next = self.write_port.wdata_i.read()
        self.w_next = self.read_port0.width_i.read()

    def _tick(self):
        we = self.we_next
        addr = self.addr_next
        wdata = self.wdata_next
        w = self.w_next

        if we:
            # Handle MMIO writes
            if NPU_MMIO_BASE <= addr < NPU_MMIO_BASE + NPU_MMIO_SIZE:
                logger.debug(f"NPUControlMemory: Intercepted MMIO write at {addr:08X}, data: {wdata:08X}")
                if addr == REG_DOORBELL:
                    if self.scheduler:
                        self.scheduler.handle_doorbell(wdata)
                    else:
                        logger.warning("NPUControlMemory: Doorbell triggered but no scheduler is attached!")
                elif addr == REG_QUEUE_BASE_LO:
                    self.queue_base = (self.queue_base & 0xFFFFFFFF00000000) | wdata
                elif addr == REG_QUEUE_BASE_HI:
                    self.queue_base = (self.queue_base & 0x00000000FFFFFFFF) | \
                        (wdata << 32)
                elif addr == REG_QUEUE_TAIL_LO:
                    self.queue_tail = (self.queue_tail & 0xFFFFFFFF00000000) | wdata
                    logger.info(
                        f"NPUControlMemory: QUEUE_TAIL set to {self.queue_tail:X}"
                    )
                elif addr == REG_QUEUE_TAIL_HI:
                    self.queue_tail = (self.queue_tail & 0x00000000FFFFFFFF) | \
                        (wdata << 32)
                    logger.info(
                        f"NPUControlMemory: QUEUE_TAIL set to {self.queue_tail:X}"
                    )
                elif addr == REG_IRQ_MASK:
                    self.irq_mask = wdata
                    logger.info(
                        f"NPUControlMemory: IRQ_MASK set to {wdata:08X}"
                    )
                elif addr == REG_IRQ_STATUS:
                    # Write-1-to-clear semantics
                    self.irq_status &= ~wdata
                    logger.info(
                        f"NPUControlMemory: IRQ_STATUS cleared by write. New status: {self.irq_status:08X}"
                    )
                else:
                    logger.warning(
                        f"NPUControlMemory: Write to unhandled MMIO address {addr:08X}"
                    )
                return

            if not (w == 1 or w == 2 or w == 4):
                raise Exception(
                    f'ERROR (Memory ({self.name}), write): Invalid width {w}')
            logger.debug(
                f"MEM {self.name}: write {wdata:08X} to address {addr:08X}"
            )

            if w == 1:  # byte
                self.mem[addr] = 0xff & wdata
            elif w == 2:  # half word
                self.mem[addr] = 0xff & wdata
                self.mem[addr + 1] = (0xff00 & wdata) >> 8
            elif w == 4:  # word
                self.mem[addr] = 0xff & wdata
                self.mem[addr + 1] = (0xff00 & wdata) >> 8
                self.mem[addr + 2] = (0xff0000 & wdata) >> 16
                self.mem[addr + 3] = (0xff000000 & wdata) >> 24

    # TODO: when memory gets loaded with program *before* simulation,
    # simulation start will cause a reset. So for now, we skip the reset here.
    def _reset(self):
        return
        """Reset memory.

        All elements are set to 0.
        """
        for i in range(0, len(self.mem)):
            self.mem[i] = 0

    

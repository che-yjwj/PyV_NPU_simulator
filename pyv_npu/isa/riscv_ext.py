
# Placeholder custom instruction encodings for RISC-V <-> NPU
from dataclasses import dataclass

@dataclass
class RiscvNpuInsn:
    mnemonic: str
    funct7: int = 0
    funct3: int = 0
    opcode: int = 0x0B   # custom-0 space (placeholder)

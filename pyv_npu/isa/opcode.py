from enum import Enum


class Opcode(str, Enum):
    """Defines the NPU operation codes."""

    # Data movement
    LOAD = "LOAD"
    STORE = "STORE"

    # Tensor Core Operations
    MATMUL = "MatMul"
    CONV = "Conv"
    GEMM_T = "GEMM_T"
    ATTN_T = "ATTN_T"

    # Vector Core Operations
    ADD = "Add"
    MUL = "Mul"
    GELU = "GELU"
    SOFTMAX = "Softmax"
    LAYERNORM = "LayerNorm"
    ERF = "Erf"

    # Fused Operations
    MATMULADDGELU = "MatMulAddGelu"

    # RISC-V Custom ISA & Control
    ENQCMD_T = "ENQCMD_T"
    TWAIT = "TWAIT"
    BARRIER = "BARRIER"

    def __str__(self) -> str:
        return self.value

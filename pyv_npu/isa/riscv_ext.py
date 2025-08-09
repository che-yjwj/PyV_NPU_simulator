from __future__ import annotations
from dataclasses import dataclass, field
from typing import List

from .npu_ir import NPUOp

# --- Tight-Coupled Custom ISA Definitions ---
# These classes represent the data associated with each custom instruction.
# They will be carried in the `args` field of an NPUOp in the instruction stream.

@dataclass
class ENQCMD_T:
    """Enqueue Command (Tensor Engine)
    Submits a descriptor of an NPU operation (like MatMul, Conv) to the TE.
    """
    # The actual NPU operation to be executed.
    npu_op_desc: NPUOp
    # A unique identifier returned to the CPU to track this command.
    ticket: int

@dataclass
class TWAIT:
    """Tensor Wait
    Stalls the CPU until a specific ticket (or group of tickets) is complete.
    """
    # The ticket to wait for.
    ticket: int
    # Optional: specify a timeout in cycles.
    timeout_cycles: int = 0

@dataclass
class TBAR:
    """Tensor Barrier
    Ensures all previously enqueued commands are complete before proceeding.
    """
    # Scope of the barrier, e.g., "tile", "cluster", "global"
    scope: str = "global"

@dataclass
class TSTAT:
    """Tensor Status
    Reads a status value from the NPU into a CPU register.
    """
    # The name of the status to read (e.g., "te_queue_depth", "ve_utilization").
    status_name: str
    # The destination CPU register (for simulation purposes).
    dest_reg: str
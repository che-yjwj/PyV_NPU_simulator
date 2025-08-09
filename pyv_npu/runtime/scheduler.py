
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any
from ..isa.npu_ir import Program, NPUOp

@dataclass
class ScheduleItem:
    op: NPUOp
    start_cycle: int
    end_cycle: int
    engine: str  # "TE" or "VE" or "CPU"

def simple_greedy_schedule(p: Program) -> List[ScheduleItem]:
    """Assign MatMul to TE; elementwise to VE; sequential, no overlap."""
    time = 0
    schedule: List[ScheduleItem] = []
    for op in p.ops:
        if op.opcode in ("MatMul", "Conv"):
            dur = 100  # placeholder cycles
            eng = "TE"
        elif op.opcode in ("GELU", "Softmax", "Add", "Mul", "LayerNorm"):
            dur = 20
            eng = "VE"
        else:
            dur = 10
            eng = "CPU"
        schedule.append(ScheduleItem(op=op, start_cycle=time, end_cycle=time+dur, engine=eng))
        time += dur
    return schedule

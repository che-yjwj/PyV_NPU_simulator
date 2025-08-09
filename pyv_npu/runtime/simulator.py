
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, List
from .scheduler import simple_greedy_schedule
from ..isa.npu_ir import Program

@dataclass
class SimulationReport:
    total_cycles: int
    engine_util: Dict[str, int]
    timeline: List[Dict[str, Any]]

def run(program: Program) -> SimulationReport:
    sched = simple_greedy_schedule(program)
    total = max((it.end_cycle for it in sched), default=0)
    util = {"TE":0, "VE":0, "CPU":0}
    timeline: List[Dict[str, Any]] = []
    for it in sched:
        util[it.engine] += it.end_cycle - it.start_cycle
        timeline.append({
            "op": it.op.opcode,
            "name": it.op.args.get("name"),
            "engine": it.engine,
            "start": it.start_cycle,
            "end": it.end_cycle,
        })
    return SimulationReport(total_cycles=total, engine_util=util, timeline=timeline)

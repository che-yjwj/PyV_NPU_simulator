from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, List

from ..config import SimConfig
from ..isa.npu_ir import Program
from .scheduler import simple_greedy_schedule, event_driven_schedule

@dataclass
class SimulationReport:
    total_cycles: int
    engine_util: Dict[str, int]
    timeline: List[Dict[str, Any]]

def run(program: Program, config: SimConfig) -> SimulationReport:
    """
    Runs the simulation for a given NPU program and configuration.

    This is the main entry point for the runtime simulation. Based on the
    SimConfig, it will select the appropriate scheduling and timing models.
    """
    print(f"Running simulation with mode='{config.mode}' and level='{config.level}'")

    # --- Scheduler Selection --- 
    if config.level in ("L2", "L3"):
      sched = event_driven_schedule(program, config)
    else:
      sched = simple_greedy_schedule(program)

    # --- Timing & Resource Calculation ---
    # In a real L2/L3 simulation, timing would be calculated here using
    # detailed models and parameters from `config` (e.g., clock_ghz, bw_dram_gbps).
    # For now, we use the pre-calculated cycles from the simple scheduler.
    total = max((it.end_cycle for it in sched), default=0)

    util = {}  # In future, engine names could come from config
    timeline: List[Dict[str, Any]] = []
    for it in sched:
        if it.engine not in util:
            util[it.engine] = 0
        util[it.engine] += it.end_cycle - it.start_cycle
        timeline.append({
            "op": it.op.opcode,
            "name": it.op.args.get("name"),
            "engine": it.engine,
            "start": it.start_cycle,
            "end": it.end_cycle,
        })

    # The report could also be made more detailed based on config
    return SimulationReport(total_cycles=total, engine_util=util, timeline=timeline)
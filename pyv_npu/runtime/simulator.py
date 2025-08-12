from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, List

from ..config import SimConfig
from ..isa.npu_ir import Program
from .scheduler import simple_greedy_schedule, event_driven_schedule

@dataclass
class SimulationReport:
    total_cycles: int
    engine_utilization: Dict[str, float] # Utilization percentage
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
      schedule = event_driven_schedule(program, config)
    else:
      schedule = simple_greedy_schedule(program)

    # --- Report Generation ---
    total_cycles = max((it.end_cycle for it in schedule), default=0)

    engine_active_cycles: Dict[str, int] = {}
    timeline: List[Dict[str, Any]] = []

    for item in schedule:
        if item.engine not in engine_active_cycles:
            engine_active_cycles[item.engine] = 0
        duration = item.end_cycle - item.start_cycle
        engine_active_cycles[item.engine] += duration
        
        timeline.append({
            "op": item.op.opcode,
            "name": item.op.name,
            "engine": item.engine,
            "start": item.start_cycle,
            "end": item.end_cycle,
        })

    utilization: Dict[str, float] = {}
    if total_cycles > 0:
        for engine, active_cycles in engine_active_cycles.items():
            utilization[engine] = round((active_cycles / total_cycles) * 100, 2)

    return SimulationReport(
        total_cycles=total_cycles, 
        engine_utilization=utilization, 
        timeline=timeline
    )
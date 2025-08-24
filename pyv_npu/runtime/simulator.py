from __future__ import annotations
from typing import Dict, Any, List, Tuple

from ..config import SimConfig
from ..isa.npu_ir import Program
from .scheduler import simple_greedy_schedule, event_driven_schedule, ScheduleItem

def run(program: Program, config: SimConfig) -> Tuple[List[ScheduleItem], Dict[str, Any]]:
    """
    Runs the simulation for a given NPU program and configuration.

    This is the main entry point for the runtime simulation. Based on the
    SimConfig, it will select the appropriate scheduling and timing models.
    """
    print(f"Running simulation with mode='{config.mode}' and level='{config.sim_level}'")

    # --- Scheduler Selection --- 
    if config.sim_level == "IA":
      schedule = simple_greedy_schedule(program)
      stats = {}
    else: # IA_TIMING, CA_HYBRID, CA_FULL
      schedule, stats = event_driven_schedule(program, config)

    return schedule, stats
from __future__ import annotations
import json
from pathlib import Path
from typing import List, Dict, Any
from ..config import SimConfig
from ..runtime.scheduler import ScheduleItem
from ..isa.opcode import Opcode
from . import viz

def _calculate_percentiles(data: List[float]) -> Dict[str, float]:
    """Calculates p50, p95, p99 without numpy."""
    if not data:
        return {}
    data.sort()
    n = len(data)
    p50 = data[int(n * 0.5)]
    p95 = data[int(n * 0.95)]
    p99 = data[int(n * 0.99)]
    return {
        "min": data[0],
        "max": data[-1],
        "p50": p50,
        "p95": p95,
        "p99": p99,
        "avg": sum(data) / n
    }

def generate_report_json(schedule: List[ScheduleItem], config: SimConfig, stats: Dict[str, Any]) -> Dict[str, Any]:
    """Generates a JSON-compatible dictionary from the schedule."""
    if not schedule:
        return {"total_cycles": 0, "engine_utilization": {}, "timeline": [], "stats": stats}

    total_cycles = max(item.end_cycle for item in schedule) if schedule else 0
    timeline = []
    engine_usage = {f"TC{i}": 0 for i in range(config.tc)}
    engine_usage.update({f"VC{i}": 0 for i in range(config.vc)})
    engine_usage.update({f"DMA{i}": 0 for i in range(config.dma_channels)})
    engine_usage.update({"CPU0": 0})

    control_op_durations = []
    dma_op_durations = []

    for item in schedule:
        duration = item.end_cycle - item.start_cycle
        if item.engine in engine_usage:
            engine_usage[item.engine] += duration
        
        if config.mode == 'tight' and item.engine.startswith("CPU"):
            control_op_durations.append(duration)
        
        if item.op.opcode in (Opcode.LOAD, Opcode.STORE):
            dma_op_durations.append(duration)

        timeline_item = {
            'op': item.op.opcode,
            'name': item.op.name,
            'engine': item.engine,
            'start': item.start_cycle,
            'end': item.end_cycle,
            'duration': duration,
            'stall_cycles': item.stall_cycles,
            'stall_breakdown': item.stall_breakdown,
        }
        if item.cycle_breakdown:
            timeline_item.update(item.cycle_breakdown)

        timeline.append(timeline_item)

    if control_op_durations:
        stats["control_overhead_stats"] = _calculate_percentiles(control_op_durations)
    
    if dma_op_durations:
        stats["dma_latency_stats"] = _calculate_percentiles(dma_op_durations)

    utilization_abs = {}
    for eng, usage in engine_usage.items():
        base_eng = ''.join(filter(str.isalpha, eng))
        utilization_abs.setdefault(base_eng, 0)
        utilization_abs[base_eng] += usage

    utilization_perc = {}
    if total_cycles > 0:
        if config.tc > 0: utilization_perc['TC_utilization'] = f"{(utilization_abs.get('TC', 0) / (total_cycles * config.tc)):.2%}"
        if config.vc > 0: utilization_perc['VC_utilization'] = f"{(utilization_abs.get('VC', 0) / (total_cycles * config.vc)):.2%}"
        if config.dma_channels > 0: utilization_perc['DMA_utilization'] = f"{(utilization_abs.get('DMA', 0) / (total_cycles * config.dma_channels)):.2%}"
        if utilization_abs.get('CPU', 0) > 0: utilization_perc['CPU_utilization'] = f"{(utilization_abs.get('CPU', 0) / total_cycles):.2%}"

    report_data = {
        "total_cycles": total_cycles,
        "engine_util_abs": utilization_abs,
        "engine_utilization": utilization_perc,
        "timeline": timeline,
        "config": config.__dict__
    }
    report_data.update(stats)
    return report_data

def generate_report(schedule: List[ScheduleItem], config: SimConfig, stats: Dict[str, Any]):
    """Generates all report artifacts."""
    report_data = generate_report_json(schedule, config, stats)
    output_dir = Path(config.report_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "report.json", "w") as f:
        json.dump(report_data, f, indent=4)

    # Use consolidated visualization functions
    viz.export_gantt(report_data['timeline'], str(output_dir / "report.html"))
    
    ascii_gantt = viz.export_gantt_ascii(report_data['timeline'])
    print(ascii_gantt)

    print(f"\nReports generated in {output_dir.absolute()}")

    if report_data.get('dram_collisions') is not None:
        print(f"DRAM Bank Collisions: {report_data['dram_collisions']}")
    if report_data.get('control_overhead_stats'):
        print("\nControl Overhead Stats (cycles):")
        for key, value in report_data['control_overhead_stats'].items():
            print(f"  {key:<5}: {value:.2f}")
    if report_data.get('dma_latency_stats'):
        print("\nDMA Latency Stats (cycles):")
        for key, value in report_data['dma_latency_stats'].items():
            print(f"  {key:<5}: {value:.2f}")
    if report_data.get('engine_utilization'):
        print("\nEngine Utilization:")
        for key, value in report_data['engine_utilization'].items():
            print(f"  {key}: {value}")
    print(f"\nTotal Cycles: {report_data['total_cycles']}")

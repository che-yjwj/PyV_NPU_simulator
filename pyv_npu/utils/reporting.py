from __future__ import annotations
import json
from pathlib import Path
from typing import List, Dict, Any
import plotly.express as px
import pandas as pd
from ..config import SimConfig
from ..runtime.scheduler import ScheduleItem

def generate_report_json(schedule: List[ScheduleItem], config: SimConfig, stats: Dict[str, Any]) -> Dict[str, Any]:
    """Generates a JSON-compatible dictionary from the schedule."""
    if not schedule:
        return {"total_cycles": 0, "engine_utilization": {}, "timeline": [], "stats": stats}

    total_cycles = max(item.end_cycle for item in schedule)
    timeline = []
    engine_usage = {f"TE{i}": 0 for i in range(config.te)}
    engine_usage.update({f"VE{i}": 0 for i in range(config.ve)})
    engine_usage.update({f"DMA{i}": 0 for i in range(config.dma_channels)})

    for item in schedule:
        duration = item.end_cycle - item.start_cycle
        if item.engine in engine_usage:
            engine_usage[item.engine] += duration
        
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

    utilization_abs = {}
    for eng, usage in engine_usage.items():
        base_eng = ''.join(filter(str.isalpha, eng))
        utilization_abs.setdefault(base_eng, 0)
        utilization_abs[base_eng] += usage

    utilization_perc = {}
    if config.te > 0:
        utilization_perc['TE_utilization'] = f"{(utilization_abs.get('TE', 0) / (total_cycles * config.te)):.2%}"
    if config.ve > 0:
        utilization_perc['VE_utilization'] = f"{(utilization_abs.get('VE', 0) / (total_cycles * config.ve)):.2%}"
    if config.dma_channels > 0:
        utilization_perc['DMA_utilization'] = f"{(utilization_abs.get('DMA', 0) / (total_cycles * config.dma_channels)):.2%}"

    report_data = {
        "total_cycles": total_cycles,
        "engine_util_abs": utilization_abs,
        "engine_utilization": utilization_perc,
        "timeline": timeline,
        "config": config.__dict__
    }
    report_data.update(stats)
    return report_data

def generate_html_report(report_data: Dict[str, Any], output_dir: Path):
    """Generates a self-contained HTML report with a Gantt chart."""
    if not report_data or not report_data['timeline']:
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(report_data['timeline'])

    required_cols = ['compute', 'fill_drain']
    for col in required_cols:
        if col not in df.columns:
            df[col] = 0

    fig = px.timeline(
        df,
        x_start="start",
        x_end="end",
        y="engine",
        color="op",
        hover_name="name",
        hover_data=['engine', 'stall_cycles', 'stall_breakdown', 'compute', 'fill_drain'],
        title="NPU Simulation Gantt Chart"
    )
    fig.update_yaxes(autorange="reversed")
    fig.write_html(output_dir / "report.html", full_html=True)

def generate_ascii_gantt(report_data: Dict[str, Any], width: int = 80):
    """Generates a simple ASCII Gantt chart."""
    timeline = report_data.get('timeline', [])
    total_cycles = report_data.get('total_cycles', 1)
    if not timeline:
        print("No timeline data to generate ASCII Gantt chart.")
        return

    engine_lanes = {}
    for item in timeline:
        engine = item['engine']
        if engine not in engine_lanes:
            engine_lanes[engine] = []
        engine_lanes[engine].append(item)

    print("--- ASCII Gantt Chart ---")
    for engine, ops in sorted(engine_lanes.items()):
        lane = ['-'] * width
        for op in ops:
            start_pos = int(op['start'] / total_cycles * width)
            end_pos = int(op['end'] / total_cycles * width)
            duration_pos = max(1, end_pos - start_pos)
            op_char = op['op'][0]
            for i in range(start_pos, start_pos + duration_pos):
                if i < width:
                    lane[i] = op_char
        print(f"{engine: >10} | {''.join(lane)} |")
    print("-------------------------")

def generate_report(schedule: List[ScheduleItem], config: SimConfig, stats: Dict[str, Any]):
    """Generates all report artifacts."""
    report_data = generate_report_json(schedule, config, stats)
    output_dir = Path(config.report_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "report.json", "w") as f:
        json.dump(report_data, f, indent=4)

    generate_html_report(report_data, output_dir)

    generate_ascii_gantt(report_data)

    print(f"Reports generated in {output_dir.absolute()}")

    # Print summary stats to console
    if report_data.get('dram_collisions') is not None:
        print(f"DRAM Bank Collisions: {report_data['dram_collisions']}")
    
    if report_data.get('engine_utilization'):
        print("\nEngine Utilization:")
        for key, value in report_data['engine_utilization'].items():
            print(f"  {key}: {value}")
    print(f"\nTotal Cycles: {report_data['total_cycles']}")
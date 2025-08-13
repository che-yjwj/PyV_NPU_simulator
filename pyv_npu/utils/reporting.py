import json
import pandas as pd
import base64
from io import BytesIO
from pathlib import Path

# Try to import plotly, but fall back to matplotlib if not available
try:
    import plotly.graph_objects as go
    import plotly.figure_factory as ff
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    import matplotlib.pyplot as plt

def generate_html_report(report_data: dict, output_dir: Path):
    """
    Generates an HTML report with a Gantt chart from the simulator's report data.
    Uses Plotly for an interactive chart if available, otherwise falls back to a static
    matplotlib chart.
    """
    output_dir.mkdir(exist_ok=True)
    timeline_df = pd.DataFrame(report_data["timeline"])
    total_cycles = report_data["total_cycles"]
    
    # Create a summary of engine utilization
    engine_util_summary = ""
    if "engine_util" in report_data:
        for k, v in report_data["engine_util"].items():
            engine_util_summary += f"<tr><td>{k}</td><td>{v}</td></tr>"

    # Generate Gantt chart
    if PLOTLY_AVAILABLE and not timeline_df.empty:
        gantt_fig = create_plotly_gantt(timeline_df)
        gantt_html = gantt_fig.to_html(full_html=False, include_plotlyjs='cdn')
    elif not timeline_df.empty:
        img_base64 = create_matplotlib_gantt(timeline_df, total_cycles)
        gantt_html = f'<img src="data:image/png;base64,{img_base64}" alt="Execution Timeline Gantt Chart" style="width: 100%;">'
    else:
        gantt_html = "<p>No timeline data to display.</p>"

    # --- Create HTML Report ---
    html_content = f"""
    <html>
    <head>
        <title>PyV-NPU Simulation Report</title>
        <style>
            body {{ font-family: sans-serif; margin: 2em; background-color: #f9f9f9; color: #333; }}
            h1, h2 {{ color: #1a237e; border-bottom: 2px solid #3f51b5; padding-bottom: 5px; }}
            table {{ border-collapse: collapse; width: 60%; margin-top: 1em; box-shadow: 0 2px 3px rgba(0,0,0,0.1); }}
            th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
            th {{ background-color: #3f51b5; color: white; }}
            tr:nth-child(even) {{ background-color: #f2f2f2; }}
            .container {{ max-width: 1400px; margin: auto; background-color: white; padding: 2em; box-shadow: 0 4px 8px rgba(0,0,0,0.1); border-radius: 8px; }}
            .chart {{ margin-top: 2em; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>PyV-NPU Simulation Report</h1>

            <h2>Summary</h2>
            <table>
                <tr><th>Metric</th><th>Value</th></tr>
                <tr><td>Total Cycles</td><td>{total_cycles:,}</td></tr>
            </table>

            <h2>Engine Utilization</h2>
            <table>
                <tr><th>Engine</th><th>Utilization (%)</th></tr>
                {engine_util_summary}
            </table>

            <div class="chart">
                <h2>Execution Timeline</h2>
                {gantt_html}
            </div>
        </div>
    </body>
    </html>
    """

    report_path = output_dir / "report.html"
    with open(report_path, "w", encoding='utf-8') as f:
        f.write(html_content)
    print(f"HTML report generated at: {report_path.resolve()}")

def create_plotly_gantt(df):
    """Creates an interactive Gantt chart using Plotly."""
    
    # Prepare data for plotly gantt chart
    gantt_data = []
    for i, row in df.iterrows():
        # Add stall bar
        if row.get('stall_cycles', 0) > 0:
            gantt_data.append(dict(
                Task=row['engine'], 
                Start=row['start'], 
                Finish=row['start'] + row['stall_cycles'], 
                Resource='Stall',
                Description=f"Op: {row['op']}<br>Stall Reason: {row['stall_reason']}"
            ))
        # Add execution bar
        gantt_data.append(dict(
            Task=row['engine'], 
            Start=row['start'] + row.get('stall_cycles', 0), 
            Finish=row['end'], 
            Resource='Execution',
            Description=f"Op: {row['op']}<br>Duration: {row['end'] - (row['start'] + row.get('stall_cycles', 0))} cycles"
        ))

    colors = {'Execution': 'rgb(63, 81, 181)', 'Stall': 'rgb(239, 83, 80)'}
    fig = ff.create_gantt(gantt_data, colors=colors, index_col='Resource', show_colorbar=True, group_tasks=True)
    fig.update_layout(
        title='Execution Timeline (Gantt Chart)',
        xaxis_title='Cycles',
        yaxis_title='Engine',
        legend_title='State'
    )
    return fig

def create_matplotlib_gantt(timeline_df, total_cycles):
    """Creates a static Gantt chart using Matplotlib."""
    fig, ax = plt.subplots(figsize=(20, 10))

    unique_engines = sorted(timeline_df["engine"].unique())
    colors = plt.cm.get_cmap("viridis", len(unique_engines))
    color_map = {engine: colors(i) for i, engine in enumerate(unique_engines)}
    stall_color = 'rgba(255, 0, 0, 0.5)'

    for i, engine in enumerate(unique_engines):
        engine_events = timeline_df[timeline_df["engine"] == engine]
        for _, event in engine_events.iterrows():
            start = event["start"]
            end = event["end"]
            stall_cycles = event.get("stall_cycles", 0)
            exec_start = start + stall_cycles
            exec_duration = end - exec_start
            op_name = event["op"]
            
            if stall_cycles > 0:
                ax.barh(i, stall_cycles, left=start, height=0.5, color=stall_color, edgecolor="gray", hatch="//")

            ax.barh(i, exec_duration, left=exec_start, height=0.5, color=color_map[engine], edgecolor="black")

            if exec_duration > total_cycles / 50:
                 ax.text(exec_start + exec_duration / 2, i, op_name, ha='center', va='center', color='white', fontsize=8, weight='bold')

    ax.set_yticks(range(len(unique_engines)))
    ax.set_yticklabels(unique_engines)
    ax.set_xlabel("Cycles")
    ax.set_ylabel("Engine")
    ax.set_title("Execution Timeline (Gantt Chart)")
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()

    img_buffer = BytesIO()
    plt.savefig(img_buffer, format='png', bbox_inches='tight')
    img_buffer.seek(0)
    img_base64 = base64.b64encode(img_buffer.read()).decode('utf-8')
    plt.close(fig)
    return img_base64

def generate_report_json(schedule: list, config: "SimConfig"):
    """
    Generates a JSON report from the final schedule.
    """
    if not schedule:
        return {}

    total_cycles = max(item.end_cycle for item in schedule) if schedule else 0

    engine_cycles = {}
    for item in schedule:
        duration = item.end_cycle - item.start_cycle
        engine_type = ''.join(filter(str.isalpha, item.engine))
        engine_cycles[engine_type] = engine_cycles.get(engine_type, 0) + duration

    engine_util = {}
    for engine, cycles in engine_cycles.items():
        num_engines_of_type = getattr(config, engine.lower(), 1)
        if engine == "DMA":
            num_engines_of_type = config.dma_channels
        
        total_possible_cycles = total_cycles * num_engines_of_type
        engine_util[engine] = (cycles / total_possible_cycles) * 100 if total_possible_cycles > 0 else 0

    timeline = [
        {
            "op": item.op.name,
            "engine": item.engine,
            "start": item.start_cycle,
            "end": item.end_cycle,
            "stall_cycles": item.stall_cycles,
            "stall_reason": item.stall_reason
        }
        for item in schedule
    ]

    report = {
        "total_cycles": total_cycles,
        "engine_util_abs": engine_cycles,
        "engine_util": {f"{k}_utilization": f"{v:.2f}%" for k,v in engine_util.items()},
        "timeline": timeline,
    }
    return report
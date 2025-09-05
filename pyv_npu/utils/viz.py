import plotly.express as px
import pandas as pd

def export_gantt(timeline, path: str):
    if not timeline:
        with open(path, "w") as f:
            f.write("<h1>Gantt Chart</h1><p>No data to display.</p>")
        return

    df = pd.DataFrame(timeline)
    # Ensure numeric types for cycle columns, coercing errors
    df['start'] = pd.to_numeric(df['start'], errors='coerce')
    df['end']   = pd.to_numeric(df['end'], errors='coerce')
    df = df.dropna(subset=['start', 'end'])

    df['duration'] = df['end'] - df['start']
    # Ensure a minimum duration for visibility
    df.loc[df['duration'] <= 0, 'duration'] = 1

    # Define hover data, checking for column existence
    hover_data_cols = ['op', 'start', 'end', 'duration', 'stall_cycles', 'stall_breakdown']
    existing_hover_cols = [c for c in hover_data_cols if c in df.columns]

    fig = px.timeline(
        df,
        x_start="start",
        x_end="end",
        y="engine",
        color="op",
        text="name",
        hover_name="name",
        hover_data=existing_hover_cols,
        title="NPU Simulation Timeline (Gantt Chart)",
        labels={"engine": "Hardware Engine", "op": "Operation"}
    )

    fig.update_yaxes(autorange="reversed", title="Engine")
    fig.update_xaxes(title="Cycle", range=[0, df['end'].max()])
    fig.update_layout(
        height=max(500, len(df['engine'].unique()) * 25),
        font=dict(family="Courier New, monospace", size=12),
        legend_title="Operation"
    )

    fig.write_html(path, include_plotlyjs="cdn", full_html=True)

def export_gantt_ascii(timeline):
    if not timeline:
        return "Timeline is empty."

    max_time = max((item['end'] for item in timeline), default=0)
    if max_time == 0:
        return "Timeline has no duration."

    # Group by engine
    engine_lanes = {}
    for item in timeline:
        if item['engine'] not in engine_lanes:
            engine_lanes[item['engine']] = []
        engine_lanes[item['engine']].append(item)

    # Sort engines for consistent output
    sorted_engines = sorted(engine_lanes.keys())

    chart = ""
    scale = 80.0 / max_time if max_time > 0 else 0 # Scale to 80 characters width

    chart += "NPU Simulation Timeline (ASCII Gantt Chart)\n"
    chart += "" + ("-" * 90) + "\n"

    for engine in sorted_engines:
        chart += f"{engine:>8} |";
        lane = ['-'] * 80
        for item in engine_lanes[engine]:
            start_pos = int(item['start'] * scale)
            end_pos = int(item['end'] * scale)
            op_char = item.get('op', '?')[0]
            for i in range(start_pos, min(end_pos, 80)):
                lane[i] = op_char
        chart += "".join(lane) + "\n"
    
    chart += "" + ("-" * 90) + "\n"
    chart += f"0 cycles {" " * 75}{max_time} cycles\n"

    return chart

def export_gantt_ascii(timeline):
    if not timeline:
        return "Timeline is empty."

    max_time = max((item['end'] for item in timeline), default=0)
    if max_time == 0:
        return "Timeline has no duration."

    # Group by engine
    engine_lanes = {}
    for item in timeline:
        if item['engine'] not in engine_lanes:
            engine_lanes[item['engine']] = []
        engine_lanes[item['engine']].append(item)

    # Sort engines for consistent output
    sorted_engines = sorted(engine_lanes.keys())

    chart = ""
    scale = 80.0 / max_time if max_time > 0 else 0 # Scale to 80 characters width

    chart += "NPU Simulation Timeline (ASCII Gantt Chart)\n"
    chart += "" + ("-" * 90) + "\n"

    for engine in sorted_engines:
        chart += f"{engine:>8} |";
        lane = ['-'] * 80
        for item in engine_lanes[engine]:
            start_pos = int(item['start'] * scale)
            end_pos = int(item['end'] * scale)
            for i in range(start_pos, end_pos):
                if i < 80:
                    lane[i] = '#'
        chart += "".join(lane) + "\n"
    
    chart += "" + ("-" * 90) + "\n"
    chart += f"0 cycles {" " * 75}{max_time} cycles\n"

    return chart


import plotly.express as px
import pandas as pd

def export_gantt(timeline, path:str):
    """Exports a timeline to an interactive HTML Gantt chart using Plotly."""
    if not timeline:
        print("Warning: Timeline is empty, cannot generate Gantt chart.")
        with open(path, "w") as f:
            f.write("<h1>Gantt Chart</h1><p>No data to display.</p>")
        return

    df = pd.DataFrame(timeline)

    # Create a text column for hover info and labels
    df['text'] = df.apply(lambda row: f"{row['name']} ({row['op']})<br>Start: {row['start']}<br>End: {row['end']}", axis=1)

    # Using integers for cycles directly
    df['start_cycle'] = pd.to_numeric(df['start'])
    df['end_cycle'] = pd.to_numeric(df['end'])

    fig = px.timeline(df,
                      x_start="start_cycle", 
                      x_end="end_cycle", 
                      y="engine", 
                      color="engine",
                      text="name", # Display op name on the bar
                      hover_name="text", # Use the custom text for hover
                      title="NPU Simulation Timeline (Gantt Chart)",
                      labels={"engine": "Hardware Engine"}
                     )

    # Improve layout
    fig.update_yaxes(autorange="reversed") # Show TE at top
    fig.update_layout(
        xaxis_title="Cycle",
        yaxis_title="Engine",
        legend_title="Engine",
        font=dict(
            family="Courier New, monospace",
            size=12,
            color="#7f7f7f"
        )
    )

    fig.write_html(path)

def export_gantt_ascii(timeline):
    if not timeline:
        return "Timeline is empty."

    # Find the max end time to determine the chart width
    max_time = 0
    for item in timeline:
        max_time = max(max_time, item['end'])

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
        chart += f"{engine:>8} |"
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



import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
import base64
from io import BytesIO

def generate_html_report(report_data: dict, output_dir: Path):
    """
    Generates an HTML report with a Gantt chart from the simulator's report data.

    Args:
        report_data: The report data dictionary from report.json.
        output_dir: The directory to save the report.html and timeline.png files.
    """
    output_dir.mkdir(exist_ok=True)
    timeline_df = pd.DataFrame(report_data["timeline"])
    total_cycles = report_data["total_cycles"]
    engine_util = report_data["engine_util"]

    # --- Create Gantt Chart ---
    fig, ax = plt.subplots(figsize=(20, 10))

    # Create a color map for engines
    unique_engines = sorted(timeline_df["engine"].unique())
    colors = plt.cm.get_cmap("tab20", len(unique_engines))
    color_map = {engine: colors(i) for i, engine in enumerate(unique_engines)}

    for i, engine in enumerate(unique_engines):
        engine_events = timeline_df[timeline_df["engine"] == engine]
        for _, event in engine_events.iterrows():
            start = event["start"]
            end = event["end"]
            duration = end - start
            op_name = event["op"]
            ax.barh(i, duration, left=start, height=0.6,
                    color=color_map[engine], edgecolor="black")
            # Add op name inside the bar if it fits
            if duration > total_cycles / 50: # Heuristic to avoid clutter
                 ax.text(start + duration / 2, i, op_name,
                         ha='center', va='center', color='white', fontsize=8, weight='bold')


    ax.set_yticks(range(len(unique_engines)))
    ax.set_yticklabels(unique_engines)
    ax.set_xlabel("Cycles")
    ax.set_ylabel("Engine")
    ax.set_title("Execution Timeline (Gantt Chart)")
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()

    # Save plot to a buffer
    img_buffer = BytesIO()
    plt.savefig(img_buffer, format='png', bbox_inches='tight')
    img_buffer.seek(0)
    img_base64 = base64.b64encode(img_buffer.read()).decode('utf-8')
    plt.close(fig)

    # --- Create HTML Report ---
    html_content = f"""
    <html>
    <head>
        <title>PyV-NPU Simulation Report</title>
        <style>
            body {{ font-family: sans-serif; margin: 2em; }}
            h1, h2 {{ color: #333; }}
            table {{ border-collapse: collapse; width: 50%; margin-top: 1em; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            .container {{ max-width: 1200px; margin: auto; }}
            .chart {{ margin-top: 2em; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>PyV-NPU Simulation Report</h1>

            <h2>Summary</h2>
            <table>
                <tr>
                    <th>Metric</th>
                    <th>Value</th>
                </tr>
                <tr>
                    <td>Total Cycles</td>
                    <td>{total_cycles:,}</td>
                </tr>
            </table>

            <h2>Engine Utilization</h2>
            <table>
                <tr>
                    <th>Engine</th>
                    <th>Cycles</th>
                    <th>Utilization (%)</th>
                </tr>
                {"".join(f"<tr><td>{k}</td><td>{v:,}</td><td>{v/total_cycles*100:.2f}%</td></tr>" for k, v in engine_util.items())}
            </table>

            <div class="chart">
                <h2>Execution Timeline (Gantt Chart)</h2>
                <img src="data:image/png;base64,{img_base64}" alt="Execution Timeline Gantt Chart" style="width: 100%;">
            </div>
        </div>
    </body>
    </html>
    """

    report_path = output_dir / "report.html"
    with open(report_path, "w") as f:
        f.write(html_content)
    print(f"HTML report generated at: {report_path.resolve()}")


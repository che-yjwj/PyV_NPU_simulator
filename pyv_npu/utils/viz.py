import plotly.express as px
import pandas as pd

import plotly.express as px
import pandas as pd

def export_gantt(timeline, path: str):
    if not timeline:
        with open(path, "w") as f:
            f.write("<h1>Gantt Chart</h1><p>No data to display.</p>")
        return

    df = pd.DataFrame(timeline)
    df['start_cycle'] = pd.to_numeric(df['start'], errors='coerce')
    df['end_cycle']   = pd.to_numeric(df['end'], errors='coerce')
    df = df.dropna(subset=['start_cycle', 'end_cycle'])

    # duration 계산
    df['duration'] = df['end_cycle'] - df['start_cycle']
    same = df['duration'] <= 0
    if same.any():
        df.loc[same, 'duration'] = 1  # 최소 1 cycle

    # hover 정보
    hover_cols = [c for c in ['op', 'start_cycle', 'end_cycle'] if c in df.columns]

    # px.bar로 간트 차트 생성
    fig = px.bar(
        df,
        x="duration",
        y="engine",
        base="start_cycle",
        color="engine",
        text="name",
        orientation='h',
        hover_data=hover_cols,
        title="NPU Simulation Timeline (Gantt Chart)",
        labels={"engine": "Hardware Engine"}
    )

    # 축 및 레이아웃 설정
    fig.update_yaxes(autorange="reversed", title="Engine")
    fig.update_xaxes(title="Cycle", range=[0, df['end_cycle'].max()])
    fig.update_layout(
        height=500,
        font=dict(family="Courier New, monospace", size=12),
        legend_title="Engine"
    )

    fig.write_html(path, include_plotlyjs="cdn", full_html=True)

    print("=== timeline sample ===")
    for row in timeline[:5]:  # 앞 5개만
        print(row)

    df = pd.DataFrame(timeline)
    print("\n=== DataFrame preview ===")
    print(df.head())
    print("\nColumns:", df.columns.tolist())

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
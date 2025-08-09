
# TODO: Graphviz/ASCII Gantt exporters
def export_gantt(timeline, path:str):
    # placeholder stub
    with open(path, "w") as f:
        f.write("# Gantt timeline placeholder\n")
        for it in timeline:
            f.write(f"{it['start']:>8} - {it['end']:>8} | {it['engine']} | {it['name']} ({it['op']})\n")

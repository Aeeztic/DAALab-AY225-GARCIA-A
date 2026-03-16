import csv
import heapq
import math
import os
import tkinter as tk
from tkinter import ttk, messagebox

# ─────────────────────────────────────────
#  DATA LOADING
# ─────────────────────────────────────────

def load_graph(filepath):
    graph = {}   # adjacency list: node -> list of (neighbor, attrs)
    nodes = set()

    with open(filepath, newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            frm  = row['From Node'].strip()
            to   = row['To Node'].strip()
            dist = float(row['Distance (km)'])
            time = float(row['Time (mins)'])
            fuel = float(row['Fuel (Liters)'])

            nodes.add(frm)
            nodes.add(to)

            graph.setdefault(frm, []).append((to,   {'distance': dist, 'time': time, 'fuel': fuel}))
            graph.setdefault(to,  []).append((frm,  {'distance': dist, 'time': time, 'fuel': fuel}))

    return graph, sorted(nodes)


# ─────────────────────────────────────────
#  DIJKSTRA
# ─────────────────────────────────────────

def dijkstra(graph, start, end, weight_key):
    """
    Returns (cost, path, totals_dict) where totals_dict has distance/time/fuel
    for the found path regardless of which weight was optimised.
    """
    # priority queue: (cost, node, path_so_far)
    pq = [(0, start, [start])]
    visited = {}

    while pq:
        cost, node, path = heapq.heappop(pq)

        if node in visited:
            continue
        visited[node] = (cost, path)

        if node == end:
            # compute full totals along the path
            totals = {'distance': 0, 'time': 0, 'fuel': 0}
            for i in range(len(path) - 1):
                a, b = path[i], path[i + 1]
                for (nb, attrs) in graph.get(a, []):
                    if nb == b:
                        totals['distance'] += attrs['distance']
                        totals['time']     += attrs['time']
                        totals['fuel']     += attrs['fuel']
                        break
            return cost, path, totals

        for (nb, attrs) in graph.get(node, []):
            if nb not in visited:
                new_cost = cost + attrs[weight_key]
                heapq.heappush(pq, (new_cost, nb, path + [nb]))

    return None, [], {}   # no path found


# ─────────────────────────────────────────
#  NODE MAP (Canvas-based)
# ─────────────────────────────────────────

NODE_POSITIONS = {
    'NOVELETA': (0.25, 0.35),
    'IMUS':     (0.42, 0.12),
    'BACOOR':   (0.75, 0.20),
    'KAWIT':    (0.25, 0.55),
    'DASMA':    (0.85, 0.55),
    'INDANG':   (0.18, 0.85),
    'SILANG':   (0.52, 0.85),
    'GENTRI':   (0.85, 0.85),
}

EDGE_COLOR      = '#94a3b8'
NODE_FILL       = '#1e40af'
NODE_OUTLINE    = '#93c5fd'
NODE_TEXT       = 'white'
HIGHLIGHT_EDGE  = '#f59e0b'
HIGHLIGHT_NODE  = '#f59e0b'
BG_COLOR        = '#0f172a'
CANVAS_W        = 720
CANVAS_H        = 480
RADIUS          = 22


def draw_map(canvas, graph, nodes, highlight_path=None):
    canvas.delete('all')
    W, H = CANVAS_W, CANVAS_H

    # background grid
    for x in range(0, W, 40):
        canvas.create_line(x, 0, x, H, fill='#1e293b', width=1)
    for y in range(0, H, 40):
        canvas.create_line(0, y, W, y, fill='#1e293b', width=1)

    def pos(name):
        px, py = NODE_POSITIONS.get(name, (0.5, 0.5))
        return int(px * W), int(py * H)

    drawn_edges = set()

    # collect highlighted edges
    hl_edges = set()
    if highlight_path and len(highlight_path) > 1:
        for i in range(len(highlight_path) - 1):
            a, b = highlight_path[i], highlight_path[i + 1]
            hl_edges.add((min(a, b), max(a, b)))

    # draw edges
    for node in graph:
        for (nb, attrs) in graph[node]:
            key = (min(node, nb), max(node, nb))
            if key in drawn_edges:
                continue
            drawn_edges.add(key)

            x1, y1 = pos(node)
            x2, y2 = pos(nb)
            is_hl  = key in hl_edges
            color  = HIGHLIGHT_EDGE if is_hl else EDGE_COLOR
            width  = 4 if is_hl else 1.5

            canvas.create_line(x1, y1, x2, y2, fill=color, width=width, smooth=True)

            # edge label (distance)
            mx, my = (x1 + x2) // 2, (y1 + y2) // 2
            canvas.create_text(mx, my - 8, text=f"{attrs['distance']}km",
                                fill='#cbd5e1', font=('Courier', 7))

    # draw nodes
    for name in nodes:
        x, y   = pos(name)
        is_hl  = highlight_path and name in highlight_path
        fill   = HIGHLIGHT_NODE if is_hl else NODE_FILL
        outline = '#fbbf24' if is_hl else NODE_OUTLINE
        lw     = 3 if is_hl else 1.5

        canvas.create_oval(x - RADIUS, y - RADIUS, x + RADIUS, y + RADIUS,
                           fill=fill, outline=outline, width=lw)
        canvas.create_text(x, y, text=name[:3], fill=NODE_TEXT,
                           font=('Courier', 8, 'bold'))
        canvas.create_text(x, y + RADIUS + 9, text=name,
                           fill='#e2e8f0', font=('Courier', 7))


# ─────────────────────────────────────────
#  GUI
# ─────────────────────────────────────────

def build_gui(graph, nodes):
    root = tk.Tk()
    root.title("Cavite Route Finder")
    root.configure(bg=BG_COLOR)
    root.resizable(False, False)

    # ── Title bar ──
    title_frm = tk.Frame(root, bg='#1e3a8a', pady=6)
    title_frm.pack(fill='x')
    tk.Label(title_frm, text="🗺  CAVITE ROUTE FINDER",
             font=('Courier', 14, 'bold'), fg='#93c5fd', bg='#1e3a8a').pack()
    tk.Label(title_frm, text="Node Map & Shortest Path Calculator",
             font=('Courier', 9), fg='#64748b', bg='#1e3a8a').pack()

    # ── Canvas ──
    canvas = tk.Canvas(root, width=CANVAS_W, height=CANVAS_H,
                       bg=BG_COLOR, highlightthickness=0)
    canvas.pack(padx=10, pady=(8, 4))

    draw_map(canvas, graph, nodes)

    # ── Controls ──
    ctrl = tk.Frame(root, bg=BG_COLOR, pady=6)
    ctrl.pack(fill='x', padx=14)

    def label(parent, text, **kw):
        return tk.Label(parent, text=text, bg=BG_COLOR,
                        fg='#94a3b8', font=('Courier', 9), **kw)

    label(ctrl, "From:").grid(row=0, column=0, sticky='e', padx=(0, 4))
    frm_var = tk.StringVar(value=nodes[0])
    ttk.Combobox(ctrl, textvariable=frm_var, values=nodes, width=12,
                 state='readonly').grid(row=0, column=1, padx=4)

    label(ctrl, "To:").grid(row=0, column=2, sticky='e', padx=(8, 4))
    to_var = tk.StringVar(value=nodes[-1])
    ttk.Combobox(ctrl, textvariable=to_var, values=nodes, width=12,
                 state='readonly').grid(row=0, column=3, padx=4)

    label(ctrl, "Optimise:").grid(row=0, column=4, sticky='e', padx=(8, 4))
    opt_var = tk.StringVar(value='distance')
    ttk.Combobox(ctrl, textvariable=opt_var,
                 values=['distance', 'time', 'fuel'],
                 width=10, state='readonly').grid(row=0, column=5, padx=4)

    # ── Result panel ──
    result_frm = tk.Frame(root, bg='#0f1e38', bd=0, pady=8)
    result_frm.pack(fill='x', padx=14, pady=(4, 10))

    result_var = tk.StringVar(value="Select nodes and click  Find Shortest Path")
    result_lbl = tk.Label(result_frm, textvariable=result_var,
                          bg='#0f1e38', fg='#e2e8f0',
                          font=('Courier', 9), justify='left',
                          wraplength=680, anchor='w')
    result_lbl.pack(padx=10)

    # ── Button ──
    def find_path():
        start = frm_var.get()
        end   = to_var.get()
        key   = opt_var.get()

        if start == end:
            messagebox.showwarning("Same Node", "Please select different From/To nodes.")
            return

        cost, path, totals = dijkstra(graph, start, end, key)

        if not path:
            result_var.set(f"❌  No path found from {start} to {end}.")
            draw_map(canvas, graph, nodes)
            return

        arrow = '  →  '
        path_str = arrow.join(path)
        units = {'distance': 'km', 'time': 'mins', 'fuel': 'L'}
        text = (
            f"✅  Shortest path ({key}) from {start} to {end}\n"
            f"Path   :  {path_str}\n"
            f"Distance:  {totals['distance']:.1f} km  │  "
            f"Time: {totals['time']:.0f} mins  │  "
            f"Fuel: {totals['fuel']:.2f} Liters"
        )
        result_var.set(text)
        draw_map(canvas, graph, nodes, highlight_path=path)

    btn = tk.Button(ctrl, text="Find Shortest Path",
                    command=find_path,
                    bg='#1d4ed8', fg='white',
                    font=('Courier', 9, 'bold'),
                    relief='flat', padx=12, pady=4,
                    activebackground='#2563eb', cursor='hand2')
    btn.grid(row=0, column=6, padx=(10, 0))

    reset_btn = tk.Button(ctrl, text="Reset",
                          command=lambda: [draw_map(canvas, graph, nodes),
                                           result_var.set("Select nodes and click  Find Shortest Path")],
                          bg='#334155', fg='#94a3b8',
                          font=('Courier', 9),
                          relief='flat', padx=8, pady=4,
                          activebackground='#475569', cursor='hand2')
    reset_btn.grid(row=0, column=7, padx=(4, 0))

    root.mainloop()


# ─────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────

if __name__ == '__main__':
    CSV_FILE = os.path.join(os.path.dirname(__file__), 'Book1.csv')
    try:
        graph, nodes = load_graph(CSV_FILE)
    except FileNotFoundError:
        print(f"ERROR: '{CSV_FILE}' not found. Place it in the same folder as this script.")
        raise

    print("Nodes loaded:", nodes)
    print(f"Edges: {sum(len(v) for v in graph.values()) // 2}")
    build_gui(graph, nodes)
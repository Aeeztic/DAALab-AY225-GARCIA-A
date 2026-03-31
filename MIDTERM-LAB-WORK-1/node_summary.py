import os
import sys

import matplotlib
matplotlib.use("TkAgg")

try:
    import pandas as pd
    import networkx as nx
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib import rcParams
    import tkinter as tk
    from tkinter import messagebox
    print("[OK] All libraries loaded.")
except ImportError as e:
    import tkinter as tk
    from tkinter import messagebox
    root = tk.Tk(); root.withdraw()
    messagebox.showerror("Missing Requirements",
        f"Missing: {e}\n\nRun: pip install pandas networkx matplotlib")
    sys.exit(1)

CYBER_BG      = "#0a0a0f"
CYBER_PANEL   = "#0d0d1a"
CYBER_ACCENT1 = "#00ffe7"
CYBER_ACCENT2 = "#ff2079"
CYBER_ACCENT3 = "#ffe600"
CYBER_PURPLE  = "#bf5fff"
CYBER_GRAY    = "#1e1e2e"
CYBER_TEXT    = "#e0e0ff"
CYBER_DIMTEXT = "#555577"
CYBER_GRID    = "#1a1a2e"
CYBER_NODEBG  = "#12122a"

rcParams.update({
    "figure.facecolor": CYBER_BG, "axes.facecolor": CYBER_BG,
    "savefig.facecolor": CYBER_BG, "text.color": CYBER_TEXT,
    "axes.labelcolor": CYBER_TEXT, "font.family": "monospace",
})


class DijkstraApp:
    def __init__(self, root):
        self.root = root
        self.root.title("◈ DIJKSTRA SHORTEST PATH — CYBERDECK v3.0 ◈")
        self.root.geometry("1150x860")
        self.root.configure(bg=CYBER_BG)

        self.graph_D = nx.DiGraph()
        self.graph_T = nx.DiGraph()
        self.graph_F = nx.DiGraph()
        self.nodes = []

        try:
            self.load_data()
        except FileNotFoundError as e:
            messagebox.showerror("Error", str(e)); self.root.destroy(); return
        except Exception as e:
            messagebox.showerror("Error", f"{e}"); self.root.destroy(); return

        self.metric_var = tk.StringVar(value='D')
        self.create_widgets()   
        self.run_dijkstra()

    def load_data(self):
        try:
            base_dir = os.path.dirname(os.path.abspath(__file__))
        except NameError:
            base_dir = os.getcwd()
        path = os.path.join(base_dir, 'dataset.csv')
        if not os.path.exists(path):
            raise FileNotFoundError(f"dataset.csv not found at: {path}")
        df = pd.read_csv(path)
        for _, row in df.iterrows():
            u, v = int(row['Node From']), int(row['Node To'])
            d, t, f = float(row['D']), float(row['T']), float(row['F'])
            self.graph_D.add_edge(u, v, weight=d)
            self.graph_T.add_edge(u, v, weight=t)
            self.graph_F.add_edge(u, v, weight=f)
            if u not in self.nodes: self.nodes.append(u)
            if v not in self.nodes: self.nodes.append(v)
        self.nodes.sort()

    def find_best_origin(self, G):
        """Return the origin node whose total shortest-path sum to all others is lowest."""
        best_origin = None
        best_total  = float('inf')
        best_lengths = {}
        best_paths   = {}

        for candidate in self.nodes:
            try:
                lengths = nx.single_source_dijkstra_path_length(G, candidate, weight='weight')
                paths   = nx.single_source_dijkstra_path(G, candidate, weight='weight')
                # Only count nodes reachable from candidate (exclude itself)
                total = sum(c for n, c in lengths.items() if n != candidate)
                # Must reach ALL other nodes to qualify
                if len(lengths) < len(self.nodes):
                    continue
                if total < best_total:
                    best_total   = total
                    best_origin  = candidate
                    best_lengths = lengths
                    best_paths   = paths
            except Exception:
                continue

        return best_origin, best_total, best_lengths, best_paths

    def run_dijkstra(self):
        graphs = {'D': self.graph_D, 'T': self.graph_T, 'F': self.graph_F}
        metric = self.metric_var.get()
        G = graphs[metric]

        origin, total, self.lengths, self.paths = self.find_best_origin(G)

        if origin is None:
            messagebox.showerror("Error", "No node can reach all others.")
            return

        self.best_origin = origin
        self.draw_graph(G, origin, metric)
        self.rebuild_table(origin, metric, total)

    def create_widgets(self):
        # Header
        hdr = tk.Frame(self.root, bg=CYBER_BG, pady=6)
        hdr.pack(side=tk.TOP, fill=tk.X)
        tk.Label(hdr, text="◈  DIJKSTRA  SHORTEST  PATH  FINDER  ◈",
                 font=("Courier New", 15, "bold"), fg=CYBER_ACCENT1, bg=CYBER_BG).pack()
        tk.Label(hdr, text="─── CYBERDECK NAVIGATION SYSTEM v3.0 ───",
                 font=("Courier New", 15), fg=CYBER_DIMTEXT, bg=CYBER_BG).pack()

        # Controls — metric only, no origin picker
        ctrl = tk.Frame(self.root, bg=CYBER_PANEL, pady=8, padx=12,
                        highlightbackground=CYBER_ACCENT1, highlightthickness=1)
        ctrl.pack(side=tk.TOP, fill=tk.X, padx=8, pady=(0, 4))

        tk.Label(ctrl, text="METRIC:", font=("Courier New", 12, "bold"),
                 fg=CYBER_ACCENT3, bg=CYBER_PANEL).pack(side=tk.LEFT, padx=(0, 6))

        btn = dict(font=("Courier New", 10, "bold"), bg=CYBER_PANEL, fg=CYBER_TEXT,
                   activebackground=CYBER_GRAY, activeforeground=CYBER_ACCENT1,
                   selectcolor=CYBER_GRAY, relief=tk.FLAT, bd=0,
                   padx=10, pady=4, cursor="crosshair", indicatoron=False)
        for lbl, val in [("▸ DISTANCE", 'D'), ("▸ TIME", 'T'), ("▸ FUEL", 'F')]:
            tk.Radiobutton(ctrl, text=lbl, variable=self.metric_var, value=val,
                           command=self.run_dijkstra, **btn).pack(side=tk.LEFT, padx=4)

        # Auto-origin badge (updated dynamically)
        tk.Label(ctrl, text="  ║  ", fg=CYBER_DIMTEXT, bg=CYBER_PANEL,
                 font=("Courier New", 12)).pack(side=tk.LEFT)
        tk.Label(ctrl, text="AUTO ORIGIN:", font=("Courier New", 10, "bold"),
                 fg=CYBER_ACCENT3, bg=CYBER_PANEL).pack(side=tk.LEFT, padx=(0, 4))
        self.origin_badge = tk.Label(ctrl, text="—",
                 font=("Courier New", 13, "bold"),
                 fg=CYBER_ACCENT2, bg=CYBER_PANEL)
        self.origin_badge.pack(side=tk.LEFT)

        # Body: graph left | table right
        body = tk.Frame(self.root, bg=CYBER_BG)
        body.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True, padx=8, pady=6)

        po = tk.Frame(body, bg=CYBER_ACCENT1, padx=1, pady=1)
        po.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        pi = tk.Frame(po, bg=CYBER_BG)
        pi.pack(fill=tk.BOTH, expand=True)
        self.figure, self.ax = plt.subplots(figsize=(7, 6))
        self.figure.patch.set_facecolor(CYBER_BG)
        self.canvas = FigureCanvasTkAgg(self.figure, master=pi)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        to = tk.Frame(body, bg=CYBER_ACCENT1, padx=1, pady=1)
        to.pack(side=tk.RIGHT, fill=tk.Y, padx=(6, 0))
        ti = tk.Frame(to, bg=CYBER_BG, padx=8, pady=8)
        ti.pack(fill=tk.BOTH, expand=True)
        tk.Label(ti, text="SHORTEST PATHS", font=("Courier New", 14, "bold"),
                 fg=CYBER_ACCENT3, bg=CYBER_BG).pack(pady=(0, 6))
        self.table_frame = tk.Frame(ti, bg=CYBER_BG)
        self.table_frame.pack(fill=tk.BOTH, expand=True)

    def rebuild_table(self, origin, metric, total):
        for w in self.table_frame.winfo_children():
            w.destroy()

        # Update origin badge in control bar
        self.origin_badge.config(text=f"NODE {origin}")

        units = {'D': 'km', 'T': 'min', 'F': 'units'}
        unit = units[metric]

        # Column headers
        for col, txt, w in [(0, "DEST", 6), (1, "COST", 14), (2, "PATH", 24)]:
            tk.Label(self.table_frame, text=txt, font=("Courier New", 12, "bold"),
                     fg=CYBER_ACCENT3, bg=CYBER_PANEL, padx=8, pady=4,
                     anchor='w', width=w).grid(row=0, column=col, padx=1, pady=1, sticky='ew')

        row_idx = 0
        for node in sorted(self.lengths.keys()):
            if node == origin:
                continue
            row_idx += 1
            cost = self.lengths[node]
            path_str = " › ".join(map(str, self.paths[node]))
            bg = CYBER_NODEBG if row_idx % 2 == 0 else CYBER_BG

            tk.Label(self.table_frame, text=f"  {node}",
                     font=("Courier New", 12, "bold"), fg=CYBER_PURPLE, bg=bg,
                     padx=8, pady=3, anchor='w', width=6
                     ).grid(row=row_idx, column=0, padx=1, pady=1, sticky='ew')
            tk.Label(self.table_frame, text=f"  {cost:.1f} {unit}",
                     font=("Courier New", 12), fg=CYBER_ACCENT1, bg=bg,
                     padx=8, pady=3, anchor='w', width=14
                     ).grid(row=row_idx, column=1, padx=1, pady=1, sticky='ew')
            tk.Label(self.table_frame, text=f"  {path_str}",
                     font=("Courier New", 12), fg=CYBER_TEXT, bg=bg,
                     padx=8, pady=3, anchor='w', width=24
                     ).grid(row=row_idx, column=2, padx=1, pady=1, sticky='ew')

        # Total row
        tr = row_idx + 1
        tk.Label(self.table_frame, text="  TOTAL", font=("Courier New", 12, "bold"),
                 fg=CYBER_ACCENT3, bg=CYBER_PANEL, padx=8, pady=4, anchor='w', width=6
                 ).grid(row=tr, column=0, padx=1, pady=(4,1), sticky='ew')
        tk.Label(self.table_frame, text=f"  {total:.1f} {unit}",
                 font=("Courier New", 12, "bold"), fg=CYBER_ACCENT3, bg=CYBER_PANEL,
                 padx=8, pady=4, anchor='w', width=14
                 ).grid(row=tr, column=1, padx=1, pady=(4,1), sticky='ew')
        tk.Label(self.table_frame, text="", bg=CYBER_PANEL, width=24
                 ).grid(row=tr, column=2, padx=1, pady=(4,1), sticky='ew')

    def draw_graph(self, G, origin, metric):
        self.ax.clear()
        self.ax.set_facecolor(CYBER_BG)
        labels_map = {'D': 'DISTANCE', 'T': 'TIME', 'F': 'FUEL'}
        pos = nx.spring_layout(G, seed=42)

        for v in [0.2, 0.4, 0.6, 0.8]:
            for s in [1, -1]:
                self.ax.axhline(s*v, color=CYBER_GRID, lw=0.5, alpha=0.4, zorder=0)
                self.ax.axvline(s*v, color=CYBER_GRID, lw=0.5, alpha=0.4, zorder=0)

        sp_edges = set()
        for node, path in self.paths.items():
            if node == origin: continue
            for i in range(len(path)-1):
                sp_edges.add((path[i], path[i+1]))

        bg_edges = [e for e in G.edges() if e not in sp_edges]
        nx.draw_networkx_edges(G, pos, ax=self.ax, edgelist=bg_edges,
                               edge_color=CYBER_DIMTEXT, alpha=0.2,
                               arrows=True, arrowsize=10,
                               connectionstyle="arc3,rad=0.08", width=0.7)
        if sp_edges:
            nx.draw_networkx_edges(G, pos, ax=self.ax, edgelist=list(sp_edges),
                                   edge_color=CYBER_ACCENT1, alpha=0.9,
                                   arrows=True, arrowsize=18,
                                   connectionstyle="arc3,rad=0.08", width=2.5)

        others = [n for n in G.nodes() if n != origin]
        nx.draw_networkx_nodes(G, pos, ax=self.ax, nodelist=[origin],
                               node_color=CYBER_ACCENT2, node_size=950,
                               edgecolors=CYBER_ACCENT3, linewidths=2.5)
        nx.draw_networkx_nodes(G, pos, ax=self.ax, nodelist=others,
                               node_color=CYBER_NODEBG, node_size=750,
                               edgecolors=CYBER_PURPLE, linewidths=2.0)
        nx.draw_networkx_labels(G, pos, ax=self.ax, font_color=CYBER_TEXT,
                                font_size=12, font_family="monospace", font_weight="bold")

        for node, cost in self.lengths.items():
            if node == origin or node not in pos: continue
            x, y = pos[node]
            self.ax.annotate(f"{cost:.0f}", xy=(x, y), xytext=(x+0.05, y+0.08),
                             fontsize=12, color=CYBER_ACCENT3,
                             fontfamily="monospace", fontweight="bold",
                             bbox=dict(boxstyle="round,pad=0.15", fc=CYBER_BG,
                                       ec=CYBER_ACCENT3, alpha=0.85, lw=0.8))

        edge_labels = nx.get_edge_attributes(G, 'weight')
        nx.draw_networkx_edge_labels(G, pos,
                                     edge_labels={k: f"{v:.0f}" for k, v in edge_labels.items()},
                                     ax=self.ax, font_color=CYBER_DIMTEXT,
                                     font_size=12, font_family="monospace",
                                     bbox=dict(boxstyle="round,pad=0.1",
                                               fc=CYBER_BG, ec="none", alpha=0.7))

        legend_items = [
            mpatches.Patch(color=CYBER_ACCENT2, label=f"Best Origin: Node {origin}"),
            mpatches.Patch(color=CYBER_PURPLE,  label="Destination nodes"),
            mpatches.Patch(color=CYBER_ACCENT1, label="Shortest path edges"),
        ]
        self.ax.legend(handles=legend_items, loc="lower right",
                       facecolor=CYBER_PANEL, edgecolor=CYBER_ACCENT1,
                       labelcolor=CYBER_TEXT, prop={"family": "monospace", "size": 8})

        self.ax.set_title(
            f"[ DIJKSTRA  ›  {labels_map[metric]}  ›  BEST ORIGIN: NODE {origin} ]",
            fontsize=12, fontweight="bold", color=CYBER_ACCENT3,
            fontfamily="monospace", pad=12)
        self.ax.axis('off')
        self.canvas.draw()


if __name__ == "__main__":
    print("[..] Starting Dijkstra App...")
    try:
        root = tk.Tk()
        root.update()
        print("[OK] Tkinter window created.")
        app = DijkstraApp(root)
        print("[OK] App initialized. Entering main loop.")
        root.mainloop()
    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()
        input("Press Enter to exit...")

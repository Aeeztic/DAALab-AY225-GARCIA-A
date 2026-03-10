import os
import sys

# Force tkinter-compatible backend BEFORE any other matplotlib import
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
    from tkinter import ttk, messagebox
    print("[OK] All libraries loaded.")
except ImportError as e:
    import tkinter as tk
    from tkinter import messagebox
    root = tk.Tk()
    root.withdraw()
    messagebox.showerror(
        "Missing Requirements",
        f"A required library is missing: {e}\n\n"
        "Please run:\n  pip install pandas networkx matplotlib"
    )
    sys.exit(1)

# ── Cyberpunk Palette ──────────────────────────────────────────────────────────
CYBER_BG      = "#0a0a0f"
CYBER_PANEL   = "#0d0d1a"
CYBER_ACCENT1 = "#00ffe7"   # neon cyan   — shortest path edges
CYBER_ACCENT2 = "#ff2079"   # hot pink    — origin node
CYBER_ACCENT3 = "#ffe600"   # electric yellow — labels / totals
CYBER_PURPLE  = "#bf5fff"   # neon purple — destination nodes
CYBER_GRAY    = "#e46060"
CYBER_TEXT    = "#e0e0ff"
CYBER_DIMTEXT = "#FF4F4F"
CYBER_GRID    = "#1a1a2e"
CYBER_NODEBG  = "#12122a"

rcParams.update({
    "figure.facecolor":  CYBER_BG,
    "axes.facecolor":    CYBER_BG,
    "savefig.facecolor": CYBER_BG,
    "text.color":        CYBER_TEXT,
    "axes.labelcolor":   CYBER_TEXT,
    "font.family":       "monospace",
})


class DijkstraApp:
    def __init__(self, root):
        self.root = root
        self.root.title("◈ DIJKSTRA SHORTEST PATH — CYBERDECK v3.0 ◈")
        self.root.geometry("1100x860")
        self.root.configure(bg=CYBER_BG)

        self.graph_D = nx.DiGraph()
        self.graph_T = nx.DiGraph()
        self.graph_F = nx.DiGraph()
        self.nodes = []

        try:
            self.load_data()
        except FileNotFoundError as e:
            messagebox.showerror("Error", str(e))
            self.root.destroy()
            return
        except Exception as e:
            messagebox.showerror("Error Reading Dataset", f"An error occurred:\n{e}")
            self.root.destroy()
            return

        self.create_widgets()
        self.run_dijkstra()   # draw with defaults

    # ── Load CSV ──────────────────────────────────────────────────────────────
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

    # ── Build UI ──────────────────────────────────────────────────────────────
    def create_widgets(self):
        # ── header ────────────────────────────────────────────────────────────
        header = tk.Frame(self.root, bg=CYBER_BG, pady=6)
        header.pack(side=tk.TOP, fill=tk.X)

        tk.Label(header,
                 text="◈  DIJKSTRA  SHORTEST  PATH  OPTIMIZER  ◈",
                 font=("Courier New", 14, "bold"),
                 fg=CYBER_ACCENT1, bg=CYBER_BG).pack()
        tk.Label(header,
                 text="─── CYBERDECK NAVIGATION SYSTEM v3.0 ───",
                 font=("Courier New", 8),
                 fg=CYBER_DIMTEXT, bg=CYBER_BG).pack()

        # ── control bar ───────────────────────────────────────────────────────
        ctrl = tk.Frame(self.root, bg=CYBER_PANEL, pady=8, padx=12,
                        highlightbackground=CYBER_ACCENT1, highlightthickness=1)
        ctrl.pack(side=tk.TOP, fill=tk.X, padx=8, pady=(0, 4))

        # Origin node selector
        tk.Label(ctrl, text="ORIGIN NODE:",
                 font=("Courier New", 10, "bold"),
                 fg=CYBER_ACCENT3, bg=CYBER_PANEL).pack(side=tk.LEFT, padx=(0, 6))

        self.origin_var = tk.IntVar(value=self.nodes[0])
        node_menu = tk.OptionMenu(ctrl, self.origin_var, *self.nodes,
                                  command=lambda _: self.run_dijkstra())
        node_menu.config(
            font=("Courier New", 10, "bold"),
            bg=CYBER_GRAY, fg=CYBER_ACCENT1,
            activebackground=CYBER_PANEL, activeforeground=CYBER_ACCENT3,
            highlightthickness=0, relief=tk.FLAT, bd=0,
            cursor="crosshair", width=4,
        )
        node_menu["menu"].config(
            bg=CYBER_GRAY, fg=CYBER_ACCENT1,
            font=("Courier New", 10),
            activebackground=CYBER_ACCENT1, activeforeground=CYBER_BG,
        )
        node_menu.pack(side=tk.LEFT, padx=6)

        # Separator
        tk.Label(ctrl, text="  ║  ",
                 font=("Courier New", 10),
                 fg=CYBER_DIMTEXT, bg=CYBER_PANEL).pack(side=tk.LEFT)

        # Metric selector
        tk.Label(ctrl, text="METRIC:",
                 font=("Courier New", 10, "bold"),
                 fg=CYBER_ACCENT3, bg=CYBER_PANEL).pack(side=tk.LEFT, padx=(0, 6))

        self.metric_var = tk.StringVar(value='D')
        btn_cfg = dict(
            font=("Courier New", 10, "bold"),
            bg=CYBER_PANEL, fg=CYBER_TEXT,
            activebackground=CYBER_GRAY, activeforeground=CYBER_ACCENT1,
            selectcolor=CYBER_GRAY,
            relief=tk.FLAT, bd=0, padx=10, pady=4,
            cursor="crosshair",
        )
        for label, val in [("▸ DISTANCE", 'D'), ("▸ TIME", 'T'), ("▸ FUEL", 'F')]:
            tk.Radiobutton(ctrl, text=label,
                           variable=self.metric_var, value=val,
                           command=self.run_dijkstra,
                           indicatoron=False, **btn_cfg).pack(side=tk.LEFT, padx=4)

        # ── results table strip ───────────────────────────────────────────────
        self.result_frame = tk.Frame(self.root, bg=CYBER_BG)
        self.result_frame.pack(side=tk.TOP, fill=tk.X, padx=8, pady=2)

        # ── canvas ────────────────────────────────────────────────────────────
        plot_outer = tk.Frame(self.root, bg=CYBER_ACCENT1, padx=1, pady=1)
        plot_outer.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True, padx=8, pady=6)
        plot_inner = tk.Frame(plot_outer, bg=CYBER_BG)
        plot_inner.pack(fill=tk.BOTH, expand=True)

        self.figure, self.ax = plt.subplots(figsize=(9, 6))
        self.figure.patch.set_facecolor(CYBER_BG)
        self.canvas = FigureCanvasTkAgg(self.figure, master=plot_inner)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # ── Dijkstra ──────────────────────────────────────────────────────────────
    def run_dijkstra(self):
        metric = self.metric_var.get()
        origin = self.origin_var.get()

        graphs = {'D': self.graph_D, 'T': self.graph_T, 'F': self.graph_F}
        units  = {'D': 'km',         'T': 'min',        'F': 'units'}
        G = graphs[metric]

        # nx.single_source_dijkstra returns (distances, paths)
        try:
            dist_map, path_map = nx.single_source_dijkstra(G, origin, weight='weight')
        except nx.NetworkXError as e:
            messagebox.showerror("Dijkstra Error", str(e))
            return

        self._rebuild_result_table(origin, dist_map, path_map, units[metric])
        self._draw(G, origin, dist_map, path_map, metric)

    # ── Result table ──────────────────────────────────────────────────────────
    def _rebuild_result_table(self, origin, dist_map, path_map, unit):
        for w in self.result_frame.winfo_children():
            w.destroy()

        # Header row
        hdr = tk.Frame(self.result_frame, bg=CYBER_BG)
        hdr.pack(side=tk.TOP, fill=tk.X, padx=8, pady=(2, 0))
        for col, width in [("DEST", 6), ("COST", 8), ("PATH", 60)]:
            tk.Label(hdr, text=col, width=width,
                     font=("Courier New", 8, "bold"),
                     fg=CYBER_ACCENT3, bg=CYBER_BG,
                     anchor='w').pack(side=tk.LEFT)

        # One row per destination
        total = 0.0
        for node in sorted(dist_map):
            if node == origin:
                continue
            cost = dist_map[node]
            total += cost
            path_str = "  ›  ".join(map(str, path_map[node]))
            color = CYBER_ACCENT1 if cost == min(
                v for k, v in dist_map.items() if k != origin) else CYBER_TEXT

            row = tk.Frame(self.result_frame, bg=CYBER_BG)
            row.pack(side=tk.TOP, fill=tk.X, padx=8)
            tk.Label(row, text=str(node), width=6,
                     font=("Courier New", 9, "bold"),
                     fg=CYBER_PURPLE, bg=CYBER_BG, anchor='w').pack(side=tk.LEFT)
            tk.Label(row, text=f"{cost:.1f} {unit}", width=10,
                     font=("Courier New", 9, "bold"),
                     fg=color, bg=CYBER_BG, anchor='w').pack(side=tk.LEFT)
            tk.Label(row, text=path_str,
                     font=("Courier New", 9),
                     fg=CYBER_TEXT, bg=CYBER_BG, anchor='w').pack(side=tk.LEFT)

        # Total row
        sep = tk.Frame(self.result_frame, bg=CYBER_DIMTEXT, height=1)
        sep.pack(fill=tk.X, padx=8, pady=2)
        tot_row = tk.Frame(self.result_frame, bg=CYBER_BG)
        tot_row.pack(side=tk.TOP, fill=tk.X, padx=8)
        tk.Label(tot_row,
                 text=f"⚡ TOTAL FROM NODE {origin}:  {total:.1f} {unit}",
                 font=("Courier New", 10, "bold"),
                 fg=CYBER_ACCENT3, bg=CYBER_BG).pack(side=tk.LEFT)

    # ── Draw graph ────────────────────────────────────────────────────────────
    def _draw(self, G, origin, dist_map, path_map, metric):
        self.ax.clear()
        self.ax.set_facecolor(CYBER_BG)

        labels_map = {'D': 'DISTANCE', 'T': 'TIME', 'F': 'FUEL'}
        pos = nx.spring_layout(G, seed=42)

        # grid lines
        for v in [0.2, 0.4, 0.6, 0.8]:
            for sign in [v, -v]:
                self.ax.axhline(sign, color=CYBER_GRID, lw=0.4, alpha=0.4, zorder=0)
                self.ax.axvline(sign, color=CYBER_GRID, lw=0.4, alpha=0.4, zorder=0)

        # all edges faded
        nx.draw_networkx_edges(G, pos, ax=self.ax,
                               edgelist=list(G.edges()),
                               edge_color=CYBER_DIMTEXT, alpha=0.2,
                               arrows=True, arrowsize=10,
                               connectionstyle="arc3,rad=0.08", width=0.7)

        # collect all shortest-path edges
        sp_edges = []
        for node, path in path_map.items():
            if node == origin:
                continue
            for i in range(len(path) - 1):
                e = (path[i], path[i + 1])
                if e not in sp_edges:
                    sp_edges.append(e)

        # draw shortest path edges in neon cyan
        nx.draw_networkx_edges(G, pos, ax=self.ax,
                               edgelist=sp_edges,
                               edge_color=CYBER_ACCENT1, width=2.5,
                               arrows=True, arrowsize=18,
                               connectionstyle="arc3,rad=0.08", alpha=0.95)

        # node colors: origin = hot pink, others = dark
        node_colors = [CYBER_ACCENT2 if n == origin else CYBER_NODEBG for n in G.nodes()]
        node_borders = [CYBER_ACCENT2 if n == origin else CYBER_PURPLE for n in G.nodes()]

        nx.draw_networkx_nodes(G, pos, ax=self.ax,
                               node_color=node_colors,
                               edgecolors=node_borders,
                               node_size=850, linewidths=2.2)

        # node labels with distance annotation
        nx.draw_networkx_labels(G, pos, ax=self.ax,
                                font_color=CYBER_BG if False else CYBER_ACCENT1,
                                font_size=11, font_family="monospace",
                                font_weight="bold")

        # distance cost labels next to each non-origin node
        offset = 0.08
        for node in G.nodes():
            if node == origin or node not in dist_map:
                continue
            x, y = pos[node]
            self.ax.text(x, y + offset,
                         f"{dist_map[node]:.0f}",
                         color=CYBER_ACCENT3,
                         fontsize=8, fontfamily="monospace",
                         ha='center', va='bottom',
                         bbox=dict(boxstyle="round,pad=0.15",
                                   fc=CYBER_BG, ec=CYBER_ACCENT3,
                                   alpha=0.85, linewidth=0.8))

        # edge weight labels
        edge_labels = nx.get_edge_attributes(G, 'weight')
        nx.draw_networkx_edge_labels(G, pos,
                                     edge_labels={k: f"{v:.0f}" for k, v in edge_labels.items()},
                                     ax=self.ax,
                                     font_color=CYBER_DIMTEXT,
                                     font_size=7, font_family="monospace",
                                     bbox=dict(boxstyle="round,pad=0.1",
                                               fc=CYBER_BG, ec="none", alpha=0.7))

        # legend
        legend_items = [
            mpatches.Patch(color=CYBER_ACCENT2, label=f"Origin  (Node {origin})"),
            mpatches.Patch(color=CYBER_ACCENT1, label="Shortest Path Edges"),
            mpatches.Patch(color=CYBER_ACCENT3, label="Cost from Origin"),
        ]
        self.ax.legend(handles=legend_items, loc="lower right",
                       facecolor=CYBER_PANEL, edgecolor=CYBER_ACCENT1,
                       labelcolor=CYBER_TEXT,
                       prop={"family": "monospace", "size": 8})

        self.ax.set_title(
            f"[ DIJKSTRA  ›  {labels_map[metric]}  ›  ORIGIN: NODE {origin} ]",
            fontsize=12, fontweight="bold",
            color=CYBER_ACCENT3, fontfamily="monospace", pad=14)
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

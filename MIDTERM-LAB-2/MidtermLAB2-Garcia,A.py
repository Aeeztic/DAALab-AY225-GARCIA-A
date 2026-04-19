import csv
import heapq
import os
import tkinter as tk
from tkinter import messagebox


def load_graph(filepath):
    graph = {}
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

            graph.setdefault(frm, []).append((to,  {'distance': dist, 'time': time, 'fuel': fuel}))
            graph.setdefault(to,  []).append((frm, {'distance': dist, 'time': time, 'fuel': fuel}))

    return graph, sorted(nodes)


def dijkstra(graph, start, end, weight_key):
    pq = [(0, start, [start])]
    visited = {}

    while pq:
        cost, node, path = heapq.heappop(pq)

        if node in visited:
            continue
        visited[node] = (cost, path)

        if node == end:
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

    return None, [], {}


# ─────────────────────────────────────────
#  DESIGN TOKENS
# ─────────────────────────────────────────

C_BG          = '#f5efe0'
C_MAP_BG      = '#ede5cc'
C_PANEL       = '#f0e8d2'
C_BORDER      = '#b08050'
C_INK         = '#3b2a1a'
C_INK_LIGHT   = '#6b4c2a'
C_GRID        = '#d4c9a8'
C_CONTOUR     = '#c9bc96'
C_EDGE        = '#9b7d5a'
C_EDGE_HL     = '#c0392b'
C_NODE_FILL   = '#3b2a1a'
C_NODE_HL     = '#c0392b'
C_NODE_RING   = '#b08050'
C_NODE_RING_HL= '#e74c3c'
C_BTN_FG      = '#f5efe0'
C_ACCENT      = '#c0392b'

FONT_LABEL    = ('Courier', 9)
FONT_RESULT   = ('Courier', 11)
FONT_NODE     = ('Courier', 7, 'bold')
FONT_EDGE_LBL = ('Courier', 8)

RADIUS        = 20

NODE_POSITIONS = {
    'NOVELETA': (0.14, 0.32),
    'IMUS':     (0.38, 0.13),
    'BACOOR':   (0.68, 0.18),
    'KAWIT':    (0.14, 0.54),
    'DASMA':    (0.80, 0.52),
    'INDANG':   (0.12, 0.84),
    'SILANG':   (0.50, 0.83),
    'GENTRI':   (0.82, 0.83),
}


# ─────────────────────────────────────────
#  MAP DRAWING  (reads live canvas size)
# ─────────────────────────────────────────

def draw_map(canvas, graph, nodes, highlight_path=None):
    canvas.update_idletasks()
    W = canvas.winfo_width()
    H = canvas.winfo_height()
    if W < 10 or H < 10:
        return

    canvas.delete('all')

    # Parchment fill
    canvas.create_rectangle(0, 0, W, H, fill=C_MAP_BG, outline='')

    # Crosshatch grid
    step = max(W, H) // 20
    for x in range(0, W, step):
        canvas.create_line(x, 0, x, H, fill=C_GRID, width=1, dash=(2, 6))
    for y in range(0, H, step):
        canvas.create_line(0, y, W, y, fill=C_GRID, width=1, dash=(2, 6))

    # Compass rose
    cx, cy, cr = W - 56, 56, 34
    canvas.create_oval(cx-cr, cy-cr, cx+cr, cy+cr,
                       outline=C_BORDER, fill=C_MAP_BG, width=1)
    for dx, dy, lbl in [(0, -(cr-6), 'N'), (0, cr-6, 'S'),
                         (cr-6, 0, 'E'), (-(cr-6), 0, 'W')]:
        canvas.create_line(cx, cy, cx+dx, cy+dy, fill=C_INK, width=1)
        canvas.create_text(cx+dx*1.5, cy+dy*1.5,
                           text=lbl, fill=C_INK, font=('Courier', 12, 'bold'))
    canvas.create_oval(cx-3, cy-3, cx+3, cy+3, fill=C_INK, outline='')

    def pos(name):
        px, py = NODE_POSITIONS.get(name, (0.5, 0.5))
        return int(px * W), int(py * H)

    hl_edges = set()
    if highlight_path and len(highlight_path) > 1:
        for i in range(len(highlight_path) - 1):
            a, b = highlight_path[i], highlight_path[i + 1]
            hl_edges.add((min(a, b), max(a, b)))

    drawn_edges = set()

    for node in graph:
        for (nb, attrs) in graph[node]:
            key = (min(node, nb), max(node, nb))
            if key in drawn_edges:
                continue
            drawn_edges.add(key)

            x1, y1 = pos(node)
            x2, y2 = pos(nb)
            is_hl  = key in hl_edges

            if is_hl:
                canvas.create_line(x1, y1, x2, y2,
                                   fill=C_EDGE_HL, width=5, capstyle='round')
                canvas.create_line(x1, y1, x2, y2,
                                   fill='#f5efe0', width=1,
                                   dash=(6, 8), capstyle='round')
            else:
                canvas.create_line(x1, y1, x2, y2,
                                   fill=C_BORDER, width=3, capstyle='round')
                canvas.create_line(x1, y1, x2, y2,
                                   fill=C_EDGE, width=1.5, capstyle='round')

            mx, my = (x1+x2)//2, (y1+y2)//2
            lbl = f"{attrs['distance']}km"
            tw  = len(lbl) * 4 + 4
            canvas.create_rectangle(mx-tw, my-8, mx+tw, my+8,
                                    fill=C_PANEL, outline=C_BORDER, width=1)
            canvas.create_text(mx, my, text=lbl, fill=C_INK, font=FONT_EDGE_LBL)

    for name in nodes:
        x, y   = pos(name)
        is_hl  = bool(highlight_path and name in highlight_path)
        is_end = bool(highlight_path and name in (highlight_path[0], highlight_path[-1]))

        fill = C_NODE_HL      if is_hl  else C_NODE_FILL
        ring = C_NODE_RING_HL if is_hl  else C_NODE_RING
        r    = RADIUS + 4     if is_end else RADIUS

        canvas.create_oval(x-r+2, y-r+2, x+r+2, y+r+2, fill='#a08060', outline='')
        canvas.create_oval(x-r,   y-r,   x+r,   y+r,   fill=fill, outline=ring, width=2)
        canvas.create_oval(x-r+5, y-r+5, x+r-5, y+r-5, fill='', outline=ring, width=1)
        canvas.create_text(x, y, text=name[:3], fill='#f5efe0', font=FONT_NODE)

        lw = len(name) * 5 + 6
        canvas.create_rectangle(x-lw, y+r+3, x+lw, y+r+17,
                                fill=C_NODE_HL if is_hl else C_INK, outline='')
        canvas.create_text(x, y+r+10, text=name,
                           fill='#f5efe0', font=('Courier', 12, 'bold'))

    canvas.create_text(W//2, H-12,
                       text="CAVITE PROVINCE  •  ROUTE MAP  •  NOT TO SCALE",
                       fill=C_CONTOUR, font=('Courier', 12, 'italic'))
    canvas.create_rectangle(2, 2, W-2, H-2, outline=C_BORDER, width=2)
    canvas.create_rectangle(6, 6, W-6, H-6, outline=C_CONTOUR, width=1)


# ─────────────────────────────────────────
#  CUSTOM WIDGETS
# ─────────────────────────────────────────

class InkCombobox(tk.Frame):
    def __init__(self, parent, textvariable, values, width=12, **kw):
        super().__init__(parent, bg=C_PANEL, **kw)
        self.var    = textvariable
        self.values = values
        self._open  = False

        self.btn = tk.Label(self, textvariable=textvariable,
                            bg=C_PANEL, fg=C_INK, font=FONT_LABEL,
                            relief='flat', bd=0, padx=6, pady=3,
                            width=width, anchor='w', cursor='hand2')
        self.btn.pack(side='left')
        tk.Label(self, text='▾', bg=C_PANEL, fg=C_INK_LIGHT,
                 font=('Courier', 12)).pack(side='left')
        tk.Frame(self, bg=C_BORDER, height=1).pack(fill='x', side='bottom')
        self.btn.bind('<Button-1>', self._toggle)

    def _toggle(self, event=None):
        if self._open:
            self._close(); return
        self._open  = True
        self._popup = tk.Toplevel(self)
        self._popup.overrideredirect(True)
        self._popup.config(bg=C_BORDER)

        # Build items first so we can measure the popup height
        for v in self.values:
            lbl = tk.Label(self._popup, text=v, bg=C_PANEL, fg=C_INK,
                           font=FONT_LABEL, anchor='w', padx=8, pady=3,
                           cursor='hand2', relief='flat')
            lbl.pack(fill='x', pady=1)
            lbl.bind('<Button-1>', lambda e, val=v: self._select(val))
            lbl.bind('<Enter>',   lambda e, w=lbl: w.config(bg=C_INK, fg=C_BTN_FG))
            lbl.bind('<Leave>',   lambda e, w=lbl: w.config(bg=C_PANEL, fg=C_INK))

        self._popup.update_idletasks()
        popup_h = self._popup.winfo_reqheight()

        x        = self.winfo_rootx()
        btn_y    = self.winfo_rooty()
        btn_h    = self.winfo_height()
        screen_h = self.winfo_screenheight()

        # Flip upward if not enough room below
        if btn_y + btn_h + popup_h > screen_h - 10:
            y = btn_y - popup_h - 1
        else:
            y = btn_y + btn_h + 1

        self._popup.geometry(f"+{x}+{y}")
        self._popup.bind('<FocusOut>', lambda e: self._close())
        self._popup.focus_set()

    def _select(self, val):
        self.var.set(val); self._close()

    def _close(self):
        self._open = False
        if hasattr(self, '_popup') and self._popup.winfo_exists():
            self._popup.destroy()

    def get(self): return self.var.get()


class InkButton(tk.Canvas):
    def __init__(self, parent, text, command, accent=False, **kw):
        w = kw.pop('width', 140)
        h = kw.pop('height', 30)
        super().__init__(parent, width=w, height=h,
                         bg=C_PANEL, highlightthickness=0, **kw)
        self.command = command
        self.text    = text
        self.accent  = accent
        self.w, self.h = w, h
        self._draw()
        self.bind('<Button-1>',        self._press)
        self.bind('<ButtonRelease-1>', self._release)
        self.bind('<Enter>',           self._hover)
        self.bind('<Leave>',           lambda e: self._draw())

    def _draw(self, pressed=False, hover=False):
        self.delete('all')
        off  = 2 if pressed else 4
        fill = C_ACCENT if self.accent else C_INK
        self.create_rectangle(off, off, self.w-1, self.h-1,
                              fill='#9b7d5a', outline='')
        self.create_rectangle(0, 0, self.w-off, self.h-off,
                              fill=fill, outline=C_BORDER, width=1)
        self.create_text((self.w-off)//2, (self.h-off)//2,
                         text=self.text, fill=C_BTN_FG,
                         font=('Courier', 12, 'bold'))

    def _press(self, e):   self._draw(pressed=True)
    def _hover(self, e):   self._draw(hover=True)
    def _release(self, e):
        self._draw(); self.command()


# ─────────────────────────────────────────
#  GUI  — fully responsive
# ─────────────────────────────────────────

def build_gui(graph, nodes):
    root = tk.Tk()
    root.title("Cavite Route Finder")
    root.configure(bg=C_BG)

    # Start maximised; works on Windows, Linux, and macOS
    try:
        root.state('zoomed')           # Windows / some Linux WMs
    except tk.TclError:
        root.attributes('-zoomed', True)  # other Linux WMs

    state = {'path': None}

    # ── HEADER (top) ──────────────────────────────────────
    hdr = tk.Frame(root, bg=C_INK)
    hdr.pack(fill='x', side='top')
    tk.Frame(hdr, bg=C_ACCENT, height=5).pack(fill='x')
    hdr_inner = tk.Frame(hdr, bg=C_INK, pady=8, padx=16)
    hdr_inner.pack(fill='x')
    t = tk.Frame(hdr_inner, bg=C_INK)
    t.pack(side='left')
    tk.Label(t, text="CAVITE ROUTE FINDER",
             font=('Georgia', 18, 'bold'), fg=C_BG, bg=C_INK).pack(anchor='w')
    tk.Label(t, text="Shortest Path Navigation System  ·  Dijkstra's Algorithm",
             font=('Courier', 12, 'italic'), fg='#9b7d5a', bg=C_INK).pack(anchor='w')
    badge = tk.Frame(hdr_inner, bg=C_ACCENT, padx=10, pady=5)
    badge.pack(side='right')
    tk.Label(badge, text="PHL · CAVITE\nPROVINCE",
             font=('Courier', 12, 'bold'), fg=C_BG, bg=C_ACCENT,
             justify='center').pack()
    tk.Frame(hdr, bg=C_BORDER, height=2).pack(fill='x')

    # ── FOOTER (bottom) ───────────────────────────────────
    footer = tk.Frame(root, bg=C_INK)
    footer.pack(fill='x', side='bottom')
    tk.Frame(footer, bg=C_ACCENT, height=2).pack(fill='x')
    tk.Label(footer,
             text="CAVITE  •  PHILIPPINES   ·   Data: Book1.csv   ·   Algorithm: Dijkstra",
             font=('Courier', 12), fg='#6b4c2a', bg=C_INK).pack(pady=3)

    # ── RESULT STRIP (above footer) ───────────────────────
    res_outer = tk.Frame(root, bg=C_BORDER, padx=1, pady=1)
    res_outer.pack(fill='x', side='bottom')
    res_inner = tk.Frame(res_outer, bg=C_PANEL)
    res_inner.pack(fill='x')
    tk.Frame(res_inner, bg=C_ACCENT, width=6).pack(side='left', fill='y')
    result_var = tk.StringVar(
        value="  Select origin and destination, then press  FIND ROUTE")
    result_lbl = tk.Label(res_inner, textvariable=result_var,
                          bg=C_PANEL, fg=C_INK_LIGHT,
                          font=FONT_RESULT, justify='left', anchor='w',
                          pady=10, padx=10)
    result_lbl.pack(fill='x')

    # ── CONTROLS BAR (above result) ───────────────────────
    ctrl_outer = tk.Frame(root, bg=C_BORDER, padx=1, pady=1)
    ctrl_outer.pack(fill='x', side='bottom')
    ctrl = tk.Frame(ctrl_outer, bg=C_PANEL, padx=14, pady=10)
    ctrl.pack(fill='x')

    def ink_label(text):
        return tk.Label(ctrl, text=text, bg=C_PANEL,
                        fg=C_INK_LIGHT, font=('Courier', 12, 'italic'))

    frm_var = tk.StringVar(value=nodes[0])
    to_var  = tk.StringVar(value=nodes[-1])
    opt_var = tk.StringVar(value='distance')

    ink_label("FROM").grid(row=0, column=0, sticky='w', padx=(0, 2))
    InkCombobox(ctrl, frm_var, nodes, width=11).grid(row=1, column=0, padx=(0, 16))

    ink_label("TO").grid(row=0, column=1, sticky='w', padx=(0, 2))
    InkCombobox(ctrl, to_var, nodes, width=11).grid(row=1, column=1, padx=(0, 16))

    ink_label("OPTIMISE BY").grid(row=0, column=2, sticky='w', padx=(0, 2))
    InkCombobox(ctrl, opt_var, ['distance', 'time', 'fuel'],
                width=11).grid(row=1, column=2, padx=(0, 24))

    btn_frame = tk.Frame(ctrl, bg=C_PANEL)
    btn_frame.grid(row=0, column=3, rowspan=2)

    # ── CANVAS (fills all remaining space) ────────────────
    canvas_border = tk.Frame(root, bg=C_BORDER, padx=1, pady=1)
    canvas_border.pack(fill='both', expand=True, padx=10, pady=(8, 4))
    canvas = tk.Canvas(canvas_border, bg=C_MAP_BG, highlightthickness=0)
    canvas.pack(fill='both', expand=True)

    # Redraw on every resize
    def on_resize(event):
        draw_map(canvas, graph, nodes, highlight_path=state['path'])

    canvas.bind('<Configure>', on_resize)

    # ── CALLBACKS ─────────────────────────────────────────
    def find_path():
        start = frm_var.get()
        end   = to_var.get()
        key   = opt_var.get()

        if start == end:
            messagebox.showwarning("Same Node",
                                   "Please select different From / To nodes.")
            return

        cost, path, totals = dijkstra(graph, start, end, key)

        if not path:
            state['path'] = None
            result_var.set(f"  ✗  No path found from {start} to {end}.")
            result_lbl.config(fg=C_ACCENT)
            draw_map(canvas, graph, nodes)
            return

        state['path'] = path
        path_str = ' → '.join(path)
        result_var.set(
            f"  ROUTE  {path_str}\n"
            f"  ────────────────────────────────────────────\n"
            f"  {totals['distance']:.1f} km   │   "
            f"{totals['time']:.0f} mins   │   "
            f"{totals['fuel']:.2f} Liters   │   "
            f"Optimised by: {key.upper()}"
        )
        result_lbl.config(fg=C_INK)
        draw_map(canvas, graph, nodes, highlight_path=path)

    def reset():
        state['path'] = None
        result_var.set("  Select origin and destination, then press  FIND ROUTE")
        result_lbl.config(fg=C_INK_LIGHT)
        draw_map(canvas, graph, nodes)

    InkButton(btn_frame, "▶  FIND ROUTE", find_path,
              accent=True, width=140, height=30).pack(pady=(0, 5))
    InkButton(btn_frame, "↺  RESET", reset,
              width=140, height=30).pack()

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

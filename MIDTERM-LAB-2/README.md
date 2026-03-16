# Cavite Route Finder
### Node Map & Shortest Path Calculator

---

## Overview

This program reads a CSV file containing route data between cities in Cavite and provides two main features:
1. A **visual node map** showing all cities and their connections
2. A **shortest path calculator** that finds the optimal route between any two cities based on distance, time, or fuel consumption

---

## Approach

### Part 1: Node Map

The node map was built using Python's built-in `tkinter` library to render a canvas-based graph. Each city is represented as a labeled circular node, and each route from the CSV is drawn as a line (edge) between two nodes.

**Steps taken:**
- Loaded all unique cities from the `From Node` and `To Node` columns to build the node list
- Built an **adjacency list** to represent connections between nodes, storing distance, time, and fuel as edge attributes
- Manually assigned screen coordinates to each node based on their approximate geographic positions in Cavite to keep the map readable
- Edge labels display the distance in km for quick reference
- When a shortest path is found, the relevant nodes and edges are **highlighted in amber/gold** on the map for visual clarity

### Part 2: Shortest Path

The shortest path is calculated using **Dijkstra's Algorithm**, a classical greedy algorithm that finds the least-cost path from a source node to a destination node in a weighted graph.

The user can choose to optimise by:
- **Distance** (km)
- **Time** (mins)
- **Fuel** (Liters)

Regardless of which weight is used for optimisation, the output always displays **all three totals** (distance, time, and fuel) for the resulting path, as the algorithm accumulates each attribute along every hop.

---

## Algorithm: Dijkstra's

Dijkstra's algorithm was chosen because:
- The graph has **non-negative edge weights**, which is a requirement for Dijkstra's to work correctly
- It guarantees the **globally optimal** (shortest/cheapest) path
- It is efficient for small-to-medium graphs like this one

**How it works:**
1. Start at the source node with cost 0
2. Use a **min-heap priority queue** to always expand the lowest-cost node next
3. For each neighbor, calculate the new cumulative cost
4. If the neighbor hasn't been visited, push it onto the queue
5. Stop when the destination node is reached
6. Trace back the path and sum up all three attributes (distance, time, fuel)

---

## Files

| File | Description |
|---|---|
| `route_finder.py` | Main Python program |
| `Book1.csv` | Route data input file |
| `README.md` | This report |

---

## How to Run

1. Place `route_finder.py` and `Book1.csv` in the **same folder**
2. Run the program:
   ```
   python route_finder.py
   ```
3. Select a **From** and **To** city from the dropdowns
4. Choose an optimisation criteria: `distance`, `time`, or `fuel`
5. Click **Find Shortest Path**
6. The result and highlighted path will appear on the map

> No additional libraries are required. The program uses only Python's standard library: `tkinter`, `csv`, and `heapq`.

---

## Challenges Faced

### 1. Node Overlap on the Canvas
The biggest visual challenge was positioning the nodes so that edges between non-adjacent cities didn't visually pass through unrelated nodes. For example, the edge between NOVELETA and KAWIT kept appearing to pass through IMUS because they were aligned on the same diagonal line on the canvas. This was resolved by carefully adjusting each node's coordinates so no straight edge would intersect an unrelated node.

### 2. Undirected vs. Directed Graph
The CSV data lists routes in one direction (e.g., `SILANG, BACOOR`), but some city pairs didn't have a reverse entry. The program treats the graph as **undirected**, meaning both directions share the same edge attributes. This was a design decision made to ensure all cities remain reachable from any starting point.

### 3. Displaying All Three Totals
The optimisation only uses one weight at a time (e.g., fuel), but the output must show distance, time, and fuel together. This required a separate accumulation pass after Dijkstra's finishes — walking the final path and summing all three attributes independently.

### 4. Edge Label Clutter
With multiple edges close together, distance labels overlapped and made the map hard to read. This was partially addressed by offsetting labels slightly above the midpoint of each edge.

---

## Algorithm Complexity

| Metric | Value |
|---|---|
| Nodes (V) | 8 |
| Edges (E) | 14 |
| Time Complexity | O((V + E) log V) |
| Space Complexity | O(V + E) |

For this graph size, performance is near-instant.


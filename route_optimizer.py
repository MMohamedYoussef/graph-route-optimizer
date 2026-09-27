"""Reproducible road-network routing demo with Dijkstra and A*."""

from __future__ import annotations

import argparse
import csv
import heapq
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

import networkx as nx

MAX_SPEED_KMH = 50.0


@dataclass(frozen=True)
class RouteResult:
    path: tuple[str, ...]
    minutes: float
    distance_km: float
    expanded_nodes: int


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius_km = 6371.0088
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = phi2 - phi1
    dlambda = math.radians(lon2 - lon1)
    value = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * radius_km * math.asin(math.sqrt(value))


def build_graph(seed: int = 42, rows: int = 6, cols: int = 6) -> nx.Graph:
    """Build a connected synthetic street grid near Cairo (no real road data)."""
    if rows < 2 or cols < 2:
        raise ValueError("Grid must contain at least two rows and two columns")
    rng = random.Random(seed)
    graph = nx.Graph()
    for row in range(rows):
        for col in range(cols):
            node = f"R{row}C{col}"
            graph.add_node(
                node,
                lat=30.040 + row * 0.011 + rng.uniform(-0.0012, 0.0012),
                lon=31.190 + col * 0.011 + rng.uniform(-0.0012, 0.0012),
            )
    for row in range(rows):
        for col in range(cols):
            source = f"R{row}C{col}"
            for next_row, next_col in ((row + 1, col), (row, col + 1)):
                if next_row >= rows or next_col >= cols:
                    continue
                target = f"R{next_row}C{next_col}"
                left, right = graph.nodes[source], graph.nodes[target]
                distance = haversine_km(left["lat"], left["lon"], right["lat"], right["lon"])
                speed = rng.choice((25.0, 35.0, 50.0))
                graph.add_edge(
                    source,
                    target,
                    distance_km=distance,
                    speed_kmh=speed,
                    minutes=distance / speed * 60,
                )
    return graph


def heuristic_minutes(graph: nx.Graph, node: str, goal: str) -> float:
    """Straight-line time at the network's maximum speed: an admissible bound."""
    left, right = graph.nodes[node], graph.nodes[goal]
    return haversine_km(left["lat"], left["lon"], right["lat"], right["lon"]) / MAX_SPEED_KMH * 60


def shortest_route(graph: nx.Graph, start: str, goal: str, algorithm: str) -> RouteResult:
    if algorithm not in ("dijkstra", "astar"):
        raise ValueError("Algorithm must be dijkstra or astar")
    if start not in graph or goal not in graph:
        raise ValueError("Start and goal must exist in the road network")
    if start == goal:
        return RouteResult((start,), 0.0, 0.0, 0)

    def estimate(node: str) -> float:
        return heuristic_minutes(graph, node, goal) if algorithm == "astar" else 0.0

    frontier: list[tuple[float, float, str]] = [(estimate(start), 0.0, start)]
    best_cost = {start: 0.0}
    previous: dict[str, str] = {}
    expanded = 0
    while frontier:
        _, elapsed, node = heapq.heappop(frontier)
        if elapsed > best_cost[node] + 1e-12:
            continue
        expanded += 1
        if node == goal:
            path = [goal]
            while path[-1] != start:
                path.append(previous[path[-1]])
            path.reverse()
            distance = sum(graph[u][v]["distance_km"] for u, v in zip(path, path[1:]))
            return RouteResult(tuple(path), elapsed, distance, expanded)
        for neighbor, attributes in graph[node].items():
            candidate = elapsed + attributes["minutes"]
            if candidate + 1e-12 < best_cost.get(neighbor, math.inf):
                best_cost[neighbor] = candidate
                previous[neighbor] = node
                heapq.heappush(frontier, (candidate + estimate(neighbor), candidate, neighbor))
    raise ValueError(f"No route from {start} to {goal}")


def render_png(graph: nx.Graph, route: RouteResult, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    navy, slate, cyan, amber = "#0d1726", "#26384f", "#55d5df", "#ffbe68"
    fig, ax = plt.subplots(figsize=(12, 8), facecolor=navy)
    ax.set_facecolor(navy)
    for u, v in graph.edges:
        left, right = graph.nodes[u], graph.nodes[v]
        ax.plot([left["lon"], right["lon"]], [left["lat"], right["lat"]], color=slate, lw=2.2, zorder=1)
    for u, v in zip(route.path, route.path[1:]):
        left, right = graph.nodes[u], graph.nodes[v]
        ax.plot([left["lon"], right["lon"]], [left["lat"], right["lat"]], color=cyan, lw=5, zorder=3)
    ax.scatter(
        [graph.nodes[n]["lon"] for n in graph],
        [graph.nodes[n]["lat"] for n in graph],
        s=28, color="#a6b8ca", zorder=4,
    )
    for node, color, label in ((route.path[0], cyan, "START"), (route.path[-1], amber, "FINISH")):
        point = graph.nodes[node]
        ax.scatter(point["lon"], point["lat"], s=180, color=color, edgecolor=navy, linewidth=2, zorder=5)
        ax.annotate(
            f"  {label} · {node}", (point["lon"], point["lat"]),
            color=color, fontsize=10, fontweight="bold", xytext=(8, 12), textcoords="offset points",
        )
    ax.set_title("FASTEST PATH / SYNTHETIC CAIRO GRID", loc="left", color="white", fontsize=18, weight="bold", pad=18)
    ax.text(
        0.0, 1.01,
        f"{route.minutes:.1f} min travel time   •   {route.distance_km:.1f} km   •   {len(route.path) - 1} road segments",
        transform=ax.transAxes, color="#a6b8ca", fontsize=11,
    )
    ax.set_xlabel("Longitude", color="#a6b8ca")
    ax.set_ylabel("Latitude", color="#a6b8ca")
    ax.tick_params(colors="#8195aa")
    for spine in ax.spines.values():
        spine.set_color(slate)
    ax.grid(alpha=0.08, color="white")
    fig.tight_layout(pad=2)
    fig.savefig(path, dpi=170, facecolor=navy)
    plt.close(fig)


def render_html(graph: nx.Graph, route: RouteResult, path: Path) -> None:
    import plotly.graph_objects as go

    figure = go.Figure()
    for u, v in graph.edges:
        left, right = graph.nodes[u], graph.nodes[v]
        figure.add_trace(go.Scatter(
            x=[left["lon"], right["lon"]], y=[left["lat"], right["lat"]],
            mode="lines", line=dict(color="#30435a", width=2),
            hoverinfo="skip", showlegend=False,
        ))
    for u, v in zip(route.path, route.path[1:]):
        left, right = graph.nodes[u], graph.nodes[v]
        figure.add_trace(go.Scatter(
            x=[left["lon"], right["lon"]], y=[left["lat"], right["lat"]],
            mode="lines", line=dict(color="#55d5df", width=6),
            hovertemplate=f"{u} → {v}<br>{graph[u][v]['minutes']:.2f} min<extra></extra>",
            showlegend=False,
        ))
    figure.add_trace(go.Scatter(
        x=[graph.nodes[n]["lon"] for n in graph],
        y=[graph.nodes[n]["lat"] for n in graph],
        mode="markers+text", text=list(graph.nodes), textposition="top center",
        marker=dict(color="#a6b8ca", size=8),
        textfont=dict(color="#a6b8ca", size=9),
        hovertemplate="%{text}<extra></extra>", showlegend=False,
    ))
    figure.update_layout(
        title=f"Fastest route · {route.minutes:.1f} min · {route.distance_km:.1f} km",
        template="plotly_dark", paper_bgcolor="#0d1726", plot_bgcolor="#0d1726",
        font=dict(color="#dbe8f1"), xaxis_title="Longitude", yaxis_title="Latitude",
        width=1100, height=740, margin=dict(l=70, r=35, t=85, b=60),
    )
    figure.write_html(path, include_plotlyjs=True, full_html=True)


def write_demo(output_dir: Path, seed: int = 42, start: str = "R0C0", goal: str = "R5C5") -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    graph = build_graph(seed=seed)
    with (output_dir / "nodes.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("node_id", "latitude", "longitude"))
        for node in sorted(graph):
            writer.writerow((node, f"{graph.nodes[node]['lat']:.6f}", f"{graph.nodes[node]['lon']:.6f}"))
    with (output_dir / "roads.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("from_node", "to_node", "distance_km", "speed_kmh", "travel_minutes"))
        for u, v in sorted(tuple(sorted(edge)) for edge in graph.edges):
            edge = graph[u][v]
            writer.writerow((u, v, f"{edge['distance_km']:.6f}", int(edge["speed_kmh"]), f"{edge['minutes']:.6f}"))
    results = {name: shortest_route(graph, start, goal, name) for name in ("dijkstra", "astar")}
    if not math.isclose(results["dijkstra"].minutes, results["astar"].minutes, abs_tol=1e-9):
        raise AssertionError("A* and Dijkstra did not agree on optimal travel time")
    render_png(graph, results["astar"], output_dir / "route-map.png")
    render_html(graph, results["astar"], output_dir / "route-map.html")
    comparison = {
        name: {
            "path": list(result.path),
            "minutes": round(result.minutes, 6),
            "distance_km": round(result.distance_km, 6),
            "expanded_nodes": result.expanded_nodes,
        }
        for name, result in results.items()
    }
    (output_dir / "comparison.json").write_text(json.dumps(comparison, indent=2) + "\n", encoding="utf-8")
    return comparison


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--start", default="R0C0")
    parser.add_argument("--goal", default="R5C5")
    args = parser.parse_args()
    comparison = write_demo(args.output_dir, args.seed, args.start, args.goal)
    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()

import json

import networkx as nx
import pytest

from route_optimizer import build_graph, heuristic_minutes, shortest_route, write_demo


def triangle():
    graph = nx.Graph()
    graph.add_node("A", lat=30.0, lon=31.00)
    graph.add_node("B", lat=30.0, lon=31.01)
    graph.add_node("C", lat=30.0, lon=31.02)
    graph.add_edge("A", "B", minutes=5.0, distance_km=1.0)
    graph.add_edge("B", "C", minutes=5.0, distance_km=1.0)
    graph.add_edge("A", "C", minutes=12.0, distance_km=2.0)
    return graph


@pytest.mark.parametrize("algorithm", ["dijkstra", "astar"])
def test_finds_fastest_route_instead_of_direct_road(algorithm):
    result = shortest_route(triangle(), "A", "C", algorithm)
    assert result.path == ("A", "B", "C")
    assert result.minutes == pytest.approx(10.0)
    assert result.distance_km == pytest.approx(2.0)


def test_start_equal_goal_has_zero_cost():
    result = shortest_route(triangle(), "A", "A", "astar")
    assert result.path == ("A",)
    assert result.minutes == 0


def test_disconnected_route_has_clear_error():
    graph = triangle()
    graph.add_node("D", lat=30.1, lon=31.1)
    with pytest.raises(ValueError, match="No route"):
        shortest_route(graph, "A", "D", "dijkstra")


def test_generated_network_heuristic_is_admissible_and_routes_agree():
    graph = build_graph(seed=17)
    goal = "R5C5"
    for node in graph:
        true_remaining = nx.shortest_path_length(graph, node, goal, weight="minutes")
        assert heuristic_minutes(graph, node, goal) <= true_remaining + 1e-9
    dijkstra = shortest_route(graph, "R0C0", goal, "dijkstra")
    astar = shortest_route(graph, "R0C0", goal, "astar")
    assert dijkstra.minutes == pytest.approx(astar.minutes)
    assert astar.minutes == pytest.approx(
        nx.shortest_path_length(graph, "R0C0", goal, weight="minutes")
    )


def test_demo_writes_reproducible_data_and_visual_artifacts(tmp_path):
    write_demo(tmp_path, seed=9)
    expected = [
        "nodes.csv", "roads.csv", "route-map.png", "route-map.html", "comparison.json"
    ]
    assert all((tmp_path / name).is_file() for name in expected)
    comparison = json.loads((tmp_path / "comparison.json").read_text(encoding="utf-8"))
    assert comparison["dijkstra"]["minutes"] == pytest.approx(
        comparison["astar"]["minutes"]
    )
    first = (tmp_path / "roads.csv").read_bytes()
    write_demo(tmp_path, seed=9)
    assert (tmp_path / "roads.csv").read_bytes() == first

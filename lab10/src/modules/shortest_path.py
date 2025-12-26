import heapq
from graph_representation import AdjacencyListGraph
from graph_traversal import restore_path


def dijkstra(graph, start):
    """Алгоритм Дейкстры для неотрицательных весов.
    Время: O((V + E) log V), память: O(V).
    """
    dist = {start: 0.0}
    parent = {start: None}
    heap = [(0.0, start)]

    while heap:
        cur_dist, v = heapq.heappop(heap)
        if cur_dist > dist.get(v, float("inf")):
            continue
        for to, w in graph.neighbors_with_weights(v):
            if w < 0:
                raise ValueError("Отрицательные веса не поддерживаются")
            new_dist = cur_dist + w
            if new_dist < dist.get(to, float("inf")):
                dist[to] = new_dist
                parent[to] = v
                heapq.heappush(heap, (new_dist, to))

    return dist, parent


if __name__ == "__main__":
    g = AdjacencyListGraph(directed=True, weighted=True)
    edges = [
        ("A", "B", 4),
        ("A", "C", 2),
        ("B", "C", 1),
        ("B", "D", 5),
        ("C", "D", 8),
        ("C", "E", 10),
        ("D", "E", 2),
    ]
    for u, v, w in edges:
        g.add_edge(u, v, w)

    dist, parent = dijkstra(g, "A")
    print("Расстояния:", dist)
    print("Путь A->E:", restore_path(parent, "A", "E"))

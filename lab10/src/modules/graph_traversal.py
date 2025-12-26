from collections import deque
from graph_representation import AdjacencyListGraph


def bfs(graph, start):
    """Поиск в ширину.

    Возвращает словари расстояний и родителей.
    Время: O(V + E), память: O(V).
    """
    # Словарь расстояний от стартовой вершины до всех остальных вершин
    dist = {start: 0}
    # Словарь родительных связей для последующего восстановления пути
    parent = {start: None}
    q = deque([start])

    while q:
        # Извлекаем первую вершину из очереди
        v = q.popleft()
        for to in graph.neighbors(v):
            if to not in dist:
                dist[to] = dist[v] + 1
                parent[to] = v
                q.append(to)

    return dist, parent


def restore_path(parent, start, target):
    """Восстановление пути по родителям, O(L) по длине пути."""
    if target not in parent:
        return None
    # Восстановление пути с конца
    path = []
    v = target
    while v is not None:
        path.append(v)
        v = parent[v]
    # Инвертируем список, чтобы вернуть правильный порядок
    path.reverse()
    if path and path[0] == start:
        return path
    return None


def _dfs_recursive(graph, v, visited, order):
    visited.add(v)  # Множество посещенных вершин
    order.append(v)
    for to in graph.neighbors(v):
        if to not in visited:
            _dfs_recursive(graph, to, visited, order)


def dfs_full_recursive(graph):
    """Полный DFS для всех компонент.
    Время: O(V + E), память: O(V).
    """
    visited = set()  # Посещённые вершины
    order = []
    for v in list(graph.adj.keys()):
        if v not in visited:
            _dfs_recursive(graph, v, visited, order)
    return order


def dfs_iterative(graph, start):
    """Итеративный DFS со стеком.
    Время: O(V + E), память: O(V).
    """
    visited = set()
    stack = [start]
    order = []

    while stack:
        v = stack.pop()
        if v in visited:
            continue
        visited.add(v)
        order.append(v)
        neighbors = list(graph.neighbors(v))
        neighbors.reverse()
        for to in neighbors:
            if to not in visited:
                stack.append(to)

    return order


def connected_components(graph):
    """Компоненты связности в неориентированном графе.
    Время: O(V + E), память: O(V).
    """
    visited = set()
    components = []  # Компоненты связности

    for v in list(graph.adj.keys()):
        if v in visited:
            continue
        # Создаём новый компонент и добавляем туда вершину
        comp = []
        stack = [v]
        visited.add(v)
        while stack:
            cur = stack.pop()
            comp.append(cur)
            for to in graph.neighbors(cur):
                if to not in visited:
                    visited.add(to)
                    stack.append(to)
        components.append(comp)

    return components


def topological_sort(graph):
    """Топологическая сортировка DAG.
    Время: O(V + E), память: O(V).
    """
    visited = set()
    temp_mark = set()
    order = []

    def visit(v):
        if v in temp_mark:
            raise ValueError("Граф содержит цикл")
        if v not in visited:
            temp_mark.add(v)
            # Рекурсивно посещаем всех потомков текущей вершины
            for to, _ in graph.neighbors_with_weights(v):
                visit(to)
            temp_mark.remove(v)
            visited.add(v)
            order.append(v)

    # Топологическая сортировка для всех вершин
    for v in list(graph.adj.keys()):
        if v not in visited:
            visit(v)

    order.reverse()
    return order


if __name__ == "__main__":
    g = AdjacencyListGraph(directed=False)
    edges = [("A", "B"), ("A", "C"), ("B", "D"), ("C", "D"), ("D", "E")]
    for u, v in edges:
        g.add_edge(u, v)

    d, p = bfs(g, "A")
    print("BFS от A:", d)
    print("Путь A->E:", restore_path(p, "A", "E"))
    print("DFS полный:", dfs_full_recursive(g))
    print("Компоненты:", connected_components(g))

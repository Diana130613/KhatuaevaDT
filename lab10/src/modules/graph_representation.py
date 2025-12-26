from collections import defaultdict


class AdjacencyMatrixGraph:
    """Граф на матрице смежности.

    Память: O(V^2).
    Проверка наличия ребра: O(1).
    Обход соседей вершины: O(V).
    """

    def __init__(self, directed=False, weighted=False):
        self.directed = directed
        self.weighted = weighted
        self.vertices = []  # Список вершин
        self.index = {}  # Словарь: вершина -> индекс в матрице
        self.matrix = []  # Матрица смежности

    def _ensure_size(self):
        """Гарантирует, что матрица имеет правильный размер (n x n).

        Вызывается после добавления новой вершины.
        Временная сложность: O(V^2) в худшем случае.
        """
        n = len(self.vertices)
        while len(self.matrix) < n:
            self.matrix.append([0] * n)
        for row in self.matrix:
            while len(row) < n:
                row.append(0)

    def add_vertex(self, v):
        """Добавление вершины,
        амортизированно O(V^2) из-за изменения размера.
        """
        if v in self.index:
            return  # Вершина уже существует
        self.index[v] = len(self.vertices)
        self.vertices.append(v)
        self._ensure_size()

    def remove_vertex(self, v):
        """Удаление вершины,
        O(V^2) на перестройку матрицы."""
        if v not in self.index:
            return  # Вершины нет в графе
        idx = self.index.pop(v)
        self.vertices.pop(idx)
        self.matrix.pop(idx)
        for row in self.matrix:
            row.pop(idx)
        self.index = {vtx: i for i, vtx in enumerate(self.vertices)}

    def add_edge(self, u, v, w=1):
        """Добавление ребра между вершинами.

        Args:
            u: начальная вершина
            v: конечная вершина
            w: вес ребра (по умолчанию 1)

        Временная сложность: O(1) после доступа к индексам вершин.
        """
        for x in (u, v):
            if x not in self.index:
                self.add_vertex(x)
        i = self.index[u]
        j = self.index[v]
        self.matrix[i][j] = w
        if not self.directed:
            self.matrix[j][i] = w

    def remove_edge(self, u, v):
        """Удаление ребра между вершинами.

        Args:
            u: начальная вершина
            v: конечная вершина

        Временная сложность: O(1).
        """
        if u not in self.index or v not in self.index:
            return
        i = self.index[u]
        j = self.index[v]
        self.matrix[i][j] = 0
        if not self.directed:
            self.matrix[j][i] = 0

    def neighbors(self, v):
        """Соседи вершины, обход строки матрицы, O(V)."""
        if v not in self.index:
            return []
        i = self.index[v]
        res = []
        for j, w in enumerate(self.matrix[i]):
            if w != 0:
                res.append(self.vertices[j])
        return res


class AdjacencyListGraph:
    """Граф на списке смежности.

    Память: O(V + E).
    Перебор соседей: O(deg(v)).
    Проверка наличия ребра: O(deg(v)).
    """

    def __init__(self, directed=False, weighted=False):
        self.directed = directed
        self.weighted = weighted
        self.adj = defaultdict(list)

    def add_vertex(self, v):
        """Добавление вершины, амортизированно O(1)."""
        _ = self.adj[v]

    def remove_vertex(self, v):
        """Удаление вершины, O(V + E)."""
        if v not in self.adj:
            return
        self.adj.pop(v)
        for u in list(self.adj.keys()):
            self.adj[u] = [(to, w) for (to, w) in self.adj[u] if to != v]

    def add_edge(self, u, v, w=1):
        """Добавление ребра, амортизированно O(1)."""
        self.adj[u].append((v, w))
        if not self.directed:
            self.adj[v].append((u, w))

    def remove_edge(self, u, v):
        """Удаление ребра, O(deg(u))."""
        self.adj[u] = [(to, w) for (to, w) in self.adj[u] if to != v]
        if not self.directed:
            self.adj[v] = [(to, w) for (to, w) in self.adj[v] if to != u]

    def neighbors(self, v):
        """Список соседей вершины, O(deg(v))."""
        return [to for (to, _) in self.adj.get(v, [])]

    def neighbors_with_weights(self, v):
        """Соседи вершины с весами, O(V)."""
        if v not in self.index:
            return []
        i = self.index[v]
        res = []
        for j, w in enumerate(self.matrix[i]):
            if w != 0:
                res.append((self.vertices[j], w))
        return res


if __name__ == "__main__":
    g = AdjacencyListGraph(directed=False)
    g.add_edge("A", "B")
    g.add_edge("A", "C")
    print("Соседи A:", g.neighbors("A"))

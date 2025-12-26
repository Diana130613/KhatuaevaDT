import time
import matplotlib.pyplot as plt
from graph_representation import AdjacencyMatrixGraph, AdjacencyListGraph
from graph_traversal import bfs


def measure_matrix_vertices(size: int):
    """Измерение времени добавления и удаления вершин для матрицы."""
    matrix = AdjacencyMatrixGraph()

    # Добавление вершин
    start = time.time()
    for i in range(size):
        matrix.add_vertex(i)
    add_time = (time.time() - start) * 1000  # мс

    # Удаление вершин
    start = time.time()
    for i in range(size-1, -1, -1):
        try:
            matrix.remove_vertex(i)
        except:
            pass
    remove_time = (time.time() - start) * 1000

    return add_time, remove_time


def measure_list_vertices(size: int):
    """Измерение времени добавления и удаления вершин для списка."""
    lst = AdjacencyListGraph()

    # Добавление вершин
    start = time.time()
    for i in range(size):
        lst.add_vertex(i)
    add_time = (time.time() - start) * 1000

    # Удаление вершин
    start = time.time()
    for i in range(size):
        try:
            lst.remove_vertex(i)
        except:
            pass
    remove_time = (time.time() - start) * 1000

    return add_time, remove_time


def measure_edges_performance(size: int, edge_prob: float = 0.3):
    """Измерение времени добавления рёбер."""
    import random

    # Матрица
    matrix = AdjacencyMatrixGraph()
    for i in range(size):
        matrix.add_vertex(i)

    start = time.time()
    for i in range(size):
        for j in range(i+1, size):
            if random.random() < edge_prob:
                matrix.add_edge(i, j)
    matrix_time = (time.time() - start) * 1000

    # Список
    lst = AdjacencyListGraph()
    for i in range(size):
        lst.add_vertex(i)

    random.seed(42)  # Для одинаковых рёбер
    start = time.time()
    for i in range(size):
        for j in range(i+1, size):
            if random.random() < edge_prob:
                lst.add_edge(i, j)
    list_time = (time.time() - start) * 1000

    return matrix_time, list_time


def measure_bfs_performance(sizes: list):
    """Измерение времени BFS."""
    import random

    matrix_times = []
    list_times = []

    for size in sizes:
        # Создаём графы
        matrix = AdjacencyMatrixGraph()
        lst = AdjacencyListGraph()

        for i in range(size):
            matrix.add_vertex(i)
            lst.add_vertex(i)

        # Добавляем рёбра
        random.seed(42)
        for i in range(size):
            for j in range(i+1, min(i+5, size)):
                if random.random() < 0.5:
                    matrix.add_edge(i, j)
                    lst.add_edge(i, j)

        # BFS на матрице
        start = time.time()
        bfs(matrix, 0)
        matrix_times.append((time.time() - start) * 1000)

        # BFS на списке
        start = time.time()
        bfs(lst, 0)
        list_times.append((time.time() - start) * 1000)

    return matrix_times, list_times


def run_analysis():
    """Запуск анализа производительности."""
    sizes = [100, 200, 500, 1000, 2000]

    print("=== Сравнение добавления вершин ===")
    for size in sizes:
        m_add, _ = measure_matrix_vertices(size)
        l_add, _ = measure_list_vertices(size)
        print(f"{size} вершин: матрица={m_add:.2f}мс, список={l_add:.2f}мс")

    print("\n=== Сравнение добавления рёбер ===")
    for size in [50, 100, 200]:
        m_time, l_time = measure_edges_performance(size, 0.2)
        print(f"{size} вершин: матрица={m_time:.2f}мс, список={l_time:.2f}мс")

    print("\n=== Сравнение BFS ===")
    bfs_sizes = [100, 500, 1000, 2000]
    m_times, l_times = measure_bfs_performance(bfs_sizes)

    for size, m_time, l_time in zip(bfs_sizes, m_times, l_times):
        print(f"{size} вершин: матрица={m_time:.2f}мс, список={l_time:.2f}мс")

    # Визуализация BFS
    plt.figure(figsize=(10, 6))
    plt.plot(bfs_sizes, m_times, 'r-', label='Матрица смежности', marker='o')
    plt.plot(bfs_sizes, l_times, 'b-', label='Список смежности', marker='s')
    plt.title('Время выполнения BFS')
    plt.xlabel('Количество вершин')
    plt.ylabel('Время (мс)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('./report/bfs_performance.png', dpi=150)
    plt.show()

# Характеристики ПК
pc_info = """
Характеристики ПК для тестирования:
- Процессор: 12th Gen Intel(R) Core(TM) i5-12450H
- Оперативная память: 16 GB DDR4
- ОС: Windows 10
- Python: 3.12.10
"""
print(pc_info)


if __name__ == "__main__":
    run_analysis()

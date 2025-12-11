from heap import Heap


class PriorityQueue:
    """Приоритетная очередь на куче
    Сложность: O(log n)"""
    def __init__(self):
        self.heap = Heap(is_min=True)

    def enqueue(self, item, priority):
        """Добавляет элемент в очередь с приоритетом.
        CСложность: O(log n)."""
        self.heap.insert((priority, item))

    def dequeue(self):
        """Извлекает элемент с наивысшим приоритетом.
        Сложность: O(log n)."""
        item = self.heap.extract()
        return item[1] if item else None

    def peek(self):
        """Возвращает элемент с наивысшим приоритетом без извлечения.
        Сложность:  # O(1)."""
        return self.heap.peek()[1] if self.heap.peek() else None

class Heap:
    """Реализация бинарной кучи (min-heap/max-heap).

    Поддерживает основные операции с кучей: вставка, извлечение корня,
    просмотр корня и построение кучи из массива.

    Сложность: O(1) для корня
    """
    def __init__(self, is_min=True):
        self.heap = []  # Массив элементов кучи
        self.is_min = is_min

    def _compare(self, x, y):
        """"Сравнивает два элемента в соответствии с типом кучи."""
        return x < y if self.is_min else x > y  # Для min-heap: <, для max-heap: >

    def insert(self, value):
        """Добавляет элемент в кучу.

        Сложность: O(log n).

        Аргумент:
            value: Элемент для вставки.
        """
        self.heap.append(value)  # Добавляем в конец
        self._sift_up(len(self.heap) - 1)  # Поднимаем вверх

    def extract(self):
        """Извлекает корневой элемент (минимальный или максимальный).

        Сложность: O(log n).
        """
        if not self.heap:
            return None
        res = self.heap[0]  # Корень (min или max в зависимости от типа)
        last = self.heap.pop()  # Последний элемент
        if self.heap:
            self.heap[0] = last  # Переносим в корень
            self._sift_down(0)  # Погружаем вниз
        return res

    def peek(self):
        """Просмотр корня.

        Сложность: O(1).
        """
        return self.heap[0] if self.heap else None

    def _sift_up(self, index):
        """Поднимает элемент на корректную позицию (всплытие).

        Аргументы:
            index: Индекс элемента для подъёма.
        """
        while index > 0:
            parent_index = (index - 1) // 2
            if not self._compare(self.heap[index], self.heap[parent_index]):
                break
            self.heap[index], self.heap[parent_index] = self.heap[parent_index], self.heap[index]
            index = parent_index

    def _sift_down(self, index):
        """Опускает элемент на корректную позицию (погружение).

        Сложность: O(log n)

        Аргументы:
            index: Индекс элемента для опускания.
        """
        size = len(self.heap)
        while True:
            left = 2 * index + 1
            right = 2 * index + 2
            idx = index
            if left < size and self._compare(self.heap[left], self.heap[idx]):
                idx = left
            if right < size and self._compare(self.heap[right], self.heap[idx]):
                idx = right
            if idx == index:
                break
            self.heap[index], self.heap[idx] = self.heap[idx], self.heap[index]
            index = idx

    def build_heap(self, array):
        """Строит кучу из произвольного массива.

        Сложность: O(n).

        Аргументы:
            array: Массив элементов.
        """
        self.heap = array[:]
        for i in range((len(self.heap) - 2)//2, -1, -1):
            self._sift_down(i)

    def is_heap(self):
        """Проверка свойства кучи, O(n)."""
        size = len(self.heap)
        for i in range(size // 2):
            left = 2 * i + 1
            right = 2 * i + 2
            ok_l = (left >= size) or not self._compare(self.heap[left], self.heap[i])
            ok_r = (right >= size) or not self._compare(self.heap[right], self.heap[i])
            if not (ok_l and ok_r):
                return False
        return True

    def show(self):
        """Визуализация кучи (текстовая).
        Сложность: O(n)."""
        def _display_node(index, pref):
            if index < len(self.heap):
                print(pref + str(self.heap[index]))
                left = 2 * index + 1
                right = 2 * index + 2
                if left < len(self.heap): _display_node(left, pref + "  ")
                if right < len(self.heap): _display_node(right, pref + "  ")
        if self.heap:
            _display_node(0, "")

from heap import Heap


def heapsort(array, is_min=True):
    """Сортирует массив с использованием кучи.

    Сложность: O(n log n).

    Аргументы:
        arr: Массив для сортировки.
        is_min: Если True, сортировка по возрастанию, иначе по убыванию.

    Возвращает:
        Отсортированный массив.
    """
    heap = Heap(is_min)
    heap.build_heap(array)  # O(n)
    out = []
    for _ in range(len(array)):
        out.append(heap.extract())  # O(log n) на каждый
    return out


def heapsort_inplace(array):
    """In-place сортировка кучей.

    Сложность: O(n log n).
    Изменяет исходный массив.

    Аргументы:
        arr: Массив для сортировки.
    """
    size = len(array)
    # Строим max-heap
    for i in range(size // 2 - 1, -1, -1):
        _sift_down_arr(array, i, size)
    # Извлекаем элементы
    for i in range(size - 1, 0, -1):
        array[0], array[i] = array[i], array[0]
        _sift_down_arr(array, 0, i)


def _sift_down_arr(array, i, size):
    """Опускает элемент в max-heap для inplace сортировки.

    Аргументы:
        array: Массив, представляющий кучу.
        i: Индекс элемента для опускания.
        size: Текущий размер кучи.
    """
    while True:
        left, right = 2 * i + 1, 2 * i + 2
        idx = i
        if left < size and array[left] > array[idx]:
            idx = left
        if right < size and array[right] > array[idx]:
            idx = right
        if idx == i:
            break
        array[i], array[idx] = array[idx], array[i]
        i = idx

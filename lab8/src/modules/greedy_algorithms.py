import heapq


def interval_scheduling(intervals):
    """
    Задача о выборе заявок: ищем макс. множество непересекающихся интервалов.
    Жадный выбор — сортировка по времени окончания.

    Args:
        intervals: Список интервалов в формате (начало, конец)

    Returns:
        Список выбранных интервалов в том же формате

    Сложность: O(n log n)
    """
    # Анонимная функция возвращает 2-й элемент
    intervals = sorted(intervals, key=lambda x: x[1])

    result = []
    last_end = float('-inf')
    for start, end in intervals:
        if start >= last_end:
            result.append((start, end))
            last_end = end
    return result


def fractional_knapsack(weights, values, capacity):
    """
    Задача о рюкзаке: можно брать дробные части предметов.
    Жадный выбор — сортировка по удельной стоимости (value/weight).

    Args:
        weights: Веса предметов
        values: Стоимости предметов
        capacity: Вместимость рюкзака

    Returns:
        Максимальная стоимость, которую можно унести

    Сложность: O(n log n)
    """
    # Список троек: удельная стоимость, вес, стоимость
    items = sorted(
        ((v / w, w, v) for w, v in zip(weights, values)),
        reverse=True
    )

    total = 0
    for ratio, weight, value in items:
        if capacity == 0:
            break

        # Берем максимально возможное количество текущего предмета
        take = min(weight, capacity)
        total += ratio * take
        capacity -= take

    return total


class HuffmanNode:
    """
    Класс узла дерева Хаффмана.

    Attributes:
        char: Символ (None для внутренних узлов)
        freq: Частота символа
        left: Левый потомок
        right: Правый потомок
    """
    def __init__(self, char, freq, left=None, right=None):
        self.char = char  # листовые узлы
        self.freq = freq
        self.left = left
        self.right = right

    def __lt__(self, other):
        # для heapq
        return self.freq < other.freq


def huffman_code(freqs):
    """
    Алгоритм Хаффмана: минимальное префиксное кодирование.
    Жадный выбор — объединяем узлы с наименьшими частотами.

    Args:
        freqs: Словарь символ->частота

    Returns:
        Кортеж (словарь кодов, корень дерева)

    Сложность: O(n log n)
    """
    heap = [HuffmanNode(char, freq) for char, freq in freqs.items()]
    heapq.heapify(heap)
    while len(heap) > 1:
        a = heapq.heappop(heap)
        b = heapq.heappop(heap)
        heapq.heappush(
            heap,
            HuffmanNode(None, a.freq + b.freq, a, b)
        )
    root = heap[0]
    codes = {}

    def gen_codes(node, prefix=""):
        if node.char is not None:
            codes[node.char] = prefix
        else:
            gen_codes(node.left, prefix + "0")
            gen_codes(node.right, prefix + "1")
    gen_codes(root)
    return codes, root


def print_huffman_tree(node, indent=""):
    """
    Визуализация дерева Хаффмана (рекурсивный вывод).

    node: Текущий узел для вывода
    indent: Отступ для текущего уровня
    """
    if node.char is not None:  # Если достигли листа
        print(indent + repr(node.char) + f":{node.freq}")
    else:
        print(indent + "*")
        print_huffman_tree(node.left, indent + "  ")
        print_huffman_tree(node.right, indent + "  ")

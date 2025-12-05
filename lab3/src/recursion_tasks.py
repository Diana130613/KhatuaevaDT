import os
from typing import List, Optional


# 1. Бинарный поиск с рекурсией
def binary_search_recursive(arr: List[int], target: int,
                            left: int = 0, right: int = None) -> Optional[int]:
    """
    Рекурсивная реализация бинарного поиска

    Временная сложность: O(log n)
    Глубина рекурсии: O(log n)

    Args:
        arr: отсортированный список целых чисел
        target: искомый элемент
        left: левая граница поиска
        right: правая граница поиска

    Return:
        Индекс элемента или None, если элемент не найден
    """
    if right is None:
        right = len(arr) - 1

    # Базовый случай: элемент не найден
    if left > right:
        return None

    # Вычисляем середину
    mid = (left + right) // 2

    # Базовый случай: элемент найден
    if arr[mid] == target:
        return mid

    # Рекурсивный шаг: ищем в левой или правой половине
    elif arr[mid] > target:
        return binary_search_recursive(arr, target, left, mid - 1)
    else:
        return binary_search_recursive(arr, target, mid + 1, right)


# 2. Рекурсивный обход файловой системы
def traverse_filesystem(path: str, indent: int = 0, max_depth: int = None,
                        current_depth: int = 0) -> None:
    """
    Рекурсивный обход файловой системы с выводом дерева каталогов

    Args:
        path: начальный путь для обхода
        indent: отступ для визуализации иерархии
        max_depth: максимальная глубина рекурсии (для исследования)
        current_depth: текущая глубина рекурсии
    """
    if max_depth is not None and current_depth > max_depth:
        return

    try:
        # Получаем список элементов в директории
        items = os.listdir(path)
    except PermissionError:
        print(" " * indent + "[Доступ запрещен]")
        return
    except FileNotFoundError:
        print(" " * indent + "[Путь не найден]")
        return

    for item in sorted(items):
        item_path = os.path.join(path, item)

        if os.path.isdir(item_path):
            # Это директория - выводим и рекурсивно обходим
            print(" " * indent + f"{item}/")
            traverse_filesystem(item_path, indent + 2, max_depth, current_depth + 1)
        else:
            # Это файл - просто выводим
            print(" " * indent + f"{item}")


# 3. Ханойские башни
def hanoi_towers(n: int, source: str = "A",
                 auxiliary: str = "B", target: str = "C") -> List[str]:
    """
    Решение задачи "Ханойские башни"

    Временная сложность: O(2^n)
    Глубина рекурсии: O(n)

    Args:
        n: количество дисков
        source: начальный стержень
        auxiliary: вспомогательный стержень
        target: целевой стержень

    Returns:
        Список ходов для решения задачи
    """
    moves = []

    def _hanoi(n: int, source: str, auxiliary: str, target: str):
        # Базовый случай: перемещаем один диск
        if n == 1:
            moves.append(f"Переместить диск 1 с {source} на {target}")
            return

        # Рекурсивный шаг:
        # 1. Переместить n-1 дисков на вспомогательный стержень
        _hanoi(n - 1, source, target, auxiliary)

        # 2. Переместить самый большой диск на целевой стержень
        moves.append(f"Переместить диск {n} с {source} на {target}")

        # 3. Переместить n-1 дисков с вспомогательного на целевой стержень
        _hanoi(n - 1, auxiliary, source, target)

    _hanoi(n, source, auxiliary, target)
    return moves


# Функции для экспериментов
def measure_max_recursion_depth(start_path: str = ".") -> int:
    """
    Измерение максимальной глубины рекурсии при обходе файловой системы
    """
    max_depth = [0]

    def _traverse_measure(path: str, current_depth: int):
        max_depth[0] = max(max_depth[0], current_depth)

        try:
            for item in os.listdir(path):
                item_path = os.path.join(path, item)
                if os.path.isdir(item_path):
                    _traverse_measure(item_path, current_depth + 1)
        except (PermissionError, FileNotFoundError):
            pass

    _traverse_measure(start_path, 0)
    return max_depth[0]


def test_binary_search():
    """Тестирование бинарного поиска"""
    print("=" * 60)
    print("Тестирование бинарного поиска")
    print("=" * 60)

    arr = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
    targets = [7, 1, 19, 8]

    for target in targets:
        result = binary_search_recursive(arr, target)
        if result is not None:
            print(f"Элемент {target} найден по индексу {result}")
        else:
            print(f"Элемент {target} не найден")


def test_hanoi_towers():
    """Тестирование Ханойских башен"""
    print("\n" + "=" * 60)
    print("Решение задачи 'Ханойские башни' для 3 дисков")
    print("=" * 60)

    moves = hanoi_towers(3)
    for i, move in enumerate(moves, 1):
        print(f"{i}. {move}")


def test_filesystem_traversal():
    """Тестирование обхода файловой системы"""
    print("\n" + "=" * 60)
    print("Обход файловой системы (ограничен глубиной 3)")
    print("=" * 60)

    # Ограничиваем глубину для наглядности
    traverse_filesystem(".", max_depth=3)


if __name__ == "__main__":
    test_binary_search()
    test_hanoi_towers()
    test_filesystem_traversal()

    # Измерение максимальной глубины рекурсии
    print("\n" + "=" * 60)
    print("Измерение максимальной глубины рекурсии")
    print("=" * 60)

    max_depth = measure_max_recursion_depth(".")
    print(f"Максимальная глубина рекурсии при обходе текущей директории: {max_depth}")

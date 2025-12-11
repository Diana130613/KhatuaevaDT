from typing import List


def merge_sort(arr: List[int]) -> List[int]:
    """
    Сортировка слиянием.

    Временная сложность:
    - Худший случай: O(n log n)
    - Средний случай: O(n log n)
    - Лучший случай: O(n log n)

    Пространственная сложность:
    - O(n) - требуется дополнительная память для временных массивов
    """
    if len(arr) <= 1:
        return arr.copy()

    # Рекурсивно делим массив на две части
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])

    # Сливаем отсортированные части
    return _merge(left, right)


def _merge(left: List[int], right: List[int]) -> List[int]:
    """Вспомогательная функция для слияния двух отсортированных массивов"""
    result = []
    i = j = 0

    # Сливаем, пока есть элементы в обоих массивах
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1

    # Добавляем оставшиеся элементы
    result.extend(left[i:])
    result.extend(right[j:])

    return result


def quick_sort(arr: List[int]) -> List[int]:
    """
    Быстрая сортировка.

    Временная сложность:
    - Худший случай: O(n²)
    - Средний случай: O(n log n)
    - Лучший случай: O(n log n) - хороший выбор опорного элемента

    Пространственная сложность:
    - O(log n) - глубина рекурсии (в среднем случае)
    - O(n) - в худшем случае (несбалансированные разбиения)
    """
    if len(arr) <= 1:
        return arr.copy()

    # Выбираем опорный элемент
    first, middle, last = arr[0], arr[len(arr) // 2], arr[-1]
    pivot = sorted([first, middle, last])[1]

    # Разделяем массив на элементы меньше, равные и больше опорного
    less = [x for x in arr if x < pivot]
    equal = [x for x in arr if x == pivot]
    greater = [x for x in arr if x > pivot]

    # Рекурсивно сортируем подмассивы и объединяем результаты
    return quick_sort(less) + equal + quick_sort(greater)

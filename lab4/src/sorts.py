from typing import List


def bubble_sort(arr: List[int]) -> List[int]:
    """
    Сортировка пузырьком.

    Временная сложность:
    - Худший случай: O(n²) - массив отсортирован в обратном порядке
    - Средний случай: O(n²)
    - Лучший случай: O(n) - массив уже отсортирован (с оптимизацией)

    Пространственная сложность:
    - O(1) - сортировка на месте, не требует дополнительной памяти
    """
    n = len(arr)
    result = arr.copy()

    for i in range(n):
        swapped = False  # условие флага для финального списка
        # Проходим по массиву, сравнивая соседние элементы
        for j in range(0, n - i - 1):
            if result[j] > result[j + 1]:
                # Меняем местами, если элементы в неправильном порядке
                result[j], result[j + 1] = result[j + 1], result[j]
                swapped = True
        # Если за проход не было обменов, массив отсортирован
        if not swapped:
            break
    return result


def selection_sort(arr: List[int]) -> List[int]:
    """
    Сортировка выбором.

    Временная сложность:
    - Худший случай: O(n²)
    - Средний случай: O(n²)
    - Лучший случай: O(n²) - всегда выполняет одинаковое количество сравнений

    Пространственная сложность:
    - O(1) - сортировка на месте
    """
    result = arr.copy()
    n = len(result)

    for i in range(n):
        # Находим минимальный элемент в неотсортированной части
        min_ind = i
        for j in range(i + 1, n):
            if result[j] < result[min_ind]:
                min_ind = j

        # Меняем местами найденный минимальный элемент с первым неотсортированным
        result[i], result[min_ind] = result[min_ind], result[i]

    return result


def insertion_sort(arr: List[int]) -> List[int]:
    """
    Сортировка вставками.

    Временная сложность:
    - Худший случай: O(n²) - массив отсортирован в обратном порядке
    - Средний случай: O(n²)
    - Лучший случай: O(n) - массив уже отсортирован

    Пространственная сложность:
    - O(1) - сортировка на месте
    """
    result = arr.copy()

    for i in range(1, len(result)):
        key = result[i]
        j = i - 1

        # Сдвигаем элементы больше key вправо
        while j >= 0 and result[j] > key:
            result[j + 1] = result[j]
            j -= 1

        # Вставляем key в правильную позицию
        result[j + 1] = key

    return result


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
    - Худший случай: O(n²) - плохой выбор опорного элемента (например, уже отсортированный массив)
    - Средний случай: O(n log n)
    - Лучший случай: O(n log n) - хороший выбор опорного элемента

    Пространственная сложность:
    - O(log n) - глубина рекурсии (в среднем случае)
    - O(n) - в худшем случае (несбалансированные разбиения)
    """
    if len(arr) <= 1:
        return arr.copy()

    # Выбираем опорный элемент (медиана из трех для улучшения производительности)
    first, middle, last = arr[0], arr[len(arr) // 2], arr[-1]
    pivot = sorted([first, middle, last])[1]

    # Разделяем массив на элементы меньше, равные и больше опорного
    less = [x for x in arr if x < pivot]
    equal = [x for x in arr if x == pivot]
    greater = [x for x in arr if x > pivot]

    # Рекурсивно сортируем подмассивы и объединяем результаты
    return quick_sort(less) + equal + quick_sort(greater)

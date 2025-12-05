# Отчет по лабораторной работе 3
# Алгоритмы сортировки

**Дата:** 2025-11-01  
**Семестр:** 3 курс 1 полугодие - 5 семестр  
**Группа:** ПИЖ-б-о-23-2(1)  
**Дисциплина:** Анализ сложности алгоритмов  
**Студент:** Хатуаева Дайана Тныбековна

## Цель работы
Изучить и реализовать основные алгоритмы сортировки. Провести их теоретический и практический сравнительный анализ по временной и пространственной сложности. Исследовать влияние начальной упорядоченности данных на эффективность алгоритмов. Получить навыки эмпирического анализа производительности алгоритмов.

## Теоретическая часть
- Сортировка пузырьком (Bubble Sort): Многократно проходит по массиву, сравнивая и меняя местами соседние элементы. Сложность: O(n²) во всех случаях.
- Сортировка выбором (Selection Sort): На каждом проходе находит минимальный элемент из неотсортированной части и ставит его на очередную позицию. Сложность: O(n²).
- Сортировка вставками (Insertion Sort): Построение окончательного массива путем пошагового вставления каждого элемента в правильную позицию в уже отсортированной части. Сложность: O(n²) в худшем и среднем, O(n) в лучшем (уже отсортированный массив).
- Сортировка слиянием (Merge Sort): Рекурсивный алгоритм "разделяй и властвуй". Массив разбивается на две части, которые сортируются рекурсивно, а затем сливаются в один отсортированный массив. Сложность: O(n log n) во всех случаях. Требует O(n) дополнительной памяти.
- Быстрая сортировка (Quick Sort): Рекурсивный алгоритм "разделяй и властвуй". Выбирается опорный элемент, массив разделяется на элементы меньше и больше опорного, которые сортируются рекурсивно. Сложность: O(n log n) в среднем, O(n²) в худшем случае (плохой выбор опорного элемента). Сортировка на месте, не требует дополнительной памяти

## Практическая часть

### Выполненные задачи
- [x] Задача 1: Реализовать 5 алгоритмов сортировки.
- [x] Задача 2: Провести теоретический анализ сложности каждого алгоритма.
- [x] Задача 3: Экспериментально сравнить время выполнения алгоритмов на различных наборах данных.
- [x] Задача 4: Проанализировать влияние начальной упорядоченности данных на эффективность сортировок.

### Ключевые фрагменты кода

#### Реализация сортировок
- Реализовать все 5 алгоритмов сортировки.
- Для каждого метода в комментарии указать временную и пространственную 
сложность в худшем, среднем и лучшем случаях.

```python
# sorts.py
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
```

####  Подготовка тестовых данных
- Сгенерировать массивы целых чисел разного размера (напр., 100, 1000, 5000, 10000 элементов).
- Разные типы данных

```python
# generate_data.py
# generate_data.py
import random
from typing import List, Dict


def generate_random_array(size: int, min_val: int = 0, max_val: int = 10000) -> List[int]:
    """Генерация массива со случайными числами"""
    return [random.randint(min_val, max_val) for _ in range(size)]


def generate_sorted_array(size: int, min_val: int = 0, max_val: int = 10000) -> List[int]:
    """Генерация отсортированного массива"""
    return sorted(generate_random_array(size, min_val, max_val))


def generate_reversed_array(size: int, min_val: int = 0, max_val: int = 10000) -> List[int]:
    """Генерация массива, отсортированного в обратном порядке"""
    return list(reversed(generate_sorted_array(size, min_val, max_val)))


def generate_almost_sorted_array(size: int, sorted_percentage: float = 0.95,
                                 min_val: int = 0, max_val: int = 10000) -> List[int]:
    """
    Генерация почти отсортированного массива

    Args:
        size: размер массива
        sorted_percentage: процент отсортированных элементов (от 0 до 1)
        min_val: минимальное значение
        max_val: максимальное значение
    """
    # Сначала создаём отсортированный массив
    arr = generate_sorted_array(size, min_val, max_val)

    # Вычисляем количество элементов для перемешивания
    num_to_shuffle = int(size * (1 - sorted_percentage))

    # Выбираем случайные индексы для перемешивания
    if num_to_shuffle > 0:
        indices_to_shuffle = random.sample(range(size), num_to_shuffle)

        # Для выбранных индексов генерируем новые случайные значения
        for ind in indices_to_shuffle:
            arr[ind] = random.randint(min_val, max_val)

    return arr


def generate_all_test_arrays(sizes: List[int] = None) -> Dict[str, Dict[int, List[int]]]:
    """
    Генерация всех тестовых массивов для разных размеров и типов

    Returns:
        'название': {размер[содержание]}
    """
    if sizes is None:
        sizes = [100, 1000, 5000, 10000]

    test_data = {
        'random': {},
        'sorted': {},
        'reversed': {},
        'almost_sorted': {}
    }

    for size in sizes:
        print(f"Генерация массивов размера {size}...")

        test_data['random'][size] = generate_random_array(size)
        test_data['sorted'][size] = generate_sorted_array(size)
        test_data['reversed'][size] = generate_reversed_array(size)
        test_data['almost_sorted'][size] = generate_almost_sorted_array(
            size, sorted_percentage=0.95
        )
    return test_data
```

#### Эмпирический анализ производительности
- Замерить время выполнения каждой сортировки на всех типах данных и для всех размеров.
- Использовать модуль timeit для точных замеров.
```python
# performance_test.py
import timeit
from sorts import (
    bubble_sort,
    selection_sort,
    insertion_sort,
    merge_sort,
    quick_sort
)
from generate_data import generate_all_test_arrays

# Характеристики ПК
pc_info = """
    Характеристики ПК для тестирования:
    - Процессор: 12th Gen Intel(R) Core(TM) i5-12450H
    - Оперативная память: 16 GB DDR4
    - ОС: Windows 10
    - Python: 3.12.10
    """
print(pc_info)

SORTING_ALGORITHMS = [
    ("bubble_sort", bubble_sort),
    ("selection_sort", selection_sort),
    ("insertion_sort", insertion_sort),
    ("merge_sort", merge_sort),
    ("quick_sort", quick_sort)
]


def measure_time(sort_func, data):
    """Замер времени выполнения конкретного алгоритма."""
    start_time = timeit.default_timer()
    sort_func(data)
    end_time = timeit.default_timer()
    return end_time - start_time


def run_performance_tests(test_data):
    results = {}  # Хранение результатов замера времени

    for algo_name, algo_func in SORTING_ALGORITHMS:
        print(f"Тестирование алгоритма {algo_name}")

        # Пробегаемся по каждому типу данных и размеру
        for array_type, arrays_by_size in test_data.items():
            type_results = {}

            for size, data in arrays_by_size.items():
                copy_data = data.copy()  # Работаем с копией данных

                elapsed_time = measure_time(algo_func, copy_data)
                type_results[size] = elapsed_time

            results.setdefault(array_type, {})[algo_name] = type_results

    return results


if __name__ == "__main__":
    # Генерируем тестовые данные
    test_data = generate_all_test_arrays()

    # Запускаем тесты производительности
    perf_results = run_performance_tests(test_data)

    # Сохраняем результаты для дальнейшего анализа
    from pprint import pprint
    pprint(perf_results)
```

#### Визуализация
- Построить графики зависимости времени выполнения от размера массива для каждого алгоритма на одном типе данных (например, случайные данные).
- Построить графики зависимости времени выполнения от типа данных для фиксированного размера массива (например, n=5000).
- Создать сводную таблицу результатов.

```python
# plot_results.py
import matplotlib.pyplot as plt
import numpy as np
from performance_test import run_performance_tests, generate_all_test_arrays

# Генерируем данные и запускаем тесты один раз
test_data = generate_all_test_arrays()
perf_results = run_performance_tests(test_data)

# Подготовим данные для графиков
sizes = [100, 1000, 5000, 10000]
algorithms = ["bubble_sort", "selection_sort", "insertion_sort", "merge_sort", "quick_sort"]

# 1. График для случайных данных
plt.figure(figsize=(12, 8))

# График 1: Зависимость от размера (случайные данные)
plt.subplot(2, 2, 1)
for algo in algorithms:
    times = []
    for size in sizes:
        times.append(perf_results['random'][algo][size])
    plt.plot(sizes, times, 'o-', label=algo, linewidth=2)
plt.title("Случайные данные")
plt.xlabel("Размер массива")
plt.ylabel("Время (сек)")
plt.grid(True, alpha=0.3)
plt.legend(loc='upper left', fontsize=8)

# График 2: Для отсортированных данных
plt.subplot(2, 2, 2)
for algo in algorithms:
    times = []
    for size in sizes:
        times.append(perf_results['sorted'][algo][size])
    plt.plot(sizes, times, 'o-', label=algo, linewidth=2)
plt.title("Отсортированные данные")
plt.xlabel("Размер массива")
plt.ylabel("Время (сек)")
plt.grid(True, alpha=0.3)

# График 3: Для обратно отсортированных данных
plt.subplot(2, 2, 3)
for algo in algorithms:
    times = []
    for size in sizes:
        times.append(perf_results['reversed'][algo][size])
    plt.plot(sizes, times, 'o-', label=algo, linewidth=2)
plt.title("Обратно отсортированные данные")
plt.xlabel("Размер массива")
plt.ylabel("Время (сек)")
plt.grid(True, alpha=0.3)

# График 4: Сравнение для размера 5000
plt.subplot(2, 2, 4)
size = 5000
x = np.arange(len(['random', 'sorted', 'reversed', 'almost_sorted']))
width = 0.15

for i, algo in enumerate(algorithms):
    times = [
        perf_results['random'][algo][size],
        perf_results['sorted'][algo][size],
        perf_results['reversed'][algo][size],
        perf_results['almost_sorted'][algo][size]
    ]
    plt.bar(x + i*width - width*2, times, width, label=algo)

plt.title(f"Сравнение типов данных (n={size})")
plt.xlabel("Тип данных")
plt.ylabel("Время (сек)")
plt.xticks(x, ['random', 'sorted', 'reversed', 'almost_sorted'])
plt.legend(loc='upper left', fontsize=8)
plt.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()

# Вывод сводной таблицы
print("\n" + "="*80)
print("СВОДНАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ (n=5000):")
print("="*80)

headers = ["Алгоритм", "Random", "Sorted", "Reversed", "Almost Sorted"]
print(f"{headers[0]:<15} {headers[1]:<12} {headers[2]:<12} {headers[3]:<12} {headers[4]:<12}")
print("-"*80)

for algo in algorithms:
    row = [algo]
    for data_type in ['random', 'sorted', 'reversed', 'almost_sorted']:
        time = perf_results[data_type][algo][5000]
        row.append(f"{time:.6f}")
    print(f"{row[0]:<15} {row[1]:<12} {row[2]:<12} {row[3]:<12} {row[4]:<12}")

```

## Результаты выполнения

![Текст](./report/figure.png)

### Пример работы программы
```
Характеристики ПК для тестирования:
    - Процессор: 12th Gen Intel(R) Core(TM) i5-12450H
    - Оперативная память: 16 GB DDR4
    - ОС: Windows 10
    - Python: 3.12.10

Генерация массивов размера 100...
Генерация массивов размера 1000...
Генерация массивов размера 5000...
Генерация массивов размера 10000...
Тестирование алгоритма bubble_sort
Тестирование алгоритма selection_sort
Тестирование алгоритма insertion_sort
Тестирование алгоритма merge_sort
Тестирование алгоритма quick_sort

================================================================================
СВОДНАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ (n=5000):
================================================================================
Алгоритм        Random       Sorted       Reversed     Almost Sorted
--------------------------------------------------------------------------------
bubble_sort     0.801792     0.000202     0.920044     0.480883
selection_sort  0.310086     0.310133     0.336228     0.312335
insertion_sort  0.325583     0.000312     0.621299     0.021495
merge_sort      0.006250     0.004878     0.004433     0.005451
quick_sort      0.005144     0.003939     0.004092     0.003939
```

## Ответы на контрольные вопросы
1. Какие алгоритмы сортировки имеют сложность O(n²) в худшем случае, а какие - O(n log n)?

Алгоритмы сортировки с временной сложностью O(n²) в худшем случае включают:
- Bubble Sort (пузырьковая сортировка)
- Selection Sort (сортировка выбором)
- Insertion Sort (сортировка вставками)
Алгоритмы сортировки с временной сложностью O(n log n) в среднем и худшем случаях:
- Merge Sort (сортировка слиянием)
- Heap Sort (сортировка кучей)
- Quick Sort (быстрая сортировка)

2. Почему сортировка вставками (Insertion Sort) эффективна для маленьких или почти
отсортированных массивов?

Сортировка вставками эффективно работает для небольших массивов и частично упорядоченных последовательностей благодаря своей простоте реализации и линейному поведению на почти отсортированном массиве (лучший случай имеет сложность O(n)). Это связано с тем, что каждый элемент перемещается лишь на небольшие расстояния назад, уменьшая количество необходимых сравнений и перестановок.

3. В чем разница между устойчивой (stable) и неустойчивой (unstable) сортировкой? Приведите
пример устойчивого и неустойчивого алгоритма.

Устойчивая сортировка сохраняет относительный порядок элементов с одинаковыми ключами. Например, если в исходном массиве два элемента равны, их порядок останется таким же после сортировки.
Неустойчивая сортировка не гарантирует сохранение порядка равных элементов.
Примеры:
- Stable (устойчивый): Merge Sort, Insertion Sort
- Unstable (неустойчивый): Quick Sort, Heap Sort

4. Опишите принцип работы алгоритма быстрой сортировки (Quick Sort). Что такое "опорный
элемент" и как его выбор влияет на производительность?

Алгоритм быстрой сортировки основан на методе разделения массива на две части относительно выбранного опорного элемента ("pivot"). Элементы меньше опорного размещаются слева, больше — справа. Затем процесс рекурсивно повторяется для каждой части.

Выбор опорного элемента существенно влияет на производительность: хороший выбор уменьшает глубину рекурсии и улучшает среднее время работы. Если выбрать плохой опорный элемент (например, минимальный или максимальный элемент), то сложность алгоритма дойдёт до квадратичной сложности O(n²).

5. Сортировка слиянием (Merge Sort) гарантирует время O(n log n), но требует дополнительной
памяти. В каких ситуациях этот алгоритм предпочтительнее быстрой сортировки?

Сортировка слиянием имеет преимущества там, где важна стабильность сортировки и гарантированное время выполнения O(n log n), ведь даже в худших условиях (почти отсортированный массив) сортировка слиянием остаётся эффективной.
Сортировка слиянием также обладает усточйчивостью, сохраняет порядок элементов с одинаковым значением ключа.
Таким образом, сортировку слиянием выбирают, когда важны гарантии устойчивости и предсказуемое время выполнения, особенно при работе с большими объемами данных или критическими приложениями, где важен детерминированный характер обработки.

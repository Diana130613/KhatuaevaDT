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

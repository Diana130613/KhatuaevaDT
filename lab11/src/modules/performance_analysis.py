import matplotlib.pyplot as plt
import random
import string
import timeit
from prefix_function import prefix_function
from z_function import z_function
from kmp_search import kmp_search
from string_matching import naive_search, rabin_karp


# Генерация случайных строк
def generate_random_string(length):
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for _ in range(length))


# Графики зависимости времени выполнения от длины текста
def plot_execution_time_vs_text_length(algorithms, max_len=10000, step=1000):
    lengths = range(step, max_len + 1, step)

    # Словарь для хранения времени выполнения
    execution_times = {alg_name: [] for alg_name in algorithms.keys()}

    for length in lengths:
        text = generate_random_string(length)
        pattern = generate_random_string(10)

        for alg_name, func in algorithms.items():
            exec_time = timeit.timeit(lambda: func(text, pattern), number=10)
            execution_times[alg_name].append(exec_time / 10)

    # Построение графика
    plt.figure(figsize=(10, 6))
    for alg_name, times in execution_times.items():
        plt.plot(lengths, times, label=alg_name)

    plt.xlabel('Длина текста')
    plt.ylabel('Время выполнения (сек)')
    plt.title('Зависимость времени выполнения от длины текста')
    plt.legend()
    plt.grid(True)
    plt.show()


# Графики зависимости времени выполнения от длины паттерна
def plot_execution_time_vs_pattern_length(algorithms, max_len=100, step=10):
    lengths = range(step, max_len + 1, step)

    # Словарь для хранения времени выполнения
    execution_times = {alg_name: [] for alg_name in algorithms.keys()}

    for length in lengths:
        text = generate_random_string(10000)
        pattern = generate_random_string(length)

        for alg_name, func in algorithms.items():
            exec_time = timeit.timeit(lambda: func(text, pattern), number=10)
            execution_times[alg_name].append(exec_time / 10)

    # Построение графика
    plt.figure(figsize=(10, 6))
    for alg_name, times in execution_times.items():
        plt.plot(lengths, times, label=alg_name)

    plt.xlabel('Длина паттерна')
    plt.ylabel('Время выполнения (сек)')
    plt.title('Зависимость времени выполнения от длины паттерна')
    plt.legend()
    plt.grid(True)
    plt.show()


# Визуализация работы префикс-функции
def visualize_prefix_function(s):
    pi = prefix_function(s)
    x = range(len(pi))
    y = pi

    plt.figure(figsize=(10, 6))
    plt.bar(x, y, align='center', alpha=0.7)
    plt.xlabel('Позиция символа')
    plt.ylabel('Значение префикс-функции')
    plt.title('Работа префикс-функции для строки "{}"'.format(s))
    plt.grid(axis='y')
    plt.show()


# Визуализация работы Z-функции
def visualize_z_function(s):
    z = z_function(s)
    x = range(len(z))
    y = z

    plt.figure(figsize=(10, 6))
    plt.bar(x, y, align='center', alpha=0.7)
    plt.xlabel('Позиция символа')
    plt.ylabel('Значение Z-функции')
    plt.title('Работа Z-функции для строки "{}"'.format(s))
    plt.grid(axis='y')
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
    # Определение алгоритмов для тестирования
    algorithms = {
        'Naive Search': naive_search,
        'KMP Search': kmp_search,
        'Rabin-Karp': rabin_karp,
        'Z-function Search': lambda t, p: z_function(t + '#' + p),
    }

    # Графики зависимости времени выполнения от длины текста
    plot_execution_time_vs_text_length(algorithms)

    # Графики зависимости времени выполнения от длины паттерна
    plot_execution_time_vs_pattern_length(algorithms)

    # Визуализация работы префикс-функции
    visualize_prefix_function("ababcababcababc")

    # Визуализация работы Z-функции
    visualize_z_function("ababcababcababc")

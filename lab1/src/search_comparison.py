import matplotlib.pyplot as plt
import time
import random


def linear_search(arr, target):
    # Цикл проходится по каждому элементу массива => O(n)
    for i in range(len(arr)):  # O(n)
        if arr[i] == target:  # O(1)
            return i  # O(1)
    return -1  # O(1)
# Общая сложность линейного поиска => O(n)


def binary_search(arr, target):
    lower_bound = 0  # O(1)
    upper_bound = len(arr) - 1  # O(1)

    while lower_bound <= upper_bound:  # O(logn)
        center = (lower_bound + upper_bound) // 2  # O(1)
        if arr[center] == target:  # O(1)
            return center  # O(1)
        elif arr[center] < target:  # O(1)
            lower_bound = center + 1  # O(1)
        elif arr[center] > target:  # O(1)
            upper_bound = center - 1  # O(1)
    return -1  # O(1)
# Общая сложность бинарного поиска - O(logn)


# Характеристики ПК
pc_info = """
    Характеристики ПК для тестирования:
    - Процессор: 12th Gen Intel(R) Core(TM) i5-12450H
    - Оперативная память: 16 GB DDR4
    - ОС: Windows 10
    - Python: 3.12.10
    """
print(pc_info)

sizes = [1000, 2000, 5000, 10000, 20000, 50000, 100000,
         200000, 500000]  # размеры массивов
linear_times = []
binary_times = []

print("Замеры времени выполнения")
print("Размер(N)", "      Линейный(мкс)", " Бинарный(мкс)",
      "   Линейный/Размер", " Бинарный/Размер")
print("=" * 78)

for size in sizes:
    data = sorted(random.sample(range(1, size * 2), size))

    target = data[len(data) // 2]  # средний элемент

    # Измерение времени для линейного поиска
    start_time = time.perf_counter()
    for _ in range(10):
        linear_search(data, target)
    end_time = time.perf_counter()
    linear_time = (end_time - start_time) / 10 * 1_000_000
    linear_times.append(linear_time)

    # Измерение времени для бинарного поиска
    start_time = time.perf_counter()
    for _ in range(10):
        binary_search(data, target)
    end_time = time.perf_counter()
    binary_time = (end_time - start_time) / 10 * 1_000_000
    binary_times.append(binary_time)

    linear_ratio = linear_time / size if size > 0 else 0
    binary_ratio = binary_time / size if size > 0 else 0

    print(f"{size:8,}\t{linear_time:.8f}\t{binary_time:.8f}\t{
        linear_ratio:.12f}\t{binary_ratio:.12f}")

# Построение графика

plt.figure(figsize=(15, 6))

plt.subplot(1, 2, 1)
plt.plot(sizes, linear_times, 'o-', label='Линейный поиск O(n)',
         linewidth=2, markersize=6)
plt.plot(sizes, binary_times, 's-', label='Бинарный поиск O(logn)',
         linewidth=2, markersize=6)
plt.xlabel('Размер массива (N)')
plt.ylabel('Время выполнения в микросекундах')
plt.title('Сравнение времени выполнения алгоритмов поиска\n(линейный масштаб)')
plt.grid(True, alpha=0.3)
plt.legend()

plt.subplot(1, 2, 2)
plt.semilogy(sizes, linear_times, 'o-', label='Линейный поиск O(n)',
             linewidth=2, markersize=6)
plt.semilogy(sizes, binary_times, 's-', label='Бинарный поиск O(logn)',
             linewidth=2, markersize=6)
plt.xlabel('Размер массива (N)')
plt.ylabel('Время выполнения (микросекунды) - логарифмическая шкала')
plt.title('Сравнение времени выполнения алгоритмов'
          'поиска\n(логарифмический масштаб по Y)')
plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()

plt.show()

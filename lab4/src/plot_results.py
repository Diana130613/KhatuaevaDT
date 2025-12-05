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

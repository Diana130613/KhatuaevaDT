import time
import tracemalloc
import matplotlib.pyplot as plt
import numpy as np
from dynamic_programming import fib_memo, fib_table, knapsack_01, lcs
from tasks import coin_change, lis_quadratic


"""Сравнительный анализ подходов динамического программирования."""


def compare_fib(n_values):
    """Сравнение memo (top-down) и table (bottom-up) для Фибоначчи."""
    times_memo = []
    times_table = []
    mem_memo = []
    mem_table = []

    for n in n_values:
        # Измерение для memo
        tracemalloc.start()
        start = time.perf_counter()
        fib_memo(n)
        time_m = time.perf_counter() - start
        mem_m, _ = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Измерение для table
        tracemalloc.start()
        start = time.perf_counter()
        fib_table(n)
        time_t = time.perf_counter() - start
        mem_t, _ = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        times_memo.append(time_m)
        times_table.append(time_t)
        mem_memo.append(mem_m / 1024)  # KB
        mem_table.append(mem_t / 1024)  # KB

        print(f"n={n}: memo={time_m:.6f}s, {mem_m/1024:.1f}KB | "
              f"table={time_t:.6f}s, {mem_t/1024:.1f}KB")

    return times_memo, times_table, mem_memo, mem_table


def greedy_fractional_knapsack(weights, values, capacity):
    """Жадный алгоритм дробного рюкзака для сравнения.
    Время: O(n log n),
    память: O(n).
    """
    items = []
    for i, (w, v) in enumerate(zip(weights, values)):
        items.append((v / w, w, v, i))
    items.sort(reverse=True)

    remaining = capacity
    total_value = 0.0

    for ratio, w, v, _ in items:
        if remaining <= 0:
            break
        if w <= remaining:
            total_value += v
            remaining -= w
        else:
            frac = remaining / w
            total_value += v * frac
            remaining = 0

    return total_value


def compare_knapsack_algorithms():
    """Сравнение 0-1 рюкзака (ДП) и дробного рюкзака (жадный)."""
    print("\n" + "="*60)
    print("Сравнение алгоритмов для рюкзака")
    print("="*60)

    weights = [1, 3, 4, 5]
    values = [1, 4, 5, 7]
    capacity = 7

    # 0-1 рюкзак (ДП)
    start = time.perf_counter()
    dp_value, dp_items, _ = knapsack_01(weights, values, capacity)
    dp_time = time.perf_counter() - start

    # Дробный рюкзак (жадный)
    start = time.perf_counter()
    greedy_value = greedy_fractional_knapsack(weights, values, capacity)
    greedy_time = time.perf_counter() - start

    print(f"0-1 Knapsack (DP):")
    print(f"  Макс стоимость: {dp_value}")
    print(f"  Выбранные предметы: {dp_items}")
    print(f"  Время выполнения: {dp_time:.6f}s")

    print(f"\nFractional Knapsack (Greedy):")
    print(f"  Макс стоимость: {greedy_value:.2f}")
    print(f"  Время выполнения: {greedy_time:.6f}s")

    print(f"\nРазница: {greedy_value - dp_value:.2f}")

    return dp_value, greedy_value


def scale_test_knapsack(max_items=20, capacity_multiplier=10):
    """Тестирование масштабируемости алгоритма рюкзака."""
    print("\n" + "="*60)
    print("Тестирование масштабируемости алгоритма рюкзака")
    print("="*60)

    item_counts = list(range(5, max_items + 1, 5))
    times = []

    for n in item_counts:
        # Генерация случайных данных
        np.random.seed(42)
        weights = np.random.randint(1, 20, n)
        values = np.random.randint(1, 50, n)
        capacity = np.sum(weights) // capacity_multiplier

        start = time.perf_counter()
        knapsack_01(weights.tolist(), values.tolist(), capacity)
        elapsed = time.perf_counter() - start

        times.append(elapsed)
        print(f"n={n}: время={elapsed:.4f}s")

    # Построение графика
    plt.figure(figsize=(10, 6))
    plt.plot(item_counts, times, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('Количество предметов')
    plt.ylabel('Время выполнения (сек)')
    plt.title('Масштабируемость алгоритма 0-1 рюкзака')
    plt.grid(True, alpha=0.3)
    plt.savefig('knapsack_scaling.png', dpi=100)
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


def test_all_algorithms():
    """Тестирование всех реализованных алгоритмов."""
    print("\n" + "="*60)
    print("Тестирование всех алгоритмов ДП")
    print("="*60)

    # 1. Размен монет
    print("\n1. Размен монет:")
    coins = [1, 2, 5, 10]
    amount = 27
    result = coin_change(coins, amount)
    if result == -1:
        print(f"Невозможно разменять сумму {amount}")
    else:
        print(f"Минимальное количество монет: {result}")
    print(f"Монеты: {coins}, сумма: {amount}")
    print(f"Минимальное количество монет: {result}")

    # 2. Наибольшая возрастающая подпоследовательность
    print("\n2. Наибольшая возрастающая подпоследовательность:")
    seq = [10, 22, 9, 33, 21, 50, 41, 60, 80]
    lis_len, lis_seq = lis_quadratic(seq)
    print(f"Последовательность: {seq}")
    print(f"Длина LIS: {lis_len}")
    print(f"LIS: {lis_seq}")

    # 3. Визуализация таблицы LCS
    print("\n3. Визуализация таблицы LCS:")
    str1 = "ABCD"
    str2 = "ACBAD"
    lcs_len, lcs_seq, dp_table = lcs(str1, str2)
    print(f"Строка 1: {str1}")
    print(f"Строка 2: {str2}")
    print(f"LCS: {lcs_seq} (длина: {lcs_len})")


if __name__ == "__main__":
    # Часть 1: Сравнение подходов для Фибоначчи
    print("Часть 1: Сравнение top-down и bottom-up для чисел Фибоначчи")
    print("-" * 60)

    n_values = [10, 20, 30, 35, 40]
    times_memo, times_table, mem_memo, mem_table = compare_fib(n_values)  # Передаем список

    # Визуализация результатов
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # График времени
    ax1.plot(n_values, times_memo, 'ro-', label='Top-down (memo)', linewidth=2)
    ax1.plot(n_values, times_table, 'bo-', label='Bottom-up (table)', linewidth=2)
    ax1.set_xlabel('n')
    ax1.set_ylabel('Время (сек)')
    ax1.set_title('Время выполнения')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # График памяти
    ax2.plot(n_values, mem_memo, 'ro-', label='Top-down (memo)', linewidth=2)
    ax2.plot(n_values, mem_table, 'bo-', label='Bottom-up (table)', linewidth=2)
    ax2.set_xlabel('n')
    ax2.set_ylabel('Память (KB)')
    ax2.set_title('Использование памяти')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('fib_comparison.png', dpi=100)
    plt.show()

    # Часть 2: Сравнение алгоритмов рюкзака
    compare_knapsack_algorithms()

    # Часть 3: Тестирование всех алгоритмов
    test_all_algorithms()

    # Часть 4: Тестирование масштабируемости
    scale_test_knapsack(max_items=50, capacity_multiplier=3)

    # Дополнительный анализ сложности
    print("\n" + "="*60)
    print("Анализ временной и пространственной сложности")
    print("="*60)
    print("""
    Числа Фибоначчи:
      - Наивная рекурсия: O(2^n) время, O(n) память (стек)
      - Top-down с мемоизацией: O(n) время, O(n) память
      - Bottom-up табличный: O(n) время, O(n) память
      - Оптимизированный bottom-up: O(n) время, O(1) память

    Рюкзак 0-1:
      - Bottom-up: O(n*W) время, O(n*W) память

    LCS (наибольшая общая подпоследовательность):
      - Bottom-up: O(m*n) время, O(m*n) память

    Расстояние Левенштейна:
      - Bottom-up: O(m*n) время, O(m*n) память

    Размен монет:
      - Bottom-up: O(amount * len(coins)) время, O(amount) память

    LIS (наибольшая возрастающая подпоследовательность):
      - Квадратичный алгоритм: O(n^2) время, O(n) память
    """)

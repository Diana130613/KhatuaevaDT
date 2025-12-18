import itertools
from .greedy_algorithms import fractional_knapsack


def brute_force_01_knapsack(weights, values, capacity):
    """Полный перебор для задачи 0-1 рюкзака"""
    n = len(weights)
    best_value = 0
    best_combination = None

    # Перебираем все возможные комбинации предметов
    for r in range(n + 1):
        for combo in itertools.combinations(range(n), r):
            total_weight = sum(weights[i] for i in combo)
            total_value = sum(values[i] for i in combo)

            if total_weight <= capacity and total_value > best_value:
                best_value = total_value
                best_combination = combo
    return best_value, best_combination


def compare_knapsack():
    """Сравнение дробного и дискретного рюкзаков"""
    weights = [10, 20, 30]
    values = [60, 100, 120]
    capacity = 50

    # Дробный рюкзак (жадный)
    fractional_val = fractional_knapsack(weights, values, capacity)

    # Дискретный рюкзак (полный перебор)
    discrete_val, combo = brute_force_01_knapsack(weights, values, capacity)

    print("Сравнение рюкзаков:")
    print(f"Веса: {weights}, Стоимости: {values}, Вместимость: {capacity}")
    print(f"Дробный рюкзак (жадный): {fractional_val:.2f}")
    print(f"Дискретный рюкзак (0-1): {discrete_val}")
    print(f"Комбинация для 0-1: {combo}")

    # Пример, где жадный алгоритм для 0-1 не оптимален
    print("\nПример неоптимальности жадного для 0-1:")
    weights2 = [30, 20, 10]
    values2 = [120, 100, 60]

    # Жадное решение для 0-1 (берём по убыванию удельной стоимости)
    items = sorted(zip(values2, weights2),
                   key=lambda x: x[0]/x[1], reverse=True)

    cap = 50
    greedy_01_val = 0
    greedy_01_weight = 0

    for v, w in items:
        if greedy_01_weight + w <= cap:
            greedy_01_val += v
            greedy_01_weight += w

    # Оптимальное решение
    opt_val, opt_combo = brute_force_01_knapsack(weights2, values2, cap)

    print(f"Жадный для 0-1: {greedy_01_val}")
    print(f"Оптимальный для 0-1: {opt_val}")
    print(f"Разница: {opt_val - greedy_01_val}")

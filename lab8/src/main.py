from modules.greedy_algorithms import (
    interval_scheduling, fractional_knapsack,
    huffman_code, print_huffman_tree
)
from modules.analysis import compare_knapsack
from modules.task_solution import min_coins
from modules.performance_analysis import perf_huffman


def main():
    # Характеристики ПК
    pc_info = """
    Характеристики ПК для тестирования:
    - Процессор: 12th Gen Intel(R) Core(TM) i5-12450H
    - Оперативная память: 16 GB DDR4
    - ОС: Windows 10
    - Python: 3.12.10
    """
    print(pc_info)

    """
    Демонстрация всех жадных алгоритмов и их результатов.
    """
    print("=" * 50)
    print("Демонстрация жадных алгоритмов")
    print("=" * 50)

    print("\n1. Задача о выборе интервалов:")
    intervals = [(1, 4), (2, 6), (4, 7), (5, 9), (8, 10)]
    print(f"Интервалы: {intervals}")
    print(f"Макс.непересекающиеся интервалы: {interval_scheduling(intervals)}")

    print("\n2. Задача о дробном рюкзаке:")
    weights = [10, 6, 2]
    values = [40, 30, 6]
    cap = 17
    print(f"Веса: {weights}, Стоимости: {values}, Вместимость: {cap}")
    result = fractional_knapsack(weights, values, cap)
    print(f"Макс. стоимость: {result}")

    print("\n3. Кодирование Хаффмана:")
    freqs = {'A': 10, 'B': 15, 'C': 30, 'D': 16, 'E': 29}
    print(f"Частоты символов: {freqs}")
    codes, root = huffman_code(freqs)
    print(f"Коды: {codes}")
    print("Дерево Хаффмана:")
    print_huffman_tree(root)

    print("\n4. Задача о сдаче:")
    coins = [50, 10, 5, 1]
    amount = 66
    print(f"Монеты: {coins}, Сумма: {amount}")
    print(f"Минимальные монеты: {min_coins(amount, coins)}")

    print("\n" + "=" * 50)
    print("\n5. Сравнение рюкзаков:")
    compare_knapsack()

    print("\n" + "=" * 50)
    print("Анализ производитлеьности")
    print("=" * 50)
    sizes, times = perf_huffman()
    print(f"Размеры алфавита: {sizes}")
    print(f"Время выполнения (сек): {times}")


if __name__ == "__main__":
    main()

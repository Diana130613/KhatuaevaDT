from typing import Dict
import time
import matplotlib.pyplot as plt

# Для отслеживания количества вызовов
call_count = 0
memo_call_count = 0

# Характеристики ПК
pc_info = """
    Характеристики ПК для тестирования:
    - Процессор: 12th Gen Intel(R) Core(TM) i5-12450H
    - Оперативная память: 16 GB DDR4
    - ОС: Windows 10
    - Python: 3.12.10
    """
print(pc_info)


def fibonacci_naive(n: int) -> int:
    """
    Вычисление n-го числа Фибоначчи (наивная версия)
    """
    global call_count
    call_count += 1

    if n < 0:
        raise ValueError("Число Фибоначчи для отрицательного индекса не определено")

    # Базовые случаи
    if n == 0:
        return 0
    if n == 1:
        return 1

    # Рекурсивный шаг
    return fibonacci_naive(n - 1) + fibonacci_naive(n - 2)


def fibonacci_memoized(n: int, memo: Dict[int, int] = None) -> int:
    """
    Оптимизированная версия с мемоизацией
    Временная сложность: O(n)
    """
    global memo_call_count
    memo_call_count += 1

    if n < 0:
        raise ValueError("Число Фибоначчи для отрицательного индекса не определено")

    if memo is None:
        memo = {0: 0, 1: 1}

    # Если результат уже вычислен, возвращаем его
    if n in memo:
        return memo[n]

    # Вычисляем и сохраняем результат
    memo[n] = fibonacci_memoized(n - 1, memo) + fibonacci_memoized(n - 2, memo)
    return memo[n]


def compare_fibonacci():
    """Сравнение производительности наивной и мемоизированной версий"""
    n = 35

    print("=" * 60)
    print("Сравнение вычисления числа Фибоначчи для n = 35")
    print("=" * 60)

    # Сброс счётчиков
    global call_count, memo_call_count
    call_count = 0
    memo_call_count = 0

    # Наивная версия
    start_time = time.time()
    result_naive = fibonacci_naive(n)
    naive_time = time.time() - start_time

    print("\nНаивная рекурсия:")
    print(f"  Результат: {result_naive}")
    print(f"  Время выполнения: {naive_time:.6f} секунд")
    print(f"  Количество рекурсивных вызовов: {call_count:,}")

    # Мемоизированная версия
    start_time = time.time()
    result_memo = fibonacci_memoized(n)
    memo_time = time.time() - start_time

    print("\nС мемоизацией:")
    print(f"  Результат: {result_memo}")
    print(f"  Время выполнения: {memo_time:.6f} секунд")
    print(f"  Количество рекурсивных вызовов: {memo_call_count:,}")

    print(f"Сокращение вызовов: {call_count/memo_call_count:.2f} раз")


# Экспериментальное исследование для разных n
def measure_for_different_n():
    """Замер времени выполнения для разных n"""
    test_values = list(range(1, 36))

    # Массивы для хранения временных метрик
    times_naive = []
    times_memoized = []

    for n in test_values:
        # Наивная версия
        start_time = time.time()
        fibonacci_naive(n)
        naive_time = time.time() - start_time

        # Версия с мемоизацией
        start_time = time.time()
        fibonacci_memoized(n)
        memo_time = time.time() - start_time

        # Добавляем результаты измерений
        times_naive.append(naive_time)
        times_memoized.append(memo_time)

    # Строим график
    plt.figure(figsize=(10, 6))
    plt.plot(test_values, times_naive, 'ro-', label='Наивная версия')
    plt.plot(test_values, times_memoized, 'bo-', label='Версия с мемоизацией')

    plt.title('Сравнение времени выполнения рекурсии Фибоначчи')
    plt.xlabel('Значение n')
    plt.ylabel('Время выполнения (секунды)')
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    compare_fibonacci()
    measure_for_different_n()

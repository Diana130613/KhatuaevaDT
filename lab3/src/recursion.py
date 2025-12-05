def factorial(n: int) -> int:
    """
    Вычисление факториала числа n рекурсивным способом

    Временная сложность: O(n)
    Глубина рекурсии: O(n)
    """
    if n < 0:
        raise ValueError("Факториал отрицательного числа не определен.")

    # Базовый случай
    if n == 0 or n == 1:
        return 1

    # Рекурсивный шаг
    return n * factorial(n - 1)


def fibonacci(n: int) -> int:
    """
    Вычисление n-го числа Фибоначчи (наивная версия)

    Последовательность Фибоначчи:
    F(0) = 0, F(1) = 1
    F(n) = F(n-1) + F(n-2) для n > 1

    Временная сложность: O(2^n) - экспоненциальная
    Глубина рекурсии: O(n)
    """
    if n < 0:
        raise ValueError("Число Фибоначчи для отрицательного индекса не определено")

    # Базовые случаи
    if n == 0:
        return 0
    if n == 1:
        return 1

    # Рекурсивный шаг
    return fibonacci(n - 1) + fibonacci(n - 2)


def fast_power(a: float, n: int) -> float:
    """
    Быстрое возведение числа a в степень n через степень двойки

    Использует свойство:
    a^n = (a^(n/2))^2, если n - чётное
    a^n = a * a^(n-1), если n - нечётное

    Временная сложность: O(log n)
    Глубина рекурсии: O(log n)
    """
    if n < 0:
        return fast_power(1 / a, -n)

    # Базовый случай
    if n == 0:
        return 1
    if n == 1:
        return a

    # Рекурсивный шаг
    if n % 2 == 0:  # n - чётное
        half_power = fast_power(a, n // 2)
        return half_power * half_power
    else:  # n - нечётное
        return a * fast_power(a, n - 1)

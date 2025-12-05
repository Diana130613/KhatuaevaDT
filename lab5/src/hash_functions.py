from typing import Union


def simple(key: Union[str, bytes], table_size: int) -> int:
    """
    Простейшая хеш-функция: сумма ASCII-кодов символов.

    Сложность: O(n), где n - длина ключа

    Args:
        key: Входная строка или байты для хеширования
        table_size: Размер хеш-таблицы (модуль для конечного результата)

    Returns:
        Хеш-значение в диапазоне [0, table_size-1]
    """
    hash_value = 0  # Инициализация результата
    for _ in key:
        hash_value += ord(_)  # Прибавляем ASCII-код символа
    return hash_value % table_size


def poly(key: Union[str, bytes],
         table_size: int,
         base: int = 31,
         large_prime: int = 10**9 + 7) -> int:
    """
    Полиномиальная rolling hash функция.

    Принцип работы:
    Каждый символ умножается на основание в определённой степени:
    h = s[0]*b^0 + s[1]*b^1 + s[2]*b^2 + ... + s[n-1]*b^(n-1)

    Сложность: O(n), где n - длина ключа

    Args:
        key: Входная строка для хеширования
        table_size: Размер хеш-таблицы
        base: Основание полинома
        large_prime: Большое простое число для предотвращения переполнения

    Returns:
        Хеш-значение в диапазоне [0, table_size-1]
    """
    hash_value = 0  # Инициализация полиномиального хеша
    power = 1       # Текущая степень основания (base^0 = 1)

    for _ in key:
        hash_value = (hash_value + ord(_) * power) % large_prime

        # Обновляем степень для следующего символа
        power = (power * base) % large_prime

    return hash_value % table_size


def djb2(key: Union[str, bytes], table_size: int) -> int:
    """
    Хеш-функция DJB2

    Сложность: O(n), где n - длина ключа

    Args:
        key: Входная строка для хеширования
        table_size: Размер хеш-таблицы

    Returns:
        Хеш-значение в диапазоне [0, table_size-1]
    """
    # Инициализация магическим числом 5381
    hash_value = 5381

    for char in key:
        # Берём модуль на каждом шаге для предотвращения переполнения
        hash_value = ((hash_value << 5) + hash_value + ord(char)) % table_size

    return hash_value

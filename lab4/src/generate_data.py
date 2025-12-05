# generate_data.py
import random
from typing import List, Dict


def generate_random_array(size: int, min_val: int = 0, max_val: int = 10000) -> List[int]:
    """Генерация массива со случайными числами"""
    return [random.randint(min_val, max_val) for _ in range(size)]


def generate_sorted_array(size: int, min_val: int = 0, max_val: int = 10000) -> List[int]:
    """Генерация отсортированного массива"""
    return sorted(generate_random_array(size, min_val, max_val))


def generate_reversed_array(size: int, min_val: int = 0, max_val: int = 10000) -> List[int]:
    """Генерация массива, отсортированного в обратном порядке"""
    return list(reversed(generate_sorted_array(size, min_val, max_val)))


def generate_almost_sorted_array(size: int, sorted_percentage: float = 0.95,
                                 min_val: int = 0, max_val: int = 10000) -> List[int]:
    """
    Генерация почти отсортированного массива

    Args:
        size: размер массива
        sorted_percentage: процент отсортированных элементов (от 0 до 1)
        min_val: минимальное значение
        max_val: максимальное значение
    """
    # Сначала создаём отсортированный массив
    arr = generate_sorted_array(size, min_val, max_val)

    # Вычисляем количество элементов для перемешивания
    num_to_shuffle = int(size * (1 - sorted_percentage))

    # Выбираем случайные индексы для перемешивания
    if num_to_shuffle > 0:
        indices_to_shuffle = random.sample(range(size), num_to_shuffle)

        # Для выбранных индексов генерируем новые случайные значения
        for ind in indices_to_shuffle:
            arr[ind] = random.randint(min_val, max_val)

    return arr


def generate_all_test_arrays(sizes: List[int] = None) -> Dict[str, Dict[int, List[int]]]:
    """
    Генерация всех тестовых массивов для разных размеров и типов

    Returns:
        'название': {размер[содержание]}
    """
    if sizes is None:
        sizes = [100, 1000, 5000, 10000]

    test_data = {
        'random': {},
        'sorted': {},
        'reversed': {},
        'almost_sorted': {}
    }

    for size in sizes:
        print(f"Генерация массивов размера {size}...")

        test_data['random'][size] = generate_random_array(size)
        test_data['sorted'][size] = generate_sorted_array(size)
        test_data['reversed'][size] = generate_reversed_array(size)
        test_data['almost_sorted'][size] = generate_almost_sorted_array(
            size, sorted_percentage=0.95
        )
    return test_data

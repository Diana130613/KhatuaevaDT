import random
import time
import matplotlib.pyplot as plt
from hash_table_chaining import HashTableChaining
from hash_table_open_addressing import HashTableOpenAddressing
from hash_functions import simple, poly, djb2


def perf_test():
    """Анализирует производительность при разных значениях коэффициента заполнения"""
    load_factors = [0.1, 0.5, 0.7]  # Различные коэффициенты заполнения
    results = {
        'Chain': [],
        'Lin': [],
        'Dbl': []
    }

    # Генерация случайных ключей
    keys = [f"key_{i}" for i in range(10000)]
    random.shuffle(keys)

    for load_factor in load_factors:
        num_elements = int(load_factor * 10007)

        tables = {
            'Chain': HashTableChaining(10007),
            'Lin': HashTableOpenAddressing(10007, 'lin'),
            'Dbl': HashTableOpenAddressing(10007, 'dbl')
        }

        for name, table in tables.items():
            insert_time = search_time = 0

            # Измерение времени вставки
            start_time = time.time()
            for key in keys[:num_elements]:
                try:
                    table.ins(key, f"value_{key}")
                except Exception:
                    break  # Если таблица переполнена
            insert_time = time.time() - start_time

            # Измерение времени поиска
            start_time = time.time()
            for key in keys[:num_elements]:
                table.srch(key)
            search_time = time.time() - start_time

            results[name].append((insert_time, search_time))

    # График времени вставки
    fig, ax = plt.subplots(figsize=(10, 6))
    for method in results.keys():
        insertion_times = [t[0] for t in results[method]]
        ax.plot(load_factors, insertion_times, marker='o', label=f"{method} Insert")

    ax.set_xlabel('Коэффициент заполнения (α)')
    ax.set_ylabel('Время (секунды)')
    ax.set_title('Время вставки vs Коэффициент заполнения')
    ax.legend()
    ax.grid(True)
    plt.savefig('perf_insert.png', dpi=150, bbox_inches='tight')
    plt.close()

    # График времени поиска
    fig, ax = plt.subplots(figsize=(10, 6))
    for method in results.keys():
        search_times = [t[1] for t in results[method]]
        ax.plot(load_factors, search_times, marker='s', label=f"{method} Search")

    ax.set_xlabel('Коэффициент заполнения (α)')
    ax.set_ylabel('Время (секунды)')
    ax.set_title('Время поиска vs Коэффициент заполнения')
    ax.legend()
    ax.grid(True)
    plt.savefig('perf_search.png', dpi=150, bbox_inches='tight')
    plt.close()


def col_test():
    """Исследует распределение коллизий для разных хэш-функций"""
    hashes = {}
    size = 1009
    keys = [f"test_{i}" for i in range(10000)]

    # Применяем каждую хэш-функцию ко всем ключам
    hashes['Simple'] = [simple(k, size) for k in keys]
    hashes['Poly'] = [poly(k, size) for k in keys]
    hashes['DJB2'] = [djb2(k, size) for k in keys]

    # Используем улучшенные гистограммы
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']  # Разные цвета для каждой функции
    
    for i, (name, values) in enumerate(hashes.items()):
        # Создаем гистограмму с разумным количеством бинов
        n_bins = min(50, size)  # Не более 50 бинов
        axes[i].hist(values, bins=n_bins, alpha=0.7, edgecolor='black', 
                    color=colors[i], density=False)
        
        axes[i].set_title(f'{name} Hash Distribution')
        axes[i].set_xlabel('Hash Value')
        axes[i].set_ylabel('Count')
        axes[i].grid(True, alpha=0.3)
        
        # Добавляем статистическую информацию
        unique_hashes = len(set(values))
        collision_rate = (len(values) - unique_hashes) / len(values) * 100
        axes[i].text(0.05, 0.95, f'Уникальных: {unique_hashes}\nКоллизии: {collision_rate:.1f}%',
                    transform=axes[i].transAxes, fontsize=9,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig('hash_dist.png', dpi=150, bbox_inches='tight')
    plt.close()


def col_hc():
    """Сравнивает методы разрешения коллизий по количеству столкновений"""
    chain_ht = HashTableChaining(1009)
    lin_ht = HashTableOpenAddressing(1009, 'lin')
    dbl_ht = HashTableOpenAddressing(1009, 'dbl')

    methods = {'Chaining': chain_ht, 'Linear': lin_ht, 'Double': dbl_ht}
    keys = [f"key_{i}" for i in range(700)]  # α ≈ 0.69

    collisions = []
    for name, ht in methods.items():
        for key in keys:
            try:
                ht.ins(key, key)
            except Exception:
                break
        collisions.append(ht.col)

    names = ['Chaining', 'Linear', 'Double']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    # Создаем столбчатую диаграмму
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(names, collisions, color=colors, alpha=0.7, edgecolor='black')

    # Добавляем значения на столбцы
    for bar, col in zip(bars, collisions):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                str(col), ha='center', va='bottom', fontsize=12, fontweight='bold')

    ax.set_ylabel('Количество коллизий')
    ax.set_title('Сравнение количества коллизий (α ≈ 0.69)')
    ax.grid(True, alpha=0.3, axis='y')
    plt.savefig('collisions_compare.png', dpi=150, bbox_inches='tight')
    plt.close()


def run_all():
    """Запуск полного набора тестов и анализа производительности"""
    import unittest
    from tests import TestHashTables

    # Запуск unit-тестов
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestHashTables)
    runner = unittest.TextTestRunner(verbosity=1)
    result = runner.run(suite)

    if result.wasSuccessful():
        print("Unit-тесты пройдены успешно. Запуск анализа производительности...")

        # Запуск анализа производительности
        perf_test()
        col_test()
        col_hc()

        print("\nАнализ производительности завершен.")
        print("Созданные графики:")
        print("1. perf_insert.png - время вставки")
        print("2. perf_search.png - время поиска")
        print("3. hash_dist.png - распределение хешей")
        print("4. collisions_compare.png - сравнение коллизий")
    else:
        print("Unit-тесты не пройдены. Анализ производительности пропущен.")

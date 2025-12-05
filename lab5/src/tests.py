import unittest
from hash_table_chaining import HashTableChaining
from hash_table_open_addressing import HashTableOpenAddressing


class TestHashTables(unittest.TestCase):
    """Набор тестов для хеш-таблиц с разными методами разрешения коллизий."""

    def test_chaining_basic(self) -> None:
        """Базовый тест вставки и поиска для метода цепочек."""
        ht = HashTableChaining(101)  # Создаем таблицу с методом цепочек
        ht.ins('a', 1)  # Вставляем первый элемент
        ht.ins('b', 2)  # Вставляем второй элемент
        self.assertEqual(ht.srch('a'), 1)  # Проверяем первый поиск
        self.assertEqual(ht.srch('b'), 2)  # Проверяем второй поиск

    def test_chaining_update(self) -> None:
        """Тест обновления существующего ключа для метода цепочек."""
        ht = HashTableChaining(101)  # Создаем таблицу
        ht.ins('x', 10)  # Первая вставка
        ht.ins('x', 20)  # Обновляем значение
        self.assertEqual(ht.srch('x'), 20)  # Проверяем, что значение обновилось

    def test_chaining_delete(self) -> None:
        """Тест удаления элемента для метода цепочек."""
        ht = HashTableChaining(101)  # Создаем таблицу
        ht.ins('y', 99)
        self.assertTrue(ht.rm('y'))  # Удаляем и проверяем
        self.assertIsNone(ht.srch('y'))  # Проверяем, что элемента больше нет

    def test_chaining_resize(self) -> None:
        """Тест автоматического масштабирования таблицы для метода цепочек."""
        ht = HashTableChaining(11)  # Маленькая таблица
        for i in range(10):
            ht.ins(f'k{i}', i)  # Добавляем 10 элементов
        self.assertGreater(ht.sz, 11)  # Проверяем, что размер увеличился

    def test_linear_basic(self) -> None:
        """Базовый тест для линейного пробирования."""
        ht = HashTableOpenAddressing(101, 'lin')  # Таблица с лин.пробированием
        ht.ins('b', 2)
        ht.ins('d', 6)
        self.assertEqual(ht.srch('b'), 2)  # Проверяем первый поиск
        self.assertEqual(ht.srch('d'), 6)  # Проверяем второй поиск

    def test_linear_delete(self) -> None:
        """Тест удаления для линейного пробирования."""
        ht = HashTableOpenAddressing(101, 'lin')
        ht.ins('r', 7)
        self.assertTrue(ht.rm('r'))  # Удаляем и проверяем
        self.assertIsNone(ht.srch('r'))  # Проверяем, что элемента нет

    def test_double_basic(self) -> None:
        """Базовый тест для двойного хеширования."""
        ht = HashTableOpenAddressing(101, 'dbl')  # Таблица с двойным хеш-м
        ht.ins('y', 2)
        ht.ins('r', 5)
        self.assertEqual(ht.srch('y'), 2)  # Проверяем первый поиск
        self.assertEqual(ht.srch('r'), 5)  # Проверяем второй поиск

    def test_double_delete(self) -> None:
        """Тест удаления для двойного хеширования."""
        ht = HashTableOpenAddressing(101, 'dbl')
        ht.ins('u', 10)
        self.assertTrue(ht.rm('u'))  # Удаляем и проверяем успех
        self.assertIsNone(ht.srch('u'))  # Проверяем, что элемента нет

    def test_collisions(self) -> None:
        """Тест обработки коллизий для метода цепочек."""
        ht = HashTableChaining(101)
        keys = [f'k{i}' for i in range(20)]  # Генерируем 20 ключей
        for i, k in enumerate(keys):
            ht.ins(k, i)  # Вставляем пару
        for i, k in enumerate(keys):  # Проверяем все элементы
            self.assertEqual(ht.srch(k), i)  # Проверяем значение

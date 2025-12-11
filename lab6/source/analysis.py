import sys
import time
import random
import matplotlib.pyplot as plt
import unittest
from binary_search_tree import BinarySearchTree
from tree_traversal import inorder_recursive, inorder_iterative

sys.setrecursionlimit(10000)  # Добавьте эту строку в начале файла


class TestBST(unittest.TestCase):

    def test_ins_srch(self):
        """Тест вставки и поиска"""
        bst = BinarySearchTree()
        bst.insert(5)
        bst.insert(3)
        bst.insert(7)
        self.assertEqual(bst.search(3), 3)
        self.assertEqual(bst.search(7), 7)
        self.assertIsNone(bst.search(10))

    def test_delete(self):
        """Тест удаления"""
        bst = BinarySearchTree()
        for v in [5, 3, 7, 1, 4]:
            bst.insert(v)
        bst.delete(3)
        self.assertIsNone(bst.search(3))

    def test_find_min_max(self):
        """Тест поиска минимума и максимума"""
        bst = BinarySearchTree()
        for v in [5, 3, 7, 1, 9]:
            bst.insert(v)
        self.assertEqual(bst.find_min(), 1)
        self.assertEqual(bst.find_max(), 9)

    def test_is_valid(self):
        """Тест проверки корректности BinarySearchTree"""
        bst = BinarySearchTree()
        for v in [5, 3, 7]:
            bst.insert(v)
        self.assertTrue(bst.is_valid())

    def test_inord(self):
        """Тест in-order обхода"""
        bst = BinarySearchTree()
        for v in [5, 3, 7, 1, 4]:
            bst.insert(v)
        res = inorder_recursive(bst)
        self.assertEqual(res, [1, 3, 4, 5, 7])

    def test_height(self):
        """Тест вычисления высоты"""
        bst = BinarySearchTree()
        bst.insert(5)
        bst.insert(3)
        bst.insert(7)
        self.assertEqual(bst.height(), 2)

    def test_inord_iter(self):
        """Тест итеративного in-order обхода"""
        bst = BinarySearchTree()
        for v in [5, 3, 7, 1, 4]:
            bst.insert(v)
        res = inorder_iterative(bst)
        self.assertEqual(res, [1, 3, 4, 5, 7])


def perf_test():
    """Анализ производительности. O(n*m) где n - размер, m - операции"""
    sizes = [100, 500, 1000, 2000]
    operations_per_size = 1000
    balanced_times = []  # Времена для сбалансированного дерева
    degenerate_times = []  # Времена для вырожденного дерева

    for size in sizes:
        balanced_bst = BinarySearchTree()
        degenerate_bst = BinarySearchTree()
        values = list(range(1, size + 1))
        random.shuffle(values)  # Перемешиваем для сбалансированного
        for v in values:
            balanced_bst.insert(v)  # Вставляем значение
        for v in range(1, size + 1):
            degenerate_bst.insert(v)
        search_values = [random.randint(1, size) for _ in range(operations_per_size)]
        balanced_bst.operation_count = 0
        t0 = time.time()
        for v in search_values:
            balanced_bst.search(v)
        balanced_time = time.time() - t0  # Время поиска в сбалансированном

        degenerate_bst.operation_count = 0
        t0 = time.time()
        for v in search_values:
            degenerate_bst.search(v)
        degenerate_time = time.time() - t0  # Время поиска в вырожденном

        balanced_times.append(balanced_time)
        degenerate_times .append(degenerate_time)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(sizes, balanced_times, marker='o', label='Balanced BST', linewidth=2)
    ax.plot(sizes, degenerate_times, marker='s', label='Degenerate BST', linewidth=2)
    ax.set_xlabel('Размер дерева (n)')
    ax.set_ylabel('Время поиска (s)')
    ax.set_title('Поиск: Сбалансированное и Вырожденное BST')
    ax.legend()
    ax.grid(True)
    plt.savefig('perf_bst.png', dpi=150)
    plt.close()


# Характеристики ПК
pc_info = """
Характеристики ПК для тестирования:
- Процессор: 12th Gen Intel(R) Core(TM) i5-12450H
- Оперативная память: 16 GB DDR4
- ОС: Windows 10
- Python: 3.12.10
"""
print(pc_info)


def run_all():
    suite = unittest.TestLoader().loadTestsFromTestCase(TestBST)
    runner = unittest.TextTestRunner(verbosity=1)
    result = runner.run(suite)
    if result.wasSuccessful():
        perf_test()  # Анализ производительности


if __name__ == "__main__":
    run_all()

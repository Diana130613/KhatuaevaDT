class TreeNode:
    """Node class для бинарного дерева поиска"""
    def __init__(self, value):  # Конструктор узла. O(1)
        self.value = value  # Значение узла
        self.left = None  # Левое поддерево
        self.right = None  # Правое поддерево
        self.height = 1  # Высота узла


class BinarySearchTree:
    """Бинарное дерево поиска.
    Сложность: средняя O(log n), худшая O(n)"""

    def __init__(self):  # Инициализация пустого дерева. O(1)
        self.root = None  # Корневой узел
        self.operation_count = 0  # Счетчик операций

    def insert(self, value):
        """Вставка.
        Сложность: средняя O(log n), худшая O(n)"""
        if self.root is None:  # Если дерево пусто
            self.root = TreeNode(value)  # Создаём корень
            return
        self._insert_recursive(self.root, value)

    def _insert_recursive(self, node, value):
        """Рекурсивная вставка.
        Сложность: средняя O(log n), худшая O(n)"""
        self.operation_count += 1  # Увеличиваем счётчик операций
        if value < node.value:
            if node.left is None:  # Если левого поддерева нет
                node.left = TreeNode(value)
            else:
                self._insert_recursive(node.left, value)
        elif value > node.value:  # Если значение больше текущего узла
            if node.right is None:
                node.right = TreeNode(value)  # Создём правый узел
            else:
                self._insert_recursive(node.right, value)
        self._update_height(node)

    def search(self, value):
        return self._search_recursive(self.root, value)

    def _search_recursive(self, node, value):
        """Рекурсивный поиск.
        Сложность: O(log n), O(n) худшая"""
        self.operation_count += 1
        if node is None:
            return None
        if value == node.value:
            return node.value
        if value < node.value:
            return self._search_recursive(node.left, value)  # Ищем слева
        return self._search_recursive(node.right, value)  # Ищем справа

    def delete(self, value):
        """Удаление. Средняя O(log n), худшая O(n)"""
        self.root = self._delete_recursive(self.root, value)

    def _delete_recursive(self, node, value):
        """Рекурсивное удаление. O(log n) средняя, O(n) худшая"""
        self.operation_count += 1
        if node is None:
            return None
        if value < node.value:  # Если значение меньше
            node.left = self._delete_recursive(node.left, value)  # Рекурсивно удаляем слева
        elif value > node.value:  # Если значение больше
            node.right = self._delete_recursive(node.right, value)  # Рекурсивно удаляем справа
        else:  # Если найден узел для удаления
            if node.left is None:  # Если левого потомка нет
                return node.right
            if node.right is None:  # Если правого потомка нет
                return node.left
            min_value = self._find_min_recursive(node.right)  # Находим минимум в правом поддереве
            node.value = min_value
            node.right = self._delete_recursive(node.right, min_value)
        if node:
            self._update_height(node)
        return node

    def find_min(self, node=None):
        """Поиск минимума.
        O(log n) средняя, O(n) худшая"""
        if self.root is None:
            return None
        return self._find_min_recursive(self.root)

    def _find_min_recursive(self, node):
        """Рекурсивный поиск минимума.
        O(log n) средняя, O(n) худшая"""
        self.operation_count += 1
        if node.left is None:  # Если левого потомка нет
            return node.value
        return self._find_min_recursive(node.left)

    def find_max(self):
        """Поиск максимума.
        O(log n) средняя, O(n) худшая"""
        if self.root is None:
            return None
        return self._find_max_recursive(self.root)

    def _find_max_recursive(self, node):
        """Рекурсивный поиск максимума.
        O(log n) средняя, O(n) худшая"""
        self.operation_count += 1
        if node.right is None:  # Если правого потомка нет
            return node.value  # Возвращаем значение узла
        return self._find_max_recursive(node.right)

    def is_valid(self):
        """Проверка корректности BST. O(n)"""
        def _check(node, min, max):  # Проверка с границами, O(n)
            if node is None:
                return True
            if node.value <= min or node.value >= max:  # Если значение вне границ
                return False
            return _check(node.left, min, node.value) and _check(node.right, node.value, max)
        return _check(self.root, float('-inf'), float('inf'))

    def height(self):
        """Получение высоты дерева. O(1)"""
        if self.root is None:
            return 0
        return self.root.height

    def _update_height(self, node):
        """Обновление высоты узла. O(1)"""
        if node is None:
            return
        left_height = node.left.height if node.left else 0
        right_height = node.right.height if node.right else 0
        node.height = 1 + max(left_height, right_height)

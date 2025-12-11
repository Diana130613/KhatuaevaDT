from binary_search_tree import BinarySearchTree


def inorder_recursive(bst):
    """In-order рекурсивный обход (левый-корень-правый). O(n)"""
    res = []

    def _traverse(node):
        if node is None:
            return
        _traverse(node.left)  # Обход левого поддерева
        res.append(node.value)
        _traverse(node.right)  # Обход правого поддерева
    _traverse(bst.root)  # Начинаем обход с корня
    return res


def preorder_recursive(bst):
    """Pre-order рекурсивный обход (корень-левый-правый). O(n)"""
    res = []

    def _traverse(node):  # Рекурсивный обход. O(n)
        if node is None:
            return
        res.append(node.value)
        _traverse(node.left)  # Обход левого поддерева
        _traverse(node.right)  # Обход правого поддерева
    _traverse(bst.root)  # Начинаем обход с корня
    return res


def postorder_recursive(bst):
    """Post-order рекурсивный обход (левый-правый-корень). O(n)"""
    res = []

    def _traverse(node):  # Рекурсивный обход. O(n)
        if node is None:
            return
        _traverse(node.left)  # Обход левого поддерева
        _traverse(node.right)  # Обход правого поддерева
        res.append(node.value)
    _traverse(bst.root)  # Начинаем обход с корня
    return res


def inorder_iterative(bst):
    """In-order итеративный обход (левый-корень-правый). O(n)"""
    res = []
    stack = []  # Стек для обхода
    current = bst.root
    while current or stack:
        while current:
            stack.append(current)
            current = current.left  # Переходим влево
        current = stack.pop()  # Извлекаем узел из стека
        res.append(current.value)
        current = current.right  # Переходим вправо
    return res


if __name__ == "__main__":
    bst = BinarySearchTree()
    values = [5, 3, 7, 2, 4, 6, 8]
    for v in values:
        bst.insert(v)

    print("In-order (рекурсивный):", inorder_recursive(bst))
    print("Pre-order (рекурсивный):", preorder_recursive(bst))
    print("Post-order (рекурсивный):", postorder_recursive(bst))
    print("In-order (итеративный):", inorder_iterative(bst))

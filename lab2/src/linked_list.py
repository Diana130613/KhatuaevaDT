class Node:
    """Класс для представления узла связного списка"""
    def __init__(self, value):
        self.value = value  # данные узла
        self.next = None  # ссылка на следующий узел


class LinkedList:
    """Класс для представления связного списка"""
    def __init__(self):
        self.head = None  # начало списка
        self.tail = None  # конец списка

    def is_empty(self):
        """Проверка на пустоту списка - O(1)"""
        return self.head is None

    def insert_at_start(self, value):
        """Вставка в начало списка - O(1)"""
        # Новый узел
        new_node = Node(value)

        if self.is_empty():
            # Если список пустой, новый узел становится и головой и хвостом
            self.head = new_node
            self.tail = new_node
        else:
            # Новый узел становится головой, next указывает на старую голову
            new_node.next = self.head
            self.head = new_node

    def insert_at_end(self, value):
        """Вставка в конец списка - O(1), (tail)"""
        new_node = Node(value)

        if self.is_empty():
            self.head = new_node
            self.tail = new_node
        else:
            # Текущий хвост указывает на новый узел
            self.tail.next = new_node
            # Новый узел становится хвостом
            self.tail = new_node

    def delete_from_start(self):
        """Удаление из начала списка - O(1)"""
        if self.is_empty():
            return None

        # Сохраняем данные удаляемого узла
        deleted_value = self.head.value

        if self.head == self.tail:
            # Если в списке только один элемент
            self.head = None
            self.tail = None
        else:
            # Головой становится следующий узел
            self.head = self.head.next

        return deleted_value

    def traversal(self):
        """Обход списка - O(n)"""
        current = self.head
        elements = []
        while current is not None:
            elements.append(current.value)
            current = current.next
        return elements

    def display(self):
        """Вывод списка для визуализации - O(n)"""
        elements = self.traversal()
        print(" -> ".join(map(str, elements)) if elements else "Пустой список")

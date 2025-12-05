import collections
from linked_list import LinkedList


def is_balanced_parentheses(expression):
    """
    Проверка сбалансированности скобок с использованием стека

    Сложность: O(n), где n - длина строки
    Используется стек (реализованный на list), так как нужен LIFO
    """
    stack = []
    matching_brackets = {')': '(', '}': '{', ']': '['}

    for char in expression:
        if char in '({[':
            # Открывающая скобка - добавляем в стек
            stack.append(char)
        elif char in ')}]':
            # Закрывающая скобка - проверяем соответствие
            if not stack or stack[-1] != matching_brackets[char]:
                return False
            stack.pop()

    # Если стек пуст - все скобки сбалансированы
    return len(stack) == 0


def simulate_print_queue():
    """
    Симуляция очереди печати

    Сложность: O(n) для n задач
    Используется deque, так как нужна FIFO с эффективным удалением из начала
    """
    print_queue = collections.deque()

    # Добавляем задачи в очередь
    tasks = ['Документ1.pdf', 'Отчет.docx', 'Презентация.pptx', 'Фото.jpg']
    for task in tasks:
        print_queue.append(task)
        print(f"Добавлена задача: {task}")

    print(f"\nОчередь печати: {list(print_queue)}")

    # Обрабатываем задачи
    print("\nОбработка задач:")
    while print_queue:
        current_task = print_queue.popleft()
        print(f"Печатается: {current_task}")
        print(f"Осталось задач: {len(print_queue)}")

    print("Все задачи выполнены!")


def is_palindrome(sequence):
    """
    Проверка, является ли последовательность палиндромом

    Сложность: O(n)
    Используется deque, так как нужен эффективный доступ к обоим концам
    """
    deq = collections.deque(sequence)

    while len(deq) > 1:
        # Сравниваем первый и последний элементы
        if deq.popleft() != deq.pop():
            return False

    return True


def demonstrate_linked_list():
    """Демонстрация работы связного списка"""
    print("\n=== Демонстрация LinkedList ===")
    ll = LinkedList()

    # Вставка в начало
    print("Вставка в начало: 1, 2, 3")
    ll.insert_at_start(1)
    ll.insert_at_start(2)
    ll.insert_at_start(3)
    ll.display()  # 3 -> 2 -> 1

    # Вставка в конец
    print("Вставка в конец: 4, 5")
    ll.insert_at_end(4)
    ll.insert_at_end(5)
    ll.display()  # 3 -> 2 -> 1 -> 4 -> 5

    # Удаление из начала
    print(f"Удален из начала: {ll.delete_from_start()}")
    ll.display()  # 2 -> 1 -> 4 -> 5


# Тестирование функций
if __name__ == "__main__":
    # Тест проверки скобок
    print("=== Проверка сбалансированности скобок ===")
    test_cases = [
        "()",  # True
        "()[]{}",  # True
        "([{}])",  # True
        "([)]",  # False
        "((())",  # False
    ]

    for test in test_cases:
        result = is_balanced_parentheses(test)
        print(f"'{test}': {result}")

    # Тест палиндромов
    print("\n=== Проверка палиндромов ===")
    test_sequences = [
        "радар",
        "level",
        "мадам",
        "hello",
        [1, 2, 3, 2, 1]
    ]

    for test in test_sequences:
        result = is_palindrome(test)
        print(f"'{test}': {result}")

    # Демонстрация очереди печати
    print("\n=== Симуляция очереди печати ===")
    simulate_print_queue()

    # Демонстрация связного списка
    demonstrate_linked_list()

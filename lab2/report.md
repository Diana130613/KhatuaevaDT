# Отчет по лабораторной работе 2
# Основные структуры данных. Анализ и применение

**Дата:** 2025-11-01  
**Семестр:** 3 курс 1 полугодие - 5 семестр  
**Группа:** ПИЖ-б-о-23-2(1)  
**Дисциплина:** Анализ сложности алгоритмов  
**Студент:** Хатуаева Дайана Тныбековна

## Цель работы
Изучить понятие и особенности базовых абстрактных типов данных (стек, очередь, дек, связный список) и их реализаций в Python. Научиться выбирать оптимальную структуру данных для решения конкретной задачи, основываясь на анализе теоретической и практической сложности операций. Получить навыки измерения производительности и применения структур данных для решения практических задач

## Теоретическая часть
- Список (list) в Python: Реализация динамического массива. Обеспечивает амортизированное время O(1) для добавления в конец (append). Вставка и удаление в середину имеют сложность O(n) из-за сдвига элементов. Доступ по индексу - O(1).
- Связный список (Linked List): Абстрактная структура данных, состоящая из узлов, где каждый узел содержит данные и ссылку на следующий элемент. Вставка и удаление в известное место (например, начало списка) выполняются за O(1). Доступ по индексу и поиск - O(n).
- Стек (Stack): Абстрактный тип данных, работающий по принципу LIFO (Last-In-First-Out). Основные операции: push (добавление, O(1)), pop (удаление с вершины, O(1)), peek (просмотр вершины, O(1)). В Python может быть реализован на основе списка.
- Очередь (Queue): Абстрактный тип данных, работающий по принципу FIFO (First-In-First-Out). Основные операции: enqueue (добавление в конец, O(1)), dequeue (удаление из начала, O(1)). В Python для эффективной реализации используется collections.deque.
- Дек (Deque, двусторонняя очередь): Абстрактный тип данных, позволяющий добавлять и удалять элементы как в начало, так и в конец. Все основные операции - O(1). В Python реализован в классе collections.deque.

## Практическая часть

### Выполненные задачи
- [x] Задача 1: Реализовать класс LinkedList (связный список) для демонстрации принципов его работы.
- [x] Задача 2: Используя встроенные типы данных (list, collections.deque), проанализировать эффективность операций, имитирующих поведение стека, очереди и дека.
- [x] Задача 3: Провести сравнительный анализ производительности операций для разных структур данных (list vs LinkedList для вставки, list vs deque для очереди).
- [x] Задача 4: Решить 2-3 практические задачи, выбрав оптимальную структуру данных.


### Ключевые фрагменты кода

####  Класс Node и класс LinkedList
Реализованы методы: insert_at_start (O(1)), insert_at_end (O(1) с хвостом), delete_from_start (O(1)), traversal (O(n)).

```python
# linked_list.py
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
```
#### Анализ производительности (на основе встроенных структур) и Визуализация
Построение графиков зависимости времени выполнения операций от количества элементов, наглядно демонстрирующих разницу в асимптотике.

```python
# performance_analysis.py
import timeit
import random
import collections
import matplotlib.pyplot as plt
from linked_list import LinkedList


# Функция для добавления элементов в начало списка
def list_prepend(size):
    lst = []  # Создаём новый список внутри функции
    for i in range(size):
        lst.insert(0, random.randint(0, 1000))
    return lst


def linked_list_prepend(size):
    linked_list = LinkedList()  # Создаем новый linked list внутри функции
    for i in range(size):
        linked_list.insert_at_start(random.randint(0, 1000))
    return linked_list


def comparison(sizes):
    """
    Основная функция сравнения производительности
    """
    # Характеристики ПК
    pc_info = """
    Характеристики ПК для тестирования:
    - Процессор: 12th Gen Intel(R) Core(TM) i5-12450H
    - Оперативная память: 16 GB DDR4
    - ОС: Windows 10
    - Python: 3.12.10
    """
    print(pc_info)

    # Подготовка списков для хранения результатов времени
    times_list_insert = []
    times_linked_list_insert = []
    times_list_pop = []
    times_deque_pop = []

    print(
        """Замеры времени выполнения для list и
        linked_list (добавление N элементов в начало):"""
    )
    print(
        "{:>10} {:>19} {:>30}".format(
            "N", "Время (мкс) - list", "Время (мкс) - linked_list"
        )
    )

    # Замеры времени выполнения при сравнении list и linked_list
    for size in sizes:
        # Используем lambda для передачи параметра size
        time_list = timeit.timeit(lambda: list_prepend(size), number=10) * 1000 / 10
        times_list_insert.append(time_list)

        time_linked = timeit.timeit(lambda: linked_list_prepend(size), number=10) * 1000 / 10
        times_linked_list_insert.append(time_linked)

        print(f"{size:>10} {time_list:>19.4f} {time_linked:>30.4f}")

    print("\n" + "="*60)
    print(
        """Замеры времени выполнения для list и deque
        (удаление из начала N количества элементов):"""
    )
    print("{:>10} {:>19} {:>30}".format(
        "N", "Время (мкс) - list", "Время (мкс) - deque"))

    # Замеры времени выполнения при сравнении list и deque
    for size in sizes:
        # Подготовка данных для теста удаления
        def prepare_and_test_list_pop():
            lst = list(range(size))
            for _ in range(size):
                lst.pop(0) if lst else None
            return lst

        def prepare_and_test_deque_pop():
            dq = collections.deque(range(size))
            for _ in range(size):
                dq.popleft() if dq else None
            return dq

        time_list_pop = timeit.timeit(prepare_and_test_list_pop, number=10) * 1000 / 10
        times_list_pop.append(time_list_pop)

        time_deque_pop = timeit.timeit(prepare_and_test_deque_pop, number=10) * 1000 / 10
        times_deque_pop.append(time_deque_pop)

        print(f"{size:>10} {time_list_pop:>19.4f} {time_deque_pop:>30.4f}")

    # График 1: Сравнение list и linked_list (вставка в начало)
    plt.figure(figsize=(10, 6))
    plt.plot(sizes, times_list_insert, "bo-", label="list")
    plt.plot(sizes, times_linked_list_insert, "ro-", label="linked_list")
    plt.xlabel("Количество элементов (N)")
    plt.ylabel("Время выполнения (мкс)")
    plt.title(
        """Зависимость времени выполнения от количества элементов
        \n(Сравнение list и linked_list для вставки в начало)"""
    )
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        "time_complexity_plot_linked.png", dpi=300, bbox_inches="tight")
    plt.show()

    # График 2: Сравнение list и deque (удаление из начала)
    plt.figure(figsize=(10, 6))
    plt.plot(sizes, times_list_pop, "bo-", label="list")
    plt.plot(sizes, times_deque_pop, "go-", label="deque")
    plt.xlabel("Количество элементов (N)")
    plt.ylabel("Время выполнения (мкс)")
    plt.title(
        """Зависимость времени выполнения от количества элементов
        \n(Сравнение list и deque для удаления из начала)"""
    )
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig("time_complexity_plot_deque.png", dpi=300, bbox_inches="tight")
    plt.show()
```

#### Решение задач:
 Реализовать проверку сбалансированности скобок ({[()]}) с использованием стека
 (реализованного на list).
 Реализовать симуляцию обработки задач в очереди печати (использовать deque).
 Решить задачу "Палиндром" (проверка, является ли последовательность палиндромом) с
 использованием дека (deque).

```python
# task_solutions.py
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
```

### Пример работы программы
```python
# main.py
from performance_analysis import comparison
from task_solutions import (
    is_balanced_parentheses,
    simulate_print_queue,
    is_palindrome
)


if __name__ == "__main__":
    print("Лабораторная работа: Основные структуры данных")
    print("=" * 50)

    # Запуск анализа производительности
    comparison([100, 500, 1000, 2000, 5000])

    # Запуск решения задач
    print("\n" + "=" * 50)
    print("РЕШЕНИЕ ПРАКТИЧЕСКИХ ЗАДАЧ")
    print("=" * 50)

    # Тестирование всех функций из task_solutions
    import task_solutions
    task_solutions.demonstrate_linked_list()

    print("\nПроверка сбалансированности скобок:")
    test_expr = "({[]})"
    print(f"'{test_expr}': {is_balanced_parentheses(test_expr)}")

    print("\nПроверка палиндрома:")
    test_word = "радар"
    print(f"'{test_word}': {is_palindrome(test_word)}")

    print("\nСимуляция очереди печати:")
    simulate_print_queue()
```
```
Лабораторная работа: Основные структуры данных
==================================================

    Характеристики ПК для тестирования:
    - Процессор: 12th Gen Intel(R) Core(TM) i5-12450H
    - Оперативная память: 16 GB DDR4
    - ОС: Windows 10
    - Python: 3.12.10

Замеры времени выполнения для list и linked_list (добавление N элементов в начало):
         N  Время (мкс) - list      Время (мкс) - linked_list
       100              0.0315                         0.0445
       500              0.1821                         0.3218
      1000              0.7069                         0.6155
      2000              0.9991                         0.8610
      5000              3.9760                         2.1064

============================================================
Замеры времени выполнения для list и deque
        (удаление из начала N количества элементов):
         N  Время (мкс) - list            Время (мкс) - deque
       100              0.0068                         0.0035
       500              0.0284                         0.0159
      1000              0.0665                         0.0326
      2000              1.5380                         0.0682
      5000             12.3802                         0.1766


```

## Ответы на контрольные вопросы

1. В чем ключевое отличие динамического массива (list в Python) от связного списка с точки зрения сложности операций вставки в начало и доступа по индексу?

У динамического массива (list) операция Вставка в начало имеет сложность O(n), потому что при добавлении элемента в начало все остальные элементы нужно сдвинуть на одну позицию вправо.
У связного списка вставка в начало выполняется за O(1), так как достаточно создать новый узел и изменить ссылки головы списка.

В динамическом массиве доступ по индексу выполняется за O(1), так как элементы хранятся в непрерывной области памяти, и адрес вычисляется по смещению.
В связном списке доступ по индексу требует O(n), потому что нужно последовательно пройти от начала списка до нужного узла.

2. Объясните принцип работы стека (LIFO) и очереди (FIFO). Приведите по два примера их практического использования.

Стек (LIFO — Last In, First Out): элементы добавляются и извлекаются с одного конца (вершины стека).
Примеры использования:

Система отмены действий в редакторах — последнее выполненное действие отменяется первым.

Управление вызовами функций в программах — последняя вызванная функция завершается первой.

Очередь (FIFO — First In, First Out): элементы добавляются в конец и извлекаются из начала.
Примеры использования:

Очередь печати документов — первый отправленный на печать документ печатается первым.

Обработка запросов на сервере — запросы обрабатываются в порядке поступления.

3. Почему операция удаления первого элемента из списка (list) в Python имеет сложность O(n), а из дека (deque) - O(1)?

Удаление первого элемента из списка (list) имеет сложность O(n), потому что после удаления все оставшиеся элементы нужно сдвинуть на одну позицию влево для сохранения непрерывности памяти.

Удаление первого элемента из дека (deque) выполняется за O(1), так как deque реализован на основе двусвязного списка или циклического буфера, где операции с обоими концами оптимизированы и не требуют сдвига элементов.

4. Какую структуру данных вы бы выбрали для реализации системы отмены действий (undo) в текстовом редакторе? Обоснуйте свой выбор.

Для системы отмены действий в текстовом редакторе подходит стек, так как операции отмены требуют порядка LIFO: последнее выполненное действие должно отменяться первым, а также стек обеспечивает быструю вставку и удаление с вершины за O(1), что критично для мгновенного отклика интерфейса.

5. Замеры показали, что вставка 1000 элементов в начало списка заняла значительно больше времени, чем вставка в начало вашей реализации связного списка. Объясните результаты с точки зрения асимптотической сложности.

Вставка в начало списка (list) имеет сложность O(n) для каждой операции, так как требует сдвига всех существующих элементов. При вставке 1000 элементов общая сложность составит примерно O(1 + 2 + ... + n) ≈ O(n²), что приводит к значительным временным затратам.

Вставка в начало связного списка выполняется за O(1) для каждого элемента, так как не требует сдвига данных. Общая сложность для 1000 операций составит O(n), что значительно быстрее, особенно при больших n.
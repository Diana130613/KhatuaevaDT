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

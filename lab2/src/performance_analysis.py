import timeit
import random
import collections
import matplotlib.pyplot as plt
from linked_list import LinkedList


# Функция для добавления элементов в начало списка
def list_prepend(size):
    lst = []  # Создаем новый список внутри функции
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

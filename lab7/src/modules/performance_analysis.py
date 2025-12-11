import time
import random
import matplotlib.pyplot as plt
from heap import Heap
from heapsort import heapsort, heapsort_inplace
from priority_queue import PriorityQueue
from sorts import merge_sort, quick_sort


def perf_test():
    """Замеры времени: построение кучи, сортировка."""
    ns = [2000, 4000, 6000, 8000, 10000]
    times_seq = []
    times_build = []
    hs_times = []
    ms_times = []
    qs_times = []

    for n in ns:
        arr = [random.randint(1, 100000) for _ in range(n)]

        # Замер времени последовательной вставки
        t0 = time.time()
        heap = Heap(is_min=True)
        for x in arr:
            heap.insert(x)
        t1 = time.time()
        times_seq.append(t1 - t0)

        # Замер времени быстрого построения кучи
        t0 = time.time()
        h2 = Heap(is_min=True)
        h2.build_heap(arr)
        t1 = time.time()
        times_build.append(t1 - t0)

        # Замер времени вашей реализации Heapsort
        t0 = time.time()
        heapsort(arr)
        t1 = time.time()
        hs_times.append(t1 - t0)

        # Замер времени вашей реализации Mergesort
        t0 = time.time()
        merge_sort(arr)
        t1 = time.time()
        ms_times.append(t1 - t0)

        # Замер времени вашей реализации Quicksorth
        t0 = time.time()
        quick_sort(arr)
        t1 = time.time()
        qs_times.append(t1 - t0)

    # График времени построения кучи
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(ns, times_seq, marker='o', label='Insert')
    ax.plot(ns, times_build, marker='^', label='Build')
    ax.set_title('Время построения кучи')
    ax.set_xlabel('Размер')
    ax.set_ylabel('Время (с)')
    ax.legend()
    plt.savefig('heap_build.png', dpi=150)
    plt.close()

    # График времени сортировки
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(ns, hs_times, marker='o', label='Heapsort')
    ax.plot(ns, ms_times, marker='s', label='Merge Sort')
    ax.plot(ns, qs_times, marker='^', label='Quick Sort')
    ax.set_title('Сравнение сортировок')
    ax.set_xlabel('Размер')
    ax.set_ylabel('Время (с)')
    ax.legend()
    plt.savefig('sort_perf.png', dpi=150)
    plt.close()


def demo():
    arr = [12, 7, 8, 3, 5, 1, 18]
    heap = Heap(is_min=True)
    for x in arr:
        heap.insert(x)
    heap.show()
    arr2 = [5, 1, 7, 3, 2]
    print(heapsort(arr2))
    arr3 = [5, 2, 8, 3, 10, 1]
    heapsort_inplace(arr3)
    print(arr3)
    pq = PriorityQueue()
    pq.enqueue("task1", 3)
    pq.enqueue("task2", 1)
    pq.enqueue("task3", 2)
    print([pq.dequeue() for _ in range(3)])  # "task2", "task3", "task1"


if __name__ == "__main__":
    # Характеристики ПК
    pc_info = """
    Характеристики ПК для тестирования:
    - Процессор: 12th Gen Intel(R) Core(TM) i5-12450H
    - Оперативная память: 16 GB DDR4
    - ОС: Windows 10
    - Python: 3.12.10
    """
    print(pc_info)

    perf_test()  # Запускаем замер производительности
    demo()

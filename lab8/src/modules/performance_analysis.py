import time
import random
import matplotlib.pyplot as plt
from .greedy_algorithms import huffman_code


def perf_huffman():
    """
    Эксперимент по замеру времени работы алгоритма Хаффмана.
    """
    # Размеры алфавита для тестирования
    sizes = [50, 100, 500, 1000, 3000]
    times = []

    for size in sizes:
        freqs = {str(i): random.randint(10, 1000) for i in range(size)}
        t0 = time.time()
        codes, _ = huffman_code(freqs)
        t1 = time.time()
        times.append(t1 - t0)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(sizes, times, label='Время кодирования Хаффмана')
    ax.set_xlabel('Размер алфавита (количество символов)')
    ax.set_ylabel('Время выполнения (секунды)')
    ax.set_title('Зависимость времени работы алгоритма Хаффмана от размера алфавита')
    ax.legend()
    plt.savefig('huffman_time.png')
    plt.close()

    return sizes, times

import time
from kmp_search import kmp_search
from z_function import z_search


def naive_search(text, pattern):
    """Наивный поиск подстроки.

    Худший случай: O(n * m), память: O(1).
    """
    n = len(text)
    m = len(pattern)
    if m == 0:
        return list(range(n + 1))

    res = []
    for i in range(n - m + 1):
        match = True
        for j in range(m):
            if text[i + j] != pattern[j]:
                match = False
                break
        if match:
            res.append(i)
    return res


def rabin_karp(text, pattern, base=257, mod=10**9 + 7):
    """Алгоритм Рабина–Карпа.

    Среднее время: O(n + m), худшее: O(n * m).
    Память: O(1) дополнительно.
    """
    n = len(text)
    m = len(pattern)
    if m == 0:
        return list(range(n + 1))
    if m > n:
        return []

    pow_base = 1
    for _ in range(m - 1):
        pow_base = (pow_base * base) % mod

    pat_hash = 0
    win_hash = 0
    for i in range(m):
        pat_hash = (pat_hash * base + ord(pattern[i])) % mod
        win_hash = (win_hash * base + ord(text[i])) % mod

    res = []
    for i in range(n - m + 1):
        if pat_hash == win_hash:
            if text[i:i + m] == pattern:
                res.append(i)
        if i < n - m:
            left = ord(text[i]) * pow_base % mod
            win_hash = (win_hash - left + mod) % mod
            win_hash = (win_hash * base + ord(text[i + m])) % mod

    return res


def find_period(s):
    """Поиск наименьшего периода строки через префикс-функцию.

    Время: O(n), память: O(n).
    """
    from prefix_function import prefix_function

    n = len(s)
    if n == 0:
        return None
    pi = prefix_function(s)
    k = n - pi[-1]
    if k != n and n % k == 0:
        return k
    return None


def is_cyclic_shift(a, b):
    """Проверка, является ли b циклическим сдвигом a.

    Использует KMP-поиск в строке a + a.
    Время: O(n), память: O(n).
    """
    if len(a) != len(b):
        return False
    from kmp_search import kmp_search

    doubled = a + a
    positions = kmp_search(doubled, b)
    return len(positions) > 0


def benchmark():
    """Сравнение наивного поиска и KMP.

    Для построения графиков следует вызывать эту функцию
    для разных длин текста и паттерна.
    """
    text = "ab" * 10000 + "aba"
    pattern = "aba"

    start = time.time()
    naive_res = naive_search(text, pattern)
    naive_time = time.time() - start

    start = time.time()
    kmp_res = kmp_search(text, pattern)
    kmp_time = time.time() - start

    print("Наивный поиск:")
    print(f"  вхождения: {len(naive_res)}, время: {naive_time:.6f} с")
    print("KMP:")
    print(f"  вхождения: {len(kmp_res)}, время: {kmp_time:.6f} с")


if __name__ == "__main__":
    text = "ababcababcababc"
    pattern = "ababc"
    print("KMP:", kmp_search(text, pattern))
    print("Z-поиск:", z_search(text, pattern))
    print("Наивный:", naive_search(text, pattern))
    print("Рабин–Карп:", rabin_karp(text, pattern))

    s = "abcabcabcabc"
    print("Период строки:", find_period(s))
    print("Циклический сдвиг:", is_cyclic_shift("abcd", "cdab"))

    benchmark()

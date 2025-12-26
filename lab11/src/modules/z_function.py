def z_function(s):
    """Строит массив z для строки s.

    z[i] — длина наибольшего общего префикса s и суффикса s[i:].

    Время: O(n), память: O(n).
    """
    n = len(s)
    z = [0] * n
    l = 0
    r = 0
    for i in range(1, n):
        if i <= r:
            z[i] = min(r - i + 1, z[i - l])
        while i + z[i] < n and s[z[i]] == s[i + z[i]]:
            z[i] += 1
        if i + z[i] - 1 > r:
            l = i
            r = i + z[i] - 1
    return z


def z_search(text, pattern):
    """Поиск подстроки с использованием Z-функции.

    Время: O(n + m), память: O(n + m).
    """
    if not pattern:
        return list(range(len(text) + 1))

    s = pattern + "#" + text
    z = z_function(s)
    m = len(pattern)
    res = []
    for i in range(m + 1, len(s)):
        if z[i] == m:
            pos = i - m - 1
            res.append(pos)
    return res


if __name__ == "__main__":
    print(z_function("aabcaabxaaaz"))
    print(z_search("ababcababcababc", "ababc"))

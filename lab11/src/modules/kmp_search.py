from prefix_function import prefix_function


def kmp_search(text, pattern):
    """Ищет все вхождения pattern в text.

    Время: O(n + m), память: O(m).
    """
    if not pattern:
        return list(range(len(text) + 1))

    combined = pattern + "#" + text
    # Префикс-функция для объединённой строки
    pi = prefix_function(combined)
    m = len(pattern)  # Длина шаблона
    result = []  # Список для хранения индексов вхождений шаблона

    for i in range(m + 1, len(combined)):
        if pi[i] == m:
            pos = i - 2 * m
            result.append(pos)

    return result


if __name__ == "__main__":
    print(kmp_search("ababcababcababc", "ababc"))

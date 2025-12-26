def fib_naive(n):
    """Наивная рекурсия.
    Время: O(2^n),
    память: O(n).
    """
    if n <= 1:
        return n
    return fib_naive(n - 1) + fib_naive(n - 2)


def fib_memo(n):
    """Нисходящий подход с мемоизацией (top-down).
    Время: O(n),
    память: O(n).
    """
    memo = {}

    def fib_helper(k):
        # Проверяем, есть ли результат в кэше
        if k in memo:
            return memo[k]

        # Базовые случаи
        if k <= 1:
            result = k
        else:
            # Рекурсивный вызов с сохранением результатов
            result = fib_helper(k - 1) + fib_helper(k - 2)

        # Сохраняем результат в кэш
        memo[k] = result
        return result

    return fib_helper(n)


def fib_table(n):
    """Табличное решение (bottom-up).
    Время: O(n),
    память: O(n).
    """
    if n <= 1:
        return n
    dp = [0] * (n + 1)
    dp[1] = 1
    for i in range(2, n + 1):
        dp[i] = dp[i - 1] + dp[i - 2]
    return dp[n]


def knapsack_01(weights, values, capacity):
    """Рюкзак 0-1 с восстановлением решения.
    Время: O(n*W),
    память: O(n*W).
    """
    n = len(weights)
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        wi = weights[i - 1]
        vi = values[i - 1]
        for w in range(capacity + 1):
            if wi <= w:
                without_item = dp[i - 1][w]
                with_item = dp[i - 1][w - wi] + vi
                dp[i][w] = max(without_item, with_item)
            else:
                dp[i][w] = dp[i - 1][w]

    # Восстановление решения
    items = []
    w = capacity
    for i in range(n, 0, -1):
        if dp[i][w] != dp[i - 1][w]:
            items.append(i - 1)
            w -= weights[i - 1]
    items.reverse()

    return dp[n][capacity], items, dp


def lcs(str1, str2):
    """LCS с восстановлением подпоследовательности.
    Время: O(m*n),
    память: O(m*n).
    """
    m, n = len(str1), len(str2)
    # Таблица размером (m+1)x(n+1)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if str1[i - 1] == str2[j - 1]:  # Если символы совпадают
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    # Восстановление самой последовательности
    seq = []
    i, j = m, n
    while i > 0 and j > 0:
        if str1[i - 1] == str2[j - 1]:
            seq.append(str1[i - 1])
            i -= 1  # Переход в предыдущую строку
            j -= 1  # Переход в предыдущий столбец
        elif dp[i - 1][j] >= dp[i][j - 1]:  # Если максимум сверху
            i -= 1
        else:
            j -= 1
    seq.reverse()

    return dp[m][n], "".join(seq), dp


def levenshtein_distance(s1, s2):
    """Расстояние Левенштейна.
    Время: O(m*n), память: O(m*n).
    """
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,  # Удаление
                dp[i][j - 1] + 1,  # Вставка
                dp[i - 1][j - 1] + cost,  # Замена
            )

    return dp[m][n], dp


if __name__ == "__main__":
    # Тестирование функций
    print("Fibonacci:")
    print(f"fib_naive(10) = {fib_naive(10)}")
    print(f"fib_memo(10) = {fib_memo(10)}")
    print(f"fib_table(10) = {fib_table(10)}")

    print("\nKnapsack 0-1:")
    weights = [1, 3, 4, 5]
    values = [1, 4, 5, 7]
    capacity = 7
    max_value, items, table = knapsack_01(weights, values, capacity)
    print(f"Max value: {max_value}")
    print(f"Items selected: {items}")

    print("\nLCS:")
    str1 = "ABCDGH"
    str2 = "AEDFHR"
    lcs_len, lcs_seq, _ = lcs(str1, str2)
    print(f"LCS length: {lcs_len}")
    print(f"LCS: {lcs_seq}")

    print("\nLevenshtein distance:")
    s1 = "kitten"
    s2 = "sitting"
    dist, _ = levenshtein_distance(s1, s2)
    print(f"Distance between '{s1}' and '{s2}': {dist}")

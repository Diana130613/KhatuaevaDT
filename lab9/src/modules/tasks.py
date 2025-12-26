def coin_change(coins, amount):
    """Минимальное количество монет для суммы."""
    dp = [float("inf")] * (amount + 1)
    dp[0] = 0

    for cur_sum in range(1, amount + 1):
        for coin in coins:
            if coin <= cur_sum:
                candidate = dp[cur_sum - coin] + 1
                if candidate < dp[cur_sum]:
                    dp[cur_sum] = candidate

    if dp[amount] == float("inf"):
        return -1
    return dp[amount]


def lis_quadratic(seq):
    """LIS (наибольшая возрастающая подпоследовательность).
    Время: O(n^2),
    память: O(n).
    """
    n = len(seq)
    if n == 0:
        return 0, []

    dp = [1] * n
    parent = [-1] * n

    for i in range(n):
        for j in range(i):
            if seq[j] < seq[i] and dp[j] + 1 > dp[i]:
                dp[i] = dp[j] + 1
                parent[i] = j

    max_len = max(dp)
    pos = dp.index(max_len)

    lis_seq = []
    while pos != -1:
        lis_seq.append(seq[pos])
        pos = parent[pos]
    lis_seq.reverse()

    return max_len, lis_seq

def min_coins(amount, coins):
    """
    Жадный алгоритм минимального числа монет для сдачи.

    Args:
        amount: Сумма для размена
        coins: Доступные номиналы монет

    Returns:
        Список монет для размена или None если размен невозможен

    Сложность: O(n) (число типов монет)
    """
    coins = sorted(coins, reverse=True)  # Сортировка по убывания номинала
    result = []

    for coin in coins:
        while amount >= coin:
            amount -= coin
            result.append(coin)

    # Если не смогли разменять полностью
    if amount > 0:
        return None
    return result

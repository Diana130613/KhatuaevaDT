from typing import Any
from hash_functions import poly, djb2


class HashTableOpenAddressing:
    """Хеш-таблица с разрешением коллизий методом цепочек."""

    EMPTY = None  # Маркер пустой ячейки
    DEL = "__DELETED__"  # Маркер удаленной ячейки

    def __init__(self, sz=1009, mt='lin', hf1=djb2, hf2=poly):
        """
        Инициализация таблицы. Сложность: O(sz)

        Args:
            sz: Размер таблицы (рекомендуется простое число)
            mt: Метод пробирования: 'lin' - линейное, 'dbl'-двойное хеширование
            hf1: Первая хеш-функция
            hf2: Вторая (используется только при двойном хешировании)
        """
        self.sz = sz  # Размер таблицы
        self.tbl = [self.EMPTY] * sz  # Массив с пустыми ячейками
        self.mt = mt  # Метод: 'lin' или 'dbl'
        self.hf1 = hf1
        self.hf2 = hf2
        self.cnt = 0  # Счетчик элементов
        self.col = 0  # Счетчик коллизий

    def _lf(self) -> float:
        """
        Вычисление коэффициента заполнения (load factor). Сложность: O(1)

        Returns:
            Коэффициент заполнения таблицы (α = cnt / sz)
        """
        return self.cnt / self.sz  # Возвращаем коэффициент заполнения

    def _pr_lin(self, key: Any, i: int) -> int:
        """
        Линейное пробирование. Сложность: O(1)

        Args:
            key: Ключ
            i: Номер попытки (проба)

        Returns:
            Индекс в таблице для i-й попытки
        """
        return (self.hf1(key, self.sz) + i) % self.sz  # h(key,i)=(h1(key)+i) mod m

    def _pr_dbl(self, key: Any, i: int) -> int:
        """
        Двойное хеширование. Сложность: O(1)

        Args:
            key: Ключ
            i: Номер попытки (проба)

        Returns:
            Индекс в таблице для i-й попытки
        """
        h1 = self.hf1(key, self.sz)  # Вычисляем первый хеш
        h2 = self.hf2(key, self.sz) + 1  # Вычисляем второй хеш (+1, чтобы избежать 0)
        return (h1 + i * h2) % self.sz  # h(key,i) = (h1(key)+i*h2(key)) mod m

    def _pr(self, key: Any, i: int) -> int:
        """
        Выбор метода пробирования. Сложность: O(1)

        Args:
            key: Ключ
            i: Номер попытки (проба)

        Returns:
            Индекс в таблице для i-й попытки
        """
        if self.mt == 'lin':
            return self._pr_lin(key, i)
        return self._pr_dbl(key, i)  # Иначе используем двойное

    def ins(self, key: Any, value: Any) -> None:
        """
        Вставка или обновление пары ключ-значение.
        Сложность: O(1/(1-α)) в среднем

        Args:
            key: Ключ для вставки
            value: Значение, ассоциированное с ключом

        Raises:
            Exception: Если таблица переполнена (α > 0.7)
            Exception: Если не удалось найти свободную ячейку

        При α > 0.7 выбрасывается исключение
        """
        if self._lf() > 0.7:
            raise Exception("Table full")
        i = 0  # Инициализируем счётчик пробирования
        while i < self.sz:  # Перебор таблицы
            idx = self._pr(key, i)  # Вычисляем позицию пробирования
            if self.tbl[idx] is self.EMPTY or self.tbl[idx] == self.DEL:  # Если ячейка свободна
                self.tbl[idx] = (key, value)  # Вставляем пару
                self.cnt += 1
                return
            if self.tbl[idx][0] == key:  # Если ключ уже существует
                self.tbl[idx] = (key, value)  # Обновляем значение
                return
            self.col += 1
            i += 1  # Переходим к следующему пробированию
        raise Exception("Insert failed")  # Не смогли вставить

    def srch(self, key):
        """
        Поиск значения по ключу. Сложность: O(1/(1-α)) в среднем

        Args:
            key: Ключ для поиска

        Returns:
            Значение, ассоциированное с ключом, или None если ключ не найден
        """
        i = 0  # Инициализируем счётчик пробирования
        while i < self.sz:  # Перебор таблицы
            idx = self._pr(key, i)  # Вычисляем позицию пробирования
            if self.tbl[idx] is self.EMPTY:  # Если ячейка пустая
                return None  # Ключа нет
            if self.tbl[idx] == self.DEL:  # Если ячейка удалена
                i += 1  # Пропускаем, продолжаем поиск
                continue  # Переход к следующей итерации
            if self.tbl[idx][0] == key:  # Нужный ключ
                return self.tbl[idx][1]  # Возвращаем значение
            i += 1  # Переходим к следующему пробированию
        return None  # Ключ не найден

    def rm(self, key: Any) -> bool:
        """
        Удаление пары ключ-значение. Сложность: O(1/(1-α)) в среднем

        Args:
            key: Ключ для удаления

        Returns:
            True если удаление успешно, False если ключ не найден
        """
        i = 0  # Инициализируем счетчик пробирования
        while i < self.sz:
            idx = self._pr(key, i)
            if self.tbl[idx] is self.EMPTY:
                return False  # Ключа нет
            if self.tbl[idx][0] == key:  # Если найден нужный ключ
                self.tbl[idx] = self.DEL  # Помечаем как удаленную
                self.cnt -= 1  # Уменьшаем счетчик
                return True  # Успешно удалили
            i += 1  # Переходим к следующему пробированию
        return False  # Ключ не найден

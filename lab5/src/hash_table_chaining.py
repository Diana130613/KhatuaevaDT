from hash_functions import djb2


class HashTableChaining:
    """Хеш-таблица с разрешением коллизий методом цепочек."""
    def __init__(self, sz: int = 1009, hf=djb2) -> None:
        """
        Инициализация таблицы. Сложность: O(sz)

        Args:
            sz: Начальный размер таблицы (простое число)
            hf: Хеш-функция, принимает ключ и размер таблицы, возвращает индекс
        """
        self.sz = sz  # Размер таблицы
        self.tbl = [[] for _ in range(sz)]  # Массив списков для цепочек
        self.hf = hf  # Ссылка на хеш-функцию
        self.cnt = 0  # Счётчик элементов
        self.col = 0  # Счётчик коллизий

    def _lf(self) -> float:
        """
        Вычисление коэффициента заполнения (load factor). Сложность: O(1)

        Returns:
            Коэффициент заполнения таблицы (α = cnt / sz)
        """
        return self.cnt / self.sz

    def _res(self) -> None:
        """
        Увеличение размера таблицы при α > 0.7. Сложность: O(n)
        Производит перехеширование всех существующих элементов
        """
        old = self.tbl  # Сохраняем старую таблицу
        self.sz = self.sz * 2 + 1  # Новый размер (нечётное число)
        self.tbl = [[] for _ in range(self.sz)]
        self.cnt = 0  # Сбрасываем счетчик
        for ch in old:  # Перебираем все цепочки
            for key, value in ch:  # Перебираем пары в цепочке
                self.ins(key, value)  # Перехешируем в новую таблицу

    def ins(self, key, value) -> None:
        """
        Вставка или обновление пары ключ-значение. Сложность: O(1+α) в среднем

        Args:
            key: Ключ для вставки
            value: Значение, ассоциированное с ключом

        Note:
            Если ключ уже существует, его значение обновляется.
            При α > 0.7 автоматически выполняется ресайз таблицы.
        """
        if self._lf() > 0.7:
            self._res()
        idx = self.hf(key, self.sz)  # Вычисляем индекс через хеш
        ch = self.tbl[idx]  # Получаем цепочку
        # Проверяем, есть ли уже такой ключ
        for i, (existing_key, existing_value) in enumerate(ch):
            if existing_key == key:  # Если ключ найден
                ch[i] = (key, value)  # Обновляем значение
                return
        ch.append((key, value))  # Добавляем новую пару в конец цепочки
        self.cnt += 1  # Увеличиваем счетчик элементов
        self.col += len(ch) - 1  # Подсчитываем коллизию

    def srch(self, key):
        """
        Поиск значения по ключу. Сложность: O(1+α) в среднем

        Args:
            key: Ключ для поиска
        """
        idx = self.hf(key, self.sz)  # Вычисляем индекс через хеш
        for existing_key, existing_value in self.tbl[idx]:  # Перебираем пары
            if existing_key == key:  # Если ключ совпадает
                return existing_value  # Возвращаем значение
        return None  # Ключ не найден

    def rm(self, key):
        """
        Поиск значения по ключу. Сложность: O(1+α) в среднем

        Args:
            key: Ключ для поиска
        """
        idx = self.hf(key, self.sz)
        ch = self.tbl[idx]  # Получаем цепочку
        for i, (existing_key, existing_value) in enumerate(ch):
            if existing_key == key:
                del ch[i]  # Удаляем элемент из цепочки
                self.cnt -= 1  # Уменьшаем счетчик
                return True
        return False  # Ключ не найден

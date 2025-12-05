# main.py (альтернативная версия)

from performance_analysis import run_all


def main():
    """
    Упрощённая версия главной функции
    """
    print("ХЕШ-ФУНКЦИИ И ХЕШ-ТАБЛИЦЫ")
    print("=" * 60)

    # Характеристики ПК
    pc_info = """
    Характеристики ПК для тестирования:
    - Процессор: 12th Gen Intel(R) Core(TM) i5-12450H
    - Оперативная память: 16 GB DDR4
    - ОС: Windows 10
    - Python: 3.12.10
    """
    print(pc_info)

    print("\nЗАПУСК ПОЛНОГО НАБОРА ТЕСТОВ...")
    print("=" * 30)

    run_all()

    print("\n" + "=" * 60)
    print("ВСЕ ТЕСТЫ УСПЕШНО ВЫПОЛНЕНЫ!")
    print("Графики сохранены в текущей директории.")
    print("=" * 60)


if __name__ == "__main__":
    main()

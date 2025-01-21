import os
import argparse
import importlib

import cv2


def main():
    # Настройка парсера аргументов
    parser = argparse.ArgumentParser(description='Обработка файлов в каталоге с заданной функцией.')
    parser.add_argument('file_name', type=str, help='Имя файла, где находится функция (без .py)')
    parser.add_argument('function_name', type=str, help='Имя функции для вызова')

    args = parser.parse_args()

    try:
        # Импортируем файл с функциями
        file_processor = importlib.import_module(args.file_name)

        # Получаем ссылку на функцию по имени
        function = getattr(file_processor, args.function_name, None)

        if function:
            # Перебираем файлы в указанном каталоге и вызываем функцию
            directory = "bars"  # Убедитесь, что этот путь существует
            for filename in os.listdir(directory):
                file_path = os.path.join(directory, filename)  # Полный путь к файлу
                if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".jfif"):
                    # Вызываем функцию с полным путем к файлу
                    try:
                        img = cv2.imread(file_path)
                        print(function(img))
                    except Exception as e:
                        print(f"Произошла ошибка: {e}")
                        continue
                else:
                    continue
        else:
            print(f"Функция '{args.function_name}' не найдена в модуле '{args.file_name}'.")
    except ImportError:
        print(f"Не удалось импортировать модуль '{args.file_name}'. Проверьте имя файла и путь.")
    except Exception as e:
        print(f"Произошла ошибка: {e}")

if __name__ == "__main__":
    main()

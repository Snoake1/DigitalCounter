import cv2
import numpy as np
import matplotlib.pyplot as plt

def extract_screen(image_path):
    image = cv2.imread(image_path)

    # Конвертируем в градации серого
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # plt.figure()
    # plt.title("Градации серого")
    # plt.imshow(gray, cmap='gray')

    # Применяем размытие для сглаживания бликов
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # plt.figure()
    # plt.title("Размытие")
    # plt.imshow(blurred, cmap='gray')

    # Используем адаптивную бинаризацию для выделения контуров
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    plt.figure()
    plt.title("Адаптивная бинаризация")
    plt.imshow(thresh, cmap='gray')

    # Убираем шумы морфологической обработкой
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    processed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    plt.figure()
    plt.title("Морфологическая обработка")
    plt.imshow(processed, cmap='gray')

    # Ищем контуры
    contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Отображаем контуры на изображении
    contours_image = image.copy()
    cv2.drawContours(contours_image, contours, -1, (0, 255, 0), 2)
    plt.figure()
    plt.title("Все контуры")
    plt.imshow(cv2.cvtColor(contours_image, cv2.COLOR_BGR2RGB))

    # Сортируем контуры по площади (по убыванию)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # Ищем контур, похожий на экран
    # screen_contour = None
    # for contour in contours:
    #     x, y, w, h = cv2.boundingRect(contour)
    #     aspect_ratio = w / float(h)
    #
    #     # Условие для экрана: прямоугольная форма с характерными пропорциями
    #     if 2 < aspect_ratio < 8 and 200 < w and 50 < h:
    #         roi = gray[y:y + h, x:x + w]
    #
    #         # Проверяем наличие цифр с помощью пороговой обработки
    #         digit_thresh = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    #         plt.figure()
    #         plt.title("Пороговая обработка экрана")
    #         plt.imshow(digit_thresh, cmap='gray')
    #
    #         num_contours, _ = cv2.findContours(digit_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #
    #         # Проверяем, есть ли достаточно маленьких контуров (предположительно цифры)
    #         digit_like = sum(1 for c in num_contours if 15 < cv2.boundingRect(c)[2] < 100 and 30 < cv2.boundingRect(c)[3] < 100)
    #         if digit_like >= 5:  # Минимум 5 цифр
    #             screen_contour = (x, y, w, h)
    #             break
    #
    # # Если экран найден, обрезаем и отображаем его
    # if screen_contour:
    #     x, y, w, h = screen_contour
    #     screen = image[y:y + h, x:x + w]
    #
    #     plt.figure()
    #     plt.title("Обрезанный экран")
    #     plt.imshow(cv2.cvtColor(screen, cv2.COLOR_BGR2RGB))
    #     plt.axis('off')
    #     plt.show()
    # else:
    #     print("Экран не найден.")

    plt.show()
# Пример использования
extract_screen("test_image/test2.jpg")
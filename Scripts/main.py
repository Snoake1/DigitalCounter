import argparse
import cv2
from barcodeReader import barcode_reader
from ultralytics import YOLO
import matplotlib.pyplot as plt
from paddleocr import PaddleOCR, draw_ocr
import re

ocr = PaddleOCR(lang='en') # need to run only once to download and load model into memory

def filter_res(result, regex):
    return list(filter(lambda x: x[-1][-1] > 0.8 and re.match(regex, x[-1][0]) is not None, result))

def predict_and_draw(im):
    result = ocr.ocr(im, cls=False)[0]
    if result is None:
        return ""
    result = filter_res(result, r'^[N№]?[\d.,]{6,}$')
    boxes = [line[0] for line in result]
    txts = [line[1][0] for line in result]
    scores = [line[1][1] for line in result]
    im_show = draw_ocr(im, boxes, txts, scores, font_path='ARIAL.TTF')
    plt.figure()
    plt.imshow(im_show)
    plt.show()

    return result[0][-1][0]

def apply_filter(image):
    # Конвертируем в градации серого
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    print(gray.shape)
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
    # plt.figure()
    # plt.title("Адаптивная бинаризация")
    # plt.imshow(thresh, cmap='gray')

    # Убираем шумы морфологической обработкой
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    processed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    # processed = thresh
    plt.figure()
    plt.title("Морфологическая обработка")
    plt.imshow(processed, cmap='gray')

    # Ищем контуры
    contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Отображаем контуры на изображении
    contours_image = image.copy()
    cv2.drawContours(contours_image, contours, -1, (0, 255, 0), 2)
    # plt.figure()
    # plt.title("Все контуры")
    # plt.imshow(cv2.cvtColor(contours_image, cv2.COLOR_BGR2RGB))

    # Сортируем контуры по площади (по убыванию)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    plt.show()

    return cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)

def main():
    parser = argparse.ArgumentParser(description='Получение данных счетчика по изображение')
    parser.add_argument('file_name', type=str, help='Имя файла, где находится изображение')

    args = parser.parse_args()
    name = args.file_name.split("\\")[-1]
    # read the image in numpy array using cv2
    img = cv2.imread(name)

    model = YOLO("../yolo.pt")
    res = model.predict(img)[0]

    num_roi = img
    disp_roi = img
    for i, cls in enumerate(res.boxes.cls):
        x1, y1, x2, y2 = res.boxes.xyxy[i].int()
        if cls == 0:
            disp_roi = img[y1:y2, x1:x2]
        elif cls == 1:
            num_roi = img[y1:y2, x1:x2]


    disp_proc = apply_filter(disp_roi)
    print(disp_roi.shape, disp_proc.shape)
    value = predict_and_draw(disp_proc)
    print(value)

    barcode_value = barcode_reader(num_roi)
    if barcode_value.isdigit():
        print(f"Заводской номер: {barcode_value}")
    else:
        #TODO Отображать номер, полученный OCR
        print(f"Не удалось прочитать заводской номер")


if __name__ == "__main__":
    main()

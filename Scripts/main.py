import argparse
import cv2
from barcodeReader import barcode_reader
from ultralytics import YOLO


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

    value = 0  # TODO Получать значение на дисплее
    barcode_value = barcode_reader(num_roi)
    if barcode_value.isdigit():
        print(f"Заводской номер: {barcode_value}")
    else:
        # TODO Отображать номер, полученный OCR
        print(f"Не удалось прочитать заводской номер")


if __name__ == "__main__":
    main()

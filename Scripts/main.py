import argparse
import cv2
from barcodeReader import barcode_reader
from ultralytics import YOLO
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import requests


def get_value(img, processor, model):
    pixel_values = processor(images=img, return_tensors="pt").pixel_values

    generated_ids = model.generate(pixel_values)
    return processor.batch_decode(generated_ids, skip_special_tokens=True)[0]


def get_rois(img, model):
    res = model.predict(img)[0]
    num_roi = img
    disp_roi = img
    for i, cls in enumerate(res.boxes.cls):
        x1, y1, x2, y2 = res.boxes.xyxy[i].int()
        if cls == 0:
            disp_roi = img[y1:y2, x1:x2]
        elif cls == 1:
            num_roi = img[y1:y2, x1:x2]
    return disp_roi, num_roi


def main():
    parser = argparse.ArgumentParser(description='Получение данных счетчика по изображение')
    parser.add_argument('file_name', type=str, help='Имя файла, где находится изображение')

    args = parser.parse_args()
    name = args.file_name
    # read the image in numpy array using cv2
    img = cv2.imread(name)

    model = YOLO("../models/yolo.pt")
    processor_disp = TrOCRProcessor.from_pretrained('../models/disps/weights')
    model_trocr_disp = VisionEncoderDecoderModel.from_pretrained('../models/disps/weights')
    processor_bar = TrOCRProcessor.from_pretrained('../models/bars/weights')
    model_trocr_bar = VisionEncoderDecoderModel.from_pretrained('../models/bars/weights')

    disp_roi, num_roi = get_rois(img, model)

    disp_res = get_value(disp_roi, processor_disp, model_trocr_disp)
    print(disp_res)
    barcode_value = barcode_reader(num_roi)
    if barcode_value.isdigit():
        print(f"Заводской номер: {str(barcode_value)[2:-1]}")
    else:
        barcode_value = get_value(num_roi, processor_bar, model_trocr_bar)
        if barcode_value.isdigit():
            print(f"Заводской номер: {barcode_value}")
        else:
            print("Не удалось распознать номер")


if __name__ == "__main__":
    main()

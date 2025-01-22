import argparse
import cv2
from barcodeReader import barcode_reader
from ultralytics import YOLO
import matplotlib.pyplot as plt
from paddleocr import PaddleOCR, draw_ocr
import re
import os


parser = argparse.ArgumentParser(description='Получение данных счетчика по изображение')
parser.add_argument('file_name', type=str, help='Имя файла, где находится изображение')

args = parser.parse_args()
name = args.file_name.split("\\")[-1]
# read the image in numpy array using cv2
directory = os.fsencode(args.file_name)
model = YOLO("../yolo.pt")

if not os.path.exists("./disps"):
    os.makedirs("./disps")
if not os.path.exists("./bars"):
    os.makedirs("./bars")

for file in os.listdir(directory):
    filename = os.fsdecode(file)
    img = cv2.imread(args.file_name + "\\" + filename)
    last_name = filename.split('\\')[-1]

    res = model.predict(img)[0]

    num_roi = None
    disp_roi = None
    for i, cls in enumerate(res.boxes.cls):
        x1, y1, x2, y2 = res.boxes.xyxy[i].int()
        if cls == 0:
            disp_roi = img[y1:y2, x1:x2]
        elif cls == 1:
            num_roi = img[y1:y2, x1:x2]

    if disp_roi is not None:
        cv2.imwrite("./disps/" + last_name, disp_roi)
    if num_roi is not None:
        cv2.imwrite("./bars/" + last_name, num_roi)

import argparse
import cv2
from ultralytics import YOLO
import os
from main import apply_filter

parser = argparse.ArgumentParser(description='Получение данных счетчика по изображение')
parser.add_argument('file_name', type=str, help='Имя файла, где находится изображение')

args = parser.parse_args()
name = args.file_name.split("\\")[-1]
# read the image in numpy array using cv2
directory = os.fsencode(args.file_name)
model = YOLO("../models/yolo.pt")

if not os.path.exists("./fil_disps"):
    os.makedirs("./fil_disps")
if not os.path.exists("./fil_bars"):
    os.makedirs("./fil_bars")

for file in os.listdir(directory)[:100]:
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

    disp_proc = apply_filter(disp_roi, False)
    if disp_proc is not None:
        cv2.imwrite("./fil_disps/" + last_name, disp_proc)
    num_proc = apply_filter(num_roi, False)
    if num_proc is not None:
        cv2.imwrite("./fil_bars/" + last_name, num_proc)

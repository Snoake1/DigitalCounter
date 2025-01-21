from ultralytics import YOLO
import cv2
from paddleocr import PaddleOCR, draw_ocr
from PIL import Image
import re

def filter_res(result, regex):
    return list(filter(lambda x: x[-1][-1] > 0.8 and re.match(regex, x[-1][0]) is not None, result))

def predict_and_draw(im, outfile):
    result = ocr.ocr(im, cls=False)[0]
    if result is None:
        return
    result = filter_res(result, r'^[N№]?[\d.,]{6,}$')
    boxes = [line[0] for line in result]
    txts = [line[1][0] for line in result]
    scores = [line[1][1] for line in result]
    im_show = draw_ocr(im, boxes, txts, scores, font_path='en_standard.ttf')
    im_show = Image.fromarray(im_show)
    im_show.save(outfile)

model = YOLO("yolo.pt")
img = cv2.imread("./samples/test1.jpg")
res = model.predict(img)[0]
disp_roi = None
num_roi = None

for i, cls in enumerate(res.boxes.cls):
    x1, y1, x2, y2 = res.boxes.xyxy[i].int()
    if cls == 0:
        disp_roi = img[y1:y2, x1:x2]
    elif cls == 1:
        num_roi = img[y1:y2, x1:x2]

ocr = PaddleOCR(lang='en') # need to run only once to download and load model into memory

if disp_roi is not None:
    predict_and_draw(disp_roi, "res_disp.jpg")
if num_roi is not None:
    predict_and_draw(num_roi, "res_num.jpg")

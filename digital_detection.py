import cv2
from ultralytics import YOLO
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from pyzbar.pyzbar import decode

model = YOLO("../models/yolo.pt")
processor_disp = TrOCRProcessor.from_pretrained('../models/disps/weights')
model_trocr_disp = VisionEncoderDecoderModel.from_pretrained('../models/disps/weights')
processor_bar = TrOCRProcessor.from_pretrained('../models/bars/weights')
model_trocr_bar = VisionEncoderDecoderModel.from_pretrained('../models/bars/weights')

# Make one method to decode the barcode
def barcode_reader(img):

    # Decode the barcode image 
    detected_barcodes = decode(img)

    # If not detected then print the message 
    if not detected_barcodes:
        return f"Серийный номер не найден"
    else:

        # Traverse through all the detected barcodes in image
        for (i, barcode) in enumerate(detected_barcodes):

            # Locate the barcode position in image 
            (x, y, w, h) = barcode.rect
            # Определяем координаты для обрезки изображения
            top_left = (x+10, y+10)
            bottom_right = (x + w + 10, y + h + 10)

            #cv2.imwrite(f"ProcessResults/barcoder{i}{name}", img[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]])
            
            # Put the rectangle in image using  
            # cv2 to highlight the barcode 
            rect = cv2.rectangle(img, (x - 10, y - 10),
                                (x + w + 10, y + h + 10),
                                (255, 0, 0), 2)

            if barcode.data != "":
                # Print the barcode data
                return barcode.data

def get_value(img, processor, model):
    pixel_values = processor(images=img, return_tensors="pt").pixel_values

    generated_ids = model.generate(pixel_values)
    return processor.batch_decode(generated_ids, skip_special_tokens=True)[0]


def get_rois(img, model):
    res = model.predict(img)[0]
    num_roi = None
    disp_roi = None
    for i, cls in enumerate(res.boxes.cls):
        x1, y1, x2, y2 = res.boxes.xyxy[i].int()
        if cls == 0:
            disp_roi = img[y1:y2, x1:x2]
        elif cls == 1:
            num_roi = img[y1:y2, x1:x2]
    return disp_roi, num_roi


def detect_digit(img):
    disp_roi, num_roi = get_rois(img, model)

    if disp_roi is None:
        print("Дисплей не найден")
        disp_res = None
    else:
        disp_res = get_value(disp_roi, processor_disp, model_trocr_disp)
        print(disp_res)

        
    if num_roi is None:
        print("Серийный номер не найден")
        barcode_value = None
    else:
        barcode_value = barcode_reader(num_roi)
        if barcode_value.isdigit():
            print(f"Заводской номер: {str(barcode_value)[2:-1]}")
        else:
            barcode_value = get_value(num_roi, processor_bar, model_trocr_bar)
            if barcode_value.isdigit():
                print(f"Заводской номер: {barcode_value}")
            else:
                print("Не удалось распознать номер")

    return disp_res, barcode_value

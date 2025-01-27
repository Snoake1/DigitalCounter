from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import csv
from difflib import SequenceMatcher

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

processor = TrOCRProcessor.from_pretrained('../disp_model/weights')
model = VisionEncoderDecoderModel.from_pretrained('../disp_model/weights')

all_images = 0
correct = 0.

with open('labels_full.csv', 'r', encoding='utf-8') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        try:
            image = Image.open('./disps_full/' + row[0]).convert("RGB")
            pixel_values = processor(images=image, return_tensors="pt").pixel_values

            generated_ids = model.generate(pixel_values)
            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

            all_images += 1
            # if generated_text == row[1].strip():
            correct += similar(generated_text, row[1].strip())
            print(generated_text, row[1].strip(), similar(generated_text, row[1].strip()), f'{correct}/{all_images}')
        except:
            pass

print(all_images, correct, correct/all_images)

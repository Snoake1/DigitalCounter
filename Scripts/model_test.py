from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import csv

processor = TrOCRProcessor.from_pretrained('../disp_model/weights')
model = VisionEncoderDecoderModel.from_pretrained('../disp_model/weights')

all_images = 0
correct = 0

images = []
texts = []

with open('labels_full.csv', 'r', encoding='utf-8') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        try:
            image = Image.open('./disps_full/' + row[0]).convert("RGB")

            images.append(image)

            texts.append(row[1].strip())

            all_images += 1
            # if all_images == 30:
            #     break
        except:
            pass

pixel_values = processor(images=images, return_tensors="pt").pixel_values

generated_ids = model.generate(pixel_values)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)

print(generated_text)

for i, t in enumerate(texts):
    if generated_text[i] == t:
        correct += 1

print(all_images, correct, correct/all_images)

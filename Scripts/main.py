from paddleocr import PaddleOCR,draw_ocr
import json, re
ocr = PaddleOCR(lang='en') # need to run only once to download and load model into memory
img_path = 'samples/test16.jpg'
result = ocr.ocr(img_path, cls=False)
regex = r'^[N№]?[\d.,]{6,}$'

filtered = list(filter(lambda x: x[-1][-1] > 0.8 and re.match(regex, x[-1][0]) is not None, result[0]))

with open('result.json', 'w+') as f:
    f.write(json.dumps(filtered))

# draw result

from PIL import Image
result = filtered
image = Image.open(img_path).convert('RGB')
boxes = [line[0] for line in result]
txts = [line[1][0] for line in result]
scores = [line[1][1] for line in result]
im_show = draw_ocr(image, boxes, txts, scores, font_path='Scripts\\ARIAL.TTF')
im_show = Image.fromarray(im_show)
im_show.save('result.jpg')

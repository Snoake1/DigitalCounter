from paddleocr import PaddleOCR,draw_ocr

ocr = PaddleOCR(rec_model_dir="./model", use_angle_cls=False, lang='en', rec_char_dict_path='Scripts\\dict.txt') # need to run only once to download and load model into memory
img_path = 'Scripts\\disps\\acce5dc913254e0291872e25028579ad.jpg'
result = ocr.ocr(img_path, cls=False)
print(result)
for idx in range(len(result)):
    res = result[idx]
    for line in res:
        print(line)


# draw result
from PIL import Image
result = result[0]
image = Image.open(img_path).convert('RGB')
boxes = [line[0] for line in result]
txts = [line[1][0] for line in result]
scores = [line[1][1] for line in result]
print(txts)

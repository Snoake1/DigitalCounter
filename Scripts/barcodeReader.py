# Importing library 
import cv2
import imutils
from pyzbar.pyzbar import decode
import numpy as np


# Make one method to decode the barcode
def barcode_reader(image):
    name = image.split("\\")[-1]
    # read the image in numpy array using cv2
    img = cv2.imread(image)

    # Decode the barcode image 
    detected_barcodes = decode(img)

    # If not detected then print the message 
    if not detected_barcodes:
        print(f"Barcode Not Detected or your barcode is blank/corrupted! File name: {name}")
    else:

        # Traverse through all the detected barcodes in image
        for (i, barcode) in enumerate(detected_barcodes):

            # Locate the barcode position in image 
            (x, y, w, h) = barcode.rect
            # Определяем координаты для обрезки изображения
            top_left = (x - 100, y - 100)
            bottom_right = (x + w + 100, y + h + 100)

            cv2.imwrite(f"ProcessResults/barcoder{i}{name}", img[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]])
            
            # Put the rectangle in image using  
            # cv2 to highlight the barcode 
            rect = cv2.rectangle(img, (x - 100, y - 100),
                                (x + w + 100, y + h + 100),
                                (255, 0, 0), 2)

            if barcode.data != "":
                # Print the barcode data
                print(barcode.data)
                print(barcode.type)
            else:
                pass    #TODO find digits

    cv2.imwrite(f"ProcessResults/barcoder{name}", img)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # convert_filenames_to_lowercase("")
    # Take the image from user
    image = "../Dataset/IMG_20240924_074029.jpg"
    barcode_reader(image)

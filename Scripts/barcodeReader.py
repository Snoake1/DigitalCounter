# Importing library 
import cv2
import imutils
from pyzbar.pyzbar import decode


# Make one method to decode the barcode
def barcode_reader(img):

    # Decode the barcode image 
    detected_barcodes = decode(img)

    # If not detected then print the message 
    if not detected_barcodes:
        return f"Barcode Not Detected or your barcode is blank/corrupted!"
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

    #cv2.imwrite(f"ProcessResults/barcoder{name}", img)
    #cv2.destroyAllWindows()


if __name__ == "__main__":
    # convert_filenames_to_lowercase("")
    # Take the image from user
    image = "../Dataset/IMG_20240924_074029.jpg"
    image = cv2.imread(image)
    barcode_reader(image)

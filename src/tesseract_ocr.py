import os
import pytesseract
from pytesseract import Output
import argparse
import cv2


def text_localization(image_path):
    image = cv2.imread(image_path)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pytesseract.image_to_data(rgb, output_type=Output.DICT)

    for i in range(0, len(results["text"])):
        x = results["left"][i]
        y = results["top"][i]
        w = results["width"][i]
        h = results["height"][i]
        text = results["text"][i]
        conf = int(results["conf"][i])

        if conf > 0:
            text = "".join([c if ord(c) < 128 else "" for c in text]).strip()
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    return image

if __name__ == "__main__":
    image_path = "../data/test/chart_images/split_2/images/PMC2424136___g008.jpg"
    pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
    for file in os.listdir("../data/test/chart_images/split_2/images"):
        image_path = "../data/test/chart_images/split_2/images/" + file
        image = text_localization(image_path)
        print("1")
        save_path = "../data/test/chart_images/split_2/ocr_images/" + file
        cv2.imwrite(save_path, image)


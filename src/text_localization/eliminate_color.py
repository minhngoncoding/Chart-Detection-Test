import cv2
import matplotlib.pyplot as plt
import os
import numpy as np
from src.utils.img_utils import *
from tesseract_ocr import tesseract_ocr

THRESHOLD_AREA = 0.01 # If decrease -> take less components
ASPECT_RAITO = [1/5, 10] # If increase -> take more large components
AREA = [30, 2000]


def read_image(image_path):
    image = cv2.imread(image_path)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return rgb


def color_distance(color1, color2):
    return np.sqrt(np.sum((color1 - color2) ** 2))


def eliminate_color(image):
    # Eliminate color
    black = np.array([0, 0, 0], dtype=np.uint8)
    white = np.array([255, 255, 255], dtype=np.uint8)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i, j][0] > 40 or image[i, j][1] > 40 or image[i, j][2] > 40:
                image[i, j] = white
    return image


def thresholding(image, binaryThresh=180, kernel_size=(3, 3), iter=3, otsu_flag=False):
    # imgFloat = image.astype(np.float32) / 255
    #
    # # Extracting the k channel (black channel)
    # kChannel = 1 - np.max(imgFloat, axis=2)
    # kChannel = (255 * kChannel).astype(np.uint8)  # Convert back to unit 8

    grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Thresholding the k channel
    if otsu_flag:
        _, binaryImage = cv2.threshold(grayImage, binaryThresh, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    else:
        binaryImage = cv2.adaptiveThreshold(grayImage,
                                   255,  # maximum value assigned to pixel values exceeding the threshold
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,  # gaussian weighted sum of neighborhood
                                   cv2.THRESH_BINARY_INV,  # thresholding type
                                   3,  # block size (5x5 window)
                                   3)  # constant

    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    # mask_image = cv2.morphologyEx(binaryImage, cv2.MORPH_DILATE, kernel, iterations=iter)

    return binaryImage


def erode_specific_component(image, label):
    cv2.imshow("Before", image)

    _, labels, _, _ = cv2.connectedComponentsWithStats(image, connectivity=8)

    blank_image = np.zeros_like(image)

    blank_image[labels == label] = 255

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    eroded_image = cv2.erode(blank_image, kernel, iterations=3)

    image[labels == label] = eroded_image[labels == label]

    cv2.imshow("Eroded", image)

    return image


def remove_vertical_line(input_image):
    removed = input_image.copy()
    thresh = cv2.threshold(removed, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
    remove_vertical = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
    cnts = cv2.findContours(remove_vertical, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    for c in cnts:
        cv2.drawContours(removed, [c], -1, (255, 255, 255), 15)

    return removed


def check_connected_component(mask_image):
    total_area = mask_image.shape[0] * mask_image.shape[1]

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_image, connectivity=4)
    remove_cc_image = mask_image.copy()

    for label in range(1, num_labels):
        x, y, w, h, area = stats[label]

        roi = mask_image[y:y+h, x:x+w]

        # print(f"Area {area}: {check_area(area)} and Raito {h/w :.2f}: {check_aspect_raito(h, w)}")
        component_area = w*h

        if not ((component_area / total_area) < THRESHOLD_AREA) or not check_aspect_raito(h,w) or not check_area(area):
            # TODO: Need to check something before removing (sometimes large component connected with other texts)
            remove_cc_image = remove_cc(remove_cc_image, labels, label)
        else:
            # TODO: Remove noises by checking concentration of ROI
            pass
            # if check_concentration_of_components(roi) > 1.5:
            #     remove_cc_image = remove_cc(remove_cc_image, labels, label)

    return remove_cc_image


def remove_cc(mask, labels, label):
    removed_cc = np.zeros_like(mask)
    removed_cc[labels == label] = 255
    # cv2.imshow("REMOVED", removed_cc)
    # cv2.waitKey(0)
    mask_img = cv2.bitwise_xor(mask, removed_cc, mask=None)

    return mask_img


def check_area(area):
    if area < AREA[0] or area > AREA[1]:
        return False

    return True


def check_aspect_raito(height, width):
    aspect_raito = height / width

    if aspect_raito < ASPECT_RAITO[0] or aspect_raito > ASPECT_RAITO[1]:
        return False

    return True


def count_black_and_white_pixels(binary_image):
    h, w = binary_image.shape
    white_pixel = cv2.countNonZero(binary_image)
    black_pixel = h*w - white_pixel

    return white_pixel, black_pixel


def check_concentration_of_components(component):
    roi = component.copy()

    cv2.namedWindow("ROI", cv2.WINDOW_NORMAL)
    cv2.imshow("ROI", roi)

    white_pixel, black_pixel = count_black_and_white_pixels(roi)
    if black_pixel == 0:
        black_pixel = 1

    concentration_score = white_pixel / black_pixel

    print(f"white_pixel: {white_pixel}, black_pixel: {black_pixel}, score: {(white_pixel/black_pixel):.2f}")

    return concentration_score


if __name__ == "__main__":
    test_path = "../image_test"

    image_paths = "../../data/test/chart_images/split_2/images"

    train_paths = "../../data"

    for image_path in os.listdir(image_paths):
        if image_path == "processed":
            continue

        original_image = cv2.imread(f"{image_paths}/{image_path}")
        image = original_image.copy()

        image = cv2.resize(image, (1000, 1000))

        # First mask
        mask_image_otsu = thresholding(image, 250, kernel_size=(1, 1), iter=1, otsu_flag=True)
        # mask_image_normal = thresholding(image, 100, kernel_size=(1,1), iter=3)
        # mask_image = cv2.bitwise_and(mask_image_normal, mask_image_otsu)

        # cv2.imshow("Otsu", mask_image_otsu)
        remove_large_area_image = check_connected_component(mask_image_otsu)


        # ocr_image = tesseract_ocr(remove_large_area_imag)
        result_image = cv2.bitwise_and(image, image, mask=remove_large_area_image)

        vis = stackImages(1, [original_image, mask_image_otsu, remove_large_area_image])
        cv2.imwrite(f"{image_paths}/processed/{image_path}",vis)
    # original_image = f"{image_paths}/processed/PMC5832805___1_HTML.jpg"
    # original_image = cv2.imread(original_image)
    # cv2.imshow("Org", original_image)
    # cv2.waitKey(0)

    print("------------------Finish------------------")

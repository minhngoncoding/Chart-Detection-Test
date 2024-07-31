import cv2
import os
import numpy as np


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


def masking(image):
    imgFloat = image.astype(np.float32) / 255

    # Extracting the k channel (black channel)
    kChannel = 1 - np.max(imgFloat, axis=2)
    kChannel = (255 * kChannel).astype(np.uint8)  # Convert back to unit 8

    # Thresholding the k channel
    binaryThresh = 190
    _, binaryImage = cv2.threshold(kChannel, binaryThresh, 255, cv2.THRESH_BINARY)

    cv2.imshow("binaryImage", binaryImage)

    # Morphology
    kernelSize = (3, 3)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernelSize)
    mask_image = cv2.morphologyEx(binaryImage, cv2.MORPH_CLOSE, kernel, iterations=5)
    mask_image = cv2.morphologyEx(mask_image, cv2.MORPH_DILATE, kernel, iterations=5)

    # cv2.imshow("mask_image", mask_image)
    # cv2.waitKey(0)

    return mask_image

def draw_roi_on_image(image, mask):
    # Find connected components in the mask
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=4)

    # Iterate over each label
    for label in range(1, num_labels):
        # Get the bounding box of the label
        area = stats[label, cv2.CC_STAT_AREA]
        print(f"Area of {label}: {area}")

    return image

if __name__ == "__main__":
    for image_path in os.listdir("../../data/test/chart_images/split_2/images"):
        image = cv2.imread(f"../../data/test/chart_images/split_2/images/{image_path}")
        # Do something with the image
        # cv2.imshow("Before", image)

        mask_image = masking(image)
        # cv2.imshow("Mask", mask_image)

        image = cv2.bitwise_and(image, image, mask=mask_image)
        # cv2.imshow("After", image)
        # cv2.waitKey(0)
        print("Saving image")
        cv2.imwrite(f"../../data/eleminate_color/{image_path}", image)


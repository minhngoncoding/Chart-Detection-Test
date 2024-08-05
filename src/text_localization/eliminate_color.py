import cv2
import matplotlib.pyplot as plt
import os
import numpy as np

THRESHOLD_AREA = 0.1


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


def masking(image, kernel_size=(3, 3), iter=3):
    imgFloat = image.astype(np.float32) / 255

    # Extracting the k channel (black channel)
    kChannel = 1 - np.max(imgFloat, axis=2)
    kChannel = (255 * kChannel).astype(np.uint8)  # Convert back to unit 8

    # Thresholding the k channel
    binaryThresh = 255
    _, binaryImage = cv2.threshold(kChannel, binaryThresh, 255, cv2.THRESH_OTSU)
    # _, binaryImage = cv2.threshold(kChannel, binaryThresh, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    mask_image = cv2.morphologyEx(binaryImage, cv2.MORPH_DILATE, kernel, iterations=iter)

    return mask_image


def visualize_connected_components(mask_image):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_image, connectivity=8)

    # Create an empty RGB image
    output_image = np.zeros((mask_image.shape[0], mask_image.shape[1], 3), dtype=np.uint8)

    # For each label (excluding the background), assign a different color
    for label in range(1, num_labels):
        output_image[labels == label] = [np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)]

    # Display the image
    plt.imshow(output_image)
    plt.show()


def erode_specific_component(image, label):
    cv2.imshow("Before", image)

    _, labels, _, _ = cv2.connectedComponentsWithStats(image, connectivity=8)

    blank_image = np.zeros_like(image)


    blank_image[labels == label] = 255


    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    eroded_image = cv2.erode(blank_image, kernel, iterations=3)

    image[labels == label] = eroded_image[labels == label]

    cv2.imshow("Eroded", image)
    cv2.waitKey(0)

    return image


def check_area_of_mask(mask_image):
    total_area = mask_image.shape[0] * mask_image.shape[1]

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_image, connectivity=8)

    # Iterate over each label
    for label in range(1, num_labels):
        # Get the bounding box of the label
        x, y, w, h, _ = stats[label]
        component_area = w * h
        if component_area / total_area < THRESHOLD_AREA:
            cv2.rectangle(mask_image, (x, y), (x + w, y + h), (255, 255, 255), 2)
        else:
            removed_cc = np.zeros_like(mask_image)
            removed_cc[labels == label] = 255
            cv2.imshow("Removed CC", removed_cc)

            mask_image = cv2.bitwise_xor(mask_image, removed_cc, mask=None)

    cv2.imshow("After removed", mask_image)

    return mask_image


def draw_roi_on_image(image):
    # Find connected components in the mask

    return image


if __name__ == "__main__":
    test_path = "../image_test"

    image_paths = "../../data/test/chart_images/split_2/images/"

    for image_path in os.listdir(test_path):
        try:
            # image_path = "PMC3338705___g005.jpg"
            original_image = cv2.imread(f"{test_path}/{image_path}")
            image = original_image.copy()
        except:
            continue

        # First mask
        mask_image = masking(image, kernel_size=(3, 3), iter=1)

        # Remove unnecessary components (too large to be a text)
        mask_image = check_area_of_mask(mask_image)

        # Dilate
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask_image = cv2.morphologyEx(mask_image, cv2.MORPH_DILATE, kernel, iterations=3)

        # cv2.imshow("Dilated", mask_image)

        result_image = cv2.bitwise_and(image, image, mask=mask_image)

        cv2.imshow("Final Result", result_image)
        cv2.waitKey(0)
        # vis = np.concatenate((original_image,result_image), axis=1)
        # cv2.imwrite(f"{test_path}/processed/{image_path}", vis)


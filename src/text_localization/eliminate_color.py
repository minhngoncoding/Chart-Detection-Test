import cv2
import matplotlib.pyplot as plt
import os
import numpy as np
from src.utils.img_utils import stackImages

THRESHOLD_AREA = 0.005
ASPECT_RAITO = [1/10, 15]
AREA = [20, 2000]


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


def masking(image, binaryThresh=180, kernel_size=(3, 3), iter=3, otsu_flag=False):
    imgFloat = image.astype(np.float32) / 255

    # Extracting the k channel (black channel)
    kChannel = 1 - np.max(imgFloat, axis=2)
    kChannel = (255 * kChannel).astype(np.uint8)  # Convert back to unit 8

    # Thresholding the k channel
    if otsu_flag:
        _, binaryImage = cv2.threshold(kChannel, binaryThresh, 255, cv2.THRESH_OTSU)
    else:
        _, binaryImage = cv2.threshold(kChannel, binaryThresh, 255, cv2.THRESH_BINARY)

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

    # # Display the image
    # plt.imshow(output_image)
    # plt.show()
    return output_image


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


def check_connected_component(mask_image):
    total_area = mask_image.shape[0] * mask_image.shape[1]

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_image, connectivity=4)

    temp = mask_image.copy()
    for label in range(1, num_labels):
        x, y, w, h, area = stats[label]

        roi = mask_image[y:y+h, x:x+w]

        # print(f"Area {area}: {check_area(area)} and Raito {h/w :.2f}: {check_aspect_raito(h, w)}")
        # cv2.imshow("ROI", roi)
        component_area = w*h

        temp = cv2.rectangle(temp, (x,y), (x+w, y+h), (255,255,255), 2)
        if not ((component_area / total_area) < THRESHOLD_AREA) or not check_aspect_raito(h,w) or not check_area(area):
            # TODO: Need to check something before removing (sometimes large component connected with other texts)
            mask_image = remove_cc(mask_image, labels, label)
            # mask_image = cv2.rectangle(mask_image, (x,y), (x+w, y+h), (0,255,0), 2)
            # cv2.imshow("Visualize:", mask_image)
            # cv2.waitKey(0)
        else:
            # TODO: Remove noises by checking concentration of ROI
            pass

    return temp, mask_image


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


def check_concentration_of_components(org_image, m_image):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(m_image, connectivity=4)
    for label in range(1, num_labels):

        x, y, w, h, area = stats[label]
        roi = original_image[y:y+h, x:x+w]

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (3,3), 0)

        (T, threshInv) = cv2.threshold(blurred, 200, 255,
                                       cv2.THRESH_BINARY_INV)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        erode_roi = cv2.morphologyEx(threshInv, cv2.MORPH_ERODE, kernel, iterations=2)

        cv2.rectangle(org_image, (x, y), (x + w, y + h), (0, 255, 0), 1)

        # cv2.imshow("ROI", roi)
        # cv2.imshow("Erode", erode_roi)
        # cv2.imshow("ROI Binary", threshInv)
        # cv2.imshow("Original", org_image)

        # cv2.waitKey(0)
    return org_image, m_image


if __name__ == "__main__":
    test_path = "../image_test"

    image_paths = "../../data/test/chart_images/split_2/images/"

    for image_path in os.listdir(test_path):
        try:
            # image_path = "PMC3338705___g005.jpg"
            original_image = cv2.imread(f"{test_path}/{image_path}")
            image = original_image.copy()

            image = cv2.resize(image, (1000, 1000))
        except:
            continue

        print(image.shape)
        # First mask
        mask_image_otsu = masking(image, 250, kernel_size=(3,3), iter=1, otsu_flag=True)
        mask_image_normal = masking(image, 100, kernel_size=(1,1), iter=2)

        # all_masks = stackImages(1, [original_image, mask_image_otsu, mask_image_normal])
        # cv2.imshow("All Mask", all_masks)

        mask_image = cv2.bitwise_and(mask_image_normal, mask_image_otsu)

        # visualize = visualize_connected_components(mask_image)
        # cv2.imshow("Final Mask", mask_image)
        # cv2.waitKey(0)
        # Remove unnecessary components (too large to be a text)
        alo, remove_large_area_image = check_connected_component(mask_image)


        # cv2.imshow("After remove: ", remove_large_area_image)
        # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        # remove_large_area_image = cv2.morphologyEx(mask_image, cv2.MORPH_CLOSE, kernel, iterations=3)
        result_image = cv2.bitwise_and(image, image, mask=remove_large_area_image)

        concat = stackImages(1, [original_image, result_image])
        cv2.imshow("Result", concat)
        # cv2.waitKey()
        # cv2.waitKey(0)
        vis = stackImages(1, [original_image, remove_large_area_image, alo])
        cv2.imwrite(f"{test_path}/processed/{image_path}", vis)

    print("------------------Finish------------------")

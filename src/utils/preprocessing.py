import os
from rotate import ImageRotator
import cv2


def resize(input_image_path, size):
    image = cv2.imread(input_image_path)

    if image is None:
        print(f"Error: Unable to load image from {input_image_path}")
        return None

    # Resize the image
    resized_image = cv2.resize(image, size)

    return resized_image


def otsu_thresholding(resized_image):
    # Load the image in grayscale
    if resized_image is None:
        print("Error: The input image is not valid.")
        return None, None

        # Convert the input image to grayscale
    grayscale_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

    # Apply Otsu's thresholding
    threshold_value, thresholded_image = cv2.threshold(
        grayscale_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Save the thresholded image to the specified output path

    # Return the thresholded image and threshold value
    return thresholded_image, threshold_value


if __name__ == "__main__":
    input_image_dir = "../../data/raw_data/train/final_data/otsu"
    output_image_dir = "../../data/raw_data/train/final_data/rotated_images"

    input_label_dir = "../../data/raw_data/train/final_data/yolo_label"
    output_label_dir = "../../data/raw_data/train/final_data/yolo_rotated_label"


    # full_test_path = "../../data/test/chart_images/split_2/images"
    # output_test_path = "../../data/test/chart_images/split_2/otsu"

    rotator = ImageRotator(input_image_dir, input_label_dir, output_image_dir, output_label_dir)
    rotator.process_all_images()


    # for image_fname in os.listdir(input_image_dir):
    #     image_path = os.path.join(input_image_dir, image_fname)
    #     resized_img = resize(input_image_path=image_path, size=(1200, 1200))
    #     thresholded_image, _ = otsu_thresholding(resized_img)


import json
import os
from PIL import Image


def convert_to_yolo(task_output, image_width, image_height, class_id=0):
    yolo_annotations = []

    for text_block in task_output["text_blocks"]:
        polygon = text_block["polygon"]

        # Calculate the bounding box coordinates
        x_min = min(polygon["x0"], polygon["x1"], polygon["x2"], polygon["x3"])
        x_max = max(polygon["x0"], polygon["x1"], polygon["x2"], polygon["x3"])
        y_min = min(polygon["y0"], polygon["y1"], polygon["y2"], polygon["y3"])
        y_max = max(polygon["y0"], polygon["y1"], polygon["y2"], polygon["y3"])

        # Calculate center, width, and height
        x_center = (x_min + x_max) / 2.0
        y_center = (y_min + y_max) / 2.0
        width = x_max - x_min
        height = y_max - y_min

        # Normalize the values
        x_center /= image_width
        y_center /= image_height
        width /= image_width
        height /= image_height

        # Create the YOLO formatted string
        yolo_annotation = f"{class_id} {x_center} {y_center} {width} {height}"
        yolo_annotations.append(yolo_annotation)

    return yolo_annotations


# Example usage:
task_output = {
    "text_blocks": [
        {
            "id": 0,
            "polygon": {"x0": 248, "x1": 279, "x2": 279, "x3": 248, "y0": 10, "y1": 10, "y2": 24, "y3": 24},
            "text": "93.6"
        },
        {
            "id": 1,
            "polygon": {"x0": 23, "x1": 46, "x2": 46, "x3": 23, "y0": 0, "y1": 0, "y2": 13, "y3": 13},
            "text": "6.4"
        }
    ]
}

import os
import random


def create_train_valid_txt(image_folder, train_split=0.8):
    # Get a list of all image filenames in the folder
    image_fnames = [f for f in os.listdir(image_folder) if
                    f.endswith('.jpg') and os.path.isfile(os.path.join(image_folder, f))]

    # Shuffle the images to ensure random distribution
    random.shuffle(image_fnames)

    # Calculate the split index
    split_idx = int(len(image_fnames) * train_split)

    # Split the images into training and validation sets
    train_images = image_fnames[:split_idx]
    valid_images = image_fnames[split_idx:]

    # Paths to train.txt and valid.txt
    train_txt_path = os.path.join(image_folder, "train.txt")
    valid_txt_path = os.path.join(image_folder, "valid.txt")

    # Write the training images to train.txt
    with open(train_txt_path, "w") as f:
        for index, image_fname in enumerate(train_images):
            if index == len(train_images) - 1:
                f.write(image_folder+ image_fname)
            else:
                f.write(image_folder+ image_fname + "\n")

    # Write the validation images to valid.txt
    with open(valid_txt_path, "w") as f:
        for index, image_fname in enumerate(valid_images):
            if index == len(valid_images) - 1:
                f.write(image_folder+ image_fname)
            else:
                f.write(image_folder+ image_fname + "\n")


# Usage example:
# create_train_valid_txt("/path/to/your/image/folder")

def create_yolo_folder(image_folder, json_folder, yolo_folder):
    for json_fname in os.listdir(json_folder):
        file_path = os.path.join(json_folder, json_fname)
        image_found = False
        base_name = os.path.splitext(json_fname)[0]
        image_fname = base_name + ".jpg"
        image_path = os.path.join(image_folder, image_fname)

        if not os.path.exists(yolo_folder):
            os.makedirs(yolo_folder)
        yolo_path = os.path.join(yolo_folder, base_name+".txt")

        if os.path.exists(image_path):
            image_found = True
            with Image.open(image_path) as img:
                width, height = img.size
                print(f"Image {base_name} has height: {height} and width: {width}")

        if not image_found:
            print(f"No corresponding image found for {base_name}.")
            continue

        with open(file_path, "r") as f:
            data = json.load(f)
            annotation_data = convert_to_yolo(data["task2"]["output"], image_width=width, image_height=height)

            with open(yolo_path, "w") as f:
                for annotate in annotation_data:
                    f.write(annotate + "\n")


if __name__ == "__main__":
    json_folder = "../../data/raw_data/train/final_data/json"
    image_folder = "../../data/raw_data/train/final_data/images"
    yolo_folder = "../../data/raw_data/train/final_data/yolo"

    test_image_folder = "../../data/test/chart_images/split_2/images"
    test_json = "../../data/test/final_full_GT/split_2/annotations_JSON"
    yolo_test_folder = "../../data/test/final_full_GT/split_2/yolo"

    # create_yolo_folder(test_image_folder, test_json, yolo_test_folder)

    create_train_valid_txt(image_folder="../../data/darknet_rotated/train/",
                           train_split=0.8)


    # create_train_valid_txt(image_folder)
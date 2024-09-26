import json
import os
import shutil
import pickle
from typing import List


def read_json_file(fpath, fname, output):
    f = open(fpath)
    dct_data = {}

    data = json.load(f)

    dct_data["file_name"] = fname
    dct_data["text_data"] = data["task2"]["output"]["text_blocks"]

    # List of Dictionary: [{file_name: str, text_block = []}, {}, {}]

    output.append(dct_data)

    return output

def delete_1kb_files(directory):
    """Deletes files of exactly 1KB size within the specified directory.

    Args:
        directory (str): The path to the directory to be scanned.
    """

    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            file_size = os.path.getsize(file_path)
            if file_size <= 1024:  # 1KB in bytes
                os.remove(file_path)
                print(f"Deleted {file_path}")

def get_all_fname(input):
    files_name = []
    for file in input:
        files_name.append(os.path.splitext(file["file_name"])[0])

    return files_name


def copy_images_by_name(source_root_folder, image_names_list, destination_folder):
    # Check if destination folder exists, if not, create it
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    # Walk through all subfolders and files in the source root folder
    for root, dirs, files in os.walk(source_root_folder):
        print(len(files))
        for file_name in files:
            # Check if the current file (without extension) exists in the image_names_list
            if os.path.splitext(file_name)[0] in image_names_list:
                # Full path of the image file
                source_image_path = os.path.join(root, file_name)
                # Destination path where the image will be copied
                destination_image_path = os.path.join(destination_folder, file_name)
                # Copy image to destination folder
                shutil.copy(source_image_path, destination_image_path)
                print(f"Copied: {file_name}")



def combine_folders(source_folder, destination_folder):
    """Combines files from multiple folders into a single destination folder.

    Args:
      source_folder: The path to the folder containing the subfolders.
      destination_folder: The path to the destination folder.
    """

    for root, dirs, files in os.walk(source_folder):
        for file in dirs:
            print(file)
            source_path = os.path.join(root, file)
            destination_path = os.path.join(destination_folder, file)
            shutil.move(source_path, destination_path)


if __name__ == "__main__":
    folder_to_delete = ["horizontal_bar", 'line', "scatter", 'vertical_bar', "vertical_box"]
    directory_path = "../../data/train/annotations_JSON"
    image_path = "../../data/train/images"
    data_path = "../../data/train/final_data/json"

    with open("all_fname.pkl", "rb") as f:
        filename_list = pickle.load(f)

    destination_folder ="../../data/train/final_data/images"
    source_image_folder = "../../data/train/images"

    copy_images_by_name(source_image_folder, filename_list, destination_folder)





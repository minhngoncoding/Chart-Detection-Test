import os
import shutil
import json
import collections

def read_json_file(file_path):
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
        return data
    except FileNotFoundError:
        print(f"The file at {file_path} was not found.")
    except json.JSONDecodeError:
        print(f"The file at {file_path} is not a valid JSON file.")
    except Exception as e:
        print(f"An error occurred: {e}")

def make_dirs(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path, exist_ok=True)

def create_data_structure(source, source_images, destination, destination_img):
    for folder in os.listdir(source):
        folder_path = os.path.join(source, folder)
        folder_img_path = os.path.join(source_images, folder)
        destination_folder = os.path.join(destination, folder)
        destination_img_folder = os.path.join(destination_img, folder)
        make_dirs(destination_folder)
        make_dirs(destination_img_folder)
        for file in os.listdir(folder_path):
            print(file)
            file_img = str(file)[0:-5] + '.jpg'
            file_path = os.path.join(folder_path, file)
            img_path = os.path.join(folder_img_path, file_img)
            data = read_json_file(file_path)
            output = data['task2']
            if output is not None:
                shutil.copy(file_path, destination_folder)
                shutil.copy(img_path, destination_img_folder)
    
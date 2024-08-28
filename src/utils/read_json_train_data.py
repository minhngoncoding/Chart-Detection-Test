import json
import os
import shutil
import pickle

def read_json_file(fpath, fname, output):
    f = open(fpath)
    dct_data = {}

    data = json.load(f)

    dct_data["file_name"] = fname
    dct_data["text_data"] = data["task2"]["output"]["text_blocks"]

    # List of Dictionary: [{file_name: str, text_block = []}, {}, {}]

    output.append(dct_data)

    return output

def get_image_has_json(image_dir, fname_lst):
    pass

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
    print(input)
    for file in input:
        files_name.append(file["file_name"])

    return files_name


def find_files_in_subfolders(folder_path, fname_list):
  """Finds files in subfolders whose names are in the given list.

  Args:
    folder_path: The path to the root folder.
    fname_list: A list of file names to search for.

  Returns:
    A list of file paths that match the criteria.
  """

  matching_files = []
  for root, dirs, files in os.walk(folder_path):
    for file in files:
      if file in fname_list:
        matching_files.append(os.path.join(root, file))
  return matching_files

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
    data_path = "../../data/train/annotations_JSON/real_data/json"

    # output = []
    # for fname in os.listdir(data_path):
    #     read_json_file(f"{data_path}/{fname}", fname=fname, output=output)
    #
    # all_fname = get_all_fname(output)

    with open("all_fname.pkl", "rb") as f:
        filename_list = pickle.load(f)

    find_files_in_subfolders("", filename_list)
    # for fname in filename_list:
    #     print(fname[:-5])




import os
import random
import shutil


def split_dataset(images_dir, labels_dir, output_dir, train_ratio=0.8):
    # Get list of all image files
    image_files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]

    # Shuffle the list to ensure random distribution
    random.shuffle(image_files)

    # Calculate split index
    split_index = int(len(image_files) * train_ratio)

    # Split into train and valid lists
    train_files = image_files[:split_index]
    valid_files = image_files[split_index:]

    # Create output directories
    train_images_dir = os.path.join(output_dir, 'images', 'train')
    valid_images_dir = os.path.join(output_dir, 'images', 'valid')
    train_labels_dir = os.path.join(output_dir, 'labels', 'train')
    valid_labels_dir = os.path.join(output_dir, 'labels', 'valid')

    os.makedirs(train_images_dir, exist_ok=True)
    os.makedirs(valid_images_dir, exist_ok=True)
    os.makedirs(train_labels_dir, exist_ok=True)
    os.makedirs(valid_labels_dir, exist_ok=True)

    # Function to copy files
    def copy_files(file_list, src_images_dir, src_labels_dir, dst_images_dir, dst_labels_dir):
        for filename in file_list:
            # Copy image file
            src_image_path = os.path.join(src_images_dir, filename)
            dst_image_path = os.path.join(dst_images_dir, filename)
            shutil.copyfile(src_image_path, dst_image_path)

            # Copy label file with the same base name but .txt extension
            label_filename = os.path.splitext(filename)[0] + '.txt'
            src_label_path = os.path.join(src_labels_dir, label_filename)
            dst_label_path = os.path.join(dst_labels_dir, label_filename)

            if os.path.exists(src_label_path):
                shutil.copyfile(src_label_path, dst_label_path)
            else:
                print(f"Warning: Label file '{label_filename}' not found for image '{filename}'.")

    # Copy training files
    copy_files(train_files, images_dir, labels_dir, train_images_dir, train_labels_dir)

    # Copy validation files
    copy_files(valid_files, images_dir, labels_dir, valid_images_dir, valid_labels_dir)

    print("Dataset splitting completed successfully.")
    print(f"Total images: {len(image_files)}")
    print(f"Training images: {len(train_files)}")
    print(f"Validation images: {len(valid_files)}")


if __name__ == '__main__':

    images_dir = "../../data/raw_data/train/final_data/rotated_images"
    labels_dir = "../../data/raw_data/train/final_data/yolo_rotated_label"
    output_dir = "../../data/yolov8"


    split_dataset(images_dir,labels_dir,output_dir)

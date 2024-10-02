import os
import cv2
import numpy as np

class ImageRotator:
    def __init__(self, input_img_dir, input_label_dir, output_img_dir, output_label_dir, angles=[-45, 45, -90, 90]):
        self.input_img_dir = input_img_dir
        self.input_label_dir = input_label_dir
        self.output_img_dir = output_img_dir
        self.output_label_dir = output_label_dir
        self.angles = angles

        # Create output directories if they don't exist
        if not os.path.exists(self.output_img_dir):
            os.makedirs(self.output_img_dir)
        if not os.path.exists(self.output_label_dir):
            os.makedirs(self.output_label_dir)

    def rotate_image(self, image, angle):
        """Rotate image by the specified angle."""
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h))
        return rotated, M

    def rotate_bbox(self, bbox, M, img_w, img_h):
        """Rotate bounding box using the transformation matrix and normalize coordinates to [0, 1]."""
        x_center, y_center, width, height = bbox

        # Convert the center to absolute coordinates
        x_center_abs = x_center * img_w
        y_center_abs = y_center * img_h
        width_abs = width * img_w
        height_abs = height * img_h

        # Get the four corners of the bounding box
        top_left = np.array([x_center_abs - width_abs / 2, y_center_abs - height_abs / 2, 1])
        top_right = np.array([x_center_abs + width_abs / 2, y_center_abs - height_abs / 2, 1])
        bottom_left = np.array([x_center_abs - width_abs / 2, y_center_abs + height_abs / 2, 1])
        bottom_right = np.array([x_center_abs + width_abs / 2, y_center_abs + height_abs / 2, 1])

        # Apply the rotation matrix to each corner
        top_left_rot = np.dot(M, top_left)
        top_right_rot = np.dot(M, top_right)
        bottom_left_rot = np.dot(M, bottom_left)
        bottom_right_rot = np.dot(M, bottom_right)

        # Find the new bounding box from the rotated corners
        x_coords = [top_left_rot[0], top_right_rot[0], bottom_left_rot[0], bottom_right_rot[0]]
        y_coords = [top_left_rot[1], top_right_rot[1], bottom_left_rot[1], bottom_right_rot[1]]

        x_min = max(min(x_coords), 0)  # Ensure coordinates are within bounds
        x_max = min(max(x_coords), img_w)
        y_min = max(min(y_coords), 0)
        y_max = min(max(y_coords), img_h)

        # Calculate the new center, width, and height
        x_center_rot = (x_min + x_max) / 2
        y_center_rot = (y_min + y_max) / 2
        width_rot = x_max - x_min
        height_rot = y_max - y_min

        # Convert back to normalized coordinates (ensure they are within [0, 1])
        x_center_rot_norm = max(min(x_center_rot / img_w, 1), 0)
        y_center_rot_norm = max(min(y_center_rot / img_h, 1), 0)
        width_rot_norm = max(min(width_rot / img_w, 1), 0)
        height_rot_norm = max(min(height_rot / img_h, 1), 0)

        return [x_center_rot_norm, y_center_rot_norm, width_rot_norm, height_rot_norm]

    def update_labels(self, label_path, M, img_w, img_h):
        """Update labels by rotating bounding boxes."""
        updated_bboxes = []
        with open(label_path, 'r') as file:
            lines = file.readlines()
            for line in lines:
                values = line.strip().split()
                class_id = int(values[0])
                bbox = list(map(float, values[1:]))
                rotated_bbox = self.rotate_bbox(bbox, M, img_w, img_h)
                updated_bboxes.append([class_id] + rotated_bbox)
        return updated_bboxes

    def save_updated_labels(self, label_path, updated_bboxes):
        """Save updated labels to file."""
        with open(label_path, 'w') as file:
            for bbox in updated_bboxes:
                line = " ".join(map(str, bbox))
                file.write(f"{line}\n")

    def process_image(self, img_name):
        """Process a single image and its corresponding label."""
        img_path = os.path.join(self.input_img_dir, img_name)
        label_path = os.path.join(self.input_label_dir, os.path.splitext(img_name)[0] + '.txt')

        image = cv2.imread(img_path)
        img_h, img_w = image.shape[:2]

        for angle in self.angles:
            # Rotate image and get the transformation matrix
            rotated_img, M = self.rotate_image(image, angle)
            rotated_img_name = f"{os.path.splitext(img_name)[0]}_{angle}.jpg"
            cv2.imwrite(os.path.join(self.output_img_dir, rotated_img_name), rotated_img)

            # Update the labels using the rotation matrix
            updated_bboxes = self.update_labels(label_path, M, img_w, img_h)
            rotated_label_name = f"{os.path.splitext(img_name)[0]}_{angle}.txt"
            self.save_updated_labels(os.path.join(self.output_label_dir, rotated_label_name), updated_bboxes)

    def process_all_images(self):
        """Process all images in the input directory."""
        for img_name in os.listdir(self.input_img_dir):
            self.process_image(img_name)

    @staticmethod
    def check_rotated_bounding_boxes(image_path, label_path):
        # Load the image
        image = cv2.imread(image_path)
        img_h, img_w = image.shape[:2]

        # Read the label file
        with open(label_path, 'r') as f:
            lines = f.readlines()

        for line in lines:
            # YOLO format: class_id x_center y_center width height (normalized)
            class_id, x_center, y_center, width, height = map(float, line.strip().split())

            # Convert normalized coordinates to pixel values
            x_center_abs = int(x_center * img_w)
            y_center_abs = int(y_center * img_h)
            width_abs = int(width * img_w)
            height_abs = int(height * img_h)

            # Calculate the top-left and bottom-right corners of the bounding box
            x_min = int(x_center_abs - width_abs / 2)
            y_min = int(y_center_abs - height_abs / 2)
            x_max = int(x_center_abs + width_abs / 2)
            y_max = int(y_center_abs + height_abs / 2)

            # Draw the bounding box on the image
            color = (0, 255, 0)  # Green color for the box
            thickness = 2
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, thickness)

            # Add label text above the bounding box
            cv2.putText(image, "text", (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        # Display the image with bounding boxes
        cv2.imshow('Image with Bounding Boxes', image)
        cv2.waitKey(0)

        # Save the output image if needed
        # cv2.imwrite('output_image_with_boxes.jpg', image)

    # Example usage


# Usage Example:
if __name__ == '__main__':
    # Directories for input and output images and labels

    input_image_dir = "../../data/raw_data/train/final_data/otsu"
    output_image_dir = "../../data/raw_data/train/final_data/rotated_images"

    input_label_dir = "../../data/raw_data/train/final_data/yolo_label"
    output_label_dir = "../../data/raw_data/train/final_data/yolo_rotated_label"
    # Initialize the ImageRotator class
    rotator = ImageRotator(input_image_dir, input_label_dir, output_image_dir, output_label_dir)
    # rotator.process_all_images()

    #Check rotated result
    for angle in ["-45", "45", "-90", "90"]:
        image_path = f"../../data/raw_data/train/final_data/rotated_images/PMC514554___1471-2458-4-32-2_{angle}.jpg"
        label_path = f"../../data/raw_data/train/final_data/yolo_rotated_label/PMC514554___1471-2458-4-32-2_{angle}.txt"
        rotator.check_rotated_bounding_boxes(image_path, label_path)
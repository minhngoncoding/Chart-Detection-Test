import os
import numpy as np
import cv2
from src.utils.img_utils import *


def calculate_iou(box1, box2):
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0  # No intersection

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    iou = intersection_area / float(box1_area + box2_area - intersection_area)
    return iou


def read_bounding_boxes(txt_file, img_width, img_height):
    boxes = []
    with open(txt_file, 'r') as file:
        lines = file.readlines()
        for line in lines:
            _ , x_center, y_center, width, height = map(float, line.strip().split())
            x_center *= img_width
            y_center *= img_height
            width *= img_width
            height *= img_height
            x1 = x_center - (width / 2)
            y1 = y_center - (height / 2)
            x2 = x_center + (width / 2)
            y2 = y_center + (height / 2)
            boxes.append([x1, y1, x2, y2])  # Append class ID along with box
    return boxes


def draw_bounding_boxes(image, boxes, color=(0, 255, 0), thickness=2):
    """
    Draw bounding boxes on the image.

    Args:
    image: The image on which to draw the bounding boxes.
    boxes: List of bounding boxes [(x1, y1, x2, y2), ...]
    color: The color of the bounding box (default is green).
    thickness: Thickness of the bounding box lines (default is 2).

    Returns:
    image: The image with bounding boxes drawn.
    """
    for (x1, y1, x2, y2) in boxes:
        image = cv2.rectangle(image, (int(x1),int(y1)), (int(x2),int(y2)), color, thickness)
    return image


def create_bb_image_folder(image_folder, label_folder, output_folder):
    # Ensure output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Loop through all images
    for image_file in os.listdir(image_folder):
        if image_file.endswith(('.jpg', '.png', '.jpeg')):  # Process image files
            # Get the full image file path
            image_path = os.path.join(image_folder, image_file)

            # Read the image
            image = cv2.imread(image_path)
            img_height, img_width = image.shape[:2]  # Get image dimensions

            # Corresponding label file (assuming .txt file with the same name as image)
            label_file = os.path.splitext(image_file)[0] + '.txt'
            label_path = os.path.join(label_folder, label_file)

            # Check if the label file exists
            if os.path.exists(label_path):
                # Read bounding boxes from the label file
                boxes = read_bounding_boxes(label_path, img_width, img_height)
                # Draw the bounding boxes on the image
                image_with_boxes = draw_bounding_boxes(image, boxes)

                # Save the image with bounding boxes to the output folder
                output_image_path = os.path.join(output_folder, image_file)
                cv2.imwrite(output_image_path, image_with_boxes)
                print(f'Saved image with bounding boxes: {output_image_path}')
            else:
                print(f'No label file found for image: {image_file}')


def infer_iou_from_txt(gt_file_path, pred_file_path):
    ground_truth_boxes = read_bounding_boxes(gt_file_path, img_width, img_height)
    predicted_boxes = read_bounding_boxes(pred_file_path, img_width, img_height)

    # Keep track of matched boxes
    matched_gt_boxes = set()
    matched_pred_boxes = set()
    all_ious = list()

    # Loop through ground truth boxes
    for i, gt_box in enumerate(ground_truth_boxes):
        best_iou = 0
        best_pred_idx = -1

        # Loop through predicted boxes to find the best match by IoU
        for j, pred_box in enumerate(predicted_boxes):
            iou = calculate_iou(gt_box, pred_box)
            if iou > best_iou:
                best_iou = iou
                best_pred_idx = j

        # If a good match is found, store it
        if best_iou > 0.5:
            # print(f'Image {gt_filename}, GT Box {i} matched with Pred Box {best_pred_idx}, IoU: {best_iou}')
            all_ious.append(best_iou)
            matched_gt_boxes.add(i)
            matched_pred_boxes.add(best_pred_idx)

    # Handle unmatched boxes
    missed_gt_boxes = len(ground_truth_boxes) - len(matched_gt_boxes)  # Unmatched ground truth (false negatives)
    false_pred_boxes = len(predicted_boxes) - len(matched_pred_boxes)  # Unmatched predictions (false positives)

    average_ious = round(sum(all_ious) / max(len(ground_truth_boxes), len(predicted_boxes)) , 3) if all_ious else 0

    # print(f"Average IOUs: {average_ious}")
    # print(f'Image {gt_filename}: Missed GT Boxes: {missed_gt_boxes}, False Predicted Boxes: {false_pred_boxes}')

    #TODO: handle over-segmentation

    return average_ious, missed_gt_boxes, false_pred_boxes

if __name__ == "__main__":
    ground_truth_dir = "labels/ground_truth"
    predict_dir = "labels/predicted"

    img_width, img_height = 1200, 1200

    label_ground_truth = "test_infer/labels/ground_truth"
    label_predicted = "test_infer/labels/predicted"

    image_ground_truth= "test_infer/images/ground_truth"
    image_predicted = "test_infer/images/predicted"

    output_folder = "output/"
    all_avg = []
    for gt_fname in os.listdir(label_ground_truth):
        if gt_fname.endswith('.txt'):
            pred_file_path = os.path.join(label_predicted, gt_fname)
            gt_file_path = os.path.join(label_ground_truth, gt_fname)

            fname =  os.path.splitext(gt_fname)[0]
            image_fname = fname + ".jpg"

            if os.path.exists(pred_file_path) and os.path.exists(gt_file_path):
                average_iou, missed_gt_boxes, false_pred_boxes = infer_iou_from_txt(gt_file_path, pred_file_path)

                gt_image = cv2.imread(os.path.join(image_ground_truth, image_fname))
                pred_image = cv2.imread(os.path.join(image_predicted, image_fname))

                vis = stackImages(1, [gt_image, pred_image])

                text_pos = (img_width - 400, 50)
                all_avg.append(average_iou)
                vis = cv2.putText(vis, "Average IOU: " + str(average_iou), text_pos, cv2.FONT_HERSHEY_SIMPLEX, 1, color=(255, 0, 0), thickness=2 )
                vis = cv2.putText(vis, "Misses GT: " + str(missed_gt_boxes), (img_width-400, 100),cv2.FONT_HERSHEY_SIMPLEX,1, color=(255, 0, 0), thickness=2)
                vis = cv2.putText(vis, "False Pred: " + str(false_pred_boxes), (img_width-400,150),cv2.FONT_HERSHEY_SIMPLEX,1, color=(255, 0, 0), thickness=2)

                output_name = "join_" + image_fname
                # cv2.imwrite(os.path.join(output_folder,output_name), vis)
            else:
                print(f"File {pred_file_path} not found")

    print(all_avg)
    print(f"All Average IOU: {round(sum(all_avg) / len(all_avg), 3)}")
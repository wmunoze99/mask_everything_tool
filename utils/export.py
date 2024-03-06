import cv2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np


def _find_contours(mask):
    convert_to_gray_scale = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
    contours, _ = cv2.findContours(convert_to_gray_scale, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    segmentation = []

    for contour in contours:
        contour = contour.flatten().tolist()
        contour_pairs = [(contour[i], contour[i + 1]) for i in range(0, len(contour), 2)]

        segmentation.append([int(coord) for pair in contour_pairs for coord in pair])

    return contours, segmentation


def export_to_coco(mask, category):
    contours, segmentation = _find_contours(mask)

    annotation = {
        "id": 1,
        "image_id": 1,
        "category_id": category,
        "segmentation": segmentation,
        "area": int(cv2.contourArea(contours[0])),
        "bbox": [int(x) for x in cv2.boundingRect(contours[0])],
        "iscrowd": 0
    }

    return annotation


def export_to_yolo(mask, category):
    # Convert to gray scale
    img_on_gray_scale = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
    # Convert image to numpy array
    img_array = np.array(img_on_gray_scale)

    # Get the dimensions of the image
    height, width = img_array.shape

    # Find the coordinates of the non-zero (white) pixels
    non_zero_cords = np.argwhere(img_array > 0)

    # Calculate the bounding box of the non-zero elements
    y_min, x_min = non_zero_cords.min(axis=0)
    y_max, x_max = non_zero_cords.max(axis=0)

    # Calculate the center, width and height of the bounding box
    center_x = (x_min + x_max) / 2
    center_y = (y_min + y_max) / 2
    bbox_width = x_max - x_min
    bbox_height = y_max - y_min

    # Uncomment to test bbox and center calculation
    """
    testing = cv2.rectangle(mask, (x_min, y_min), (x_max, y_max), (255, 0, 255), 2)
    testing = cv2.circle(testing, (int(center_x), int(center_y)), 5, (255,255,0), -1)

    image = Image.fromarray(testing)
    image.show()
    """

    # Normalize these values by the dimensions of the image
    norm_center_x = center_x / width
    norm_center_y = center_y / height
    norm_width = bbox_width / width
    norm_height = bbox_height / height

    return f"{category-1} {norm_center_x} {norm_center_y} {norm_width} {norm_height}\n".strip()

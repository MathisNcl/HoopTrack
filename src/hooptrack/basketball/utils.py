"""Utils functions for InferencePipeline"""
import os
from typing import Any

import numpy as np


def iou(box1: list[Any], box2: list[Any]) -> float:
    """Compute intersection over union metric

    Args:
        box1 (list[Any]): bbox1 coords, label and score
        box2 (list[Any]): bbox2 coords, label and score

    Returns:
        float: iou
    """
    intersection = np.maximum(0.0, np.minimum(box1[2], box2[2]) - np.maximum(box1[0], box2[0])) * np.maximum(
        0.0, np.minimum(box1[3], box2[3]) - np.maximum(box1[1], box2[1])
    )
    union = (box1[2] - box1[0]) * (box1[3] - box1[1]) + (box2[2] - box2[0]) * (box2[3] - box2[1]) - intersection
    return intersection / union


def determine_filename(root: str = "") -> str:
    """Determine the filename of the outputed video

    Args:
        root (str): root path. Default "".

    Returns:
        str: filename
    """
    i = 0
    while os.path.exists(os.path.join(root, f"output_vid_{i}.mp4")):
        i += 1
    return os.path.join(root, f"output_vid_{i}.mp4")


def calculate_center_distance(box1: list[int], box2: list[int]) -> float:
    """Calculate Euclidean distance between centers of two bounding boxes.

    Args:
        box1 (list[int]): 4 coords bbox
        box2 (list[int]): 4 coords bbox

    Returns:
        float: distance
    """
    x1, y1, _, _ = box1
    x2, y2, _, _ = box2

    center_distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return center_distance

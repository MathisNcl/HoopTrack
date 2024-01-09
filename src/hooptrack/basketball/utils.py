"""Utils functions for InferencePipeline"""
import os
from typing import Any


def iou(box1: list[Any], box2: list[Any]) -> float:
    """Compute intersection over union metric

    Args:
        box1 (list[Any]): bbox1 coords, label and score
        box2 (list[Any]): bbox2 coords, label and score

    Returns:
        float: iou
    """
    return intersection(box1, box2) / union(box1, box2)


def union(box1: list[Any], box2: list[Any]) -> float:
    """Compute union area over two boxes

    Args:
        box1 (list[Any]): bbox1 coords, label and score
        box2 (list[Any]): bbox2 coords, label and score

    Returns:
        float: union area
    """
    box1_x1, box1_y1, box1_x2, box1_y2 = box1[:4]
    box2_x1, box2_y1, box2_x2, box2_y2 = box2[:4]
    box1_area = (box1_x2 - box1_x1) * (box1_y2 - box1_y1)
    box2_area = (box2_x2 - box2_x1) * (box2_y2 - box2_y1)
    return box1_area + box2_area - intersection(box1, box2)


def intersection(box1: list[Any], box2: list[Any]) -> float:
    """Compute intersection area over two boxes

    Args:
        box1 (list[Any]): bbox1 coords, label and score
        box2 (list[Any]): bbox2 coords, label and score

    Returns:
        float: intersection area
    """
    box1_x1, box1_y1, box1_x2, box1_y2 = box1[:4]
    box2_x1, box2_y1, box2_x2, box2_y2 = box2[:4]
    x1 = max(box1_x1, box2_x1)
    y1 = max(box1_y1, box2_y1)
    x2 = min(box1_x2, box2_x2)
    y2 = min(box1_y2, box2_y2)
    return (x2 - x1) * (y2 - y1)


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

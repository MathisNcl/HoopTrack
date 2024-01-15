"""Implementation of the Tracker class"""
from collections import defaultdict
from typing import Any, Optional

import numpy as np
from scipy.optimize import linear_sum_assignment

from hooptrack.basketball.utils import iou
from hooptrack.schemas.inference import Result


class Tracker:
    """Tracker class

    Args:
            iou_importance (float, optional): Weight for iou in cost. Defaults to 0.65.
            color_importance (float, optional): Weight for color_distance in cost. Defaults to 0.35.
    """

    def __init__(self, iou_importance: float = 0.65, color_importance: float = 0.35) -> None:
        """Constructor"""
        if iou_importance + color_importance != 1:
            raise ValueError(f"iou_importance + color_importance should be 1, got {iou_importance + color_importance}")
        self.iou_importance: float = iou_importance
        self.color_importance: float = color_importance
        # key is label, value is dict of id:list of boxes
        self.tracks: dict = {}
        self.previous_boxes: list = []
        self.current_image: Optional[np.ndarray] = None

    def __repr__(self) -> str:
        """Representation of the object"""
        return "\n".join(
            [
                f"Tracker(iou_importance={self.iou_importance}, " + f"color_importance={self.color_importance})",
                f"Actual tracks: {self.tracks}",
            ]
        )

    @staticmethod
    def calculate_color_similarity(box1: list[Any], box2: list[Any], image1: np.ndarray, image2: np.ndarray) -> float:
        """Compute the color similarity between two sub images

        Args:
            box1 (list[Any]): xyxy coords of box 1 to consider in image1
            box2 (list[Any]): xyxy coords of box 2 to consider in image2
            image1 (np.ndarray): image containing objects from box1
            image2 (np.ndarray): image containing objects from box2

        Returns:
            float: color similarity between the two object detected,
            metric is from 0 to 1, the closer to 0 the more identical
        """
        box1 = [0 if b < 0 else b for b in box1]
        box2 = [0 if b < 0 else b for b in box2]
        mean_color_box1: Any = np.mean(image1[int(box1[1]) : int(box1[3]), int(box1[0]) : int(box1[2])], axis=(0, 1))
        mean_color_box2: Any = np.mean(image2[int(box2[1]) : int(box2[3]), int(box2[0]) : int(box2[2])], axis=(0, 1))
        color_difference: Any = np.linalg.norm(mean_color_box1 - mean_color_box2)

        return 1 / (1 + np.exp(-color_difference))

    def create_cost_matrix(
        self, boxes1: list[list[int]], boxes2: list[list[int]], image1: np.ndarray, image2: np.ndarray
    ) -> np.ndarray:
        """

        Args:
            boxes1 (list[list[int]]): list of xyxy coords bbox to consider in image1
            boxes2 (list[list[int]]): list of xyxy coords bbox to consider in image2
            image1 (np.ndarray): image containing objects from boxes1
            image2 (np.ndarray): image containing objects from boxes2

        Returns:
            np.ndarray: cost matrix of every association
        """
        cost_matrix = np.zeros((len(boxes1), len(boxes2)))
        for i, box1 in enumerate(boxes1):
            for j, box2 in enumerate(boxes2):
                iou_cost: float = 1 - iou(box1, box2)
                color_similarity_cost: float = self.calculate_color_similarity(box1, box2, image1, image2)

                cost_matrix[i, j] = self.iou_importance * iou_cost + self.color_importance * color_similarity_cost
        return cost_matrix

    def set_previous_boxes(self) -> tuple[list[str], list[str]]:
        """Set previous_boxes and get labels and tracking_id of current tracking

        Returns:
            tuple[list[str], list[str]]: labels class and their tracking id
        """
        self.previous_boxes = []
        labels, tracking_id = [], []
        for label, dict_id in self.tracks.items():
            for id_track, boxes in dict_id.items():
                self.previous_boxes.append(boxes[-1])
                tracking_id.append(id_track)
                labels.append(label)
        return labels, tracking_id

    def populate_tracks(
        self, current_boxes: list[list[int]], current_labels: list[str], selected_ids: list[str]
    ) -> None:
        """Add the tracking results from hungarian algorithm into tracks.
           len(current_boxes)=len(current_labels)=len(selected_ids)

        Args:
            current_boxes (list[list[int]]): every object bouding boxes detected in the image
            current_labels (list[str]): label class of every object detected in the image
            selected_ids (list[str]): tracking id attributed to every object in the image
        """
        for curr_box, curr_label, sel_id in zip(current_boxes, current_labels, selected_ids):
            if curr_label not in self.tracks:
                self.tracks[curr_label] = {}
            if sel_id not in self.tracks[curr_label]:
                self.tracks[curr_label][sel_id] = []
            # populate
            self.tracks[curr_label][sel_id].append(curr_box)

    def get_infos_and_set_images(self, result_predictions: Result) -> tuple[list[list[int]], list[str]]:
        """Set previous_image and current_image in the class.
           Return current boxes and current labels detected in the image

        Args:
            result_predictions (Result): Result from the model pipeline

        Returns:
            tuple[list[list[int]], list[str]]: current boxes and current labels
        """
        self.previous_image = self.current_image
        self.current_image = np.array(result_predictions.image)
        return [box.bbox for box in result_predictions.boxes], [box.label_name for box in result_predictions.boxes]

    def track_objects(self, result_predictions: Result, cost_threshold: float = 0.5) -> None:
        """Track objects detected into the tracker at `tracks` variable. Main function of the class.

        Args:
            result_predictions (Result): Result from the model pipeline
            cost_threshold (float, optional): Max threshold to accept the cost. Defaults to 0.5.

        """
        labels, tracking_id = self.set_previous_boxes()
        current_boxes, current_labels = self.get_infos_and_set_images(result_predictions)
        selected_ids: list[str] = []
        if len(self.tracks) == 0:
            cumulative_counts: dict = defaultdict(int)

            for lab in current_labels:
                selected_ids.append(f"{lab}_{cumulative_counts[lab]}")
                cumulative_counts[lab] += 1

            self.populate_tracks(current_boxes, current_labels, selected_ids)
            return

        cost_matrix = self.create_cost_matrix(
            current_boxes, self.previous_boxes, self.current_image, self.previous_image
        )

        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        is_association = [cost_matrix[i, j] < cost_threshold for i, j in zip(row_ind, col_ind)]

        # i is idx for current
        for i, j, is_associated in zip(row_ind, col_ind, is_association):
            if is_associated and current_labels[i] == labels[j]:
                selected_ids.append(tracking_id[j])
            else:
                selected_ids.append(f"{current_labels[i]}_{len(self.tracks[current_labels[i]])}")

        self.populate_tracks(current_boxes, current_labels, selected_ids)

    # im1 = Image.open("243.jpg")
    # im2 = Image.open("244.jpg")
    # create_cost_matrix(
    #     b1,
    #     b2,
    #     np.array(im1),
    #     np.array(im2),
    # )

    # object_tracking(
    #     b1,
    #     b2,
    #     np.array(im1),
    #     np.array(im2),
    # )


# from hooptrack.basketball.tracker import Tracker
# import json
# from PIL import Image
# from hooptrack.schemas.inference import Result

# with open("output_vid_2.json") as f:
#     r = json.load(f)
# tracker = Tracker(0.65, 0.35)
# im1 = Image.open("243.jpg")
# im2 = Image.open("244.jpg")
# res1 = Result(
#     **{
#         "image": im1,
#         "boxes": [
#             {"bbox": [2542, 435, 2786, 687], "label_name": "rim", "label_id": 2, "score": 1},
#             {"bbox": [2193, 1418, 2297, 1507], "label_name": "ball", "label_id": 0, "score": 1},
#             {"bbox": [-1, 543, 96, 820], "label_name": "rim", "label_id": 2, "score": 1},
#         ],
#     }
# )
# res2 = Result(
#     **{
#         "image": im2,
#         "boxes": [
#             {"bbox": [2181, 1347, 2282, 1432], "label_name": "ball", "label_id": 0, "score": 1},
#             {"bbox": [2543, 432, 2784, 686], "label_name": "rim", "label_id": 2, "score": 1},
#             {"bbox": [-1, 536, 92, 819], "label_name": "rim", "label_id": 2, "score": 1},
#         ],
#     }
# )
# tracker.object_tracking(res1)
# tracker.object_tracking(res2)

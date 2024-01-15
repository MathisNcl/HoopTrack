"""Tests for the Tracker class"""
from unittest import TestCase

import pytest
from PIL import Image

from hooptrack.basketball.tracker import Tracker
from hooptrack.schemas.inference import Result


def test_error_param():
    with pytest.raises(ValueError) as e:
        Tracker(iou_importance=1, color_distance_importance=1)
        "iou_importance + distance_importance should be  1" in e


def test_track():
    tracker = Tracker(iou_importance=0.65, color_distance_importance=0.35)
    im1 = Image.open("tests/units/fixture/243.jpg")
    im2 = Image.open("tests/units/fixture/244.jpg")
    res1 = Result(
        **{
            "image": im1,
            "boxes": [
                {"bbox": [2542, 435, 2786, 687], "label_name": "rim", "label_id": 2, "score": 1},
                {"bbox": [2193, 1418, 2297, 1507], "label_name": "ball", "label_id": 0, "score": 1},
                {"bbox": [-1, 543, 96, 820], "label_name": "rim", "label_id": 2, "score": 1},
            ],
        }
    )
    res2 = Result(
        **{
            "image": im2,
            "boxes": [
                {"bbox": [2181, 1347, 2282, 1432], "label_name": "ball", "label_id": 0, "score": 1},
                {"bbox": [2543, 432, 2784, 686], "label_name": "rim", "label_id": 2, "score": 1},
                {"bbox": [-1, 536, 92, 819], "label_name": "rim", "label_id": 2, "score": 1},
            ],
        }
    )
    tracker.track_objects(res1)
    TestCase().assertDictEqual(
        tracker.tracks,
        {
            "rim": {
                "rim_0": [[2542, 435, 2786, 687]],
                "rim_1": [[-1, 543, 96, 820]],
            },
            "ball": {"ball_0": [[2193, 1418, 2297, 1507]]},
        },
    )

    tracker.track_objects(res2)
    TestCase().assertDictEqual(
        tracker.tracks,
        {
            "rim": {
                "rim_0": [[2542, 435, 2786, 687], [2543, 432, 2784, 686]],
                "rim_1": [[-1, 543, 96, 820], [-1, 536, 92, 819]],
            },
            "ball": {"ball_0": [[2193, 1418, 2297, 1507]], "ball_1": [[2181, 1347, 2282, 1432]]},
        },
    )

    assert "Actual tracks:" in tracker.__repr__()

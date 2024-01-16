"""Tests for the Tracker class"""
from unittest import TestCase

import pandas as pd
from PIL import Image

from hooptrack.basketball.tracker import Tracker
from hooptrack.schemas.inference import Result


class TestTracker(TestCase):
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

    def test_error_params(self):
        with self.assertRaises(ValueError) as e:
            Tracker(iou_importance=1, color_importance=1)
            self.assertIn("iou_importance + color_importance should be  1", e)

    def test_track(self):
        tracker = Tracker(iou_importance=0.65, color_importance=0.35)

        tracking_id = tracker.track_objects(self.res1)
        self.assertDictEqual(
            tracker.tracks,
            {
                "rim": {
                    "rim_0": [[2542, 435, 2786, 687]],
                    "rim_1": [[-1, 543, 96, 820]],
                },
                "ball": {"ball_0": [[2193, 1418, 2297, 1507]]},
            },
        )
        self.assertListEqual(tracking_id, ["rim_0", "ball_0", "rim_1"])
        self.assertDictEqual(
            tracker.tracks_image_id,
            {
                "rim": {
                    "rim_0": [0],
                    "rim_1": [0],
                },
                "ball": {
                    "ball_0": [0],
                },
            },
        )

        # next prediction
        tracking_id = tracker.track_objects(self.res2)
        self.assertDictEqual(
            tracker.tracks,
            {
                "rim": {
                    "rim_0": [[2542, 435, 2786, 687], [2543, 432, 2784, 686]],
                    "rim_1": [[-1, 543, 96, 820], [-1, 536, 92, 819]],
                },
                "ball": {"ball_0": [[2193, 1418, 2297, 1507]], "ball_1": [[2181, 1347, 2282, 1432]]},
            },
        )
        self.assertListEqual(tracking_id, ["ball_1", "rim_0", "rim_1"])
        self.assertDictEqual(
            tracker.tracks_image_id,
            {
                "rim": {
                    "rim_0": [0, 1],
                    "rim_1": [0, 1],
                },
                "ball": {
                    "ball_0": [0],
                    "ball_1": [1],
                },
            },
        )
        assert "Actual tracks:" in tracker.__repr__()

        df = tracker.track_to_df(2160)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(df.shape, (6, 6))

    def test_update_dict_error(self):
        with self.assertRaises(ValueError) as e:
            Tracker.update_dict({}, "wrong")
            self.assertIn("key_str should contains to element as `elm1.elm2`.", e)

        with self.assertRaises(ValueError) as e:
            Tracker.update_dict({}, "wrong.wrong.wrong")
            self.assertIn("key_str should contains to element as `elm1.elm2`.", e)

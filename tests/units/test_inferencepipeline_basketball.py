import os
from unittest import mock

import numpy as np
from PIL import Image

from hooptrack.basketball.inference_pipeline import InferencePipelineBasketBall
from hooptrack.schemas.config import BasketballDetectionConfig
from hooptrack.schemas.inference import Box


def test_pipeline(img_dummy, caplog, mocked_session):
    with mock.patch("onnxruntime.InferenceSession", return_value=mocked_session):
        pipeline = InferencePipelineBasketBall(config=BasketballDetectionConfig(model="mock_model.onnx"))

    assert "mock_model.onnx loaded in CPU." in caplog.text

    res = pipeline.run(buffer=np.asarray(img_dummy))

    # assert something
    assert len(res.boxes) == 2
    assert res.boxes == [
        Box(bbox=[68, 26, 75, 37], label_name="rim", label_id=2, score=0.7722053527832031),
        Box(bbox=[69, 13, 72, 18], label_name="ball", label_id=0, score=0.4473499655723572),
    ]


def test_prepare_input(img_dummy, mocked_session):
    with mock.patch("onnxruntime.InferenceSession", return_value=mocked_session):
        pipeline = InferencePipelineBasketBall(config=BasketballDetectionConfig(model="mock_model.onnx"))

    img_dummy.save("tests/test.jpg")
    res = pipeline.prepare_input("tests/test.jpg")
    assert isinstance(res[1], Image.Image)
    assert isinstance(res[0], np.ndarray)
    assert res[0].shape == (1, 3, 640, 640)

    os.remove("tests/test.jpg")


# TODO: add live_streaming testing

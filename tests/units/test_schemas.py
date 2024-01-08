import pytest
from pydantic_core import ValidationError

from hooptrack.schemas.config import BasketballDetectionConfig
from hooptrack.schemas.inference import Box, Result


@pytest.mark.parametrize(
    "config",
    [
        {"model": "dummy.onnx"},
        {
            "model": "dummy.onnx",
            "threshold_confidence": 0.8,
            "image_size": (1, 2),
            "onnx_mode": False,
            "yolo_classes": [],
            "classes_keeped": [],
            "frame_processed_every": 2,
        },
    ],
)
def test_config(config):
    assert isinstance(BasketballDetectionConfig(**config), BasketballDetectionConfig)


@pytest.mark.parametrize(
    "config",
    [
        {},
        {"model": 0},
        {"model": "dummy.onnx", "threshold_confidence": "six"},
        {"model": "dummy.onnx", "onnx_mode": "not activated"},
        {"model": "dummy.onnx", "image_size": 0.8},
        {"model": "dummy.onnx", "image_size": ("six", "two")},
        {"model": "dummy.onnx", "classes_keeped": "rim"},
        {"model": "dummy.onnx", "classes_keeped": 9},
        {"model": "dummy.onnx", "yolo_classes": None},
        {"model": "dummy.onnx", "yolo_classes": "0"},
        {"model": "dummy.onnx", "yolo_classes": 21},
        {"model": "dummy.onnx", "frame_processed_every": None},
    ],
)
def test_config_error(config):
    with pytest.raises(ValidationError):
        BasketballDetectionConfig(**config)


def test_box_schema():
    bbox = [10, 20, 50, 60]
    label_name = "Test"
    label_id = 1
    score = 0.9

    # Create a valid Box instance
    box = Box(bbox=bbox, label_name=label_name, label_id=label_id, score=score)

    # Test attribute values
    assert box.bbox == bbox
    assert box.label_name == label_name
    assert box.label_id == label_id
    assert box.score == score


def test_box_schema_error():
    with pytest.raises(ValidationError):
        Box(bbox=[10, 20], label_name="rim", label_id=2, score=0.8)


def test_result_schema(img_dummy):
    box = Box(bbox=[10, 20, 30, 40], label_name="ball", label_id=3, score=0.5)

    result = Result(image=img_dummy, boxes=[box])

    assert result.image == img_dummy
    assert len(result.boxes) == 1
    assert result.boxes[0] == box


def test_result_schema_error_img(img_dummy):
    # Test Field constraints
    with pytest.raises(ValidationError):
        # Invalid image type
        Result(image="invalid", boxes=[Box(bbox=[10, 20, 30, 40], label_name="ball", label_id=3, score=0.5)])


def test_result_schema_error_box(img_dummy):
    with pytest.raises(ValidationError):
        Result(image=img_dummy, boxes=[{"invalid": "box"}])

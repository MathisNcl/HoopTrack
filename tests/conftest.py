"""Conftest containing fixtures for testing purpose"""
import pytest
from PIL import Image


@pytest.fixture
def bball_model_output_only_rim():
    return [{"bbox": [2736, 567, 2970, 805], "label_name": "rim", "label_id": 2, "score": 0.801576554775238}]


@pytest.fixture
def bball_model_output_rim_bball():
    return [
        {"bbox": [2541, 550, 2784, 806], "label_name": "rim", "label_id": 2, "score": 0.7226023077964783},
        {"bbox": [1998, 1497, 2090, 1586], "label_name": "ball", "label_id": 0, "score": 0.707162618637085},
    ]


@pytest.fixture
def img_dummy():
    return Image.new("RGB", (100, 100), color="white")

"""Conftest containing fixtures for testing purpose"""
import pickle
from typing import Any

import pytest
from PIL import Image

from hooptrack.schemas.inference import Result


@pytest.fixture
def bball_model_output_only_rim():
    return [{"bbox": [2736, 567, 2970, 805], "label_name": "rim", "label_id": 2, "score": 0.801576554775238}]


@pytest.fixture
def bball_model_output_rim_bball():
    return [
        {"bbox": [10, 10, 20, 20], "label_name": "rim", "label_id": 2, "score": 0.7226023077964783},
        {"bbox": [50, 50, 70, 70], "label_name": "ball", "label_id": 0, "score": 0.707162618637085},
    ]


@pytest.fixture
def img_dummy():
    return Image.new("RGB", (100, 100), color="white")


@pytest.fixture
def sample_result():
    img = Image.new("RGB", (100, 100), color="white")
    return Result(image=img, boxes=[{"bbox": [10, 10, 50, 50], "label_id": 1, "label_name": "rim", "score": 0.8}])


class MockedInferenceSession:
    def __init__(self, model: str, providers: list[str], to_return: list = []) -> None:
        self.model = model
        self.providers = providers
        self.to_return = to_return

    def run(self, output_names: Any, input_feed: Any, run_options: Any | None = None) -> Any:
        return self.to_return


@pytest.fixture
def mocked_session():
    with open("tests/units/fixture/run_ort.pkl", "rb") as f:
        to_return = pickle.load(f).reshape(1, 1, 8, 8400)
    return MockedInferenceSession(model="mock_model.onnx", providers=["provider1", "provider2"], to_return=to_return)

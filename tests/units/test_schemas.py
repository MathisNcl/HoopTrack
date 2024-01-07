import pytest

from hooptrack.schemas.config import BasketballDetectionConfig


@pytest.mark.parametrize("config", [{"model": "dummy.onnx"}])
def test_config(config):
    assert isinstance(BasketballDetectionConfig(**config), BasketballDetectionConfig)

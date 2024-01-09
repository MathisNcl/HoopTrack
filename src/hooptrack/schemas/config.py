"""Module containing config schemas used for modeling"""
from pydantic import BaseModel, Field


class BasketballDetectionConfig(BaseModel):
    """Basketball config"""

    model: str = Field(description="Model full path")
    threshold_confidence: float = Field(default=0.25, description="Lowest score to approve label inference")
    image_size: tuple[int, int] = Field(default=(640, 640), description="Image size handled by the model")
    onnx_mode: bool = Field(default=True, description="Whether the model is onnx")
    yolo_classes: list[str] = Field(
        default=["ball", "made", "rim", "shoot"], description="Classes of the model, order matters"
    )
    classes_keeped: list[str] = Field(default=["ball", "made", "rim", "shoot"], description="Classes to keep")

    frame_processed_every: int = 1

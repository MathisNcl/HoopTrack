"""Module containing schemas used in inference"""
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field


class Box(BaseModel):
    """Bounding box schema"""

    bbox: list[int] = Field(min_length=4, max_length=4, description="x1, y1, x2, y2 coords of bbox")
    label_name: str = Field(description="Label name")
    label_id: int = Field(description="Label id")
    score: float = Field(description="Score of the label")


class Result(BaseModel):
    """Result from inference schema"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    image: Image.Image = Field(description="Original image")
    boxes: list[Box] = Field(description="List of boxes with all infos")

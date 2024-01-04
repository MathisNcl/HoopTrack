"""Implementation file of the InferencePipeline class"""
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import onnxruntime as ort
from PIL import Image

from hooptrack.basketball.utils import iou
from hooptrack.schemas.config import BasketballDetectionConfig


class InferencePipeline(ABC):
    """Abstract class to follow to make an inference pipeline"""

    @abstractmethod
    def prepare_input(self, buffer: str | bytes) -> Any:
        """Prepare input data for model"""
        raise NotImplementedError

    @abstractmethod
    def process_output(self, output: Any, img_width: int, img_height: int) -> Any:
        """Process output of the model"""
        raise NotImplementedError

    @abstractmethod
    def run_model(self, input: np.ndarray) -> Any:
        """Run the model"""
        raise NotImplementedError

    @abstractmethod
    def run(self, buffer: str | bytes) -> Any:
        """Run all the pipeline"""
        raise NotImplementedError


# @see https://dev.to/andreygermanov/how-to-create-yolov8-based-object-
# detection-web-service-using-python-julia-nodejs-javascript-go-and-rust-4o8e#python
class InferencePipelineBasketBall(InferencePipeline):
    """InferencePipeline for detecting basketball objects"""

    def __init__(self, config: BasketballDetectionConfig):
        """Constructor"""
        self.config: BasketballDetectionConfig = config

        # TODO: se renseigner sur les providers des InferenceSession
        self.model = ort.InferenceSession(config.model, providers=["CPUExecutionProvider"])

    def run(self, buffer: str | bytes) -> Any:
        """Run the entire pipeline on an image

        Args:
            buffer (str | bytes): image, either path or bytes

        Returns:
            Any: result from model and pipeline
        """
        input, img_width, img_height = self.prepare_input(buffer)
        output: Any = self.run_model(input)

        res: list[list[Any]] = self.process_output(output, img_width, img_height)

        return self.post_process(res)

    def prepare_input(self, buffer: str | bytes) -> tuple[np.ndarray, int, int]:
        """Prepare input by resizing and reshaping in np array

        Args:
            buffer (bytes): image bytes to prepare

        Returns:
            tuple[np.ndarray, int, int]: resized input array on onnx format, width and height of the original image
        """
        img: Image = Image.open(buffer)
        img_width, img_height = img.size
        img = img.resize(self.config.image_size).convert("RGB")
        input: np.ndarray = np.array(img).transpose(2, 0, 1).reshape(1, 3, 640, 640) / 255.0
        return input.astype(np.float32), img_width, img_height

    def run_model(self, input: np.ndarray) -> Any:
        """Infer model over the input

        Args:
            input (np.ndarray): to infer

        Returns:
            Any: results from onnx model
        """
        return self.model.run(["output0"], {"images": input})[0]

    def process_output(self, output: Any, img_width: int, img_height: int) -> list[list[Any]]:
        """Extract boxes detected from the onnx output

        Args:
            output (Any): model output
            img_width (int): original image width
            img_height (int): original image height

        Returns:
            list[list[Any]]: list of boxes with the four coordinates, the label and the score detected
        """
        output_reformatted: np.ndarray = output[0].astype(float).transpose()

        boxes: list[list[Any]] = []
        for row in output_reformatted:
            prob: float = row[4:].max()
            if prob < self.config.threshold_confidence:
                continue
            label: str = self.config.yolo_classes[row[4:].argmax()]
            xc, yc, w, h = row[:4]
            x1: int = (xc - w / 2) / 640 * img_width
            y1: int = (yc - h / 2) / 640 * img_height
            x2: int = (xc + w / 2) / 640 * img_width
            y2: int = (yc + h / 2) / 640 * img_height
            boxes.append([x1, y1, x2, y2, label, prob])

        boxes.sort(key=lambda x: x[5], reverse=True)
        result: list[list[Any]] = []

        # Non-maximum suppression
        while len(boxes) > 0:
            result.append(boxes[0])
            boxes = [box for box in boxes if iou(box, boxes[0]) < 0.7]

        return result

    def post_process(self, res: Any) -> Any:
        """Post process predictions, maybe needed later"""
        return res

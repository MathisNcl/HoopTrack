"""Implementation file of the InferencePipeline class"""
import json
import queue
import threading
from typing import Any

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image
from tqdm import tqdm

from hooptrack import logger
from hooptrack.basketball.utils import determine_filename, iou
from hooptrack.basketball.visualiser import Visualiser
from hooptrack.schemas.config import BasketballDetectionConfig
from hooptrack.schemas.inference import Result


# @see https://dev.to/andreygermanov/how-to-create-yolov8-based-object-
# detection-web-service-using-python-julia-nodejs-javascript-go-and-rust-4o8e#python
class InferencePipelineBasketBall:
    """InferencePipeline for detecting basketball objects"""

    def __init__(self, config: BasketballDetectionConfig):
        """Constructor"""
        self.config: BasketballDetectionConfig = config

        self.model = ort.InferenceSession(config.model, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        logger.info(f"{config.model} loaded in CPU.")

    def run(self, buffer: str | bytes) -> Any:
        """Run the entire pipeline on an image

        Args:
            buffer (str | bytes): image, either path or bytes

        Returns:
            Any: result from model and pipeline
        """
        input, img = self.prepare_input(buffer)
        output: Any = self.run_model(input)

        img_width, img_height = img.size
        res: list[list[Any]] = self.process_output(output, img_width, img_height)

        return self.post_process(res, img)

    def _inference_worker(self) -> None:
        """Inference worker for live streaming"""
        cpt = 0
        logger.info("Inference worker started...")
        while True:
            try:
                _, frame = self.cap.read()
                if cpt % self.config.frame_processed_every == 0:
                    res = self.run(frame)

                    self.result_queue.put(res)
                cpt += 1
            except AttributeError:
                logger.info("Thread stopped, error reading image")
                res = None
                break

    def live_streaming(self, source: int | str, show: bool = True, save: bool = False) -> None:
        """live stream result on your laptop by chosing between a file or your camera device (0)

        Args:
            source (int | str): int if cam, selecting your device or a path str to a video
            show (bool): Whether the output should be shown in a window during processing. Default True.
            save (bool): Whether the output should be saved, auto-naming. Default False.

        """
        if not show and not save:
            raise ValueError("show or save should be set to True.")
        self.cap: Any = cv2.VideoCapture(source)
        frame_count: int = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) / self.config.frame_processed_every)
        fps: int = int(round(self.cap.get(cv2.CAP_PROP_FPS) / self.config.frame_processed_every))

        self.result_queue: queue.Queue = queue.Queue()

        # start inference thread
        inference_thread: Any = threading.Thread(
            target=self._inference_worker,
        )
        inference_thread.start()
        visualiser: Visualiser = Visualiser()

        if save:
            # determine path and video settings
            bboxes: list[Any] = []
            filename: str = determine_filename()
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(
                filename,
                fourcc,
                fps,
                (int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))),
            )
        logger.info("Source stream mode" if frame_count == 0 else "Source video mode")
        for _ in tqdm(range(frame_count)) if frame_count > 0 else iter(int, 1):
            res: Result = self.result_queue.get()
            output_image: np.ndarray = visualiser.plot(res)
            if save:
                bboxes.append([b.model_dump() for b in res.boxes])
                out.write(output_image)
            if show:
                cv2.imshow("Basketball detection", output_image)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        # Release resources
        self.cap.release()
        cv2.destroyAllWindows()
        [cv2.waitKey(1) for i in range(4)]  # for macos purpose, window not killed unless
        inference_thread.join()
        if save:
            out.release()
            with open(filename.split(".")[0] + ".json", "w") as f:
                json.dump({i: bboxes[i] for i in range(len(bboxes))}, f)
            logger.info(f"New video named: {filename} + txt")

    def prepare_input(self, buffer: str | bytes | np.ndarray) -> tuple[np.ndarray, Image.Image]:
        """Prepare input by resizing and reshaping in np array

        Args:
            buffer (bytes): image bytes to prepare

        Returns:
            tuple[np.ndarray, Image]: resized input array on onnx format, original image
        """
        img: Image.Image
        if isinstance(buffer, np.ndarray):
            img = Image.fromarray(buffer)
        else:
            img = Image.open(buffer)
        img_resized: Any = img.resize(self.config.image_size).convert("RGB")
        input: np.ndarray = np.array(img_resized).transpose(2, 0, 1).reshape(1, 3, 640, 640) / 255.0
        return input.astype(np.float32), img

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
            label_id: int = row[4:].argmax()
            label: str = self.config.yolo_classes[label_id]
            xc, yc, w, h = row[:4]
            x1: int = (xc - w / 2) / 640 * img_width
            y1: int = (yc - h / 2) / 640 * img_height
            x2: int = (xc + w / 2) / 640 * img_width
            y2: int = (yc + h / 2) / 640 * img_height
            boxes.append([x1, y1, x2, y2, label_id, label, prob])

        boxes.sort(key=lambda x: x[6], reverse=True)
        result: list[list[Any]] = []

        # Non-maximum suppression
        while len(boxes) > 0:
            result.append(boxes[0])
            boxes = [box for box in boxes if iou(box, boxes[0]) < 0.7]

        return result

    def post_process(self, res: Any, img: Image.Image) -> Result:
        """Post process predictions, maybe needed later"""
        post_processed_list: list[dict[str, Any]] = []

        for box in res:
            post_processed_list.append(
                {"bbox": [int(b) for b in box[:4]], "label_id": box[4], "label_name": box[5], "score": box[6]}
            )
        return Result(image=img, boxes=post_processed_list)

"""Module implementation of Visualiser class"""

from typing import Any, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from hooptrack.schemas.inference import Result


class Colors:
    """
    Ultralytics default color palette https://ultralytics.com/.

    This class provides methods to work with the Ultralytics color palette, including converting hex color codes to
    RGB values.

    Attributes:
        palette (list of tuple): List of RGB color values.
        n (int): The number of colors in the palette.
        pose_palette (np.array): A specific color palette array with dtype np.uint8.
    """

    def __init__(self) -> None:
        """Initialize colors as hex = matplotlib.colors.TABLEAU_COLORS.values()."""
        hexs = (
            "FF3838",
            "FF9D97",
            "FF701F",
            "FFB21D",
            "CFD231",
            "48F90A",
            "92CC17",
            "3DDB86",
            "1A9334",
            "00D4BB",
            "2C99A8",
            "00C2FF",
            "344593",
            "6473FF",
            "0018EC",
            "8438FF",
            "520085",
            "CB38FF",
            "FF95C8",
            "FF37C7",
        )
        self.palette = [self.hex2rgb(f"#{c}") for c in hexs]
        self.n = len(self.palette)
        self.pose_palette = np.array(
            [
                [255, 128, 0],
                [255, 153, 51],
                [255, 178, 102],
                [230, 230, 0],
                [255, 153, 255],
                [153, 204, 255],
                [255, 102, 255],
                [255, 51, 255],
                [102, 178, 255],
                [51, 153, 255],
                [255, 153, 153],
                [255, 102, 102],
                [255, 51, 51],
                [153, 255, 153],
                [102, 255, 102],
                [51, 255, 51],
                [0, 255, 0],
                [0, 0, 255],
                [255, 0, 0],
                [255, 255, 255],
            ],
            dtype=np.uint8,
        )

    def __call__(self, i: Any, bgr: bool = False) -> Any:
        """Converts hex color codes to RGB values."""
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h: Any) -> Any:
        """Converts hex color codes to RGB values (i.e. default PIL order)."""
        return tuple(int(h[1 + i : 1 + i + 2], 16) for i in (0, 2, 4))


colors = Colors()


class Visualiser:
    """Visualiser class for drawing result

    Args:
            line_width (float, optional): The line width of the bounding boxes. If None, it is scaled to the image size.
            font_size (float, optional): The font size of the text. If None, it is scaled to the image size.
            font (str): The font to use for the text.
    """

    def __init__(self, line_width: Optional[float] = None, font_size: Optional[float] = None, font: str = "Arial.ttf"):
        """Constructor"""
        self.line_width: Optional[float] = line_width
        self.font_size: Optional[float] = font_size
        self.font_str: str = font
        self.font: Optional[Any] = None

    def set_font_and_linewidth(self, img: Image.Image) -> None:
        """Define font and line_width if it is not, scaled to image size

        Args:
            img (Image.Image): _description_
        """
        if self.font is None:
            try:
                self.font = ImageFont.truetype(str(self.font_str), max(round(sum(img.size) / 2 * 0.035), 12))
            except OSError:
                self.font = ImageFont.load_default()

        if self.line_width is None:
            self.line_width = max(round(sum(img.size) / 2 * 0.003), 2)

    # @see: https://github.com/ultralytics/ultralytics/blob/main/ultralytics/engine/results.py#L167
    def plot(
        self,
        result: Result,
        show_id: bool = False,
        show_conf: bool = True,
        show_labels: bool = True,
        show_boxes: bool = True,
    ) -> np.ndarray:
        """
        Plots the detection results on an input RGB image.

        Args:
            result (Result): Inference result
            show_conf (bool): Whether to plot the detection confidence score.
            show_labels (bool): Whether to plot the label of bounding boxes.
            show_boxes (bool): Whether to plot the bounding boxes.

        Returns:
            (numpy.ndarray): A numpy array of the annotated image.

        Example:
            ```python
            from hooptrack.basketball.inference_pipeline import InferencePipelineBasketBall
            from hooptrack.schemas.config import BasketballDetectionConfig
            from hooptrack.basketball.visualiser import Visualiser
            from PIL import Image

            inference_pipeline = InferencePipelineBasketBall(
                config=BasketballDetectionConfig(
                    model="/models/best_nano_22h21.onnx"
                )
            )
            results = inference_pipeline.run("my_image.png")

            im_array = Visualiser().plot(results, show_id=True)
            im = Image.fromarray(im_array)  # RGB PIL image
            im.show()  # show image
            im.save('results.jpg')  # save image
            ```
        """

        img: Image.Image = result.image
        self.draw: ImageDraw.Draw = ImageDraw.Draw(img)
        self.set_font_and_linewidth(img)

        # Plot Detect results
        if result.boxes and show_boxes:
            for box in result.boxes:
                label: str = ""
                if show_id:
                    label += f"id:{box.label_id} "
                if show_labels:
                    label += f"{box.label_name} "
                if show_conf:
                    label += f"{box.score:.2f}"
                # inplace for img
                self.box_label(box.bbox, label, color=colors(box.label_id))

        return np.asarray(img)

    def box_label(
        self, box: list[int], label: str = "", color: tuple = (128, 128, 128), txt_color: tuple = (255, 255, 255)
    ) -> None:
        """Add one xyxy box to image with label inplace."""
        p1 = (box[0], box[1])
        self.draw.rectangle(box, width=self.line_width, outline=color)  # box
        if label:
            w, h = self.font.getbbox(label)[2:4]  # type: ignore # because of PIL>'9.2.0'
            outside = p1[1] - h >= 0  # label fits outside box
            self.draw.rectangle(
                (p1[0], p1[1] - h if outside else p1[1], p1[0] + w + 1, p1[1] + 1 if outside else p1[1] + h + 1),
                fill=color,
            )
            # self.draw.text((box[0], box[1]), label, fill=txt_color, font=self.font, anchor='ls')  # for PIL>8.0
            self.draw.text((p1[0], p1[1] - h if outside else p1[1]), label, fill=txt_color, font=self.font)

import numpy as np

from hooptrack.basketball.visualiser import Visualiser


def test_set_font_and_linewidth(sample_result):
    visualiser = Visualiser(font="Lato-Medium.ttf")  # doesn exist for macOS

    assert visualiser.font is None
    assert visualiser.line_width is None

    img = sample_result.image
    visualiser.set_font_and_linewidth(img)
    assert visualiser.font is not None
    assert visualiser.line_width is not None


def test_plot(sample_result):
    visualiser = Visualiser(font="Arial.ttf")
    output_image = visualiser.plot(sample_result, show_id=True)
    assert isinstance(output_image, np.ndarray)
    assert output_image.shape == (100, 100, 3)

    # visual verification
    # from PIL import Image
    # Image.fromarray(output_image).save("tests/test_plot.jpeg")

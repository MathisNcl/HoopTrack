import os

from hooptrack.basketball.utils import determine_filename


def test_determine_filename():
    assert determine_filename("tests") == "tests/output_vid_0.mp4"

    with open("tests/output_vid_0.mp4", "w"):
        assert determine_filename("tests") == "tests/output_vid_1.mp4"

    os.remove("tests/output_vid_0.mp4")

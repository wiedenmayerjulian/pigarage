import time
from queue import Queue
from threading import Thread

import easyocr
import pytest
from pigarage import ocr_detector

from . import download_lnpr_plate

VISUAL_DEBUG = False


@pytest.fixture(scope="session")
def reader():
    return easyocr.Reader(["en"], verbose=True)


@pytest.mark.parametrize(
    "id, expected_ocr",
    [
        ("08HdV8ArxuVKXgxdUor1", "K SC124"),
        ("uJsY6e391eOodCkFLMJA", "1A777AK77"),
        ("qTPu96zhef7AbiBSFFTD", "R275 ULO"),
    ],
)
def test_ocr(id, expected_ocr, reader):
    img = download_lnpr_plate(id)
    plate = ocr_detector.cv2_improve_plate_img(img, min_char_height_ratio=0)
    assert plate is not None
    ocr = ocr_detector.plate2text(plate, reader=reader)
    assert expected_ocr == ocr


@pytest.mark.parametrize(
    "id, expected_ocr",
    [
        ("08HdV8ArxuVKXgxdUor1", "K SC124"),
        ("uJsY6e391eOodCkFLMJA", "1A777AK77"),
        ("qTPu96zhef7AbiBSFFTD", "R275 ULO"),
    ],
)
def test_ocr_detector(id, expected_ocr):
    detected_plates = Queue(maxsize=0)
    detected_plates.put(download_lnpr_plate(id))
    detector = ocr_detector.OcrDetector(
        detected_plates=detected_plates,
        allowed_plates=expected_ocr,
        ocr_regex=".*",
        cv2_improve_plate_img_kwargs={"min_char_height_ratio": 0},
    )
    Thread(target=lambda: time.sleep(0.3) or detector.start()).start()
    detector.wait()
    assert detector.detected_ocrs.get_nowait() == expected_ocr

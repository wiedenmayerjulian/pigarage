import time
from threading import Thread
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest
from pigarage.plate_detector import PlateDetector

from . import download_lnpr_image


def test_no_plate_detected():
    mock = MagicMock()
    mock.capture_array.return_value = np.zeros((480, 640, 3), dtype=np.uint8)
    detector = PlateDetector(cam=mock)
    Thread(target=lambda: time.sleep(0.3) or detector.start()).start()
    detector.wait(timeout=2)
    detector.pause()
    assert mock.capture_array.call_count >= 1


@pytest.mark.parametrize(
    "id",
    ["08HdV8ArxuVKXgxdUor1", "uJsY6e391eOodCkFLMJA", "qTPu96zhef7AbiBSFFTD"],
)
def test_plate_detected(id):
    img = download_lnpr_image(id)
    mock = MagicMock()
    mock.capture_array.return_value = img
    detector = PlateDetector(cam=mock)
    detector._on_notifying = lambda: detector.pause()
    Thread(target=lambda: time.sleep(0.3) or detector.start()).start()
    detector.wait()
    assert mock.capture_array.call_count == 1
    assert detector.detected_plates.qsize() >= 1


def hstack_images(imgs):
    """Horizontally stack images, resizing heights as needed."""
    heights = [img.shape[0] for img in imgs]
    min_height = min(heights)
    resized_imgs = [
        cv2.resize(img, (img.shape[1] * min_height // img.shape[0], min_height))
        for img in imgs
    ]

    return np.hstack(resized_imgs)


def test_multiple_plates_detected():
    img = hstack_images(
        [
            download_lnpr_image(id)
            for id in ("7Vv1s1IXaN5CNlOb3rP2", "qTPu96zhef7AbiBSFFTD")
        ]
    )

    detector = PlateDetector(cam=None)
    boxes = detector._detect_plate_boxes(img)
    assert len(boxes) == 2
    detector._update_history(img, boxes)
    assert detector.detected_plates.qsize() == 2

    # Test again with adjusted positions
    for box in boxes:
        box.x1 += 1
    detector._update_history(img, boxes)
    assert detector.detected_plates.qsize() == 2

    # Test again with one plate removed
    detector._update_history(img, boxes[1:])
    assert detector.detected_plates.qsize() == 2


def test_direction_detected():
    img = download_lnpr_image("7Vv1s1IXaN5CNlOb3rP2")
    imgs = [
        np.concatenate([np.zeros((20 * i, img.shape[1], img.shape[2])), img], axis=0)
        for i in range(6)
    ]
    mock = MagicMock()
    mock.capture_array.side_effect = imgs
    detector = PlateDetector(cam=mock)
    detector._on_notifying = (
        lambda: detector.detected_directions.qsize() > 0 and detector.pause()
    )
    Thread(target=lambda: time.sleep(1) or detector.start()).start()
    while detector.detected_directions.qsize() == 0:
        detector.wait()
    assert detector.detected_directions.get_nowait() == "arriving"


def test_multiple_plates_direction_detected():
    imgs = [
        download_lnpr_image(id)
        for id in ("7Vv1s1IXaN5CNlOb3rP2", "qTPu96zhef7AbiBSFFTD")
    ]
    imgs = [
        hstack_images(
            [
                np.concatenate(
                    [
                        imgs[0][30 * i :, :],
                        np.zeros((30 * i, imgs[0].shape[1], imgs[0].shape[2])),
                    ],
                    axis=0,
                ),
                imgs[1],
            ]
        )
        for i in range(3)
    ]
    detector = PlateDetector(cam=None)
    for i, img in enumerate(imgs):
        cv2.imwrite(f"image_{i}.jpg", img)
        boxes = detector._detect_plate_boxes(img)
        detector._update_history(img, boxes)
        detector._detect_direction(img, boxes)

    assert detector.detected_plates.qsize() == 4
    assert detector.detected_directions.get_nowait() == "leaving"

import argparse
import logging
import re
import time
from pathlib import Path
from queue import Queue

import cv2
import easyocr
import numpy as np

from .config import VISUAL_DEBUG
from .config import config as pigarage_config
from .util import PausableNotifingThread

TRACE = 5
"""Custom logging level for trace messages."""
logging.addLevelName(TRACE, "TRACE")


def cv2_improve_plate_img(
    plate: cv2.typing.MatLike,
    blur: int = 5,
    block_size: int = 151,
    c: float = 2,
    min_plate_area: float = 0.3,
) -> cv2.typing.MatLike | None:
    plate = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(
        src=cv2.medianBlur(plate, blur),
        maxValue=255,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY,
        blockSize=block_size,
        C=c,
    )

    # Calculate contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # Calculate areas of contours and sort them descending
    areas = np.array([cv2.contourArea(c) for c in contours])
    if np.all(areas == 0):
        logging.getLogger(__name__).log(TRACE, "All contours have zero area")
        return None
    idxs = np.argsort(areas)[::-1]
    # Check that plate area is large enough
    if areas[idxs[0]] / (plate.shape[0] * plate.shape[1]) < min_plate_area:
        logging.getLogger(__name__).log(
            TRACE,
            "Plate area too small: %.2e",
            areas[idxs[0]] / (plate.shape[0] * plate.shape[1]),
        )
        return None

    masked = cv2.drawContours(
        cv2.bitwise_not(np.zeros(thresh.shape).astype(thresh.dtype)),
        [contours[i] for i in idxs[1:]],
        -1,
        color=(0, 0, 0),
        thickness=cv2.FILLED,
    )
    # Remove everything outside the largest contour (the plate)
    plate_background = cv2.drawContours(
        np.zeros(thresh.shape).astype(thresh.dtype),
        [contours[idxs[0]]],
        -1,
        color=(255, 255, 255),
        thickness=cv2.FILLED,
    )
    masked = cv2.bitwise_or(masked, cv2.bitwise_not(plate_background))
    if VISUAL_DEBUG:
        cv2.imshow("masked", masked)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return masked


def plate2text(plate: cv2.typing.MatLike, reader: easyocr.Reader | None = None) -> str:
    reader = reader or easyocr.Reader(["en"], verbose=False)
    result = reader.readtext(
        plate,
        allowlist="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ.o",
        text_threshold=0.5,
        slope_ths=0.01,
        add_margin=0,
    )
    filtered = re.sub(
        r"[o. ]+",
        " ",
        " ".join(
            text
            for _, text in sorted(
                (
                    (box, text)
                    for box, text, _confidence in result
                    # Drop all results with low height
                    if abs(box[-1][1] - box[0][1]) > 0.3 * plate.shape[0]
                ),
                key=lambda x: x[0][0][0],  # Sort by x coordinate
            )
        ),
    )
    logging.getLogger(__name__).debug(f"OCR result: {result} => {filtered}")
    return filtered


class OcrDetector(PausableNotifingThread):
    def __init__(
        self,
        detected_plates: Queue,
        allowed_plates: list[str],
        ocr_regex: str = r"[A-Z]{1,2}\.? ?\.?[A-Z]{0,2} ?[0-9]{2,4}$",
        *,
        debug: bool = False,
    ) -> None:
        super().__init__()
        self._debug = debug
        self._detected_plates = detected_plates
        self.detected_ocrs = Queue(maxsize=1)
        self._ocr_regex = ocr_regex
        self._reader = easyocr.Reader(["en"], gpu=False, verbose=False)
        self.allowed_plates = allowed_plates

    def resume(self) -> None:
        while self.detected_ocrs.qsize() > 0:
            self.detected_ocrs.get_nowait()
        return super().resume()

    def _postprocess(self, ocr: str) -> str:
        ocr = re.search(self._ocr_regex, ocr)
        if ocr:
            return ocr.group(0)
        return None

    def process(self) -> None:
        plate = self._detected_plates.get()

        if self._debug:
            cv2.imwrite(
                pigarage_config.logdir
                / f"{time.strftime('%Y-%m-%d_%H-%M-%S')}_ocr_pre.jpg",
                plate,
            )
        plate = cv2_improve_plate_img(plate)
        if plate is None:
            return
        if self._debug:
            cv2.imwrite(
                pigarage_config.logdir
                / f"{time.strftime('%Y-%m-%d_%H-%M-%S')}_ocr_post.jpg",
                plate,
            )
        result = plate2text(plate, reader=self._reader)
        ocr = self._postprocess(result)
        self._log.info(f"OCR: '{result.strip()}' -> '{ocr}'")
        if ocr is not None and ocr in self.allowed_plates:
            self.detected_ocrs.put(ocr)
            self._notify_waiters()
            self.pause()


def main() -> None:
    logging.basicConfig(level=TRACE)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "action",
        choices=("improve", "ocr"),
        help="Action to perform",
    )
    parser.add_argument("input", type=Path, help="Path to the image file")
    parser.add_argument(
        "--output",
        type=Path,
        help="Path to save the output image. "
        "If not provided, the image will be shown using cv2.imshow.",
    )
    parser.add_argument(
        "--visual-debug",
        action="store_true",
        help="Enable visual debug mode with additional output.",
    )
    args = parser.parse_args()

    global VISUAL_DEBUG  # noqa: PLW0603
    VISUAL_DEBUG = args.visual_debug
    img = cv2.imread(str(args.input))

    match args.action:
        case "improve":
            out = cv2_improve_plate_img(img)
        case "ocr":
            result = plate2text(img)
            logging.getLogger(__name__).info(f"OCR result: '{result}'")
            return

    if args.output:
        cv2.imwrite(str(args.output), out)
    else:
        cv2.imshow("output", out)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

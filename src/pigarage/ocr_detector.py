import argparse
import logging
import re
import time
from collections.abc import Callable
from http import HTTPStatus
from http.client import HTTPException
from pathlib import Path
from queue import Queue

import cv2
import easyocr
import numpy as np
import requests

from .config import VISUAL_DEBUG
from .config import config as pigarage_config
from .util import PausableNotifingThread

TRACE = 5
"""Custom logging level for trace messages."""
logging.addLevelName(TRACE, "TRACE")


def cv2_contours_append_children(
    hierarachy: np.ndarray,
    idx: int,
    idxs: set[int],
) -> None:
    if (child := hierarachy[0][idx][2]) != -1:
        idxs.add(child)
        cv2_contours_append_children(hierarachy, child, idxs)
    if (sibling := hierarachy[0][idx][0]) != -1:
        idxs.add(sibling)
        cv2_contours_append_children(hierarachy, sibling, idxs)


def cv2_improve_plate_img(  # noqa: PLR0913
    plate: cv2.typing.MatLike,
    blur: int = 3,
    block_size: int = 181,
    c: float = 2,
    clahe_clip: float = 2.0,
    clahe_tile: int = 5,
    min_plate_area: float = 0.3,
    min_char_height_ratio: float = 0.5,
) -> cv2.typing.MatLike | None:
    plate = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY) if len(plate.shape) == 3 else plate  # noqa: PLR2004
    plate = cv2.createCLAHE(
        clipLimit=clahe_clip,
        tileGridSize=(clahe_tile, clahe_tile),
    ).apply(plate)
    thresh = cv2.adaptiveThreshold(
        src=cv2.medianBlur(plate, blur),
        maxValue=255,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY,
        blockSize=block_size,
        C=c,
    )

    # Calculate contours
    contours, hierarachy = cv2.findContours(
        thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
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

    heights = np.array([cv2.boundingRect(c)[3] for c in contours], dtype=np.float32)
    heights /= heights[idxs[0]]

    idxs_to_draw = {i for i in idxs[1:] if heights[i] > min_char_height_ratio}
    for i in idxs_to_draw.copy():
        cv2_contours_append_children(hierarachy, i, idxs_to_draw)

    masked = cv2.drawContours(
        cv2.bitwise_not(np.zeros(thresh.shape).astype(thresh.dtype)),
        [contours[i] for i in idxs_to_draw],
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
    def __init__(  # noqa: PLR0913
        self,
        detected_plates: Queue,
        allowed_plates: list[str],
        ocr_regex: str = r"[A-Z]{1,2}\.? ?\.?[A-Z]{0,2} ?[0-9]{2,4}$",
        on_ocr_detected: Callable[[str], None] = lambda _: None,
        *,
        cv2_improve_plate_img_kwargs: dict | None = None,
        debug: bool = False,
    ) -> None:
        super().__init__()
        self._debug = debug
        self._detected_plates = detected_plates
        self.detected_ocrs = Queue(maxsize=1)
        self._ocr_regex = ocr_regex
        self._reader = easyocr.Reader(["en"], gpu=False, verbose=False)
        self.allowed_plates = allowed_plates
        self._on_ocr_detected = on_ocr_detected
        self._remote_session = requests.Session()
        self._cv2_improve_plate_img_kwargs = cv2_improve_plate_img_kwargs

    def resume(self) -> None:
        while self.detected_ocrs.qsize() > 0:
            self.detected_ocrs.get_nowait()
        return super().resume()

    def _postprocess(self, ocr: str) -> str:
        ocr = re.search(self._ocr_regex, ocr)
        if ocr:
            return ocr.group(0)
        return None

    def _process_remote(self, plate: cv2.typing.MatLike) -> str | None:
        logging.getLogger(__name__).debug("Using remote OCR")
        plate = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
        response = self._remote_session.post(
            f"{pigarage_config.client['url']}/ocr",
            headers={"auth": pigarage_config.client["auth"]},
            data=cv2.imencode(".jpg", plate)[1].tobytes(),
        )
        if response.status_code != HTTPStatus.OK:
            raise HTTPException(f"Remote OCR failed with {response.status_code}")
        return response.json().get("result")

    def _process_local(self, plate: cv2.typing.MatLike) -> str | None:
        logging.getLogger(__name__).debug("Using local OCR")
        plate = cv2_improve_plate_img(
            plate, **(self._cv2_improve_plate_img_kwargs or {})
        )
        if plate is None:
            return None
        if self._debug:
            cv2.imwrite(
                pigarage_config.logdir
                / f"{time.strftime('%Y-%m-%d_%H-%M-%S')}_ocr_post.jpg",
                plate,
            )
        return plate2text(plate, reader=self._reader)

    def process(self) -> None:
        plate = self._detected_plates.get()

        if self._debug:
            cv2.imwrite(
                pigarage_config.logdir
                / f"{time.strftime('%Y-%m-%d_%H-%M-%S')}_ocr_pre.jpg",
                plate,
            )

        if "url" in pigarage_config.client:
            try:
                result = self._process_remote(plate)
            except (requests.RequestException, HTTPException) as e:
                logging.getLogger(__name__).warning(
                    f"Remote OCR failed, falling back to local OCR: {e}"
                )
                result = self._process_local(plate)
        else:
            result = self._process_local(plate)

        ocr = self._postprocess(result)
        self._log.info(f"OCR: '{result.strip()}' -> '{ocr}'")
        if ocr is not None:
            self._on_ocr_detected(ocr)
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

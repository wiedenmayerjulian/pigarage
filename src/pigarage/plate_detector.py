import argparse
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from queue import Queue
from typing import Literal

import cv2
import ultralytics
from huggingface_hub import hf_hub_download
from ultralytics import YOLO

try:
    from picamera2 import Picamera2
except ImportError:
    from unittest.mock import MagicMock

    Picamera2 = MagicMock()

from .config import config as pigarage_config
from .util import PausableNotifingThread


def increment_path_exists_ok(
    path: str | Path,
    *,
    exist_ok: bool = False,  # noqa: ARG001
    sep: str = "",  # noqa: ARG001
    mkdir: bool = False,  # noqa: ARG001
) -> Path:
    return path


ultralytics.utils.files.increment_path = increment_path_exists_ok
ultralytics.utils.LOGGER.setLevel(logging.WARNING)


@dataclass
class Box:
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def center_x(self) -> float:
        return (self.x1 + self.x2) / 2

    @property
    def center_y(self) -> float:
        return (self.y1 + self.y2) / 2


class PlateHistory:
    def __init__(self, different_plate_distance: int) -> None:
        self._boxes: list[Box] = []
        self._different_plate_distance = different_plate_distance

    def get(self, box: Box) -> Box | None:
        """Get existing plate close to center_x, if any."""
        for prev_box in self._boxes:
            if abs(box.center_x - prev_box.center_x) < self._different_plate_distance:
                return prev_box
        return None

    def update_x(self, box: Box) -> None:
        """Update x position of existing plates to account for movement."""
        for prev_box in self._boxes:
            if abs(box.center_x - prev_box.center_x) < self._different_plate_distance:
                prev_box.x1 = box.x1
                prev_box.x2 = box.x2

    def add_or_update(self, box: Box) -> bool:
        """Add new plate position or update existing one."""
        if self.get(box) is not None:
            self.update_x(box)
            return False
        self._boxes.append(box)
        return True

    def clear(self) -> None:
        self._boxes = []


class PlateDetector(PausableNotifingThread):
    def __init__(  # noqa: PLR0913
        self,
        cam: Picamera2,
        cam_setting: str = "main",
        on_resume: Callable[[], None] = lambda: None,
        on_notifying: Callable[[], None] = lambda: None,
        on_direction: Callable[[Literal["arriving", "leaving"]], None] = lambda _: None,
        direction_min_distance: int = 50,
        direction_ignore_distance: int = 5,
        different_plate_distance: int = 100,
        *,
        debug: bool = False,
    ) -> None:
        super().__init__(on_resume=on_resume, on_notifying=on_notifying)
        self.model = YOLO(
            hf_hub_download(
                "morsetechlab/yolov11-license-plate-detection",
                "license-plate-finetune-v1n.pt",
            )
        )
        self._cam = cam
        self._cam_setting = cam_setting
        self._on_direction = on_direction
        self._debug = debug
        self.detected_plates = Queue(maxsize=10)
        self.detected_directions = Queue(maxsize=1)
        self._direction_min_distance = direction_min_distance
        self._direction_ignore_distance = direction_ignore_distance
        self._different_plate_distance = different_plate_distance
        self._history = PlateHistory(different_plate_distance)

    def resume(self) -> None:
        while self.detected_plates.qsize() > 0:
            self.detected_plates.get_nowait()
        while self.detected_directions.qsize() > 0:
            self.detected_directions.get_nowait()
        self._history.clear()
        return super().resume()

    def _detect_plate_boxes(self, img: cv2.typing.MatLike) -> list[Box]:
        results = self.model.predict(
            source=img,
            verbose=False,
            save=self._debug,
            save_crop=self._debug,
            project="/tmp",  # noqa: S108
            name="plate_detector",
            imgsz=512,
        )
        boxes: list[Box] = []
        # Detect plates
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            box = Box(x1, y1, x2, y2)  # noqa: PLW2901
            boxes.append(box)
        return boxes

    def _update_history(self, img: cv2.typing.MatLike, boxes: list[Box]) -> None:
        """Update plate history and notify about new plates."""
        for i, box in enumerate(boxes):
            if self._history.add_or_update(box):
                self._log.debug(
                    f"New plate detected at ({len(self._history._boxes)}, "  # noqa: SLF001
                    f"{box.center_x}, {box.center_y})"
                )
                if self._debug:
                    cv2.imwrite(
                        pigarage_config.logdir
                        / f"{time.strftime('%Y-%m-%d_%H-%M-%S')}_plate_{i}.jpg",
                        cv2.resize(
                            cv2.rectangle(
                                img,
                                (box.x1, box.y1),
                                (box.x2, box.y2),
                                color=(255, 0, 0),
                                thickness=3,
                            ),
                            dsize=(int(0.2 * img.shape[1]), int(0.2 * img.shape[0])),
                        ),
                    )
                self.detected_plates.put(img[box.y1 : box.y2, box.x1 : box.x2])
                self._notify_waiters()

    def _detect_direction(self, img: cv2.typing.MatLike, boxes: list[Box]) -> None:
        """Detect direction of plates based on y position changes."""
        for box_index, box in enumerate(boxes):
            prev = self._history.get(box)
            if prev is None:
                continue
            if abs(box.center_y - prev.center_y) < self._direction_ignore_distance:
                # Continue if movement of this box is too small to update the plate
                continue
            self._log.debug(
                f"Plate updated ({box_index}, {box.center_x}, {box.center_y})"
            )

            if self.detected_plates.qsize() < self.detected_plates.maxsize:
                self.detected_plates.put(img[box.y1 : box.y2, box.x1 : box.x2])
                self._notify_waiters()

            if self.detected_directions.qsize() > 0:
                # Continue if direction already detected
                continue

            diff = box.center_y - prev.center_y
            if abs(diff) < self._direction_min_distance:
                # Continue if movement is still too small
                continue

            direction = "arriving" if diff > 0 else "leaving"
            self._log.debug(f"direction: {direction}")
            self.detected_directions.put(direction)
            self._on_direction(direction)
            self._notify_waiters()

    def process(self) -> None:
        img = self._cam.capture_array(self._cam_setting)
        boxes = self._detect_plate_boxes(img)
        if len(boxes) == 0:
            return
        self._update_history(img, boxes)
        self._detect_direction(img, boxes)


def main() -> None:
    logging.basicConfig(level=logging.DEBUG)
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path, help="Path to the image file")
    parser.add_argument(
        "--output",
        type=Path,
        help="Path to save the output image. "
        "If not provided, the image will be shown using cv2.imshow.",
    )
    args = parser.parse_args()

    img = cv2.imread(str(args.input))
    detector = PlateDetector(cam=None)  # type: ignore[arg-type]
    results = detector.model.predict(
        source=img,
        verbose=False,
        save=False,
        save_crop=False,
        imgsz=512,
    )
    if len(results[0].boxes) == 0:
        logging.getLogger(__name__).info("No plate detected")
    for i, box in enumerate(results[0].boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        logging.getLogger(__name__).info(
            "Plate detected at (%d, %d, %d, %d)", x1, y1, x2, y2
        )
        out = img[y1:y2, x1:x2]
        if args.output:
            cv2.imwrite(f"{str(args.output).rsplit('.', 1)[0]}_{i}.jpg", out)
        else:
            cv2.imshow("output", out)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

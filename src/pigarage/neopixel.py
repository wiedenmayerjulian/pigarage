import functools
import operator
import threading
import time
from queue import Queue
from typing import Literal

try:
    import spidev
except ImportError:
    from unittest.mock import MagicMock

    spidev = MagicMock()


def flatten(xss: list) -> list:
    return [x for xs in xss for x in xs]


def rgb_to_bits(rgb: tuple[int, int, int]) -> list[int]:
    r, g, b = rgb
    return flatten(
        (0xF8 if bit == "1" else 0xC0 for bit in f"{color:08b}") for color in (g, r, b)
    )


class StoppableTask(threading.Thread):
    def __init__(self, func: callable, *args: object, **kwargs: object) -> None:
        self._running = False
        self._func = func
        super().__init__(*args, **kwargs)

    def start(self) -> None:
        self._running = True
        return super().start()

    def stop(self) -> None:
        self._running = False

    def run(self) -> None:
        while True:
            if self._running is False:
                return
            self._func()


class NeopixelSpi:
    instance = None

    def __init__(self, bus: int, device: int, leds: int, spi_freq: int = 800) -> None:
        NeopixelSpi.instance = self
        self.spi = spidev.SpiDev()
        self.spi.open(bus, device)
        self.spi.max_speed_hz = spi_freq * 1024 * 8

        self.num_leds = leds
        self._task_queue = Queue(maxsize=1)
        self._task = threading.Thread(target=self._task_func)
        self._task.start()
        self.color = (0, 0, 0)
        self.clear()

    def _task_func(self) -> None:
        func = None
        while True:
            if func is None or self._task_queue.qsize() > 0:
                func = self._task_queue.get()
            if func is not None:
                func()

    def update(self, state: list[tuple[int, int, int]]) -> None:
        raw_data = functools.reduce(
            operator.iadd, (rgb_to_bits(led) for led in state), []
        )
        self.spi.xfer3(raw_data)

    def clear(self) -> None:
        self.stop()
        self.fill((0, 0, 0))

    def fill(
        self,
        color: tuple[int, int, int] | None = None,
    ) -> None:
        self.stop()
        self.color = color or self.color
        self.update(self.num_leds * [self.color])

    def fade(
        self,
        color_from: tuple[int, int, int],
        color_to: tuple[int, int, int],
        duration: float = 0.1,
        steps: int = 20,
    ) -> None:
        self.stop()
        for i in range(steps + 1):
            self.fill(
                tuple(
                    round(c1 + (c2 - c1) / steps * i)
                    for c1, c2 in zip(color_from, color_to, strict=False)
                )
            )
            time.sleep(duration / steps)

    def pulse_once(
        self,
        color: tuple[int, int, int] | None = None,
        amplitude: float = 1.0,
        duration: float = 0.5,
        steps: int | str = "auto",
    ) -> None:
        self.stop()
        if steps == "auto":
            steps = int(20 * duration)
        self.color = color or self.color
        color_to = [(1 - amplitude) * c for c in self.color]
        self.fade(self.color, color_to, duration=duration / 2, steps=steps)
        self.fade(color_to, self.color, duration=duration / 2, steps=steps)

    def pulse(
        self,
        color: tuple[int, int, int] | None = None,
        amplitude: float = 1.0,
        duration: float = 0.5,
        steps: int | str = "auto",
    ) -> None:
        self._task_queue.put(
            lambda: self.pulse_once(
                color,
                amplitude=amplitude,
                duration=duration,
                steps=steps,
            )
        )

    def stop(self) -> None:
        if threading.current_thread() is self._task:
            return
        self._task_queue.put(None)
        # Put second time to ensure previous None was consumed
        self._task_queue.put(None)

    def roll_once(
        self,
        color: tuple[int, int, int] | None = None,
        duration: float = 2.0,
    ) -> None:
        self.stop()
        self.color = color or self.color
        for i in range(self.num_leds):
            state = self.num_leds * [(0, 0, 0)]
            state[i] = self.color
            self.update(state)
            time.sleep(duration / self.num_leds)

    def roll(
        self,
        color: tuple[int, int, int] | None = None,
        duration: float = 2.0,
    ) -> None:
        self._task_queue.put(
            lambda: self.roll_once(
                color,
                duration=duration,
            )
        )

    def sweep(
        self,
        direction: Literal["ltr", "rtl", "ttb", "btt"],
        color: tuple[int, int, int] | None = None,
        duration: float = 1.0,
    ) -> None:
        self._task_queue.put(
            lambda: self.sweep_once(
                direction=direction,
                color=color,
                duration=duration,
            )
        )

    def sweep_once(
        self,
        direction: Literal["ltr", "rtl", "ttb", "btt"],
        color: tuple[int, int, int] | None = None,
        duration: float = 1.0,
    ) -> None:
        self.stop()
        count = self.num_leds // 2
        self.color = color or self.color
        for i in range(count):
            state = self.num_leds * [(0, 0, 0)]
            match direction:
                case "btt":
                    state[i] = self.color
                    state[-(i + 1)] = self.color
                case "ttb":
                    state[count - i - 1] = self.color
                    state[-(count - i)] = self.color
            self.update(state)
            time.sleep(duration / count)


if __name__ == "__main__":
    neo = NeopixelSpi(bus=0, device=0, leds=12)
    neo.clear()
    # neo.sweep("ttb", (0, 0, 255), 1.0)
    # time.sleep(10)
    # neo.clear()

import functools
import operator
import threading
import time
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

        self.state = leds * [(0, 0, 0)]
        self._task = None
        self._task_lock = threading.RLock()
        self.color = (0, 0, 0)
        self.clear()

    def update(self, newstate: list[tuple[int, int, int]] | None = None) -> None:
        if newstate:
            self.state = newstate
        raw_data = functools.reduce(
            operator.iadd, (rgb_to_bits(led) for led in self.state), []
        )
        self.spi.xfer3(raw_data)

    def clear(self) -> None:
        self.stop()
        self.fill(0, 0, 0)

    def fill(
        self,
        color: tuple[int, int, int] | None = None,
    ) -> None:
        self.stop()
        color = color or self.color
        self.state = len(self.state) * [color]
        self.update()

    def fade(
        self,
        color_from: tuple[int, int, int],
        color_to: tuple[int, int, int],
        duration: float = 0.1,
        steps: int = 20,
    ) -> None:
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
        if steps == "auto":
            steps = int(20 * duration)
        color = color or self.color
        color_to = [(1 - amplitude) * c for c in color]
        self.fade(color, color_to, duration=duration / 2, steps=steps)
        self.fade(color_to, color, duration=duration / 2, steps=steps)

    def pulse(
        self,
        color: tuple[int, int, int] | None = None,
        amplitude: float = 1.0,
        duration: float = 0.5,
        steps: int | str = "auto",
    ) -> None:
        self.stop()
        with self._task_lock:
            self._task = StoppableTask(
                func=lambda: self.pulse_once(
                    color,
                    amplitude=amplitude,
                    duration=duration,
                    steps=steps,
                )
            )
            self._task.start()

    def stop(self) -> None:
        if self._task is None or threading.current_thread() is self._task:
            return
        with self._task_lock:
            self._task.stop()
            self._task.join()
            self._task = None

    def roll_once(
        self,
        color: tuple[int, int, int] | None = None,
        duration: float = 2.0,
    ) -> None:
        self.stop()
        color = color or self.color
        for i in range(len(self.state)):
            state = len(self.state) * [(0, 0, 0)]
            state[i] = color
            self.update(state)
            time.sleep(duration / len(self.state))

    def roll(
        self,
        color: tuple[int, int, int] | None = None,
        duration: float = 2.0,
    ) -> None:
        self.stop()
        with self._task_lock:
            self._task = StoppableTask(
                func=lambda: self.roll_once(
                    color,
                    duration=duration,
                )
            )
            self._task.start()

    def sweep(
        self,
        direction: Literal["ltr", "rtl", "ttb", "btt"],
        color: tuple[int, int, int] | None = None,
        duration: float = 2.0,
    ) -> None:
        self.stop()
        with self._task_lock:
            self._task = StoppableTask(
                func=lambda: self.sweep_once(
                    direction=direction,
                    color=color,
                    duration=duration,
                )
            )
            self._task.start()

    def sweep_once(
        self,
        direction: Literal["ltr", "rtl", "ttb", "btt"],
        color: tuple[int, int, int] | None = None,
        duration: float = 2.0,
    ) -> None:
        self.stop()
        count = len(self.state) // 2
        color = color or self.color
        for i in range(count):
            state = len(self.state) * [(0, 0, 0)]
            match direction:
                case "ttb":
                    state[i] = color
                    state[-i] = color
                case "btt":
                    state[count - i] = color
                    state[-(count - i)] = color
            self.update(state)
            time.sleep(duration / count)


if __name__ == "__main__":
    neo = NeopixelSpi(bus=0, device=0, leds=12)
    neo.clear()

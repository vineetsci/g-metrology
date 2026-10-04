"""Small terminal-display helpers used by the reproducibility suite."""
from __future__ import annotations

from dataclasses import dataclass, field
import sys
import threading
import time
from typing import TextIO

SPINNER_FRAMES = ("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏")
SBC_BAR_WIDTH = 40
SPINNER_INTERVAL_SECONDS = 0.10


def format_duration(seconds: float) -> str:
    """Format elapsed/remaining time as MM:SS or HH:MM:SS."""
    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, sec = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{sec:02d}"
    return f"{minutes:02d}:{sec:02d}"


def format_sbc_progress(
    done: int,
    total: int,
    elapsed: float,
    width: int = SBC_BAR_WIDTH,
    spinner_index: int | None = None,
) -> str:
    """Return one SBC progress line without a trailing newline.

    ``spinner_index`` is independent of ``done`` so the spinner can animate
    smoothly between simulation-completion callbacks.
    """
    if total < 1:
        raise ValueError("total must be positive")
    if not 0 <= done <= total:
        raise ValueError("done must lie between 0 and total")
    if width < 1:
        raise ValueError("width must be positive")

    filled = min(width, int(width * done / total))
    fraction = done / total
    rate = done / elapsed if done > 0 and elapsed > 0 else 0.0
    remaining = (total - done) / rate if rate > 0 else 0.0
    eta = "done" if done >= total else format_duration(remaining)
    if spinner_index is None:
        spinner_index = filled
    frame = SPINNER_FRAMES[spinner_index % len(SPINNER_FRAMES)]
    bar = "|" * filled + "." * (width - filled)
    return f"{frame} {fraction:6.1%} {bar}  elapsed {format_duration(elapsed)}  ETA {eta}"


@dataclass
class SBCProgress:
    """Render SBC progress on one terminal line with an independent spinner."""

    total: int
    stream: TextIO = field(default_factory=lambda: sys.stdout)
    enabled: bool | None = None
    start_time: float | None = None
    spinner_interval: float = SPINNER_INTERVAL_SECONDS
    auto_start: bool = True
    _last_filled: int = -1
    _done: int = 0
    _spinner_index: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    _stop_event: threading.Event = field(default_factory=threading.Event, init=False, repr=False)
    _thread: threading.Thread | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.total < 1:
            raise ValueError("total must be positive")
        if self.spinner_interval <= 0:
            raise ValueError("spinner_interval must be positive")
        if self.start_time is None:
            self.start_time = time.perf_counter()
        if self.enabled is None:
            self.enabled = bool(getattr(self.stream, "isatty", lambda: False)())
        if self.enabled and self.auto_start:
            self.start()

    def start(self) -> None:
        """Start the independent spinner animation for an interactive terminal."""
        if not self.enabled or self._thread is not None:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._animate,
            name="sbc-progress-spinner",
            daemon=True,
        )
        self._thread.start()

    def _animate(self) -> None:
        while not self._stop_event.wait(self.spinner_interval):
            with self._lock:
                if self._done >= self.total:
                    return
                self._spinner_index += 1
                self._render_locked()

    def _render_locked(self) -> None:
        elapsed = time.perf_counter() - float(self.start_time)
        line = format_sbc_progress(
            self._done,
            self.total,
            elapsed,
            spinner_index=self._spinner_index,
        )
        self.stream.write("\r" + line + " " * 8)
        self.stream.flush()

    def update(self, done: int, total: int | None = None) -> None:
        actual_total = self.total if total is None else total
        if actual_total != self.total:
            raise ValueError("total cannot change during SBC progress")
        if not 0 <= done <= self.total:
            raise ValueError("done must lie between 0 and total")
        if not self.enabled:
            return

        with self._lock:
            self._done = done
            filled = min(SBC_BAR_WIDTH, int(SBC_BAR_WIDTH * done / self.total))
            if filled != self._last_filled:
                self._last_filled = filled
                self._render_locked()

        if done == self.total:
            self.finish()

    def finish(self) -> None:
        """Stop animation and leave one final completed progress line."""
        if not self.enabled:
            return
        with self._lock:
            self._done = self.total
            self._spinner_index += 1
            self._render_locked()
        self._stop_event.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=max(0.2, self.spinner_interval * 3))
        self.stream.write("\n")
        self.stream.flush()

    def __enter__(self) -> "SBCProgress":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.enabled and self._done < self.total:
            self._stop_event.set()
            thread = self._thread
            if thread is not None and thread is not threading.current_thread():
                thread.join(timeout=max(0.2, self.spinner_interval * 3))

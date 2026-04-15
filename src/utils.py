"""Utility helpers for logging and optional text-to-speech."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from queue import Empty, Queue
from threading import Event, Thread
from typing import Optional


class GestureLogger:
    """Append recognized gestures to a text log file."""

    def __init__(self, log_path: str = "gesture_log.txt") -> None:
        self.log_path = Path(log_path)

    def log(self, message: str) -> None:
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with self.log_path.open("a", encoding="utf-8") as file:
            file.write(f"[{stamp}] {message}\n")


class SpeechEngine:
    """Optional non-blocking text-to-speech helper."""

    def __init__(self, enabled: bool = False) -> None:
        self.enabled = enabled
        self.engine: Optional[object] = None
        self.queue: Queue[str] = Queue(maxsize=50)
        self.stop_event = Event()
        self.worker: Optional[Thread] = None

        if not enabled:
            return

        try:
            import pyttsx3

            self.engine = pyttsx3.init()
            self.engine.setProperty("rate", 150)
            self.worker = Thread(target=self._run, daemon=True)
            self.worker.start()
        except Exception as exc:
            print(f"[Speech] Text-to-speech unavailable: {exc}")
            self.enabled = False
            self.engine = None
            self.worker = None

    def speak(self, text: str) -> None:
        """Queue speech without blocking the video loop."""
        if not self.enabled or self.engine is None:
            return
        if self.queue.full():
            return
        self.queue.put(text)

    def close(self) -> None:
        if not self.enabled:
            return
        self.stop_event.set()
        if self.worker and self.worker.is_alive():
            self.worker.join(timeout=1.0)

    def _run(self) -> None:
        while not self.stop_event.is_set():
            try:
                text = self.queue.get(timeout=0.2)
            except Empty:
                continue
            if self.engine is None:
                continue
            try:
                self.engine.say(text)
                self.engine.runAndWait()
            except Exception:
                # Keep app alive even if local audio engine fails mid-run.
                self.enabled = False
                return

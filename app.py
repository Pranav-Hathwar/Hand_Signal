"""Real-time hand sign detection and translation using a webcam."""

from __future__ import annotations

import argparse
import time
from collections import deque

import cv2
import numpy as np

from src.gesture_recognizer import GestureRecognizer
from src.hand_detector import HandData, HandDetector
from src.utils import GestureLogger, SpeechEngine

WINDOW_NAME = "Real-Time Hand Sign Translator"
EXIT_KEYS = {ord("q"), ord("x"), 27}  # q, x, Esc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect and translate hand gestures in real time."
    )
    parser.add_argument("--camera-index", type=int, default=0, help="Webcam index.")
    parser.add_argument(
        "--max-hands",
        type=int,
        default=2,
        choices=(1, 2),
        help="Maximum hands to track.",
    )
    parser.add_argument(
        "--no-swap-handedness",
        action="store_true",
        help="Use raw MediaPipe Left/Right labels.",
    )
    parser.add_argument("--log", action="store_true", help="Save recognized gestures.")
    parser.add_argument("--speak", action="store_true", help="Enable text-to-speech.")
    return parser.parse_args()


def draw_overlay(frame, text: str, fps: float) -> None:
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 80), (20, 20, 20), -1)
    cv2.putText(
        frame,
        f"Gesture: {text}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 255),
        2,
    )
    cv2.putText(
        frame,
        f"FPS: {fps:.1f}  |  Quit: q / x / Esc",
        (20, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
    )


def should_exit(window_name: str, key: int) -> bool:
    if key in EXIT_KEYS:
        return True
    try:
        visible = cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE)
        autosize = cv2.getWindowProperty(window_name, cv2.WND_PROP_AUTOSIZE)
        if visible < 1 or autosize < 0:
            return True
    except cv2.error:
        return True
    return False


def label_hands(
    frame,
    hands: list[HandData],
    recognizer: GestureRecognizer,
    logger: GestureLogger,
    speaker: SpeechEngine,
    do_log: bool,
    now: float,
    last_label: dict[str, str],
    last_time: dict[str, float],
) -> str:
    active_keys: set[str] = set()
    labels: list[str] = []

    for hand in hands:
        key = hand.handedness
        active_keys.add(key)
        label = recognizer.get_stable_gesture(hand.landmarks_px, hand.handedness, key)
        labels.append(f"{key}: {label}")

        x_min, y_min, _, _ = hand.bbox
        cv2.putText(
            frame,
            label,
            (x_min, max(55, y_min - 35)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2,
        )

        prev_label = last_label.get(key, "")
        prev_time = last_time.get(key, 0.0)
        if label != "Unknown" and label != prev_label and now - prev_time > 0.9:
            message = f"{key}: {label}"
            print(f"Detected {message}")
            if do_log:
                logger.log(message)
            speaker.speak(message)
            last_label[key] = label
            last_time[key] = now

    removed = recognizer.prune_missing(active_keys)
    for key in removed:
        last_label.pop(key, None)
        last_time.pop(key, None)

    return " | ".join(labels) if labels else "No hand detected"


def main() -> None:
    args = parse_args()
    detector = HandDetector(
        max_num_hands=args.max_hands,
        swap_handedness=not args.no_swap_handedness,
    )
    recognizer = GestureRecognizer(smoothing_window=7)
    logger = GestureLogger()
    speaker = SpeechEngine(enabled=args.speak)

    cap = cv2.VideoCapture(args.camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. Check camera connection.")

    frame_deltas = deque(maxlen=12)
    previous_time = time.time()
    last_label: dict[str, str] = {}
    last_time: dict[str, float] = {}

    print("Starting webcam. Press q, x, Esc, close window, or Ctrl+C to quit.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Failed to read a frame from webcam.")
                break

            frame = cv2.flip(frame, 1)
            frame, hands = detector.find_hands(frame)

            now = time.time()
            frame_deltas.append(now - previous_time)
            previous_time = now
            fps = 1.0 / max(float(np.mean(frame_deltas)), 1e-6)

            if hands:
                text = label_hands(
                    frame=frame,
                    hands=hands,
                    recognizer=recognizer,
                    logger=logger,
                    speaker=speaker,
                    do_log=args.log,
                    now=now,
                    last_label=last_label,
                    last_time=last_time,
                )
            else:
                recognizer.reset()
                last_label.clear()
                last_time.clear()
                text = "No hand detected"

            draw_overlay(frame, text, fps)
            cv2.imshow(WINDOW_NAME, frame)

            key = cv2.waitKey(1) & 0xFF
            if should_exit(WINDOW_NAME, key):
                break
    except KeyboardInterrupt:
        print("Stopped by keyboard interrupt.")
    finally:
        cap.release()
        detector.close()
        speaker.close()
        try:
            cv2.destroyWindow(WINDOW_NAME)
        except cv2.error:
            pass
        cv2.destroyAllWindows()
        # Flush window events so close is immediate on more systems.
        for _ in range(3):
            cv2.waitKey(1)


if __name__ == "__main__":
    main()

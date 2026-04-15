"""Real-time hand sign detection and translation using a webcam."""

from __future__ import annotations

import argparse
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from src.gesture_recognizer import GestureRecognizer
from src.hand_detector import HandData, HandDetector
from src.utils import GestureLogger, SpeechEngine

WINDOW_NAME = "Real-Time Hand Sign Translator"
EXIT_KEYS  = {ord("q"), ord("x"), 27}   # q, x, Esc
HELP_KEY   = ord("h")
SHOT_KEY   = ord("s")
SHOT_DIR   = Path("screenshots")


# ──────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect and translate hand gestures in real time."
    )
    parser.add_argument("--camera-index", type=int, default=0, help="Webcam index.")
    parser.add_argument(
        "--max-hands", type=int, default=2, choices=(1, 2),
        help="Maximum hands to track.",
    )
    parser.add_argument(
        "--no-swap-handedness", action="store_true",
        help="Use raw MediaPipe Left/Right labels.",
    )
    parser.add_argument("--log",   action="store_true", help="Save recognized gestures.")
    parser.add_argument("--speak", action="store_true", help="Enable text-to-speech.")
    return parser.parse_args()


# ──────────────────────────────────────────────────────────────
# Drawing helpers
# ──────────────────────────────────────────────────────────────

def draw_overlay(frame, text: str, fps: float, confidence: float, session_count: int) -> None:
    """Top info bar: gesture name, confidence, FPS, session count, key hints."""
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 80), (20, 20, 20), -1)

    conf_str = f"  ({int(confidence * 100)}%)" if confidence > 0 else ""
    cv2.putText(
        frame, f"Gesture: {text}{conf_str}",
        (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2,
    )
    cv2.putText(
        frame,
        f"FPS: {fps:.1f}  |  Session: {session_count}  |  H: Guide  S: Screenshot  Q: Quit",
        (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 200, 200), 1,
    )


def draw_history_strip(frame, history: deque) -> None:
    """Thin bottom bar showing the last few detected gestures."""
    h = frame.shape[0]
    cv2.rectangle(frame, (0, h - 32), (frame.shape[1], h), (20, 20, 20), -1)
    recent = list(history)[-6:]
    hist_text = ("History: " + "  →  ".join(recent)) if recent else "History: (none yet)"
    cv2.putText(
        frame, hist_text,
        (15, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1,
    )


def draw_help_overlay(frame) -> None:
    """Semi-transparent gesture guide panel drawn over the live feed."""
    descriptions = GestureRecognizer.get_gesture_descriptions()

    # Panel bounds
    px, py = 70, 90
    pw = frame.shape[1] - px * 2
    ph = frame.shape[0] - py - 38

    overlay = frame.copy()
    cv2.rectangle(overlay, (px, py), (px + pw, py + ph), (10, 10, 10), -1)
    cv2.addWeighted(overlay, 0.80, frame, 0.20, 0, frame)

    # Title
    cv2.putText(
        frame, "GESTURE GUIDE  —  press H to close",
        (px + 18, py + 34), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 255, 255), 2,
    )
    cv2.line(frame, (px + 18, py + 46), (px + pw - 18, py + 46), (70, 70, 70), 1)

    # Two-column layout
    items  = list(descriptions.items())
    row_h  = 30
    y0     = py + 68
    col_w  = pw // 2

    for i, (name, desc) in enumerate(items):
        col = i % 2
        row = i // 2
        x   = px + 18 + col * col_w
        y   = y0 + row * row_h
        cv2.putText(frame, f"{name}:", (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (80, 220, 80), 1)
        cv2.putText(frame, desc, (x + 4, y + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (190, 190, 190), 1)


def take_screenshot(frame) -> None:
    """Save a timestamped PNG to the screenshots/ folder."""
    SHOT_DIR.mkdir(exist_ok=True)
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = SHOT_DIR / f"screenshot_{ts}.png"
    cv2.imwrite(str(path), frame)
    print(f"Screenshot saved: {path}")


# ──────────────────────────────────────────────────────────────
# Exit helper
# ──────────────────────────────────────────────────────────────

def should_exit(window_name: str, key: int) -> bool:
    if key in EXIT_KEYS:
        return True
    try:
        visible  = cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE)
        autosize = cv2.getWindowProperty(window_name, cv2.WND_PROP_AUTOSIZE)
        if visible < 1 or autosize < 0:
            return True
    except cv2.error:
        return True
    return False


# ──────────────────────────────────────────────────────────────
# Per-frame gesture labelling
# ──────────────────────────────────────────────────────────────

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
    session_stats: dict,
) -> tuple[str, float]:
    active_keys: set[str]  = set()
    labels:      list[str] = []
    confidences: list[float] = []

    for hand in hands:
        key = hand.handedness
        active_keys.add(key)

        label, confidence = recognizer.get_stable_gesture(
            hand.landmarks_px, hand.handedness, key
        )
        confidences.append(confidence)
        labels.append(f"{key}: {label}")

        # Gesture name near the bounding box
        x_min, y_min, _, _ = hand.bbox
        cv2.putText(
            frame, label,
            (x_min, max(55, y_min - 35)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2,
        )

        # Detect a new/changed gesture (with cooldown)
        prev_label = last_label.get(key, "")
        prev_time  = last_time.get(key, 0.0)
        if label != "Unknown" and label != prev_label and now - prev_time > 0.9:
            message = f"{key}: {label}"
            print(f"Detected {message}")
            if do_log:
                logger.log(message)
            speaker.speak(message)
            last_label[key] = label
            last_time[key]  = now
            session_stats["count"] += 1
            session_stats["history"].append(label)

    removed = recognizer.prune_missing(active_keys)
    for key in removed:
        last_label.pop(key, None)
        last_time.pop(key, None)

    text     = " | ".join(labels) if labels else "No hand detected"
    avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
    return text, avg_conf


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    detector  = HandDetector(
        max_num_hands=args.max_hands,
        swap_handedness=not args.no_swap_handedness,
    )
    recognizer = GestureRecognizer(smoothing_window=7)
    logger     = GestureLogger()
    speaker    = SpeechEngine(enabled=args.speak)

    cap = cv2.VideoCapture(args.camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,  720)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. Check camera connection.")

    frame_deltas  = deque(maxlen=12)
    previous_time = time.time()
    last_label:  dict[str, str]   = {}
    last_time:   dict[str, float] = {}
    session_stats = {"count": 0, "history": deque(maxlen=20)}
    show_help = False

    print("Starting webcam. Press q / x / Esc or close window to quit.")
    print("Press H for the gesture guide, S to take a screenshot.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Failed to read a frame from webcam.")
                break

            frame = cv2.flip(frame, 1)

            # Always run hand detection
            frame, hands = detector.find_hands(frame)

            now = time.time()
            frame_deltas.append(now - previous_time)
            previous_time = now
            fps = 1.0 / max(float(np.mean(frame_deltas)), 1e-6)

            if hands:
                text, confidence = label_hands(
                    frame=frame, hands=hands, recognizer=recognizer,
                    logger=logger, speaker=speaker, do_log=args.log,
                    now=now, last_label=last_label, last_time=last_time,
                    session_stats=session_stats,
                )
            else:
                recognizer.reset()
                last_label.clear()
                last_time.clear()
                text, confidence = "No hand detected", 0.0

            draw_overlay(frame, text, fps, confidence, session_stats["count"])
            draw_history_strip(frame, session_stats["history"])

            if show_help:
                draw_help_overlay(frame)

            cv2.imshow(WINDOW_NAME, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == HELP_KEY:
                show_help = not show_help
            elif key == SHOT_KEY:
                take_screenshot(frame)
            elif should_exit(WINDOW_NAME, key):
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
        for _ in range(3):
            cv2.waitKey(1)
        print(f"\nSession summary: {session_stats['count']} gesture(s) detected.")


if __name__ == "__main__":
    main()

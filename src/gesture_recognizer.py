"""Rule-based gesture recognition from MediaPipe hand landmarks."""

from __future__ import annotations

from collections import Counter, deque
from math import acos, hypot, pi
from typing import Deque, Dict, List, Tuple

Point = Tuple[int, int]


class GestureRecognizer:
    """Recognize beginner-friendly hand signs with simple geometry rules."""

    TIP = {"thumb": 4, "index": 8, "middle": 12, "ring": 16, "pinky": 20}
    PIP = {"thumb": 3, "index": 6, "middle": 10, "ring": 14, "pinky": 18}
    MCP = {"thumb": 2, "index": 5, "middle": 9, "ring": 13, "pinky": 17}

    INDEX_STRAIGHT_MIN = 160.0
    THUMB_STRAIGHT_MIN = 145.0
    WRIST_DISTANCE_MARGIN = 0.12
    THUMB_PALM_MARGIN = 0.08
    OK_PINCH_RATIO_MAX = 0.45
    THUMB_DIRECTION_MIN = 0.25

    PATTERN_TO_GESTURE = {
        "00000": "0 / Fist",
        "01000": "1",
        "01100": "2 / Peace",
        "01110": "3",
        "01111": "4",
        "11111": "5 / Hello",
        "10001": "6 / Call Me",
        "01001": "Rock",
        "11001": "I Love You",
        "11000": "L Sign",
        "00001": "Pinky",
        "00111": "W Sign",
    }

    def __init__(self, smoothing_window: int = 7) -> None:
        self.smoothing_window = smoothing_window
        self.histories: Dict[str, Deque[str]] = {}

    def reset(self, hand_key: str | None = None) -> None:
        if hand_key is None:
            self.histories.clear()
            return
        self.histories.pop(hand_key, None)

    def prune_missing(self, active_keys: set[str]) -> set[str]:
        removed = {key for key in self.histories if key not in active_keys}
        for key in removed:
            self.histories.pop(key, None)
        return removed

    def get_stable_gesture(
        self, landmarks: List[Point], handedness: str, hand_key: str
    ) -> Tuple[str, float]:
        """Return (gesture_name, confidence) where confidence is 0.0–1.0."""
        gesture = self.classify(landmarks, handedness)
        history = self.histories.setdefault(hand_key, deque(maxlen=self.smoothing_window))
        history.append(gesture)
        counter = Counter(history)
        best_gesture, best_count = counter.most_common(1)[0]
        confidence = best_count / len(history)
        return best_gesture, confidence

    def classify(self, landmarks: List[Point], handedness: str) -> str:
        if len(landmarks) < 21:
            return "Unknown"
        states = self.get_finger_states(landmarks, handedness)
        pattern = self._pattern(states)

        special = self._special(landmarks, states, pattern)
        if special:
            return special
        if pattern in self.PATTERN_TO_GESTURE:
            return self.PATTERN_TO_GESTURE[pattern]

        # Small fallbacks for slightly bent fingers.
        if states["index"] and states["middle"] and not states["ring"] and not states["pinky"]:
            return "2 / Peace"
        if states["index"] and not states["middle"] and not states["ring"] and not states["pinky"]:
            return "L Sign" if states["thumb"] else "1"
        if states["index"] and states["middle"] and states["ring"] and states["pinky"]:
            return "5 / Hello" if states["thumb"] else "4"
        return "Unknown"

    def get_finger_states(self, landmarks: List[Point], handedness: str) -> Dict[str, bool]:
        wrist = landmarks[0]
        palm = self._palm_center(landmarks)
        hand_size = max(self._distance(landmarks[0], landmarks[9]), 1.0)
        states: Dict[str, bool] = {}

        for finger in ("index", "middle", "ring", "pinky"):
            mcp = landmarks[self.MCP[finger]]
            pip = landmarks[self.PIP[finger]]
            tip = landmarks[self.TIP[finger]]
            straight = self._angle(mcp, pip, tip) > self.INDEX_STRAIGHT_MIN
            extended = self._distance(tip, wrist) > self._distance(pip, wrist) + self.WRIST_DISTANCE_MARGIN * hand_size
            states[finger] = straight and extended

        thumb_mcp = landmarks[self.MCP["thumb"]]
        thumb_pip = landmarks[self.PIP["thumb"]]
        thumb_tip = landmarks[self.TIP["thumb"]]
        straight_thumb = self._angle(thumb_mcp, thumb_pip, thumb_tip) > self.THUMB_STRAIGHT_MIN
        far_from_palm = self._distance(thumb_tip, palm) > self._distance(
            thumb_pip, palm
        ) + self.THUMB_PALM_MARGIN * hand_size
        side_open = thumb_tip[0] < thumb_pip[0] if handedness == "Right" else thumb_tip[0] > thumb_pip[0]
        states["thumb"] = straight_thumb and (far_from_palm or side_open)
        return states

    def _special(self, landmarks: List[Point], states: Dict[str, bool], pattern: str) -> str | None:
        size = max(self._distance(landmarks[0], landmarks[9]), 1.0)
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        wrist = landmarks[0]

        pinch_ratio = self._distance(thumb_tip, index_tip) / size
        if pinch_ratio < self.OK_PINCH_RATIO_MAX and states["middle"] and states["ring"] and states["pinky"]:
            return "OK"

        if pattern == "10000":
            direction = (wrist[1] - thumb_tip[1]) / size
            if direction > self.THUMB_DIRECTION_MIN:
                return "Thumbs Up"
            if direction < -self.THUMB_DIRECTION_MIN:
                return "Thumbs Down"
            return "Thumb"
        return None

    @staticmethod
    def _pattern(states: Dict[str, bool]) -> str:
        order = ("thumb", "index", "middle", "ring", "pinky")
        return "".join("1" if states[finger] else "0" for finger in order)

    @staticmethod
    def _distance(a: Point, b: Point) -> float:
        return hypot(a[0] - b[0], a[1] - b[1])

    @staticmethod
    def _angle(a: Point, b: Point, c: Point) -> float:
        ab_x, ab_y = a[0] - b[0], a[1] - b[1]
        cb_x, cb_y = c[0] - b[0], c[1] - b[1]
        ab_norm = hypot(ab_x, ab_y)
        cb_norm = hypot(cb_x, cb_y)
        if ab_norm == 0 or cb_norm == 0:
            return 0.0
        cosine = (ab_x * cb_x + ab_y * cb_y) / (ab_norm * cb_norm)
        cosine = max(-1.0, min(1.0, cosine))
        return acos(cosine) * 180.0 / pi

    @staticmethod
    def _palm_center(landmarks: List[Point]) -> Point:
        ids = (0, 5, 9, 13, 17)
        x = int(sum(landmarks[i][0] for i in ids) / len(ids))
        y = int(sum(landmarks[i][1] for i in ids) / len(ids))
        return (x, y)

    @staticmethod
    def get_gesture_descriptions() -> Dict[str, str]:
        return {
            "0 / Fist":    "All fingers are folded down.",
            "1":           "Only the index finger is up.",
            "2 / Peace":   "Index and middle fingers are up.",
            "3":           "Index, middle, and ring are up.",
            "4":           "Four fingers up, no thumb.",
            "5 / Hello":   "All five fingers are up.",
            "6 / Call Me": "Thumb and pinky are up.",
            "OK":          "Pinch thumb+index, three fingers up.",
            "Rock":        "Index and pinky are up.",
            "I Love You":  "Thumb, index, and pinky are up.",
            "L Sign":      "Thumb and index finger are up.",
            "Pinky":       "Only the pinky finger is up.",
            "W Sign":      "Middle, ring, and pinky are up.",
            "Thumbs Up":   "Thumb pointing up above the wrist.",
            "Thumbs Down": "Thumb pointing down below the wrist.",
            "Thumb":       "Only the thumb (neutral direction).",
        }

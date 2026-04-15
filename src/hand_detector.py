"""Hand detection and drawing utilities using MediaPipe."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import List, Tuple

import cv2
import mediapipe as mp

Point = Tuple[int, int]


@dataclass
class HandData:
    landmarks_px: List[Point]
    handedness: str
    bbox: Tuple[int, int, int, int]


class HandDetector:
    """Detect hands and return pixel landmarks + bounding boxes."""

    def __init__(
        self,
        max_num_hands: int = 2,
        min_detection_confidence: float = 0.7,
        min_tracking_confidence: float = 0.6,
        swap_handedness: bool = True,
    ) -> None:
        self.swap_handedness = swap_handedness
        self.mode = "solutions" if hasattr(mp, "solutions") else "tasks"

        if self.mode == "solutions":
            self._init_solutions(
                max_num_hands=max_num_hands,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )
        else:
            self._init_tasks(
                max_num_hands=max_num_hands,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )

    def _init_solutions(
        self,
        max_num_hands: int,
        min_detection_confidence: float,
        min_tracking_confidence: float,
    ) -> None:
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_styles = mp.solutions.drawing_styles
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def _init_tasks(
        self,
        max_num_hands: int,
        min_detection_confidence: float,
        min_tracking_confidence: float,
    ) -> None:
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        model_path = Path(__file__).resolve().parent.parent / "models" / "hand_landmarker.task"
        if not model_path.exists():
            raise FileNotFoundError(
                "Missing model: models/hand_landmarker.task"
            )

        self.connections = list(vision.HandLandmarksConnections.HAND_CONNECTIONS)
        options = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=max_num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.hands = vision.HandLandmarker.create_from_options(options)
        self.start_time = time.time()

    def find_hands(self, frame) -> Tuple[object, List[HandData]]:
        height, width = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        items: List[HandData] = []

        if self.mode == "solutions":
            results = self.hands.process(rgb)
            if results.multi_hand_landmarks and results.multi_handedness:
                for landmarks, handed in zip(results.multi_hand_landmarks, results.multi_handedness):
                    points = [(int(p.x * width), int(p.y * height)) for p in landmarks.landmark]
                    label = self._normalize_handedness(handed.classification[0].label)
                    hand = self._build_hand_data(points, label, width, height)
                    items.append(hand)
                    self.mp_drawing.draw_landmarks(
                        frame,
                        landmarks,
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_styles.get_default_hand_landmarks_style(),
                        self.mp_styles.get_default_hand_connections_style(),
                    )
                    self._draw_bbox(frame, hand.bbox, hand.handedness)
        else:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp = int((time.time() - self.start_time) * 1000)
            results = self.hands.detect_for_video(mp_image, timestamp)
            if results.hand_landmarks and results.handedness:
                for landmarks, handed in zip(results.hand_landmarks, results.handedness):
                    points = [(int(p.x * width), int(p.y * height)) for p in landmarks]
                    label = self._normalize_handedness(handed[0].category_name)
                    hand = self._build_hand_data(points, label, width, height)
                    items.append(hand)
                    self._draw_task_landmarks(frame, hand.landmarks_px)
                    self._draw_bbox(frame, hand.bbox, hand.handedness)

        return frame, items

    def _build_hand_data(
        self, points: List[Point], label: str, width: int, height: int
    ) -> HandData:
        x_values = [p[0] for p in points]
        y_values = [p[1] for p in points]
        pad = 20
        bbox = (
            max(min(x_values) - pad, 0),
            max(min(y_values) - pad, 0),
            min(max(x_values) + pad, width),
            min(max(y_values) + pad, height),
        )
        return HandData(landmarks_px=points, handedness=label, bbox=bbox)

    @staticmethod
    def _draw_bbox(frame, bbox: Tuple[int, int, int, int], label: str) -> None:
        x0, y0, x1, y1 = bbox
        cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 255, 0), 2)
        cv2.putText(
            frame,
            label,
            (x0, max(30, y0 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

    def _draw_task_landmarks(self, frame, points: List[Point]) -> None:
        for edge in self.connections:
            cv2.line(frame, points[edge.start], points[edge.end], (0, 200, 255), 2)
        for idx, point in enumerate(points):
            color = (255, 100, 50) if idx in (4, 8, 12, 16, 20) else (255, 255, 255)
            cv2.circle(frame, point, 4, color, -1)

    def _normalize_handedness(self, label: str) -> str:
        if not self.swap_handedness:
            return label
        if label == "Left":
            return "Right"
        if label == "Right":
            return "Left"
        return label

    def close(self) -> None:
        try:
            self.hands.close()
        except Exception:
            pass

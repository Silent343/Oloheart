"""MediaPipe hand tracking adapter with strict camera-frame privacy."""

from __future__ import annotations

import os
import threading
import time
from collections.abc import Callable
from pathlib import Path

from oloheart.domain.model import GestureKind, HandObservation


class MediaPipeHandTracker:
    """Convert live video to landmarks; pixels are never emitted, stored, or displayed."""

    def __init__(self, model_path: Path, camera_index: int = 0) -> None:
        self._model_path = model_path
        self._camera_index = camera_index
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(
        self,
        on_observation: Callable[[HandObservation], None],
        on_status: Callable[[str], None],
    ) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run,
            args=(on_observation, on_status),
            name="oloheart-spatial-input",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3.0)

    def _run(self, on_observation, on_status) -> None:
        capture = recognizer = None
        try:
            if not self._model_path.exists():
                raise FileNotFoundError(f"gesture model missing: {self._model_path}")
            os.environ.setdefault("MPLCONFIGDIR", str(self._model_path.parents[1] / ".cache" / "matplotlib"))
            import cv2
            import mediapipe as mp

            def handle_result(result, _output_image, timestamp_ms: int) -> None:
                if not self._stop.is_set():
                    on_observation(self._translate(result, timestamp_ms / 1000.0))

            options = mp.tasks.vision.HandLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(model_asset_path=str(self._model_path)),
                running_mode=mp.tasks.vision.RunningMode.LIVE_STREAM,
                num_hands=2,
                min_hand_detection_confidence=0.58,
                min_hand_presence_confidence=0.58,
                min_tracking_confidence=0.58,
                result_callback=handle_result,
            )
            recognizer = mp.tasks.vision.HandLandmarker.create_from_options(options)
            capture, active_index, backend_name = self._open_camera(cv2)
            on_status(f"CÁMARA {active_index} ACTIVA · {backend_name} · LANDMARKS ASYNC")

            last_timestamp_ms = 0
            while not self._stop.is_set():
                success, frame = capture.read()
                if not success:
                    time.sleep(0.04)
                    continue
                frame = cv2.flip(frame, 1)
                if frame.shape[1] > 480:
                    frame = cv2.resize(frame, (480, 270), interpolation=cv2.INTER_AREA)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                now = time.monotonic()
                timestamp_ms = max(last_timestamp_ms + 1, int(now * 1000))
                last_timestamp_ms = timestamp_ms
                recognizer.detect_async(
                    mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), timestamp_ms
                )
                time.sleep(0.004)
        except Exception as error:
            on_status("CÁMARA NO DISPONIBLE · " + str(error).upper())
            on_observation(HandObservation.empty(time.monotonic()))
        finally:
            if capture is not None:
                capture.release()
            if recognizer is not None:
                recognizer.close()

    def _open_camera(self, cv2):
        """Find an available Windows camera while preferring the configured device."""

        indices = [self._camera_index, *(index for index in range(4) if index != self._camera_index)]
        backends = (("DIRECTSHOW", cv2.CAP_DSHOW), ("MEDIA FOUNDATION", cv2.CAP_MSMF),
                    ("AUTO", cv2.CAP_ANY))
        for index in indices:
            for backend_name, backend in backends:
                capture = cv2.VideoCapture(index, backend)
                if not capture.isOpened():
                    capture.release()
                    continue
                capture.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
                capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 270)
                capture.set(cv2.CAP_PROP_FPS, 30)
                capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                return capture, index, backend_name
        raise RuntimeError("no se pudo abrir ninguna cámara entre los índices 0–3")

    def _translate(self, result, now: float) -> HandObservation:
        if not result.hand_landmarks:
            return HandObservation.empty(now)

        hands = result.hand_landmarks
        points_2d: list[tuple[float, float]] = []
        centers: list[tuple[float, float]] = []
        for hand in hands:
            points = [(float(point.x), float(point.y)) for point in hand]
            points_2d.extend(points)
            centers.append((sum(point[0] for point in points) / 21.0,
                            sum(point[1] for point in points) / 21.0))

        first = hands[0]
        gesture, confidence = self._classify_gesture(first)

        palm_x = sum(center[0] for center in centers) / len(centers)
        palm_y = sum(center[1] for center in centers) / len(centers)
        if len(centers) >= 2:
            dx, dy = centers[0][0] - centers[1][0], centers[0][1] - centers[1][1]
            span = min(1.8, max(0.35, math_hypot(dx, dy) * 2.5))
        else:
            xs, ys = [point[0] for point in points_2d[:21]], [point[1] for point in points_2d[:21]]
            span = min(1.4, max(0.55, ((max(xs) - min(xs)) + (max(ys) - min(ys))) * 1.5))
        if gesture == GestureKind.PINCH:
            pointer_x = (float(first[4].x) + float(first[8].x)) * 0.5
            pointer_y = (float(first[4].y) + float(first[8].y)) * 0.5
        else:
            pointer_x = float(first[8].x)
            pointer_y = float(first[8].y)
        return HandObservation(gesture, confidence, palm_x, palm_y,
                               pointer_x, pointer_y, span,
                               len(hands), tuple(points_2d), now)

    @classmethod
    def _classify_gesture(cls, hand) -> tuple[GestureKind, float]:
        """Classify semantic gestures from geometry without a second neural model."""

        palm_scale = max(cls._distance(hand, 0, 9), cls._distance(hand, 5, 17), 0.08)
        if cls._distance(hand, 4, 8) < palm_scale * 0.32:
            return GestureKind.PINCH, 0.96

        index = cls._finger_extended(hand, 8, 6)
        middle = cls._finger_extended(hand, 12, 10)
        ring = cls._finger_extended(hand, 16, 14)
        pinky = cls._finger_extended(hand, 20, 18)
        thumb_reference = cls._distance(hand, 3, 5)
        thumb_extended = cls._distance(hand, 4, 5) > thumb_reference * 1.25
        extended_count = sum((index, middle, ring, pinky))

        if extended_count == 4:
            return GestureKind.OPEN_PALM, 0.92
        if index and middle and ring and not pinky:
            return GestureKind.THREE_FINGERS, 0.94
        if pinky and not index and not middle and not ring:
            return GestureKind.PINKY, 0.93
        if index and middle and not ring and not pinky:
            return GestureKind.VICTORY, 0.92
        if index and not middle and not ring and not pinky:
            return GestureKind.POINT, 0.90
        if extended_count == 0 and thumb_extended:
            if float(hand[4].y) < float(hand[2].y) - 0.035:
                return GestureKind.THUMB_UP, 0.88
            if float(hand[4].y) > float(hand[2].y) + 0.035:
                return GestureKind.THUMB_DOWN, 0.88
        if extended_count == 0:
            return GestureKind.CLOSED_FIST, 0.86
        return GestureKind.NONE, 0.72

    @classmethod
    def _finger_extended(cls, hand, tip: int, joint: int) -> bool:
        return cls._distance(hand, tip, 0) > cls._distance(hand, joint, 0) * 1.12

    @staticmethod
    def _distance(hand, first: int, second: int) -> float:
        return sum((float(getattr(hand[first], axis)) - float(getattr(hand[second], axis))) ** 2
                   for axis in ("x", "y", "z")) ** 0.5


def math_hypot(x: float, y: float) -> float:
    """Keep the hot camera loop independent from array libraries."""

    return (x * x + y * y) ** 0.5

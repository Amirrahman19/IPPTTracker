"""Ultralytics YOLO26 Pose adapter with single-participant selection."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

from .models import Landmark, PoseFrame


class YOLO26PoseEstimator:
    """Lazily load YOLO26 Pose so geometry tests do not need PyTorch or weights."""

    def __init__(
        self,
        model_name: str = "yolo26n-pose.pt",
        confidence: float = 0.35,
        device: str | None = None,
    ) -> None:
        # Keep Ultralytics' mutable settings local to this project. This also
        # avoids writing a participant-station configuration into a user profile.
        if "YOLO_CONFIG_DIR" not in os.environ:
            config_directory = Path.cwd() / ".ultralytics"
            config_directory.mkdir(parents=True, exist_ok=True)
            os.environ["YOLO_CONFIG_DIR"] = str(config_directory)
        try:
            from ultralytics import YOLO
        except ImportError as error:  # pragma: no cover - installation error path
            raise RuntimeError("Install project requirements in the ippt environment first.") from error
        self.model: Any = YOLO(model_name)
        self.confidence = confidence
        self.device = device

    def infer(self, image: np.ndarray, frame_index: int, timestamp_s: float) -> PoseFrame | None:
        result = self.model(image, verbose=False, conf=self.confidence, device=self.device)[0]
        keypoints = result.keypoints
        if keypoints is None or keypoints.data is None or len(keypoints.data) == 0:
            return None

        raw_keypoints = keypoints.data.detach().cpu().numpy()
        boxes = result.boxes.xyxy.detach().cpu().numpy() if result.boxes is not None else None
        selected = self._select_participant(raw_keypoints, boxes)
        values = raw_keypoints[selected]
        landmarks = tuple(
            Landmark(
                x=float(value[0]),
                y=float(value[1]),
                confidence=float(value[2]) if len(value) > 2 else 1.0,
            )
            for value in values
        )
        bbox = None
        if boxes is not None:
            bbox = tuple(float(value) for value in boxes[selected])
        return PoseFrame(
            frame_index=frame_index,
            timestamp_s=timestamp_s,
            landmarks=landmarks,
            bbox=bbox,
        )

    @staticmethod
    def _select_participant(keypoints: np.ndarray, boxes: np.ndarray | None) -> int:
        """Prefer the largest well-localised pose when a frame has several people."""
        visibility = keypoints[:, :, 2].mean(axis=1) if keypoints.shape[-1] > 2 else np.ones(len(keypoints))
        if boxes is None:
            return int(np.argmax(visibility))
        areas = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
        return int(np.argmax(areas * visibility))

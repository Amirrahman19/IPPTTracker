"""Session-only participant geometry calibration from a stable starting pose."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .geometry import angle, distance, median_landmarks
from .keypoints import SIDE_KEYPOINTS, Keypoint
from .models import Calibration, Landmark, PoseFrame

REQUIRED_POINTS = ("shoulder", "elbow", "wrist", "hip", "knee", "ankle")


def _landmark(landmarks: tuple[Landmark, ...], keypoint: Keypoint) -> Landmark:
    return landmarks[int(keypoint)]


def select_camera_side(landmarks: tuple[Landmark, ...]) -> str:
    """Choose the less-occluded body side by six joint confidences."""
    scores = {
        side: float(np.mean([_landmark(landmarks, index).confidence for index in mapping.values()]))
        for side, mapping in SIDE_KEYPOINTS.items()
    }
    return max(scores, key=scores.get)


def build_calibration(samples: list[PoseFrame], min_confidence: float) -> Calibration:
    """Create a calibration vector from a median of accepted start-position frames."""
    accepted = [sample for sample in samples if sample.mean_confidence >= min_confidence]
    if not accepted:
        raise ValueError("No calibration frame met the minimum pose-confidence threshold.")

    landmarks = median_landmarks([sample.landmarks for sample in accepted])
    side = select_camera_side(landmarks)
    points = {name: _landmark(landmarks, index) for name, index in SIDE_KEYPOINTS[side].items()}
    if min(point.confidence for point in points.values()) < min_confidence:
        raise ValueError("The camera-facing arm and leg must be visible during calibration.")

    left_shoulder = _landmark(landmarks, Keypoint.LEFT_SHOULDER)
    right_shoulder = _landmark(landmarks, Keypoint.RIGHT_SHOULDER)
    return Calibration(
        side=side,
        upper_arm_length=distance(points["shoulder"], points["elbow"]),
        forearm_length=distance(points["elbow"], points["wrist"]),
        torso_length=distance(points["shoulder"], points["hip"]),
        thigh_length=distance(points["hip"], points["knee"]),
        shin_length=distance(points["knee"], points["ankle"]),
        shoulder_width=distance(left_shoulder, right_shoulder),
        baseline_elbow_angle=angle(points["shoulder"], points["elbow"], points["wrist"]),
        baseline_hip_angle=angle(points["shoulder"], points["hip"], points["knee"]),
        baseline_shoulder_y=points["shoulder"].y,
        baseline_knee_y=points["knee"].y,
        confidence=float(np.mean([point.confidence for point in points.values()])),
    )


@dataclass
class CalibrationCollector:
    """Collect a fixed number of stable frames before test scoring starts."""

    required_frames: int
    min_confidence: float
    _samples: list[PoseFrame] = field(default_factory=list)

    @property
    def progress(self) -> tuple[int, int]:
        return min(len(self._samples), self.required_frames), self.required_frames

    @property
    def complete(self) -> bool:
        return len(self._samples) >= self.required_frames

    def add(self, pose: PoseFrame | None) -> Calibration | None:
        if pose is None or pose.mean_confidence < self.min_confidence or self.complete:
            return None
        self._samples.append(pose)
        if self.complete:
            return build_calibration(self._samples, self.min_confidence)
        return None

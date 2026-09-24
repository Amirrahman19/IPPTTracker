"""Convert camera-facing landmarks into interpretable, calibrated measurements."""

from __future__ import annotations

from .geometry import angle, straightness_deviation
from .keypoints import SIDE_KEYPOINTS
from .models import Calibration, FrameMetrics, Landmark, PoseFrame


def _points(pose: PoseFrame, side: str) -> dict[str, Landmark]:
    mapping = SIDE_KEYPOINTS[side]
    return {name: pose.landmarks[int(keypoint)] for name, keypoint in mapping.items()}


def calculate_metrics(pose: PoseFrame, calibration: Calibration) -> FrameMetrics:
    points = _points(pose, calibration.side)
    elbow_angle = angle(points["shoulder"], points["elbow"], points["wrist"])
    hip_angle = angle(points["shoulder"], points["hip"], points["knee"])
    trunk_deviation = straightness_deviation(points["shoulder"], points["hip"], points["ankle"])
    # Image y grows downwards. Normalisation makes camera distance less important.
    shoulder_drop = max(0.0, points["shoulder"].y - calibration.baseline_shoulder_y)
    normalized_shoulder_drop = shoulder_drop / max(calibration.upper_arm_length, 1e-6)
    relative_hip_flexion = max(
        0.0,
        (calibration.baseline_hip_angle - hip_angle) / max(calibration.baseline_hip_angle, 1e-6),
    )
    confidence = min(point.confidence for point in points.values())
    return FrameMetrics(
        elbow_angle=elbow_angle,
        hip_angle=hip_angle,
        trunk_deviation=trunk_deviation,
        normalized_shoulder_drop=normalized_shoulder_drop,
        relative_hip_flexion=relative_hip_flexion,
        pose_confidence=confidence,
    )

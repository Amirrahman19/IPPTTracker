from __future__ import annotations

import pytest

from ippt_tracker.calibration import build_calibration
from ippt_tracker.keypoints import Keypoint
from ippt_tracker.models import Landmark, PoseFrame


def pose(frame_index: int) -> PoseFrame:
    landmarks = [Landmark(0, 0, 0.4) for _ in range(17)]
    # Camera-facing right side: a fully extended horizontal arm and straight body.
    landmarks[Keypoint.RIGHT_SHOULDER] = Landmark(0, 0, 0.95)
    landmarks[Keypoint.RIGHT_ELBOW] = Landmark(1, 0, 0.95)
    landmarks[Keypoint.RIGHT_WRIST] = Landmark(2, 0, 0.95)
    landmarks[Keypoint.RIGHT_HIP] = Landmark(1, 1, 0.95)
    landmarks[Keypoint.RIGHT_KNEE] = Landmark(2, 1, 0.95)
    landmarks[Keypoint.RIGHT_ANKLE] = Landmark(3, 1, 0.95)
    landmarks[Keypoint.LEFT_SHOULDER] = Landmark(0, 0.5, 0.4)
    landmarks[Keypoint.LEFT_ELBOW] = Landmark(1, 0.5, 0.4)
    landmarks[Keypoint.LEFT_WRIST] = Landmark(2, 0.5, 0.4)
    landmarks[Keypoint.LEFT_HIP] = Landmark(1, 1.5, 0.4)
    landmarks[Keypoint.LEFT_KNEE] = Landmark(2, 1.5, 0.4)
    landmarks[Keypoint.LEFT_ANKLE] = Landmark(3, 1.5, 0.4)
    return PoseFrame(frame_index, frame_index / 30, tuple(landmarks))


def test_calibration_uses_highest_confidence_side_and_median_geometry() -> None:
    calibration = build_calibration([pose(0), pose(1), pose(2)], min_confidence=0.35)
    assert calibration.side == "right"
    assert calibration.upper_arm_length == pytest.approx(1.0)
    assert calibration.forearm_length == pytest.approx(1.0)
    assert calibration.baseline_elbow_angle == pytest.approx(180.0)

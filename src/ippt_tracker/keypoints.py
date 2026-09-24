"""COCO keypoint definitions used by the official YOLO26 Pose models."""

from __future__ import annotations

from enum import IntEnum


class Keypoint(IntEnum):
    NOSE = 0
    LEFT_EYE = 1
    RIGHT_EYE = 2
    LEFT_EAR = 3
    RIGHT_EAR = 4
    LEFT_SHOULDER = 5
    RIGHT_SHOULDER = 6
    LEFT_ELBOW = 7
    RIGHT_ELBOW = 8
    LEFT_WRIST = 9
    RIGHT_WRIST = 10
    LEFT_HIP = 11
    RIGHT_HIP = 12
    LEFT_KNEE = 13
    RIGHT_KNEE = 14
    LEFT_ANKLE = 15
    RIGHT_ANKLE = 16


SIDE_KEYPOINTS: dict[str, dict[str, Keypoint]] = {
    "left": {
        "shoulder": Keypoint.LEFT_SHOULDER,
        "elbow": Keypoint.LEFT_ELBOW,
        "wrist": Keypoint.LEFT_WRIST,
        "hip": Keypoint.LEFT_HIP,
        "knee": Keypoint.LEFT_KNEE,
        "ankle": Keypoint.LEFT_ANKLE,
    },
    "right": {
        "shoulder": Keypoint.RIGHT_SHOULDER,
        "elbow": Keypoint.RIGHT_ELBOW,
        "wrist": Keypoint.RIGHT_WRIST,
        "hip": Keypoint.RIGHT_HIP,
        "knee": Keypoint.RIGHT_KNEE,
        "ankle": Keypoint.RIGHT_ANKLE,
    },
}

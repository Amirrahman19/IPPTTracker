"""OpenCV-only visual feedback for a station display or recorded test video."""

from __future__ import annotations

import cv2
import numpy as np

from .keypoints import SIDE_KEYPOINTS
from .models import PoseFrame
from .session import ExerciseSession


def _draw_text(image: np.ndarray, text: str, row: int, colour: tuple[int, int, int]) -> None:
    cv2.putText(
        image,
        text,
        (18, 34 + row * 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        colour,
        2,
        cv2.LINE_AA,
    )


def draw_overlay(image: np.ndarray, session: ExerciseSession, pose: PoseFrame | None) -> np.ndarray:
    """Draw only the session result and skeleton; frames are not stored by default."""
    output = image.copy()
    _draw_text(output, f"{session.exercise.value.upper()}  |  {session.phase}", 0, (255, 255, 255))
    _draw_text(output, f"VALID REPS: {session.state.valid_reps}", 1, (0, 220, 0))

    if pose is None:
        _draw_text(output, "Participant pose not detected", 2, (0, 0, 255))
        return output

    if pose.bbox:
        x1, y1, x2, y2 = (int(value) for value in pose.bbox)
        cv2.rectangle(output, (x1, y1), (x2, y2), (255, 180, 0), 2)

    if session.calibrated and session.state.calibration:
        mapping = SIDE_KEYPOINTS[session.state.calibration.side]
        ordered = ("shoulder", "elbow", "wrist", "hip", "knee", "ankle")
        points = {
            name: tuple(int(value) for value in (pose.landmarks[int(mapping[name])].x, pose.landmarks[int(mapping[name])].y))
            for name in ordered
        }
        for start, end in (("shoulder", "elbow"), ("elbow", "wrist"), ("shoulder", "hip"), ("hip", "knee"), ("knee", "ankle")):
            cv2.line(output, points[start], points[end], (0, 220, 255), 3, cv2.LINE_AA)
        for value in points.values():
            cv2.circle(output, value, 5, (0, 80, 255), -1, cv2.LINE_AA)

    if session.last_metrics:
        metrics = session.last_metrics
        _draw_text(output, f"Elbow: {metrics.elbow_angle:.1f} deg", 2, (255, 255, 255))
        _draw_text(output, f"Hip flexion: {metrics.relative_hip_flexion:.2f}", 3, (255, 255, 255))
        _draw_text(output, f"Pose confidence: {metrics.pose_confidence:.2f}", 4, (255, 255, 255))

    if session.state.last_event:
        event = session.state.last_event
        colour = (0, 220, 0) if event.decision.value == "valid" else (0, 0, 255)
        reason = "COUNTED" if not event.reasons else ", ".join(item.value for item in event.reasons)
        _draw_text(output, f"Rep {event.number}: {event.decision.value.upper()} - {reason}", 5, colour)
    return output

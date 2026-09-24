"""Numerically safe 2D pose geometry for a fixed, side-view camera."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from .models import Landmark


def point(landmark: Landmark) -> np.ndarray:
    return np.array((landmark.x, landmark.y), dtype=float)


def distance(first: Landmark, second: Landmark) -> float:
    return float(np.linalg.norm(point(first) - point(second)))


def angle(first: Landmark, vertex: Landmark, third: Landmark) -> float:
    """Return the internal angle (0–180°) at *vertex*."""
    first_vector = point(first) - point(vertex)
    second_vector = point(third) - point(vertex)
    denominator = np.linalg.norm(first_vector) * np.linalg.norm(second_vector)
    if denominator <= 1e-8:
        return 0.0
    cosine = float(np.dot(first_vector, second_vector) / denominator)
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def straightness_deviation(first: Landmark, vertex: Landmark, third: Landmark) -> float:
    """Degrees from a straight body line at *vertex*; lower is better."""
    return abs(180.0 - angle(first, vertex, third))


def median_landmarks(samples: Iterable[tuple[Landmark, ...]]) -> tuple[Landmark, ...]:
    """Aggregate stable calibration frames without retaining their images."""
    frames = list(samples)
    if not frames:
        raise ValueError("At least one pose frame is required for calibration.")
    keypoint_count = len(frames[0])
    if any(len(frame) != keypoint_count for frame in frames):
        raise ValueError("Calibration poses have incompatible keypoint counts.")

    result: list[Landmark] = []
    for index in range(keypoint_count):
        values = np.array(
            [[frame[index].x, frame[index].y, frame[index].confidence] for frame in frames],
            dtype=float,
        )
        result.append(Landmark(*np.median(values, axis=0)))
    return tuple(result)

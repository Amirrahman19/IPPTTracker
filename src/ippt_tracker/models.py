"""Typed values that cross the pose, calibration, rule, and UI layers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class Exercise(str, Enum):
    PUSHUP = "pushup"
    SITUP = "situp"


class Decision(str, Enum):
    VALID = "valid"
    INVALID = "invalid"


class ReasonCode(str, Enum):
    INSUFFICIENT_DEPTH = "insufficient_depth"
    INCOMPLETE_EXTENSION = "incomplete_extension"
    BODY_ALIGNMENT = "body_alignment"
    INSUFFICIENT_FLEXION = "insufficient_flexion"
    LOW_CONFIDENCE = "low_confidence"


@dataclass(frozen=True)
class Landmark:
    x: float
    y: float
    confidence: float


@dataclass(frozen=True)
class PoseFrame:
    """One selected participant pose from a frame of video."""

    frame_index: int
    timestamp_s: float
    landmarks: tuple[Landmark, ...]
    bbox: tuple[float, float, float, float] | None = None

    @property
    def mean_confidence(self) -> float:
        return float(np.mean([landmark.confidence for landmark in self.landmarks]))


@dataclass(frozen=True)
class Calibration:
    """Temporary geometry from a single session; never persists a participant profile."""

    side: str
    upper_arm_length: float
    forearm_length: float
    torso_length: float
    thigh_length: float
    shin_length: float
    shoulder_width: float
    baseline_elbow_angle: float
    baseline_hip_angle: float
    baseline_shoulder_y: float
    baseline_knee_y: float
    confidence: float

    @property
    def arm_length(self) -> float:
        return self.upper_arm_length + self.forearm_length


@dataclass(frozen=True)
class FrameMetrics:
    elbow_angle: float
    hip_angle: float
    trunk_deviation: float
    normalized_shoulder_drop: float
    relative_hip_flexion: float
    pose_confidence: float


@dataclass(frozen=True)
class RepEvent:
    number: int
    exercise: Exercise
    decision: Decision
    reasons: tuple[ReasonCode, ...]
    start_frame: int
    end_frame: int
    measurements: Mapping[str, float]


@dataclass
class SessionState:
    """Public session state rendered by the overlay and saved with events."""

    exercise: Exercise
    calibration: Calibration | None = None
    valid_reps: int = 0
    attempted_reps: int = 0
    last_event: RepEvent | None = None
    events: list[RepEvent] = field(default_factory=list)

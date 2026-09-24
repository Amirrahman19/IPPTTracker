"""Temporal, explainable repetition assessment for the two V1 exercises."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from .models import Decision, Exercise, FrameMetrics, ReasonCode, RepEvent


@dataclass
class _Attempt:
    start_frame: int
    peak_elbow_flexion: float = 180.0
    max_shoulder_drop: float = 0.0
    max_trunk_deviation: float = 0.0
    peak_hip_flexion: float = 0.0
    low_confidence: bool = False
    reached_target: bool = False

    def observe(self, metrics: FrameMetrics, min_confidence: float) -> None:
        self.peak_elbow_flexion = min(self.peak_elbow_flexion, metrics.elbow_angle)
        self.max_shoulder_drop = max(self.max_shoulder_drop, metrics.normalized_shoulder_drop)
        self.max_trunk_deviation = max(self.max_trunk_deviation, metrics.trunk_deviation)
        self.peak_hip_flexion = max(self.peak_hip_flexion, metrics.relative_hip_flexion)
        self.low_confidence |= metrics.pose_confidence < min_confidence


class ExerciseAssessor(ABC):
    """Counts only completed motion cycles, returning a decision for each one."""

    def __init__(self, exercise: Exercise, min_confidence: float) -> None:
        self.exercise = exercise
        self.min_confidence = min_confidence
        self._attempt: _Attempt | None = None
        self._attempted_reps = 0
        self._valid_reps = 0

    @property
    def valid_reps(self) -> int:
        return self._valid_reps

    @property
    def attempted_reps(self) -> int:
        return self._attempted_reps

    @property
    def phase(self) -> str:
        return "ACTIVE" if self._attempt else "READY"

    def update(self, frame_index: int, metrics: FrameMetrics) -> RepEvent | None:
        if self._attempt is None:
            if self._starts_attempt(metrics):
                self._attempt = _Attempt(start_frame=frame_index)
                self._attempt.observe(metrics, self.min_confidence)
            return None

        self._attempt.observe(metrics, self.min_confidence)
        self._update_target(self._attempt, metrics)
        if not self._ends_attempt(metrics):
            return None

        attempt = self._attempt
        self._attempt = None
        return self._close_attempt(frame_index, attempt)

    def _close_attempt(self, end_frame: int, attempt: _Attempt) -> RepEvent:
        reasons = self._reasons(attempt)
        decision = Decision.VALID if not reasons else Decision.INVALID
        self._attempted_reps += 1
        if decision is Decision.VALID:
            self._valid_reps += 1
        return RepEvent(
            number=self._attempted_reps,
            exercise=self.exercise,
            decision=decision,
            reasons=tuple(reasons),
            start_frame=attempt.start_frame,
            end_frame=end_frame,
            measurements=self._measurements(attempt),
        )

    @abstractmethod
    def _starts_attempt(self, metrics: FrameMetrics) -> bool: ...

    @abstractmethod
    def _update_target(self, attempt: _Attempt, metrics: FrameMetrics) -> None: ...

    @abstractmethod
    def _ends_attempt(self, metrics: FrameMetrics) -> bool: ...

    @abstractmethod
    def _reasons(self, attempt: _Attempt) -> list[ReasonCode]: ...

    @abstractmethod
    def _measurements(self, attempt: _Attempt) -> dict[str, float]: ...


class PushupAssessor(ExerciseAssessor):
    """UP → descending → bottom → UP, with depth and trunk-form evidence."""

    def __init__(self, config: dict[str, float], min_confidence: float, baseline_extension: float) -> None:
        super().__init__(Exercise.PUSHUP, min_confidence)
        self.min_extension = max(float(config["min_extension_angle_degrees"]), baseline_extension - 12.0)
        self.bottom_angle = float(config["bottom_elbow_angle_degrees"])
        self.attempt_angle = float(config["attempt_elbow_angle_degrees"])
        self.min_drop = float(config["min_normalized_shoulder_drop"])
        self.max_trunk_deviation = float(config["max_trunk_deviation_degrees"])

    def _starts_attempt(self, metrics: FrameMetrics) -> bool:
        return metrics.elbow_angle <= self.attempt_angle or metrics.normalized_shoulder_drop >= self.min_drop * 0.4

    def _update_target(self, attempt: _Attempt, metrics: FrameMetrics) -> None:
        attempt.reached_target |= metrics.elbow_angle <= self.bottom_angle and metrics.normalized_shoulder_drop >= self.min_drop

    def _ends_attempt(self, metrics: FrameMetrics) -> bool:
        return metrics.elbow_angle >= self.min_extension

    def _reasons(self, attempt: _Attempt) -> list[ReasonCode]:
        reasons: list[ReasonCode] = []
        if not attempt.reached_target:
            reasons.append(ReasonCode.INSUFFICIENT_DEPTH)
        if attempt.max_trunk_deviation > self.max_trunk_deviation:
            reasons.append(ReasonCode.BODY_ALIGNMENT)
        if attempt.low_confidence:
            reasons.append(ReasonCode.LOW_CONFIDENCE)
        return reasons

    def _measurements(self, attempt: _Attempt) -> dict[str, float]:
        return {
            "minimum_elbow_angle_degrees": attempt.peak_elbow_flexion,
            "maximum_normalized_shoulder_drop": attempt.max_shoulder_drop,
            "maximum_trunk_deviation_degrees": attempt.max_trunk_deviation,
            "required_bottom_elbow_angle_degrees": self.bottom_angle,
            "required_normalized_shoulder_drop": self.min_drop,
        }


class SitupAssessor(ExerciseAssessor):
    """DOWN → rising → UP → lowering → DOWN using calibrated hip flexion."""

    def __init__(self, config: dict[str, float], min_confidence: float) -> None:
        super().__init__(Exercise.SITUP, min_confidence)
        self.min_flexion = float(config["min_relative_hip_flexion"])
        self.attempt_flexion = float(config["attempt_relative_hip_flexion"])
        self.down_flexion = float(config["down_relative_hip_flexion"])

    def _starts_attempt(self, metrics: FrameMetrics) -> bool:
        return metrics.relative_hip_flexion >= self.attempt_flexion

    def _update_target(self, attempt: _Attempt, metrics: FrameMetrics) -> None:
        attempt.reached_target |= metrics.relative_hip_flexion >= self.min_flexion

    def _ends_attempt(self, metrics: FrameMetrics) -> bool:
        return metrics.relative_hip_flexion <= self.down_flexion

    def _reasons(self, attempt: _Attempt) -> list[ReasonCode]:
        reasons: list[ReasonCode] = []
        if not attempt.reached_target:
            reasons.append(ReasonCode.INSUFFICIENT_FLEXION)
        if attempt.low_confidence:
            reasons.append(ReasonCode.LOW_CONFIDENCE)
        return reasons

    def _measurements(self, attempt: _Attempt) -> dict[str, float]:
        return {
            "maximum_relative_hip_flexion": attempt.peak_hip_flexion,
            "required_relative_hip_flexion": self.min_flexion,
        }

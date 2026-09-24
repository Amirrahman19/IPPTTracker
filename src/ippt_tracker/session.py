"""Coordinates calibration, smoothing, assessment, and an audit-ready event log."""

from __future__ import annotations

from collections import deque
from dataclasses import asdict
from pathlib import Path

import yaml

from .assessors import ExerciseAssessor, PushupAssessor, SitupAssessor
from .calibration import CalibrationCollector
from .metrics import calculate_metrics
from .models import Exercise, FrameMetrics, PoseFrame, SessionState


class MetricsSmoother:
    def __init__(self, window: int) -> None:
        self._values: deque[FrameMetrics] = deque(maxlen=max(1, window))

    def add(self, metrics: FrameMetrics) -> FrameMetrics:
        self._values.append(metrics)
        count = len(self._values)
        return FrameMetrics(
            **{field: sum(getattr(value, field) for value in self._values) / count for field in FrameMetrics.__dataclass_fields__}
        )


class ExerciseSession:
    """A one-exercise, in-memory session. Calibration is discarded when it ends."""

    def __init__(self, exercise: Exercise, config: dict) -> None:
        self.exercise = exercise
        self.config = config
        session_config = config["session"]
        self.collector = CalibrationCollector(
            required_frames=int(session_config["calibration_frames"]),
            min_confidence=float(session_config["min_pose_confidence"]),
        )
        self.smoother = MetricsSmoother(int(session_config["smoothing_window"]))
        self.state = SessionState(exercise=exercise)
        self._assessor: ExerciseAssessor | None = None
        self.last_metrics: FrameMetrics | None = None

    @property
    def calibrated(self) -> bool:
        return self.state.calibration is not None

    @property
    def phase(self) -> str:
        if not self.calibrated:
            completed, required = self.collector.progress
            return f"CALIBRATING {completed}/{required}"
        return self._assessor.phase if self._assessor else "READY"

    def update(self, pose: PoseFrame | None) -> SessionState:
        if pose is None:
            return self.state
        if not self.calibrated:
            calibration = self.collector.add(pose)
            if calibration is not None:
                self.state.calibration = calibration
                self._assessor = self._create_assessor()
            return self.state

        assert self.state.calibration is not None and self._assessor is not None
        self.last_metrics = self.smoother.add(calculate_metrics(pose, self.state.calibration))
        event = self._assessor.update(pose.frame_index, self.last_metrics)
        if event is not None:
            self.state.last_event = event
            self.state.events.append(event)
            self.state.valid_reps = self._assessor.valid_reps
            self.state.attempted_reps = self._assessor.attempted_reps
        return self.state

    def write_events(self, output_path: Path) -> None:
        """Save decisions and geometry, not source imagery or persistent biometrics."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "exercise": self.exercise.value,
            "valid_reps": self.state.valid_reps,
            "attempted_reps": self.state.attempted_reps,
            "calibration": asdict(self.state.calibration) if self.state.calibration else None,
            "events": [
                {
                    **asdict(event),
                    "exercise": event.exercise.value,
                    "decision": event.decision.value,
                    "reasons": [reason.value for reason in event.reasons],
                }
                for event in self.state.events
            ],
        }
        output_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    def _create_assessor(self) -> ExerciseAssessor:
        assert self.state.calibration is not None
        minimum_confidence = float(self.config["session"]["min_pose_confidence"])
        if self.exercise is Exercise.PUSHUP:
            return PushupAssessor(
                self.config["pushup"],
                minimum_confidence,
                self.state.calibration.baseline_elbow_angle,
            )
        return SitupAssessor(self.config["situp"], minimum_confidence)


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)

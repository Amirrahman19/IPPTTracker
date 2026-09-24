from __future__ import annotations

from ippt_tracker.assessors import PushupAssessor, SitupAssessor
from ippt_tracker.models import Decision, FrameMetrics, ReasonCode


def metrics(
    *,
    elbow: float = 170.0,
    drop: float = 0.0,
    trunk: float = 0.0,
    flexion: float = 0.0,
    confidence: float = 0.95,
) -> FrameMetrics:
    return FrameMetrics(
        elbow_angle=elbow,
        hip_angle=90.0,
        trunk_deviation=trunk,
        normalized_shoulder_drop=drop,
        relative_hip_flexion=flexion,
        pose_confidence=confidence,
    )


PUSHUP_RULES = {
    "min_extension_angle_degrees": 155.0,
    "bottom_elbow_angle_degrees": 105.0,
    "attempt_elbow_angle_degrees": 140.0,
    "min_normalized_shoulder_drop": 0.30,
    "max_trunk_deviation_degrees": 22.0,
}


def test_pushup_counts_only_a_complete_valid_cycle() -> None:
    assessor = PushupAssessor(PUSHUP_RULES, min_confidence=0.35, baseline_extension=170.0)
    assert assessor.update(0, metrics()) is None
    assert assessor.update(1, metrics(elbow=130, drop=0.15)) is None
    assert assessor.update(2, metrics(elbow=90, drop=0.40)) is None
    event = assessor.update(3, metrics(elbow=170, drop=0.0))
    assert event is not None
    assert event.decision is Decision.VALID
    assert assessor.valid_reps == 1


def test_pushup_rejects_a_partial_motion_with_a_reason() -> None:
    assessor = PushupAssessor(PUSHUP_RULES, min_confidence=0.35, baseline_extension=170.0)
    assessor.update(1, metrics(elbow=130, drop=0.15))
    event = assessor.update(2, metrics(elbow=170))
    assert event is not None
    assert event.decision is Decision.INVALID
    assert event.reasons == (ReasonCode.INSUFFICIENT_DEPTH,)


def test_situp_counts_after_up_and_return_to_down() -> None:
    assessor = SitupAssessor(
        {
            "min_relative_hip_flexion": 0.35,
            "attempt_relative_hip_flexion": 0.15,
            "down_relative_hip_flexion": 0.10,
        },
        min_confidence=0.35,
    )
    assessor.update(0, metrics(flexion=0.20))
    assert assessor.update(1, metrics(flexion=0.45)) is None
    event = assessor.update(2, metrics(flexion=0.05))
    assert event is not None
    assert event.decision is Decision.VALID

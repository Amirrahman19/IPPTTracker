from __future__ import annotations

import pytest

from ippt_tracker.geometry import angle, distance, straightness_deviation
from ippt_tracker.models import Landmark


def landmark(x: float, y: float) -> Landmark:
    return Landmark(x, y, 1.0)


def test_distance_and_right_angle() -> None:
    assert distance(landmark(0, 0), landmark(3, 4)) == pytest.approx(5.0)
    assert angle(landmark(1, 0), landmark(0, 0), landmark(0, 1)) == pytest.approx(90.0)


def test_straightness_is_zero_for_a_straight_body() -> None:
    assert straightness_deviation(landmark(0, 0), landmark(1, 0), landmark(2, 0)) == pytest.approx(0.0)

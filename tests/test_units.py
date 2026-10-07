import numpy as np
import pandas as pd
import pytest

from ippt.geometry import Leveler, angle_deg, fit_line_angle, signed_line_distance
from ippt.reps import detect_attempts
from ippt.scoring import ScoreTable
from ippt.signal import interpolate_gaps, one_euro


def test_angle_deg():
    a, b, c = np.array([1.0, 0]), np.array([0.0, 0]), np.array([0.0, 1])
    assert angle_deg(a, b, c) == pytest.approx(90.0)
    assert angle_deg(a, b, -a) == pytest.approx(180.0)


def test_signed_line_distance_sign():
    # Line heading +x: a point above is positive.
    assert signed_line_distance(np.array([1.0, 2]), np.array([0.0, 0]), np.array([4.0, 0])) == pytest.approx(2.0)


def test_leveler_removes_roll_and_keeps_scale():
    theta = np.radians(12)
    ground = np.array([[0.0, 0], [100, 0]]) @ np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
    img = np.stack([ground[:, 0], -ground[:, 1]], -1)  # to image coords (y down)
    lev = Leveler(tuple(np.array([img[:, 0].mean(), -img[:, 1].mean()])), fit_line_angle(np.stack([img[:, 0], -img[:, 1]], -1)))
    out = lev.apply(img)
    assert abs(out[0, 1] - out[1, 1]) < 1e-6
    assert abs(abs(out[1, 0] - out[0, 0]) - 100) < 1e-6


def test_interpolate_gaps_respects_max_gap():
    x = np.array([0.0, np.nan, np.nan, 3.0, np.nan, np.nan, np.nan, np.nan, 8.0])
    out = interpolate_gaps(x, max_gap=2)
    assert out[1] == pytest.approx(1.0) and out[2] == pytest.approx(2.0)
    assert np.isnan(out[4:8]).all()


def test_one_euro_smooths_noise_and_keeps_nan():
    rng = np.random.default_rng(0)
    t = np.arange(200) / 30
    x = np.sin(t) + rng.normal(0, 0.05, 200)
    x[50] = np.nan
    y = one_euro(x, t)
    assert np.isnan(y[50])
    assert np.nanstd(np.diff(y)) < np.nanstd(np.diff(x))


def test_detect_attempts_valleys():
    elbow = np.array([172, 170, 173, 140, 100, 90, 120, 168, 172, 160, 171, 172, 120, 100, 90, 110, 130], float)
    d = 175.0 - elbow    # push-up depth coordinate, home at the top
    att = detect_attempts(d, enter_d=25, exit_d=6, min_prominence=25)
    # The dip to 160 is jitter; the last descent has no return before the clip ends.
    assert [(a, v) for a, v, _ in att] == [(2, 5), (11, 14)]
    assert att[0][2] == 8 and att[1][2] is None


def test_detect_attempts_missed_lockout_stays_separate():
    elbow = np.array([175, 120, 85, 120, 150, 120, 85, 120, 175, 175], float)   # first rep never locks out
    att = detect_attempts(175.0 - elbow, enter_d=25, exit_d=6, min_prominence=25)
    assert len(att) == 2 and att[0][2] == 4    # first attempt ends at its best point (150), not merged


def test_score_table(tmp_path):
    rows = [("pushup", "m", 20, 24, r, p) for r, p in [(10, 5), (20, 12), (30, 20)]]
    df = pd.DataFrame(rows, columns=["exercise", "sex", "age_min", "age_max", "reps", "points"])
    t = ScoreTable(df)
    assert t.points("pushup", "M", 22, 5) == 0
    assert t.points("pushup", "male", 22, 25) == 12
    assert t.points("pushup", "m", 22, 99) == 20
    with pytest.raises(LookupError):
        t.points("pushup", "f", 22, 10)

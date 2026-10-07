"""End-to-end checks on simulated sessions with known ground truth."""

import pytest

from ippt.calibration import CalibrationError
from ippt.config import load_pipeline_config, load_rules
from ippt.pipeline import analyze_sequence, calibrate_sequence
from ippt.sim import make_session

CFG = load_pipeline_config()


def run(exercise, test_reps, **kw):
    seq, segs, t0 = make_session(exercise, test_reps, **kw)
    prof = calibrate_sequence(exercise, seq, segs, CFG, load_rules(exercise))
    return analyze_sequence(exercise, seq, prof, CFG, load_rules(exercise), t_start=t0), prof


@pytest.mark.parametrize("exercise", ["pushup", "situp"])
def test_clean_reps_all_count(exercise):
    res, _ = run(exercise, [{}] * 6)
    assert res.summary["valid"] == 6 and res.summary["no_count"] == 0


@pytest.mark.parametrize("exercise", ["pushup", "situp"])
@pytest.mark.parametrize("kw", [dict(roll_deg=8.0), dict(mirror=True), dict(noise_px=2.5, dropout=0.03, seed=3)])
def test_robust_to_camera_roll_mirroring_noise(exercise, kw):
    res, _ = run(exercise, [{}] * 5, **kw)
    assert res.summary["valid"] == 5 and res.summary["no_count"] == 0


@pytest.mark.parametrize("exercise,fault,reason", [
    ("pushup", {"bottom": 125.0}, "depth"),
    ("pushup", {"top": 150.0}, "lockout"),
    ("pushup", {"sag": -0.15}, "body_sag"),
    ("pushup", {"sag": 0.15}, "body_pike"),
    ("pushup", {"knee_drop": 0.3}, "knee_contact"),
    ("situp", {"top": 50.0}, "top"),
    ("situp", {"low": 30.0}, "return"),
    ("situp", {"foot_lift": 0.2}, "feet_anchor"),
    ("situp", {"bridge": 0.15}, "hip_bridge"),
])
def test_fault_gives_specific_no_count(exercise, fault, reason):
    res, _ = run(exercise, [{}, fault, {}])
    assert res.summary["valid"] == 2 and res.summary["no_count"] == 1
    bad = next(r for r in res.reps if not r.valid)
    assert reason in bad.reasons


def test_reps_after_the_minute_do_not_count():
    # 1.6 s rep + 0.8 s rest = 2.4 s per cycle, so 30 reps overruns 60 s.
    res, _ = run("pushup", [{}] * 30)
    assert 24 <= res.summary["valid"] <= 26
    assert res.summary["outside_window"] > 0


def test_calibration_rejects_frontal_view():
    seq, segs, _ = make_session("pushup")
    # Move the far-side shoulder well away in x: the shoulders no longer overlap, so this is not a side view.
    from ippt.schema import JOINT_INDEX
    seq.xy[:, JOINT_INDEX["l_shoulder"], 0] += 250
    with pytest.raises(CalibrationError) as e:
        calibrate_sequence("pushup", seq, segs, CFG, load_rules("pushup"))
    assert any("side view" in r for r in e.value.reasons)


def test_calibration_rejects_low_confidence():
    seq, segs, _ = make_session("situp")
    seq.conf[:] = 0.2
    with pytest.raises(CalibrationError):
        calibrate_sequence("situp", seq, segs, CFG, load_rules("situp"))


def test_profile_roundtrip(tmp_path):
    _, prof = run("pushup", [{}] * 2)
    prof.save(tmp_path / "p.yaml")
    from ippt.calibration import Profile
    again = Profile.load(tmp_path / "p.yaml")
    assert again.side == prof.side and again.rom == prof.rom
    assert again.leveler.flip == prof.leveler.flip


def test_calibration_cannot_loosen_hard_limits():
    from ippt.calibration import clamp_tighter
    assert clamp_tighter(30.0, 15.0, lower_bound=False) == 15.0   # weak calibrated lying angle capped
    assert clamp_tighter(10.0, 15.0, lower_bound=False) == 10.0   # tighter calibrated value kept
    assert clamp_tighter(150.0, 160.0, lower_bound=True) == 160.0

"""Command line entry point: `ippt <command>` (or `python -m ippt.cli`)."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from .calibration import CalibrationError, Profile
from .config import load_pipeline_config, load_rules, load_yaml
from .schema import PoseSequence


def _load_pose(path: str, cfg: dict) -> PoseSequence:
    p = Path(path)
    if p.suffix == ".npz":
        return PoseSequence.load(p)
    from .pose import MediaPipePoseEstimator
    return MediaPipePoseEstimator.from_config(cfg).estimate_video(p)


def cmd_model(args, cfg):
    from .pose import ensure_model
    print(ensure_model(args.variant))


def cmd_pose(args, cfg):
    seq = _load_pose(args.video, cfg)
    out = Path(args.out or Path(args.video).with_suffix(".npz"))
    seq.save(out)
    print(f"{len(seq)} frames @ {seq.fps:.1f} fps -> {out}")


def cmd_calibrate(args, cfg):
    from .pipeline import calibrate_sequence
    seq = _load_pose(args.pose, cfg)
    segs = {k: tuple(v) for k, v in load_yaml(args.segments).items()}
    try:
        prof = calibrate_sequence(args.exercise, seq, segs, cfg, load_rules(args.exercise))
    except CalibrationError as e:
        print("calibration REJECTED:")
        for r in e.reasons:
            print(f"  - {r}")
        return 2
    prof.save(args.out)
    print(f"profile saved to {args.out} (side={prof.side}, ROM {prof.rom['lo']:.0f}-{prof.rom['hi']:.0f} [{prof.rom['source']}])")
    for w in prof.warnings:
        print(f"  warning: {w}")


def cmd_analyze(args, cfg):
    from .pipeline import analyze_sequence
    seq = _load_pose(args.pose, cfg)
    prof = Profile.load(args.profile)
    res = analyze_sequence(args.exercise, seq, prof, cfg, load_rules(args.exercise), t_start=args.t_start)
    log = res.rep_log()
    print(log.to_string(index=False) if len(log) else "no attempts detected")
    print(res.summary)
    if args.log:
        log.to_csv(args.log, index=False)
    if args.annotate:
        if Path(args.pose).suffix == ".npz":
            print("--annotate needs the video; pass the video path as the pose argument", file=sys.stderr)
        else:
            from .viz import annotate_video
            annotate_video(args.pose, res, args.annotate)


def cmd_data(args, cfg):
    from . import datasets
    if args.action == "status":
        for r in datasets.status():
            print(f"{r['name']:34s} {r['method']:7s} {'present' if r['present'] else 'missing':8s} {r['size_gb']:.2f} GB")
        print(f"budget remaining: {datasets.budget_remaining_gb():.2f} GB")
    else:
        print(datasets.fetch(args.name))


def cmd_dashboard(args, cfg):
    from .dashboard import serve
    serve(args.host, args.port, open_browser=not args.no_browser)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="ippt")
    ap.add_argument("--config", help="extra pipeline YAML merged over configs/pipeline.yaml")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("model", help="download the MediaPipe pose model")
    p.add_argument("--variant", default="heavy", choices=["lite", "full", "heavy"])
    p.set_defaults(fn=cmd_model)

    p = sub.add_parser("pose", help="run pose estimation on a video and cache the track (.npz)")
    p.add_argument("video")
    p.add_argument("-o", "--out")
    p.set_defaults(fn=cmd_pose)

    p = sub.add_parser("calibrate", help="build a participant profile from a calibration clip")
    p.add_argument("pose", help="video or cached .npz pose track")
    p.add_argument("--exercise", required=True, choices=["pushup", "situp"])
    p.add_argument("--segments", required=True, help="YAML: start/opposite/reps -> [t0, t1] seconds")
    p.add_argument("-o", "--out", required=True)
    p.set_defaults(fn=cmd_calibrate)

    p = sub.add_parser("analyze", help="count and validate reps")
    p.add_argument("pose", help="video or cached .npz pose track")
    p.add_argument("--exercise", required=True, choices=["pushup", "situp"])
    p.add_argument("--profile", required=True)
    p.add_argument("--t-start", type=float, help="clock start (s); default is the first attempt")
    p.add_argument("--log", help="write the rep log CSV here")
    p.add_argument("--annotate", help="write an annotated review video here")
    p.set_defaults(fn=cmd_analyze)

    p = sub.add_parser("data", help="dataset registry")
    p.add_argument("action", choices=["status", "fetch"])
    p.add_argument("name", nargs="?")
    p.set_defaults(fn=cmd_data)

    p = sub.add_parser("dashboard", help="live camera assessment dashboard")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8000)))
    p.add_argument("--no-browser", action="store_true", default=bool(os.environ.get("PORT")))
    p.set_defaults(fn=cmd_dashboard)

    args = ap.parse_args(argv)
    if args.cmd == "data" and args.action == "fetch" and not args.name:
        ap.error("data fetch needs a dataset name")
    return args.fn(args, load_pipeline_config(args.config)) or 0


if __name__ == "__main__":
    sys.exit(main())

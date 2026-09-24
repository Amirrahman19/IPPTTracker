# IPPT Tracker

IPPT Tracker is an early-stage computer-vision project for tracking push-ups
and sit-ups from a fixed side-view camera. It uses YOLO26 Pose to locate body
joints, calibrates its measurements to the participant's starting posture, and
evaluates complete movement cycles instead of counting individual poses.

The current version is a research prototype. Its thresholds are configurable
starting values and must not be treated as official IPPT rules or scoring.

## How it works

```text
Camera or video
    → YOLO26 human-pose estimation
    → temporary participant calibration
    → normalized joint measurements
    → push-up or sit-up movement tracking
    → valid/no-count decision with a reason
```

Calibration is session-specific and exists only in memory while the program is
running. It adjusts how body movement is measured; it does not give different
participants different exercise standards.

## Current features

- Push-up and sit-up modes.
- Webcam, video-file, or stream input.
- YOLO26 Pose with 17 COCO body keypoints.
- Automatic selection of the clearest visible side of the body.
- Start-position calibration using the median of multiple stable frames.
- Body-relative measurements that are less sensitive to camera distance and
  participant proportions.
- Temporal rep validation:
  - push-up: `UP → BOTTOM → UP`
  - sit-up: `DOWN → UP → DOWN`
- Explainable no-count reasons, including insufficient depth, insufficient
  flexion, body misalignment, and low pose confidence.
- Live skeleton, rep count, joint metrics, and decision overlay.
- Optional annotated video output.
- YAML session report containing decisions and measurements without saving the
  participant's source video by default.

## Project structure

```text
IPPTTracker/
├── config/
│   └── defaults.yaml       # Prototype calibration and movement thresholds
├── src/
│   └── ippt_tracker/       # Tracking application
├── tests/                  # Geometry, calibration, and rep-counting tests
├── .gitignore
├── LICENSE
├── PROJECT_OVERVIEW.md
├── pyproject.toml
└── requirements.txt
```

Downloaded model weights and generated caches are not part of the repository.
Ultralytics downloads `yolo26n-pose.pt` automatically the first time the
tracker is run.

## Environment setup

The dependencies are already installed in the existing `ippt` Conda
environment. To reinstall them later:

```powershell
conda run -n ippt python -m pip install -e .
```

Run the tests with:

```powershell
conda run -n ippt python -m pytest -q
```

## Running the tracker

Place the camera at the participant's side with their full body visible. Hold
the exercise's starting position while calibration completes.

```powershell
# Webcam
conda run -n ippt python -m ippt_tracker.run pushup --source 0

# Video file
conda run -n ippt python -m ippt_tracker.run situp --source sample.mp4
```

Press `q` to finish. Use `--save-video runs/review.mp4` only when an annotated
review video is wanted.

## Next steps

As the project develops, the next useful additions are an open exercise-video
dataset, assessor-labelled evaluation data, comparison against fixed-threshold
baselines, and a simple review interface. Those folders and scripts can be
introduced when that work begins instead of carrying unused scaffolding now.

## Licence note

The Ultralytics dependency is distributed under AGPL-3.0. Its licensing must be
reviewed before any closed-source or institutional deployment.

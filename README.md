# IPPT: joint-centric push-up and sit-up assessment

Side-view video in, valid / no-count rep log out, with a reason and measured
value for every no-count. MediaPipe pose is the first estimator; the pipeline is
estimator-agnostic (see `src/ippt/schema.py`), so YOLO26-pose or a fine-tuned
model can be added as another backend later.

**Status: foundation only.** Pipeline, calibration, rep logic and tests exist
and pass on simulated data. It has not yet been run on real footage, rule
thresholds are unverified placeholders, and no scoring table is bundled. Dataset
download, cleaning and training are intentionally not started.

## Environment

Everything runs in the existing `ippt` conda env (Python 3.12):

```powershell
conda activate ippt
pip install -e . --no-deps     # already done; package name is `ippt`
pytest                         # 30 tests
ippt model                     # downloads the MediaPipe model (~30 MB)
```

mediapipe 1.1.0 and scipy/pandas/tqdm/kaggle were installed into `ippt`. OpenCV
is `opencv-contrib-python<5` (mediapipe needs the contrib build; having plain
`opencv-python` installed as well breaks `cv2`). `pip check` still lists
`opencv-python` as missing for `ultralytics` and the older `ippt-tracker`
package, but both import fine.

Datasets and models live in `data/raw/` (override with `%IPPT_DATA_DIR%`); derived
caches (pose tracks, web transcodes) live in `data/interim/`. Both are gitignored,
and the dataset budget is 5 GB (`configs/datasets.yaml`). `pip install -e ".[data]"`
adds the download dependencies; the dashboard also needs `ffmpeg` on PATH.

## Sample data and dashboard

```powershell
ippt data fetch workout_vids_pushup      # 56 gym push-up clips (~70 MB, ranged read from a 4.9 GB zip)
ippt data fetch hmdb51_pushup_situp      # 100 push-up + 100 sit-up clips from HMDB51 (CC BY 4.0)
ippt dashboard                           # http://127.0.0.1:8000
```

The dashboard plays any clip under `data/raw/datasets/` (or uploaded videos) with
the MediaPipe skeleton drawn over it (near side highlighted, smoothed or raw), live
elbow / body-line / hip / knee angles, and a signal chart you can click to seek.
Pose runs on first open of a clip and is cached.

Click **Analyze & Score** to run the full pipeline on a clip: auto-calibration
detects the starting pose, reps are segmented and checked against rules, and the
result is scored using the official IPPT tables. The dashboard shows a rep-by-rep
log (valid/no-count with reasons and measured values), station points, and
calibration details. You can also upload your own video via the sidebar.

## Pipeline

```
video -> MediaPipe pose (33 pts) -> internal skeleton (23 joints)
      -> gate low-confidence, interpolate short gaps, One Euro smooth
      -> calibration: near side, ground line, facing, segment lengths, posture baseline, ROM
      -> ground-levelled, body-normalised joint features
      -> rep segmentation (valleys on the primary signal) -> fixed rule checks per rep
      -> rep log (valid / no-count + reason + value vs limit) -> scoring table lookup
```

| Module | Role |
|---|---|
| `schema.py` | Internal joint list, MediaPipe mapping, `PoseSequence` (+ `.npz` cache) |
| `pose/mediapipe_backend.py` | Image and video estimation, model download |
| `signal.py` | Gating, gap interpolation, One Euro filter |
| `geometry.py` | Angles, line distance, `Leveler` (removes camera roll, mirrors to head = +x) |
| `calibration.py` | `calibrate()` -> `Profile`; rejects bad calibrations with reasons |
| `features.py` | Elbow/knee/hip/neck angles, upper-arm incline, torso angle, body line, knee line, foot drift, hip rise |
| `reps.py` | Segmentation, per-rep checks, 60 s window |
| `scoring.py` | CSV table lookup (table not bundled) |
| `sim.py` | Kinematic simulator with injectable faults, used by tests |
| `datasets.py`, `cli.py`, `viz.py`, `pipeline.py` | Registry, CLI, review video, orchestration |

Rules live in `configs/rules/*.yaml`, pipeline settings in `configs/pipeline.yaml`,
datasets in `configs/datasets.yaml`.

## Calibration

Calibration decides **how to measure** someone, never **what to demand**.
Validity limits in the rules file are fixed; where a calibrated value feeds one,
`clamp_tighter` lets it tighten the limit but never loosen it.

**Auto-calibration** (default in the dashboard): hold the starting position still
for ~2 seconds at the beginning of the clip, then do your reps. The system
detects the static window automatically (see `auto_calibrate.py`).

**Manual calibration**: record a short clip with three windows (seconds), in a YAML file:

```yaml
start:    [0.5, 2.5]     # required. Push-up: arms locked at the top. Sit-up: lying flat, shoulder blades down.
opposite: [4.0, 6.0]     # optional. Push-up: bottom position. Sit-up: full-range top.
reps:     [7.0, 20.0]    # optional. A few slow reference reps (best source of range of motion).
```

It derives the near-side chain, ground line and facing from the start posture,
segment lengths, natural hip/knee offsets, and the range of motion that places
the rep-segmentation band. It rejects the calibration if keypoint confidence is
low, the view is not a side view, the start posture is not still, or no ground
line can be fitted.

## Usage

```powershell
ippt pose clip.mp4 -o clip.npz                                   # cache the pose track
ippt calibrate clip.npz --exercise pushup --segments calib.yaml -o profile.yaml
ippt analyze clip.mp4 --exercise pushup --profile profile.yaml --t-start 12.0 `
      --log reps.csv --annotate review.mp4
ippt data status                                                 # datasets on disk vs the 5 GB budget
```

`--t-start` is when the 60 s clock starts; without it the first attempt starts it.

## What is not done yet

- **Real footage tuning.** Auto-calibration and the full pipeline run end-to-end,
  but rule thresholds are unverified on real footage. Expect tuning once tested
  on a wider set of real-person clips.
- **Rules are placeholders.** `configs/rules/*.yaml` may need adjusting,
  especially the sit-up end position, hand position and elbow-to-knee requirement.
  Check against current MINDEF/HPB standard before relying on them.
- **Live camera feed.** Real-time pose + rep counting via webcam is planned but
  not yet implemented.
- **Not started:** data cleaning, training, the TCN/GRU rep-validity model,
  fine-tuning a keypoint model, run station.

## Known limits of the current design

- One person per frame (`num_poses=1`); the near side is chosen once at calibration.
- Lengths are 2D projections, so large camera yaw foreshortens limbs; the
  side-view check is the only guard.
- Fingertip and heel/toe points come from BlazePose and can be noisy on prone poses;
  ground-line and foot checks use them with fallbacks.
- A clip that ends within a few frames of a rep's deepest point will not register
  that final attempt.

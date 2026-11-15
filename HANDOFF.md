# HANDOFF — IPPTTracker

_Last updated: 2026-11-15 · by: Claude (dashboard home: recordings + live)_

**Home page (latest):** `/` is a dashboard: Live detection panel (fills when a phone connects) + Recordings library (click a clip -> video, estimated rep count synced to playback, signal graph). Recordings come from `data/interim/web/*.mp4` (override `IPPT_RECORDINGS_DIR`) with pose in `data/interim/pose/*.npz`; both are git-ignored so Render shows an empty library. Rep counts there are estimates (default ROM, no calibration; sit-up estimates are weak). New: `src/ippt/dashboard/recordings.py`, routes `/api/recordings`, `/api/recordings/<name>`, `/media/<name>` (Range supported).

**2026-11-15 update:** `/` is the public dashboard (panel open by default). A "Connect phone" button in the nav bar opens a dialog with the camera link. `/api/status` returns the LAN `camera_url` only to direct loopback viewers (no `X-Forwarded-For`); everyone else gets `location.origin/camera`. 500 errors no longer echo exception text. Run locally with `C:\Users\Amir Rahman\anaconda3\envs\ippt\python.exe -m ippt.cli dashboard --no-browser`. **Privacy caveat:** anyone with the URL still sees the live video feed and can press Reset / change exercise — consider an optional admin token if the site is public.

## Status in one paragraph
Current state per last recorded session: see 'Verified / Not verified' and B&A.md. Baseline before Claude work: Empty scaffold from the planning chat (earlier folder in COMPETITIONS\IPPT was deleted).

## How to run / build / test
`ippt dashboard` (port 8000) in conda env `ippt`; phone opens `http://<LAN-IP>:8000/camera`. Deploy: `render.yaml` + `Dockerfile`. **Re-verify via README.**

## Where things are
- Root: `C:\Users\Amir Rahman\Documents\GitHub\IPPTTracker`
- Top level: `.dockerignore`, `.gitignore`, `Dockerfile`, `README.md`, `The Perfect Push Up _ Do it right!.mp4`, `configs/`, `data/`, `docs/`, `pyproject.toml`, `render.yaml`, `scripts/`, `src/`, `tests/`
- Manifests: pyproject.toml, Dockerfile, render.yaml
- Git: branch `main`, 4 commits; latest: 2026-10-27 a931e61 Add Render deploy support (Dockerfile, render.yaml) and live camera dashboard; uncommitted entries: 0

## Decisions in force
- Recorded footage first, then live feed (user)
- Simple UI emphasising counts + graph (user)
- Hosted on Render via blueprint after the Vercel request (final state; reason inferred)

## Known problems & mistakes to avoid
- Dashboard text showed 127.0.0.1 instead of the LAN IP — misleading; fixed
- Per-frame exception wrote an HTTP error onto an upgraded WebSocket and killed the camera connection — fixed in `server.py` with per-frame guards
- Spawning ffmpeg repeatedly for video reading caused annoyance — removed from the test dashboard

## Verified / Not verified
**Not verified:**
- Accuracy on real reps/official scoring (tests used simulated data)
- Sit-up flow on live camera

## Open questions
- Confirm the reconstructed MVP with the user.

## Next actions (ordered)
1. Validate on real recorded pushups/sit-ups
2. Tune rule profiles
3. Document phone setup

## Read first
- `MVP.md`, `B&A.md`
- Memory record: `Claude Vault\Memory\Log - Software and Data Projects.md` (see the project record)
- `Claude Vault\Agent Harness\README.md` (skills), `Claude Vault\Memory\README.md`

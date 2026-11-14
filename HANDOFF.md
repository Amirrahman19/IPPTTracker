# HANDOFF — IPPTTracker

_Last updated: 2026-10-28 · by: backfill script (reconstructed; refresh on first real session)_

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

# B&A — Before and After — IPPTTracker

> Append-only. Backfilled 2026-10-28: the entries below are **reconstructed** (curated from Claude Vault Memory logs + live files). New entries go at the bottom using the template in `Claude Vault\Agent Harness\Skills\Project Control Files (MVP, HANDOFF, B&A).md`.

## Baseline (reconstructed)
- **Before Claude-assisted work:** Empty scaffold from the planning chat (earlier folder in COMPETITIONS\IPPT was deleted).
- **MVP alignment:** see MVP.md.

## Entries from Claude sessions (reconstructed from Memory logs)

### 2026-10-07 — Foundation pipeline
- **Before:** No code
- **Change (what/how):** Created `ippt` package: MediaPipe 33→23-joint skeleton, calibration, rule profiles; 30 tests (simulated data only)
- **Why:** User's planning chat: MediaPipe + datasets ≤ 5 GB
- **MVP alignment:** Starts the pipeline MUSTs
- **Verified / Not verified:** 30 tests; not run on a real person

### 2026-10-12 — Dashboard, phone camera, GPU, stability
- **Before:** Test dashboard with ffmpeg polling; lag; no reps shown after restart
- **Change (what/how):** Rebuilt dashboard around count + graph; LAN address fix; websocket per-frame guards; GPU check in conda env
- **Why:** User bug reports ("still wrong", lag, no count)
- **MVP alignment:** Meets dashboard/phone/stability MUSTs
- **Verified / Not verified:** User confirmed it worked; later restart regression fixed

### 2026-10-27 — Deploy to Render
- **Before:** Local only
- **Change (what/how):** Added Dockerfile, .dockerignore, render.yaml, pyproject and CLI tweak; no features changed
- **Why:** User wanted Vercel deployment without feature changes; Render blueprint worked
- **MVP alignment:** Satisfies "deployable, no feature changes"
- **Verified / Not verified:** User confirmed it is live

## Git timeline (auto-extracted, newest first; 4 commits total)
| Date | Commit | Message |
|---|---|---|
| 2026-10-27 | `a931e61` | Add Render deploy support (Dockerfile, render.yaml) and live camera dashboard |
| 2026-10-07 | `7d3f52c` | initial commit |
| 2026-09-24 | `dd8625b` | initial commit with project overview and scripts |
| 2026-09-24 | `66136b3` | Initial commit |

## New entries below this line

### 2026-11-15 — Public dashboard + Connect phone button, IP hidden
- **Before:** Dashboard landing page showed the server's LAN IP URL to every visitor in the main view; panel collapsed until data arrived; 500 responses echoed exception text.
- **Change (what/how):** `index.html`: panel open by default, "Connect phone" nav button + modal (copy/open link, no IP in page text). `server.py`: `/api/status` includes `camera_url` (LAN IP) only for direct loopback clients; generic 500 error body.
- **Why:** User: dashboard visible to anyone opening the site, phone connect via button, hide IP/sensitive info.
- **MVP alignment:** Phone camera over LAN + dashboard MUSTs preserved; no scoring changes.
- **Verified / Not verified:** Verified locally: server starts, `/api/status` shows `camera_url` on loopback and omits it with `X-Forwarded-For`, page renders, modal opens with the LAN link. Not verified: real phone stream, Render deployment after push, behind-proxy behaviour on Render.

### 2026-11-15 — Dashboard home with recordings + live detection
- **Before:** Main page was the live view only; recordings were not browsable in the UI.
- **Change (what/how):** New `dashboard/recordings.py` + routes (`/api/recordings`, analysis, `/media/` with Range, filename allow-list). `index.html`: Live detection section, Recordings grid, viewer modal (video, synced rep count, signal graph). "Connect phone" unchanged, optional.
- **Why:** User wanted a dashboard first (recordings + live), phone connect on demand.
- **MVP alignment:** Dashboard + recorded-footage MUSTs; phone path preserved.
- **Verified / Not verified:** Verified locally: 10 clips listed, playback + seek (206), traversal blocked (404), rep estimates computed, viewer UI, 30 tests pass. Not verified: real phone live stream, Render (no clips there: data is git-ignored), sit-up estimate accuracy (clip gave 0 reps).

### 2026-11-15 — Landing page demo with skeleton + counting
- **Before:** Public page showed an empty "No live camera" panel and an (empty on Render) recordings list.
- **Change (what/how):** `index.html` demo mode (video drawn to canvas, precomputed skeleton, counter, graph, hero + feature chips); `scripts/make_demo.py`; `/demo/*` routes; `.gitignore`/`.dockerignore` exceptions for the demo mp4; recordings estimate now self-calibrates ROM from the clip; recordings section hidden when empty.
- **Why:** User wants visitors to see live-style pose estimation and counting as marketing for the project.
- **MVP alignment:** Dashboard MUST; live path and Connect phone untouched.
- **Verified / Not verified:** Verified locally: demo plays with skeleton, count reaches 2 per loop, live frame takeover and demo resume, 30 tests pass. Not verified: Render deploy, mobile layout, demo-clip licence.

### 2026-11-15 — Demo clip swapped to licensed footage
- **Before:** Demo used the `workout-vids` clip (no licence stated; registry says do not redistribute).
- **Change (what/how):** Rebuilt demo from HMDB51 `pushup_test_00880` (CC BY 4.0) via `scripts/make_demo.py`; added on-page credit with link; pushed.
- **Why:** User asked for an attribution note; the original clip could not be credited as open source.
- **MVP alignment:** Public marketing demo requirement kept; licensing constraint respected.
- **Verified / Not verified:** Verified locally: demo shows a single side-view athlete, skeleton, 3 reps, credit visible; 30 tests pass. Not verified: Render deploy, phone layout.

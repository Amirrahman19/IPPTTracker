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

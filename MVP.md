# MVP — IPPTTracker

> Backfilled 2026-10-28 (curated from Claude Vault Memory logs + live files). Edit freely; log later changes below. Items marked _(reconstructed)_ come from past sessions, not from a fresh interview.

## Goal
Computer-vision IPPT (Singapore fitness test) tracker for push-ups and sit-ups: calibrate, record and score from recorded footage first, then live camera (phone or laptop), with a dashboard showing the rep count and up/down graph.

## Users and context
The user and trainees; phone as wireless camera, laptop as display.

## Requirements (MUST) _(reconstructed — tick only when verified)_
- [ ] MediaPipe pose pipeline with calibration and scoring per official HTA/SPF/SAF rubrics (see project memory `ippt-scoring-rules.md`)
- [ ] Recorded-footage path (upload → tracking + scoring) and live camera path on one codebase
- [ ] Dashboard: obvious rep count + live graph; side panel closable/resizable
- [ ] Phone camera over LAN (HTTP; HTTPS did not work for the user) with the page showing the correct reachable address
- [ ] GPU use in the `ippt` conda env where possible
- [ ] Per-frame errors never kill the connection
- [ ] Deployable (Render blueprint)

## Requirements (SHOULD / later)
- Female standards later

## Out of scope
- Heavy datasets in git (≤ 5 GB, git-ignored)

## Constraints
- Use the existing `ippt` conda env
- Do not modify features when changing deploy structure

## Acceptance criteria
- A new model can run the project from HANDOFF.md and each MUST above can be demonstrated. _(Specific acceptance tests: unknown — confirm with the user.)_

## Change requests log
| Date | Requested by | Change | Status |
|---|---|---|---|
| 2026-10-28 | system | Backfilled MVP from history | done |
| 2026-11-15 | user | Dashboard shown to anyone who opens the site; "Connect phone" button in nav bar; hide IP address and other sensitive info; deploy locally, check, commit and push | done (local-verified; Render redeploys on push) |
| 2026-11-15 | user | Main page must be a dashboard first: show existing recordings AND live detection; "Connect phone" is an optional action the user clicks to continue with the phone (supersedes the earlier layout) | done (local-verified) |

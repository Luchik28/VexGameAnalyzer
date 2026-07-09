# VexGameAnalyzer

Local pipeline that turns VEX V5RC tournament livestream VODs into per-match
spatiotemporal data — robot positions, robot archetypes, zone/goal states —
and mines it for strategy insights. Built on Push Back (2025-26); swapping in
a new season (Override) is a config + relabeled dataset, not a rewrite.

## How it works

```
YouTube VODs ──▶ [segment] overlay OCR ──▶ match boundaries + score timeline
RobotEvents API ─▶ official schedules/results joined by match number
[calibrate] 4-corner click ──▶ homography (pixels → field inches)
[detect]    fine-tuned YOLO11s (robots + blocks)
[track]     ByteTrack + starting-tile identity → per-team tracks @5 Hz
[store]     SQLite + parquet    [classify] robot archetypes per team-event
[viewer]    2D canvas replay    [analytics] heatmaps · win-conditions · clusters · scouting
```

## Setup

```bash
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env   # add your RobotEvents token (robotevents.com → API)
```

Needs macOS (Apple Vision OCR) and node ≥22 on PATH or under ~/.nvm
(yt-dlp JS challenges). No Homebrew required — ffmpeg comes from
imageio-ffmpeg, video decode from the opencv wheel.

## Pipeline runbook (one event)

```bash
V=.venv/bin/python
# 1. download a VOD (test with --section first; drop it for the full VOD)
$V -m vexga.cli download "https://www.youtube.com/watch?v=..." --section 1:00:00-1:20:00

# 2. segment into matches (OCR pass, ~1-2 h for an 8 h VOD)
$V -m vexga.cli segment data/videos/<id>.mp4 --event <robotevents_event_id>
$V -m vexga.cli matches                      # inspect what was found
$V -m vexga.cli join-re --event <id> --division <div>   # official teams+scores

# 3. calibrate the camera (once per event/camera; opens a window)
$V -m vexga.calibrate.tool data/videos/<id>.mp4 --ts <some_match_ts>

# 4. build the detection dataset + pre-labels + Label Studio tasks
$V -m vexga.cli dataset --name pushback_v1 --frames 1500
uvx label-studio   # import tasks.json, correct labels (~2-4 h), export YOLO
# unzip the export over data/datasets/pushback_v1/labels/

# 5. train the detector (overnight on M1)
$V -m vexga.detect.train data/datasets/pushback_v1/dataset.yaml

# 6. track all matches (overnight for ~300 matches)
$V -m vexga.cli track --weights models/pushback_v1/weights/best.pt

# 7. robot archetypes: collect crops, seed-label ~30-50 teams, classify
$V -c "from vexga.classify.crops import collect_crops; collect_crops('models/pushback_v1/weights/best.pt')"
$V -m vexga.classify.label_tool --event <id>
$V -c "from vexga.classify.archetype import train_and_classify; train_and_classify(<id>)"

# 8. exports + replay viewer + analytics
$V -m vexga.cli export
(cd vexga/viewer && python3 -m http.server 8093)   # open :8093/?match=../../data/exports/matches/match_1.json
$V -m vexga.analytics.scout --all
$V -c "from vexga.analytics.insights import win_condition_analysis; print(win_condition_analysis())"
```

## Layout

- `vexga/games/` — **all game-specific knowledge** (field geometry, zones,
  scoring, archetypes). `pushback.py` is complete; `override.py` is the
  template for next season.
- `vexga/acquire|segment|calibrate|detect|track|store|classify|analytics|viewer/`
  — pipeline stages, each independently runnable.
- `data/` (gitignored) — videos, frames, datasets, sqlite db, exports.
- `models/` — trained weights + archetype seed labels.

## Performance notes (M1, 16GB)

- Close Jupyter/heavy apps before training or batch tracking: under memory
  pressure torch imports take 1-2 min and the first MPS inference minutes.
- The OCR segmenter and the motion-based dry-run tracker are pure
  CPU/OpenCV/Vision and stay fast regardless.
- `vexga/track/dryrun.py` runs the full tracking path with the motion
  detector (no trained weights needed) — use it to sanity-check new events
  before spending a night on training.

## New season runbook (Override)

1. Fill `vexga/games/override.py` from the Override manual Appendix A
   (render the field-spec pages; coordinates use the audience-view frame,
   origin bottom-left, x toward blue wall).
2. Download 1-2 early-season VODs, `segment`, `calibrate`.
3. `dataset --game override` + correct labels + retrain.
4. Everything downstream (tracking, viewer, analytics) works unchanged.

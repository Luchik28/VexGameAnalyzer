"""Pre-label the detection dataset without torch: motion-based robots +
HSV blocks. Groups dataset images by source match (encoded in filenames as
m<match_id>_t<ts>.jpg), builds one background model per match, then writes
YOLO label files and Label Studio tasks.

Stationary robots (auton start, parked) are missed by motion diff — the
human correction pass adds those boxes.

    PYTHONPATH=. .venv/bin/python scripts/prelabel_motion.py --name pushback_v1
"""

import argparse
import re
from collections import defaultdict
from pathlib import Path

import cv2

from vexga.config import DATASETS
from vexga.detect.labelstudio import yolo_to_tasks
from vexga.detect.prelabel import to_yolo_line
from vexga.games.base import get_game
from vexga.store.db import connect
from vexga.track.motion import MotionRobotDetector, background_model
from vexga.track.process import calibration_for

FNAME_RE = re.compile(r"m(\d+)_t([\d.]+)\.jpg")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="pushback_v1")
    ap.add_argument("--game", default="pushback")
    args = ap.parse_args()

    game = get_game(args.game)
    ds = DATASETS / args.name
    con = connect()
    cls_ids = {c: i for i, c in enumerate(game.detect_classes)}

    by_match: dict[str, list[Path]] = defaultdict(list)
    for split in ("train", "val"):
        for p in (ds / "images" / split).glob("*.jpg"):
            m = FNAME_RE.match(p.name)
            if m:
                by_match[m.group(1)].append(p)

    # Match ids in filenames may be stale (dataset extracted before a
    # re-segmentation); fall back to any match on the same video whose span
    # covers the frame timestamp.
    videos = {v["id"]: v["path"] for v in con.execute("SELECT id, path FROM videos")}
    matches = [dict(r) for r in con.execute(
        "SELECT id, video_id, video_start_ts, video_end_ts FROM matches WHERE video_end_ts IS NOT NULL")]

    n_done = 0
    for mid, paths in sorted(by_match.items()):
        row = next((m for m in matches if str(m["id"]) == mid), None)
        if row is None:
            ts0 = float(FNAME_RE.match(paths[0].name).group(2))
            row = next((m for m in matches
                        if m["video_start_ts"] <= ts0 <= (m["video_end_ts"] or 0)), None)
        if row is None:
            print(f"match {mid}: no covering span, skipping {len(paths)} frames")
            continue
        cal = calibration_for(con, row["video_id"], max(row["video_start_ts"], 0))
        if cal is None:
            print(f"match {mid}: no calibration, skipping")
            continue
        bg = background_model(videos[row["video_id"]],
                              max(row["video_start_ts"], 0), row["video_end_ts"])
        det = MotionRobotDetector(bg, cal, game)
        for p in sorted(paths):
            frame = cv2.imread(str(p))
            if frame is None:
                continue
            h, w = frame.shape[:2]
            lines = [to_yolo_line(cls_ids[c], box, w, h)
                     for c, _tid, _conf, box in det(frame) if c in cls_ids]
            lbl = ds / "labels" / p.parent.name / f"{p.stem}.txt"
            lbl.parent.mkdir(parents=True, exist_ok=True)
            lbl.write_text("\n".join(lines) + ("\n" if lines else ""))
            n_done += 1
        print(f"match {mid}: {len(paths)} frames pre-labeled")
    print(f"total {n_done} frames")
    yolo_to_tasks(ds, game)


if __name__ == "__main__":
    main()

"""Round-2 pre-labeling: run the trained detector over frames NOT yet
human-annotated and write YOLO labels + a Label Studio tasks file for a new
project. Human-annotated frames (annotations/*.json) are left untouched.

    PYTHONPATH=. .venv/bin/python -u scripts/prelabel_model.py \
        --weights models/pushback_v1h/weights/best.pt
"""

import argparse
import json
from pathlib import Path

import cv2

from vexga.config import DATASETS
from vexga.detect.labelstudio import yolo_to_tasks
from vexga.detect.prelabel import to_yolo_line
from vexga.games.base import get_game


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--name", default="pushback_v2")
    ap.add_argument("--conf", type=float, default=0.30)
    args = ap.parse_args()

    from ultralytics import YOLO

    game = get_game("pushback")
    ds = DATASETS / args.name
    ann_path = Path("annotations") / f"{args.name}_annotations.json"
    done = set(json.loads(ann_path.read_text()).keys()) if ann_path.exists() else set()
    model = YOLO(args.weights)
    cls_ids = {c: i for i, c in enumerate(game.detect_classes)}

    n = 0
    for split in ("train", "val"):
        for p in sorted((ds / "images" / split).glob("*.jpg")):
            rel = f"{args.name}/images/{split}/{p.name}"
            if rel in done:
                continue
            frame = cv2.imread(str(p))
            if frame is None:
                continue
            h, w = frame.shape[:2]
            res = model.predict(frame, conf=args.conf, verbose=False, device="mps")[0]
            lines = []
            for b in res.boxes:
                cname = res.names[int(b.cls)]
                if cname not in cls_ids:
                    continue
                lines.append(to_yolo_line(cls_ids[cname], tuple(map(float, b.xyxy[0])), w, h))
            lbl = ds / "labels" / split / f"{p.stem}.txt"
            lbl.write_text("\n".join(lines) + ("\n" if lines else ""))
            n += 1
            if n % 200 == 0:
                print(f"{n} frames pre-labeled", flush=True)
    print(f"model pre-labels written for {n} frames (kept {len(done)} human-annotated)",
          flush=True)
    yolo_to_tasks(ds, game)
    print("NOTE: import tasks.json into a NEW Label Studio project to avoid"
          " duplicating the old tasks; already-annotated frames are included"
          " with their human labels as predictions.", flush=True)


if __name__ == "__main__":
    main()

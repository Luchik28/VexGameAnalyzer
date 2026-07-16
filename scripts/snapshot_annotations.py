"""Snapshot Label Studio annotations into the repo (annotations/).

Reads Label Studio's own sqlite (read-only) and writes one JSON per dataset:
task image path -> list of boxes (class, x/y/w/h as fractions). Small, diffs
cleanly, and doubles as a backup of labeling hours. Run any time; commit the
result. The YOLO export for training still comes from Label Studio's own
Export button (or from this snapshot via --emit-yolo).

    PYTHONPATH=. .venv/bin/python scripts/snapshot_annotations.py [--emit-yolo]
"""

import argparse
import json
import re
import sqlite3
from pathlib import Path

LS_DB = Path.home() / "Library/Application Support/label-studio/label_studio.sqlite3"
OUT_DIR = Path(__file__).resolve().parent.parent / "annotations"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--emit-yolo", action="store_true",
                    help="also write YOLO label files into the dataset's labels/")
    args = ap.parse_args()

    con = sqlite3.connect(f"file:{LS_DB}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    rows = con.execute(
        "SELECT t.data, c.result, c.updated_at FROM task_completion c"
        " JOIN task t ON t.id = c.task_id WHERE c.was_cancelled = 0"
    ).fetchall()

    per_dataset: dict[str, dict] = {}
    for r in rows:
        img = json.loads(r["data"]).get("image", "")
        m = re.search(r"d=([^&]+)", img)
        if not m:
            continue
        rel = m.group(1)                      # pushback_v2/images/train/x.jpg
        dataset = rel.split("/")[0]
        boxes = []
        classes = set()
        for res in json.loads(r["result"]):
            if res.get("type") != "rectanglelabels":
                continue
            v = res["value"]
            cls = v["rectanglelabels"][0]
            classes.add(cls)
            boxes.append({"cls": cls,
                          "x": round(v["x"] / 100, 6), "y": round(v["y"] / 100, 6),
                          "w": round(v["width"] / 100, 6), "h": round(v["height"] / 100, 6)})
        per_dataset.setdefault(dataset, {})[rel] = {
            "updated_at": r["updated_at"], "boxes": boxes}

    OUT_DIR.mkdir(exist_ok=True)
    for dataset, items in per_dataset.items():
        out = OUT_DIR / f"{dataset}_annotations.json"
        out.write_text(json.dumps(items, indent=1, sort_keys=True))
        print(f"{dataset}: {len(items)} annotated frames -> {out}")

        if args.emit_yolo:
            from vexga.games.base import get_game

            game = get_game("pushback")
            cls_ids = {c: i for i, c in enumerate(game.detect_classes)}
            root = Path(__file__).resolve().parent.parent / "data/datasets"
            n = 0
            for rel, item in items.items():
                lbl = root / rel.replace("/images/", "/labels/").replace(".jpg", ".txt")
                lines = []
                for b in item["boxes"]:
                    if b["cls"] not in cls_ids:
                        continue
                    lines.append(f"{cls_ids[b['cls']]} {b['x'] + b['w']/2:.6f}"
                                 f" {b['y'] + b['h']/2:.6f} {b['w']:.6f} {b['h']:.6f}")
                lbl.parent.mkdir(parents=True, exist_ok=True)
                lbl.write_text("\n".join(lines) + ("\n" if lines else ""))
                n += 1
            print(f"  emitted {n} YOLO label files (human-corrected)")


if __name__ == "__main__":
    main()

"""Label Studio round trip for the assisted-labeling pass.

Export: dataset images + YOLO pre-labels -> tasks.json with predictions and
a labeling config XML. Label Studio runs locally in its own venv to avoid
dependency clashes:

    uvx label-studio  # or: pipx run label-studio

In Label Studio: create project -> Labeling Setup -> paste config.xml ->
import tasks.json (enable serving local files, LABEL_STUDIO_LOCAL_FILES_
SERVING_ENABLED=true, DOCUMENT_ROOT=data/datasets). After correcting,
export as "YOLO" and unzip over the dataset's labels/ directory.
"""

import json
from pathlib import Path

import cv2

from vexga.games.base import GameConfig

COLORS = {"robot_red": "#e53935", "robot_blue": "#1e88e5",
          "block_red": "#ffb3b3", "block_blue": "#b3d1ff"}


def labeling_config(game: GameConfig) -> str:
    labels = "\n".join(
        f'    <Label value="{c}" background="{COLORS.get(c, "#aaaaaa")}"/>'
        for c in game.detect_classes
    )
    return (
        '<View>\n  <Image name="image" value="$image" zoom="true"/>\n'
        '  <RectangleLabels name="label" toName="image">\n'
        f"{labels}\n  </RectangleLabels>\n</View>\n"
    )


def yolo_to_tasks(dataset_dir: Path, game: GameConfig) -> Path:
    """Build tasks.json with pre-annotations from the YOLO label files."""
    tasks = []
    for split in ("train", "val"):
        for img_path in sorted((dataset_dir / "images" / split).glob("*.jpg")):
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            h, w = img.shape[:2]
            results = []
            lbl = dataset_dir / "labels" / split / f"{img_path.stem}.txt"
            if lbl.exists():
                for line in lbl.read_text().splitlines():
                    parts = line.split()
                    if len(parts) != 5:
                        continue
                    ci, cx, cy, bw, bh = int(parts[0]), *map(float, parts[1:])
                    results.append({
                        "type": "rectanglelabels",
                        "from_name": "label", "to_name": "image",
                        "original_width": w, "original_height": h,
                        "value": {
                            "x": (cx - bw / 2) * 100, "y": (cy - bh / 2) * 100,
                            "width": bw * 100, "height": bh * 100,
                            "rectanglelabels": [game.detect_classes[ci]],
                        },
                    })
            rel = img_path.relative_to(dataset_dir.parent)
            tasks.append({
                "data": {"image": f"/data/local-files/?d={rel}"},
                "predictions": [{"result": results}],
            })
    out = dataset_dir / "tasks.json"
    out.write_text(json.dumps(tasks, indent=1))
    (dataset_dir / "labeling_config.xml").write_text(labeling_config(game))
    print(f"{len(tasks)} tasks -> {out}")
    return out
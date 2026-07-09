"""Detection dataset utilities: extract diverse frames from segmented
matches, auto pre-label them, and emit a YOLO-format dataset that Label
Studio (or any YOLO-aware tool) can import for human correction.

Layout (data/datasets/<name>/):
    images/train|val/*.jpg
    labels/train|val/*.txt      # YOLO: cls cx cy w h (normalized)
    dataset.yaml
"""

import random
from pathlib import Path

import cv2

from vexga.config import DATASETS
from vexga.games.base import GameConfig
from vexga.store.db import connect


def sample_match_frames(n_total: int, out_dir: Path, seed: int = 7,
                        val_frac: float = 0.15) -> list[Path]:
    """Pull ~n_total frames uniformly across all segmented matches in the DB,
    spread over match time. Returns saved image paths."""
    con = connect()
    matches = con.execute(
        "SELECT m.id, m.video_start_ts, m.video_end_ts, v.path FROM matches m"
        " JOIN videos v ON v.id = m.video_id"
        " WHERE m.video_end_ts IS NOT NULL"
    ).fetchall()
    if not matches:
        raise RuntimeError("no segmented matches in DB - run `vexga.cli segment` first")
    rng = random.Random(seed)
    per_match = max(1, n_total // len(matches))
    saved: list[Path] = []
    (out_dir / "images/train").mkdir(parents=True, exist_ok=True)
    (out_dir / "images/val").mkdir(parents=True, exist_ok=True)
    by_video: dict[str, list] = {}
    for m in matches:
        by_video.setdefault(m["path"], []).append(m)
    for vpath, ms in by_video.items():
        cap = cv2.VideoCapture(vpath)
        for m in ms:
            for _ in range(per_match):
                ts = rng.uniform(m["video_start_ts"], m["video_end_ts"])
                cap.set(cv2.CAP_PROP_POS_MSEC, ts * 1000)
                ok, frame = cap.read()
                if not ok:
                    continue
                split = "val" if rng.random() < val_frac else "train"
                p = out_dir / "images" / split / f"m{m['id']}_t{ts:.1f}.jpg"
                cv2.imwrite(str(p), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
                saved.append(p)
        cap.release()
    return saved


def write_dataset_yaml(out_dir: Path, game: GameConfig) -> None:
    names = "\n".join(f"  {i}: {c}" for i, c in enumerate(game.detect_classes))
    (out_dir / "dataset.yaml").write_text(
        f"path: {out_dir}\ntrain: images/train\nval: images/val\nnames:\n{names}\n"
    )


def make_dataset(name: str, game: GameConfig, n_frames: int = 1500) -> Path:
    out = DATASETS / name
    imgs = sample_match_frames(n_frames, out)
    write_dataset_yaml(out, game)
    print(f"dataset {name}: {len(imgs)} frames at {out}")
    return out

"""Seed-labeling tool for robot archetypes (user-run, cv2 window).

    .venv/bin/python -m vexga.classify.label_tool --event 12345

Shows a montage of each team's crops; press the number key of the archetype
(listed in the game config), s to skip, u to undo, q to quit. Labels land in
models/team_labels.json (team -> archetype). Label 30-50 teams spread across
obviously different robot designs.
"""

import argparse
import json

import cv2
import numpy as np

from vexga.classify.archetype import LABELS_PATH
from vexga.config import FRAMES
from vexga.games.base import get_game


def montage(team_dir, cell: int = 200, cols: int = 4, rows: int = 3) -> np.ndarray | None:
    paths = sorted(team_dir.glob("*.jpg"))
    if not paths:
        return None
    step = max(1, len(paths) // (cols * rows))
    grid = np.zeros((rows * cell, cols * cell, 3), np.uint8)
    for i, p in enumerate(paths[::step][: cols * rows]):
        im = cv2.imread(str(p))
        if im is None:
            continue
        s = cell / max(im.shape[:2])
        im = cv2.resize(im, (int(im.shape[1] * s), int(im.shape[0] * s)))
        r, c = divmod(i, cols)
        grid[r * cell:r * cell + im.shape[0], c * cell:c * cell + im.shape[1]] = im
    return grid


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--event", type=int, required=True)
    ap.add_argument("--game", default="pushback")
    args = ap.parse_args()

    game = get_game(args.game)
    labels: dict[str, str] = json.loads(LABELS_PATH.read_text()) if LABELS_PATH.exists() else {}
    teams = sorted(p for p in (FRAMES / "crops" / str(args.event)).iterdir() if p.is_dir())
    order = [t for t in teams if t.name not in labels]
    print(f"{len(order)} unlabeled teams; keys: " +
          " ".join(f"{i+1}={a}" for i, a in enumerate(game.archetypes)) + "  s=skip u=undo q=quit")
    history: list[str] = []
    i = 0
    while i < len(order):
        team_dir = order[i]
        img = montage(team_dir)
        if img is None:
            i += 1
            continue
        cv2.putText(img, team_dir.name, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.imshow("label archetypes", img)
        k = cv2.waitKey(0) & 0xFF
        if k == ord("q"):
            break
        if k == ord("s"):
            i += 1
            continue
        if k == ord("u") and history:
            labels.pop(history.pop(), None)
            i -= 1
            continue
        idx = k - ord("1")
        if 0 <= idx < len(game.archetypes):
            labels[team_dir.name] = game.archetypes[idx]
            history.append(team_dir.name)
            i += 1
    cv2.destroyAllWindows()
    LABELS_PATH.write_text(json.dumps(labels, indent=1, sort_keys=True))
    print(f"{len(labels)} labels saved to {LABELS_PATH}")


if __name__ == "__main__":
    main()

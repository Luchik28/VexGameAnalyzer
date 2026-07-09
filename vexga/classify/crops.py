"""Collect per-team robot image crops across all of a team's matches.

A team's robot is constant within an event, so archetype classification
happens per (event, team) using many crops + behavioral features rather than
per frame. Crops come from re-running the detector on a handful of sampled
frames per match and picking the detection nearest the team's tracked
position (avoids needing stored per-frame boxes).

Output: data/frames/crops/<event_id>/<team>/m<match>_t<ts>.jpg
"""

from pathlib import Path

import cv2
import numpy as np

from vexga.config import FRAMES
from vexga.games.base import get_game
from vexga.store.db import connect
from vexga.track.process import calibration_for


def collect_crops(weights: str, game_name: str = "pushback",
                  per_match: int = 6, min_conf: float = 0.4) -> Path:
    from ultralytics import YOLO

    game = get_game(game_name)
    model = YOLO(weights)
    con = connect()
    rows = con.execute(
        "SELECT m.id, m.event_id, m.video_id, m.video_start_ts, m.video_end_ts,"
        " m.red1, m.red2, m.blue1, m.blue2, v.path FROM matches m"
        " JOIN videos v ON v.id=m.video_id WHERE m.video_end_ts IS NOT NULL"
    ).fetchall()
    out_root = FRAMES / "crops"
    for m in rows:
        cal = calibration_for(con, m["video_id"], m["video_start_ts"])
        if cal is None or not Path(m["path"]).exists():
            continue
        slot_team = {s: m[s] for s in ("red1", "red2", "blue1", "blue2") if m[s]}
        if not slot_team:
            continue
        # tracked positions to match detections against
        tracks: dict[str, list] = {}
        for r in con.execute(
            "SELECT t, slot, x_in, y_in FROM robot_tracks WHERE match_id=? AND conf>0 ORDER BY t",
            (m["id"],),
        ):
            tracks.setdefault(r["slot"], []).append((r["t"], r["x_in"], r["y_in"]))
        if not tracks:
            continue
        cap = cv2.VideoCapture(m["path"])
        dur = m["video_end_ts"] - m["video_start_ts"]
        for k in range(per_match):
            t_rel = (k + 0.5) * dur / per_match
            cap.set(cv2.CAP_PROP_POS_MSEC, (m["video_start_ts"] + t_rel) * 1000)
            ok, frame = cap.read()
            if not ok:
                continue
            res = model.predict(frame, conf=min_conf, verbose=False, device="mps")[0]
            names = res.names
            dets = []
            for b in res.boxes:
                if not names[int(b.cls)].startswith("robot"):
                    continue
                x0, y0, x1, y1 = map(float, b.xyxy[0])
                g = cal.to_field(np.array([[(x0 + x1) / 2, y1]]))[0]
                dets.append((g, (int(x0), int(y0), int(x1), int(y1))))
            for slot, team in slot_team.items():
                tr = tracks.get(slot)
                if not tr:
                    continue
                ts_arr = np.array([p[0] for p in tr])
                i = int(np.clip(np.searchsorted(ts_arr, t_rel), 0, len(tr) - 1))
                pos = np.array(tr[i][1:])
                best, best_d = None, 12.0  # inches
                for g, box in dets:
                    d = float(np.linalg.norm(g - pos))
                    if d < best_d:
                        best, best_d = box, d
                if best is None:
                    continue
                x0, y0, x1, y1 = best
                pad = int(0.15 * max(x1 - x0, y1 - y0))
                crop = frame[max(0, y0 - pad):y1 + pad, max(0, x0 - pad):x1 + pad]
                if crop.size == 0:
                    continue
                d = out_root / str(m["event_id"] or 0) / team.replace("/", "_")
                d.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(d / f"m{m['id']}_t{t_rel:.0f}.jpg"), crop)
        cap.release()
        print(f"match {m['id']}: crops collected")
    return out_root

"""(Re)calibrate all clips: cluster by camera, auto-calibrate each major
cluster (orientation from a median background frame + park-zone self-check),
copy to nearest cluster for singletons. Idempotent; run after adding events.

    PYTHONPATH=. .venv/bin/python -u scripts/recalibrate.py
"""

import json
from pathlib import Path

import cv2
import numpy as np

from vexga.calibrate.auto import auto_calibrate
from vexga.calibrate.cameras import DIST_THRESHOLD, apply_cluster_calibration, cluster_clips, fingerprint
from vexga.config import DATA
from vexga.games.base import get_game
from vexga.store.db import connect

CLUSTERS_JSON = DATA / "clusters.json"


def color_background(path: str, n: int = 15) -> np.ndarray | None:
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frames = []
    for ts in np.linspace(5, max(6.0, total / fps - 5), n):
        cap.set(cv2.CAP_PROP_POS_MSEC, ts * 1000)
        ok, f = cap.read()
        if ok:
            frames.append(f)
    cap.release()
    if not frames:
        return None
    return np.median(np.stack(frames), axis=0).astype(np.uint8)


def main() -> None:
    con = connect()
    game = get_game("pushback")
    sources = [r["source_id"] for r in con.execute(
        "SELECT DISTINCT source_id FROM videos WHERE source_id IS NOT NULL")]

    if CLUSTERS_JSON.exists():
        clusters = json.loads(CLUSTERS_JSON.read_text())
    else:
        clusters = {}
        for src in sources:
            cl = cluster_clips(con, src)
            clusters[src] = {str(k): v for k, v in cl.items()}
            print(f"{src}: {len(cl)} clusters "
                  f"({sorted((len(v) for v in cl.values()), reverse=True)[:5]}...)", flush=True)
        CLUSTERS_JSON.write_text(json.dumps(clusters))

    cluster_cal: dict[tuple[str, str], object] = {}
    applied = 0
    for src, cl in clusters.items():
        for k, ids in sorted(cl.items(), key=lambda kv: -len(kv[1])):
            if len(ids) < 3:
                continue
            rep = ids[len(ids) // 2]
            row = con.execute("SELECT path FROM videos WHERE id=?", (rep,)).fetchone()
            cap = cv2.VideoCapture(row["path"])
            cap.set(cv2.CAP_PROP_POS_MSEC, 30000)
            ok, frame = cap.read()
            cap.release()
            bg = color_background(row["path"])
            if not ok or bg is None:
                continue
            cal = auto_calibrate(frame, game, orient_frame=bg)
            if cal is None:
                print(f"NO-FIT {src[:8]} cluster {k} ({len(ids)} clips)", flush=True)
                continue
            apply_cluster_calibration(con, ids, cal.to_json(), cal.reproj_err_in)
            cluster_cal[(src, k)] = cal
            applied += len(ids)
    print(f"majors: {applied} clips calibrated", flush=True)

    assigned = 0
    for src, cl in clusters.items():
        centers = {}
        for k, ids in cl.items():
            if (src, k) not in cluster_cal:
                continue
            fps_ = [fingerprint(con.execute("SELECT path FROM videos WHERE id=?", (i,)).fetchone()["path"])
                    for i in ids[:3]]
            fps_ = [f for f in fps_ if f is not None]
            if fps_:
                centers[k] = np.mean(fps_, axis=0)
        for k, ids in cl.items():
            if len(ids) >= 3:
                continue
            for cid in ids:
                fp = fingerprint(con.execute("SELECT path FROM videos WHERE id=?", (cid,)).fetchone()["path"])
                if fp is None:
                    continue
                best, best_d = None, 2 * DIST_THRESHOLD
                for kk, c in centers.items():
                    d = float(np.mean(np.abs(fp - c)))
                    if d < best_d:
                        best, best_d = kk, d
                if best is None:
                    continue
                cal = cluster_cal[(src, best)]
                con.execute(
                    "INSERT OR REPLACE INTO calibrations (video_id, from_ts, homography, reproj_err_in)"
                    " VALUES (?,0,?,?)", (cid, cal.to_json(), cal.reproj_err_in))
                assigned += 1
    con.commit()
    n = con.execute("SELECT COUNT(DISTINCT video_id) c FROM calibrations WHERE video_id LIKE 'm%'").fetchone()["c"]
    print(f"singletons assigned: {assigned}; total calibrated clips: {n}", flush=True)


if __name__ == "__main__":
    main()

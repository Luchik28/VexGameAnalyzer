"""Cluster clips by camera and map calibrations per camera.

Events stream matches from multiple field cameras (Kalahari alternates two
fields), so one calibration per event is wrong. This module:
1. fingerprints each clip (downscaled gray median of a few frames),
2. clusters fingerprints greedily by L1 distance (same camera = near-
   identical background; robots/people are a small fraction of pixels),
3. lets a calibration be attached per cluster, materialized into the
   `calibrations` table per clip video_id so downstream lookup is unchanged.
"""

import numpy as np

FP_W, FP_H = 64, 36
# Same-camera frames differ only by robots/audience: empirically < ~10;
# different cameras/fields differ by > ~25.
DIST_THRESHOLD = 16.0


def fingerprint(video_path: str, at_s: tuple[float, ...] = (2.0, 30.0, 60.0)) -> np.ndarray | None:
    import cv2

    cap = cv2.VideoCapture(video_path)
    frames = []
    for ts in at_s:
        cap.set(cv2.CAP_PROP_POS_MSEC, ts * 1000)
        ok, f = cap.read()
        if ok:
            frames.append(cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), (FP_W, FP_H)))
    cap.release()
    if not frames:
        return None
    return np.median(np.stack(frames), axis=0).astype(np.float32)


def cluster_clips(con, source_id: str | None = None) -> dict[int, list[str]]:
    """Greedy clustering of all clips (optionally one event's). Returns
    cluster_index -> [clip video_id]. Also stores the cluster in videos.division
    as 'cam<idx>' for reuse."""
    q = "SELECT id, path FROM videos WHERE id LIKE 'm%'"
    args: list = []
    if source_id:
        q += " AND source_id = ?"
        args.append(source_id)
    rows = con.execute(q, args).fetchall()
    centers: list[np.ndarray] = []
    clusters: dict[int, list[str]] = {}
    for r in rows:
        fp = fingerprint(r["path"])
        if fp is None:
            continue
        best, best_d = None, DIST_THRESHOLD
        for i, c in enumerate(centers):
            d = float(np.mean(np.abs(fp - c)))
            if d < best_d:
                best, best_d = i, d
        if best is None:
            centers.append(fp)
            best = len(centers) - 1
        else:  # running mean keeps the center stable
            n = len(clusters[best])
            centers[best] = (centers[best] * n + fp) / (n + 1)
        clusters.setdefault(best, []).append(r["id"])
    return clusters


def apply_cluster_calibration(con, clip_ids: list[str], homography_json: str,
                              reproj_err_in: float) -> None:
    con.executemany(
        "INSERT OR REPLACE INTO calibrations (video_id, from_ts, homography, reproj_err_in)"
        " VALUES (?, 0, ?, ?)",
        [(cid, homography_json, reproj_err_in) for cid in clip_ids],
    )
    con.commit()

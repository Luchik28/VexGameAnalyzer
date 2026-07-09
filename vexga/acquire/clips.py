"""Cut per-match clips out of a full VOD so the VOD can be deleted.

Disk on this machine is scarce (~6 GB free) while a full event VOD is
~3-4 GB, so the durable artifact is one ~2.5-minute clip per match:
re-encoded (frame-accurate seek), audio stripped, 720p CRF 24 — about
10-15 MB each, ~4 GB for a 340-match pilot.

Each clip becomes its own `videos` row (id `m<match_id>`, source_id = the
VOD's id) and the match's video_id/timestamps are rewritten clip-relative.
Calibration lookups fall back to the source VOD (track/process.py), so one
calibration click-session per event still covers everything.
"""

import subprocess
import time
from pathlib import Path

from vexga.config import VIDEOS, ffmpeg_exe

PRE_ROLL = 8.0    # seconds before auton start (calibration/QA context)
POST_ROLL = 10.0  # seconds after match end (result settles)
RETRIES = 2       # macOS memory pressure SIGKILLs ffmpeg occasionally


def extract_match_clips(con, video_id: str, delete_source: bool = False) -> int:
    """Cut clips for every finished match of `video_id`; rewrite DB rows.
    Returns number of clips made."""
    src = con.execute("SELECT * FROM videos WHERE id=?", (video_id,)).fetchone()
    if src is None:
        raise KeyError(f"unknown video {video_id}")
    src_path = Path(src["path"])
    matches = con.execute(
        "SELECT id, video_start_ts, video_auton_end_ts, video_end_ts FROM matches"
        " WHERE video_id=? AND video_end_ts IS NOT NULL", (video_id,)
    ).fetchall()
    clip_dir = VIDEOS / "clips"
    clip_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    failed = 0
    for m in matches:
        start = max(0.0, m["video_start_ts"] - PRE_ROLL)
        pre = m["video_start_ts"] - start  # == PRE_ROLL unless clamped at 0
        out = clip_dir / f"m{m['id']}.mp4"
        cmd = [ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y",
               "-ss", f"{start:.3f}", "-to", f"{m['video_end_ts'] + POST_ROLL:.3f}",
               "-i", str(src_path),
               "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "24",
               str(out)]
        for attempt in range(RETRIES + 1):
            try:
                subprocess.run(cmd, check=True)
                break
            except subprocess.CalledProcessError as e:
                out.unlink(missing_ok=True)
                if attempt == RETRIES:
                    print(f"  clip m{m['id']} FAILED after {RETRIES + 1} tries: {e}",
                          flush=True)
                    failed += 1
                else:
                    time.sleep(15)  # let memory pressure subside
        else:
            continue
        clip_id = f"m{m['id']}"
        con.execute(
            "INSERT OR REPLACE INTO videos (id, event_id, source_id, division, path,"
            " title, fps, width, height) VALUES (?,?,?,?,?,?,?,?,?)",
            (clip_id, src["event_id"], video_id, src["division"], str(out),
             src["title"], src["fps"], src["width"], src["height"]),
        )
        con.execute(
            "UPDATE matches SET video_id=?, video_start_ts=?, video_auton_end_ts=?,"
            " video_end_ts=?, notes = COALESCE(notes,'') || ? WHERE id=?",
            (clip_id, pre,
             (m["video_auton_end_ts"] - start) if m["video_auton_end_ts"] is not None else None,
             m["video_end_ts"] - start,
             f" | src={video_id}@{m['video_start_ts']:.0f}", m["id"]),
        )
        con.commit()
        n += 1
        print(f"  clip m{m['id']} ({out.stat().st_size/1e6:.0f} MB)", flush=True)
    if failed:
        print(f"{failed} clips failed; source VOD kept for retry"
              f" (re-run extract_match_clips - it only processes unclipped matches)",
              flush=True)
    if delete_source and failed == 0 and n > 0:
        src_path.unlink(missing_ok=True)
        print(f"deleted source VOD {src_path.name}", flush=True)
    return n

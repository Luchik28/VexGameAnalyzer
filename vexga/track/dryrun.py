"""Tracker dry run with a stand-in detector (before the real one is trained).

Uses motion-based robot detection (track/motion.py) — pure OpenCV, no torch,
so it runs fast even when the machine is loaded. Quality is below the
fine-tuned detector, but it exercises the entire tracking -> storage ->
viewer path on real footage.

    PYTHONPATH=. .venv/bin/python -m vexga.track.dryrun --match 8
"""

import argparse

from vexga.games.base import get_game
from vexga.store.db import connect
from vexga.store.writer import store_track_result
from vexga.track.motion import MotionRobotDetector, background_model
from vexga.track.process import calibration_for
from vexga.track.tracker import track_match


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--match", type=int, required=True)
    ap.add_argument("--game", default="pushback")
    args = ap.parse_args()

    con = connect()
    m = con.execute(
        "SELECT m.*, v.path FROM matches m JOIN videos v ON v.id=m.video_id WHERE m.id=?",
        (args.match,),
    ).fetchone()
    game = get_game(args.game)
    cal = calibration_for(con, m["video_id"], max(m["video_start_ts"], 0))
    assert cal is not None, "no calibration stored for this video"
    print("building background model...", flush=True)
    bg = background_model(m["path"], max(m["video_start_ts"], 0), m["video_end_ts"])
    det = MotionRobotDetector(bg, cal, game)
    res = track_match(m["path"], max(m["video_start_ts"], 0), m["video_end_ts"], cal, game, det)
    teams = {s: m[s] for s in ("red1", "red2", "blue1", "blue2")}
    store_track_result(con, m["id"], res, teams)
    print(f"match {m['id']} ({m['name']}): quality={res.quality:.2f} {res.notes}")
    for slot, tr in res.slots.items():
        print(f"  {slot} ({teams[slot]}): {len(tr)} samples")


if __name__ == "__main__":
    main()

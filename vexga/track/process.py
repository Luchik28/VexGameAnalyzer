"""Batch-process segmented matches through the tracker."""

from pathlib import Path

from vexga.calibrate.homography import Calibration
from vexga.games.base import get_game
from vexga.store.db import connect
from vexga.store.writer import store_track_result
from vexga.track.tracker import track_match


def calibration_for(con, video_id: str, ts: float) -> Calibration | None:
    row = con.execute(
        "SELECT homography, reproj_err_in FROM calibrations WHERE video_id=? AND from_ts<=?"
        " ORDER BY from_ts DESC LIMIT 1", (video_id, ts)
    ).fetchone()
    return Calibration.from_json(row["homography"], row["reproj_err_in"]) if row else None


def process_matches(weights: str, game_name: str = "pushback",
                    video_id: str | None = None, only_missing: bool = True) -> None:
    con = connect()
    game = get_game(game_name)
    q = ("SELECT m.id, m.video_id, m.video_start_ts, m.video_end_ts,"
         " m.red1, m.red2, m.blue1, m.blue2, v.path FROM matches m"
         " JOIN videos v ON v.id = m.video_id WHERE m.video_end_ts IS NOT NULL")
    args: list = []
    if video_id:
        q += " AND m.video_id = ?"
        args.append(video_id)
    if only_missing:
        q += " AND NOT EXISTS (SELECT 1 FROM robot_tracks rt WHERE rt.match_id = m.id)"
    rows = con.execute(q, args).fetchall()
    print(f"{len(rows)} matches to process")
    for i, m in enumerate(rows):
        cal = calibration_for(con, m["video_id"], m["video_start_ts"])
        if cal is None:
            print(f"  match {m['id']}: no calibration for {m['video_id']} - run vexga.calibrate.tool")
            continue
        if not Path(m["path"]).exists():
            print(f"  match {m['id']}: missing video file {m['path']}")
            continue
        res = track_match(m["path"], m["video_start_ts"], m["video_end_ts"], cal, game, weights)
        teams = {"red1": m["red1"], "red2": m["red2"], "blue1": m["blue1"], "blue2": m["blue2"]}
        store_track_result(con, m["id"], res, teams)
        print(f"  [{i+1}/{len(rows)}] match {m['id']}: quality={res.quality:.2f} ({res.notes})")

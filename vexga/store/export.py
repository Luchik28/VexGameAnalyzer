"""Exports: parquet tables for analytics, per-match JSON for the viewer."""

import json
from pathlib import Path

import polars as pl

from vexga.config import EXPORTS
from vexga.games.base import get_game
from vexga.store.db import connect

TABLES = ("events", "matches", "robot_tracks", "zone_states", "score_timeline", "team_robots")


def to_parquet(out_dir: Path = EXPORTS) -> None:
    con = connect()
    for t in TABLES:
        rows = [dict(r) for r in con.execute(f"SELECT * FROM {t}").fetchall()]
        if rows:
            pl.DataFrame(rows).write_parquet(out_dir / f"{t}.parquet")
            print(f"{t}: {len(rows)} rows")


def match_json(match_id: int, game_name: str = "pushback") -> dict:
    """Everything the replay viewer needs for one match."""
    con = connect()
    m = con.execute("SELECT * FROM matches WHERE id = ?", (match_id,)).fetchone()
    if m is None:
        raise KeyError(f"no match {match_id}")
    game = get_game(game_name)
    tracks: dict[str, list] = {}
    for r in con.execute(
        "SELECT t, slot, x_in, y_in, conf FROM robot_tracks WHERE match_id=? ORDER BY t", (match_id,)
    ):
        tracks.setdefault(r["slot"], []).append([r["t"], r["x_in"], r["y_in"], r["conf"]])
    zones: dict[str, list] = {}
    for r in con.execute(
        "SELECT t, zone, red_blocks, blue_blocks FROM zone_states WHERE match_id=? ORDER BY t", (match_id,)
    ):
        zones.setdefault(r["zone"], []).append([r["t"], r["red_blocks"], r["blue_blocks"]])
    scores = [[r["t"], r["red_score"], r["blue_score"]] for r in con.execute(
        "SELECT t, red_score, blue_score FROM score_timeline WHERE match_id=? ORDER BY t", (match_id,))]
    return {
        "match": {k: m[k] for k in m.keys()},
        "game": {
            "name": game.name,
            "field_size": game.field_size,
            "auton_seconds": game.auton_seconds,
            "zones": [
                {"name": z.name, "kind": z.kind, "polygon": list(map(list, z.polygon))}
                for z in game.zones
            ],
        },
        "tracks": tracks,
        "zone_states": zones,
        "score_timeline": scores,
        "youtube": _youtube_ref(con, m),
    }


def _youtube_ref(con, m) -> dict:
    """YouTube id + timestamp for a match, resolving clip videos back to
    their source VOD (clips note their origin as 'src=<vid>@<ts>')."""
    import re

    vid = m["video_id"] or ""
    ts = m["video_start_ts"]
    src = con.execute("SELECT source_id FROM videos WHERE id=?", (vid,)).fetchone()
    if src and src["source_id"]:
        hit = re.search(r"src=(\S+)@(\d+)", m["notes"] or "")
        if hit:
            vid, ts = hit.group(1), float(hit.group(2))
        else:
            vid = src["source_id"]
    # Strip local suffixes: section ranges (_10000_12000) and stray format
    # parts (.f398) so the id is embeddable.
    return {"video_id": vid.split("_")[0].split(".")[0], "start_ts": ts}


def export_match_json(match_id: int, out_dir: Path | None = None) -> Path:
    out_dir = out_dir or (EXPORTS / "matches")
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"match_{match_id}.json"
    p.write_text(json.dumps(match_json(match_id)))
    return p

"""Feature extraction: one row per robot-match, one per match.

These feed win-condition analysis, strategy clustering, and scouting reports.
All features are computed from robot_tracks + zone_states + match metadata.
"""

import numpy as np
import polars as pl

from vexga.games.base import GameConfig, get_game
from vexga.store.db import connect


def robot_match_features(game: GameConfig | None = None) -> pl.DataFrame:
    game = game or get_game("pushback")
    con = connect()
    rows = con.execute(
        "SELECT rt.match_id, rt.slot, rt.team, rt.t, rt.x_in, rt.y_in, rt.conf,"
        " m.red_score, m.blue_score, m.event_id"
        " FROM robot_tracks rt JOIN matches m ON m.id = rt.match_id"
        " WHERE rt.conf > 0 ORDER BY rt.match_id, rt.slot, rt.t"
    ).fetchall()
    if not rows:
        return pl.DataFrame()
    df = pl.DataFrame([dict(r) for r in rows])
    F, half = game.field_size, game.field_size / 2
    zone_kinds = {"loader": [], "goal": [], "park": []}
    for kind in zone_kinds:
        zone_kinds[kind] = game.zones_of_kind(kind)

    feats = []
    for (mid, slot), g in df.group_by(["match_id", "slot"], maintain_order=True):
        xy = g.select(["x_in", "y_in"]).to_numpy()
        t = g["t"].to_numpy()
        team = g["team"][0]
        alliance = "red" if str(slot).startswith("red") else "blue"
        red_score, blue_score = g["red_score"][0], g["blue_score"][0]
        won = None
        if red_score is not None and blue_score is not None and red_score != blue_score:
            won = (red_score > blue_score) == (alliance == "red")
        d = np.linalg.norm(np.diff(xy, axis=0), axis=1)
        dt = np.diff(t)
        ok = (dt > 0) & (dt < 2)
        speed = d[ok] / dt[ok] if ok.any() else np.array([0.0])
        auton = xy[t <= game.auton_seconds]
        offside = ((xy[:, 0] > half) if alliance == "red" else (xy[:, 0] < half)).mean()

        def frac_in(zones, pts=xy):
            if len(pts) == 0:
                return 0.0
            return float(np.mean([any(z.contains(x, y) for z in zones) for x, y in pts]))

        park = game.zones_of_kind("park")
        own_park = [z for z in park if alliance in z.name]
        endgame = xy[t >= t.max() - 15] if len(t) else xy
        feats.append({
            "match_id": mid, "slot": str(slot), "team": team, "alliance": alliance,
            "event_id": g["event_id"][0], "won": won,
            "speed_mean": float(speed.mean()), "speed_p90": float(np.percentile(speed, 90)),
            "range_x": float(xy[:, 0].std()), "range_y": float(xy[:, 1].std()),
            "frac_offensive_half": float(offside),
            "frac_near_loader": frac_in(zone_kinds["loader"]),
            "frac_near_goal": frac_in(zone_kinds["goal"]),
            "auton_dist": float(np.linalg.norm(np.diff(auton, axis=0), axis=1).sum()) if len(auton) > 1 else 0.0,
            "endgame_in_own_park": frac_in(own_park, endgame),
            "n_samples": len(xy),
        })
    return pl.DataFrame(feats)


def match_features(game: GameConfig | None = None) -> pl.DataFrame:
    """Alliance-level rows: two per match (red, blue) for win analysis."""
    rm = robot_match_features(game)
    if rm.is_empty():
        return rm
    return (
        rm.group_by(["match_id", "alliance", "event_id"])
        .agg([
            pl.col("won").first(),
            pl.col("speed_mean").mean().alias("speed_mean"),
            pl.col("frac_offensive_half").mean().alias("frac_offensive_half"),
            pl.col("frac_near_loader").mean().alias("frac_near_loader"),
            pl.col("frac_near_goal").mean().alias("frac_near_goal"),
            pl.col("auton_dist").mean().alias("auton_dist"),
            pl.col("endgame_in_own_park").sum().alias("robots_parked_endgame"),
        ])
        .sort(["match_id", "alliance"])
    )

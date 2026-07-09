"""Write tracker output into the DB."""

from vexga.track.tracker import TrackResult


def store_track_result(con, match_id: int, res: TrackResult,
                       teams: dict[str, str | None] | None = None) -> None:
    teams = teams or {}
    con.execute("DELETE FROM robot_tracks WHERE match_id = ?", (match_id,))
    con.execute("DELETE FROM zone_states WHERE match_id = ?", (match_id,))
    con.executemany(
        "INSERT INTO robot_tracks (match_id, t, slot, team, x_in, y_in, conf) VALUES (?,?,?,?,?,?,?)",
        [
            (match_id, t, slot, teams.get(slot), x, y, c)
            for slot, samples in res.slots.items()
            for t, x, y, c in samples
        ],
    )
    con.executemany(
        "INSERT INTO zone_states (match_id, t, zone, red_blocks, blue_blocks) VALUES (?,?,?,?,?)",
        [(match_id, t, z, r, b) for t, z, r, b in res.zone_states],
    )
    con.execute("UPDATE matches SET quality = ?, notes = COALESCE(notes,'') || ' | ' || ? WHERE id = ?",
                (res.quality, res.notes, match_id))
    con.commit()

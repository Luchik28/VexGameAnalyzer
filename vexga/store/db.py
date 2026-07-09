"""SQLite data store. One database for everything; parquet export for
analytics-friendly consumption lives in export.py.

Times: `t` columns are seconds since match start (auton begins at t=0).
Video timestamps (`video_ts`) are seconds into the source video file.
Coordinates are field inches in the canonical frame (see games/base.py).
"""

import sqlite3
from pathlib import Path

from vexga.config import DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY,           -- RobotEvents event id (or negative local id)
    sku TEXT, name TEXT, season TEXT, start_date TEXT
);
CREATE TABLE IF NOT EXISTS videos (
    id TEXT PRIMARY KEY,              -- youtube id (+ optional section suffix)
    event_id INTEGER REFERENCES events(id),
    division TEXT, path TEXT, title TEXT, duration_s REAL, fps REAL,
    width INTEGER, height INTEGER
);
CREATE TABLE IF NOT EXISTS matches (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id INTEGER REFERENCES events(id),
    video_id TEXT REFERENCES videos(id),
    name TEXT,                        -- e.g. "Q12", "SF1-1" (normalized)
    re_match_id INTEGER,              -- RobotEvents match id if joined
    red1 TEXT, red2 TEXT, blue1 TEXT, blue2 TEXT,
    video_start_ts REAL, video_auton_end_ts REAL, video_end_ts REAL,
    red_score INTEGER, blue_score INTEGER,        -- official (RobotEvents)
    ocr_red_score INTEGER, ocr_blue_score INTEGER, -- from overlay/result card
    breakdown TEXT,                   -- result-card scoring table, json
    quality REAL,                     -- 0-1 pipeline quality score
    notes TEXT
);
CREATE TABLE IF NOT EXISTS score_timeline (
    match_id INTEGER REFERENCES matches(id),
    t REAL, red_score INTEGER, blue_score INTEGER
);
CREATE TABLE IF NOT EXISTS calibrations (
    video_id TEXT REFERENCES videos(id),
    from_ts REAL,                     -- valid from this video timestamp
    homography TEXT,                  -- 9 floats, row-major, json list
    reproj_err_in REAL,
    PRIMARY KEY (video_id, from_ts)
);
CREATE TABLE IF NOT EXISTS robot_tracks (
    match_id INTEGER REFERENCES matches(id),
    t REAL,
    slot TEXT,                        -- red1|red2|blue1|blue2
    team TEXT,
    x_in REAL, y_in REAL,
    conf REAL
);
CREATE TABLE IF NOT EXISTS zone_states (
    match_id INTEGER REFERENCES matches(id),
    t REAL, zone TEXT,
    red_blocks INTEGER, blue_blocks INTEGER
);
CREATE TABLE IF NOT EXISTS team_robots (
    event_id INTEGER REFERENCES events(id),
    team TEXT,
    archetype TEXT, archetype_conf REAL,
    PRIMARY KEY (event_id, team)
);
CREATE INDEX IF NOT EXISTS idx_tracks_match ON robot_tracks(match_id, t);
CREATE INDEX IF NOT EXISTS idx_zones_match ON zone_states(match_id, t);
CREATE INDEX IF NOT EXISTS idx_scores_match ON score_timeline(match_id, t);
"""


def connect(path: Path = DB_PATH) -> sqlite3.Connection:
    con = sqlite3.connect(path)
    con.row_factory = sqlite3.Row
    con.executescript(SCHEMA)
    return con

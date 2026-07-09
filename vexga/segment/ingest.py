"""Persist segmentation results and join them against RobotEvents matches."""

import json
import re
from pathlib import Path

import cv2

from vexga.segment.segmenter import MatchSpan, scan_video
from vexga.store.db import connect


def register_video(con, path: Path, event_id: int | None = None,
                   division: str | None = None) -> str:
    vid = path.stem
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    title = ""
    info = path.with_suffix(".info.json")
    if info.exists():
        title = json.loads(info.read_text()).get("title", "")
    con.execute(
        "INSERT OR REPLACE INTO videos (id, event_id, division, path, title, duration_s, fps, width, height)"
        " VALUES (?,?,?,?,?,?,?,?,?)",
        (vid, event_id, division, str(path), title, n / fps if fps else None, fps, w, h),
    )
    con.commit()
    return vid


def store_spans(con, video_id: str, spans: list[MatchSpan],
                cards: list | None = None, event_id: int | None = None) -> list[int]:
    by_name = {_canon(c.match_name): c for c in (cards or [])}
    ids = []
    for sp in spans:
        card = by_name.get(_canon(sp.name))
        red = (sp.top_teams("red", card.red_teams if card else None) + [None] * 2)[:2]
        blue = (sp.top_teams("blue", card.blue_teams if card else None) + [None] * 2)[:2]
        # Result card is authoritative; otherwise median of the last few
        # timeline readings (a single OCR glitch shouldn't set the final).
        ocr_red = card.red_score if card else _tail_median(sp.scores, 1)
        ocr_blue = card.blue_score if card else _tail_median(sp.scores, 2)
        cur = con.execute(
            "INSERT INTO matches (event_id, video_id, name, red1, red2, blue1, blue2,"
            " video_start_ts, video_auton_end_ts, video_end_ts,"
            " ocr_red_score, ocr_blue_score, breakdown, notes)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                event_id, video_id, sp.name, *red, *blue,
                sp.start_ts, sp.auton_end_ts, sp.end_ts,
                ocr_red, ocr_blue,
                json.dumps(card.breakdown) if card and card.breakdown else None,
                sp.notes,
            ),
        )
        mid = cur.lastrowid
        ids.append(mid)
        con.executemany(
            "INSERT INTO score_timeline (match_id, t, red_score, blue_score) VALUES (?,?,?,?)",
            [(mid, t, r, b) for t, r, b in sp.scores],
        )
    con.commit()
    return ids


def _tail_median(scores: list, idx: int, n: int = 5) -> int | None:
    vals = sorted(s[idx] for s in scores[-n:] if s[idx] is not None)
    return vals[len(vals) // 2] if vals else None


def normalize_re_name(re_match: dict) -> str:
    """RobotEvents match -> our normalized name (Q12, QF2-1, SF1, F1...)."""
    rnd = {1: "P", 2: "Q", 3: "QF", 4: "SF", 5: "F", 6: "R16"}.get(re_match["round"], "?")
    inst = re_match.get("instance", 1)
    num = re_match.get("matchnum", 0)
    if rnd == "Q" or rnd == "P":
        return f"{rnd}{num}"
    return f"{rnd}{inst}-{num}" if num else f"{rnd}{inst}"


def join_robotevents(con, event_id: int, division_id: int) -> tuple[int, int]:
    """Attach official teams/scores to segmented matches by normalized name.
    Returns (joined, unjoined) counts."""
    from vexga.acquire.robotevents import event_matches

    re_matches = {normalize_re_name(m): m for m in event_matches(event_id, division_id)}
    rows = con.execute(
        "SELECT id, name FROM matches WHERE event_id = ? AND re_match_id IS NULL", (event_id,)
    ).fetchall()
    joined = 0
    for row in rows:
        m = re_matches.get(_canon(row["name"]))
        if m is None:
            continue
        teams = {"red": [], "blue": []}
        for al in m["alliances"]:
            teams[al["color"]] = [t["team"]["name"] for t in al["teams"]]
        scores = {al["color"]: al["score"] for al in m["alliances"]}
        con.execute(
            "UPDATE matches SET re_match_id=?, red1=?, red2=?, blue1=?, blue2=?,"
            " red_score=?, blue_score=? WHERE id=?",
            (
                m["id"],
                *(teams["red"] + [None, None])[:2],
                *(teams["blue"] + [None, None])[:2],
                scores.get("red"), scores.get("blue"), row["id"],
            ),
        )
        joined += 1
    con.commit()
    return joined, len(rows) - joined


def _canon(name: str | None) -> str:
    if not name:
        return ""
    return re.sub(r"\s+", "", name.upper())


def segment_and_store(video_path: str, event_id: int | None = None,
                      division: str | None = None, period: float = 1.0) -> list[int]:
    path = Path(video_path)
    con = connect()
    vid = register_video(con, path, event_id, division)
    con.execute("DELETE FROM score_timeline WHERE match_id IN (SELECT id FROM matches WHERE video_id=?)", (vid,))
    con.execute("DELETE FROM matches WHERE video_id = ?", (vid,))
    con.commit()
    spans, cards = scan_video(str(path), period=period)
    print(f"{len(spans)} match spans, {len(cards)} result cards in {path.name}")
    return store_spans(con, vid, spans, cards, event_id)

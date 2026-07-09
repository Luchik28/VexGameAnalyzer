"""Layout-aware overlay parsing.

Livestream overlays differ per event (stock TM bottom bar vs. custom side
panels like Kalahari's), but share ingredients: a countdown timer, a match
label, team numbers grouped under red/blue alliance headers, and often live
scores. A locate pass finds these on full frames; afterwards each sample
OCRs only small crops.

Also parses full-screen result cards ("QUALIFICATION 120", big red/blue
scores, a breakdown table: Blocks Scored / Goals Controlled / Parked Robots /
Autonomous Bonus / AWP) which give official-quality per-match scoring.
"""

import re
from dataclasses import dataclass, field

import numpy as np

# OCR confusions in seven-segment-ish timer fonts.
_TIMER_FIX = str.maketrans({"o": "0", "O": "0", "l": "1", "I": "1", "S": "5", "B": "8"})
TIMER_RE = re.compile(r"^([0-2])[:;.]([0-5]\d)$")
TEAM_RE = re.compile(r"^(\d{1,5}[A-Z])$")
INT_RE = re.compile(r"^\d{1,3}$")
MATCH_RES = [
    (re.compile(r"\b(?:QUALIFICATION|QUAL|Q)\s*#?\s*(\d+)\b", re.I), "Q", False),
    (re.compile(r"\b(?:ROUND\s*OF\s*16|R16)\s*#?\s*(\d+)(?:\s*-\s*(\d+))?", re.I), "R16", True),
    (re.compile(r"\b(?:QUARTERFINAL|QF)\s*#?\s*(\d+)(?:\s*-\s*(\d+))?", re.I), "QF", True),
    (re.compile(r"\b(?:SEMIFINAL|SF)\s*#?\s*(\d+)(?:\s*-\s*(\d+))?", re.I), "SF", True),
    (re.compile(r"\b(?:FINAL|F)\s*#?\s*(\d+)(?:\s*-\s*(\d+))?", re.I), "F", True),
    (re.compile(r"\bPRACTICE\s*#?\s*(\d+)", re.I), "P", False),
]
UP_NEXT_RE = re.compile(r"UP\s*NEXT", re.I)
BREAKDOWN_ROWS = {
    "blocks scored": "blocks",
    "long goals controlled": "long_goals",
    "upper goal controlled": "upper_goal",
    "lower goal controlled": "lower_goal",
    "parked robots": "parked",
    "autonomous bonus": "auton_bonus",
    "autonomous win point": "awp",
}

Det = tuple[str, float, tuple[float, float, float, float]]


def parse_timer(text: str) -> int | None:
    m = TIMER_RE.match(text.strip().translate(_TIMER_FIX))
    if not m:
        return None
    secs = int(m.group(1)) * 60 + int(m.group(2))
    return secs if secs <= 120 else None


def parse_match_name(text: str) -> str | None:
    if UP_NEXT_RE.search(text):
        return None  # "UP NEXT: QUAL 119" is a preview, not the running match
    for rx, prefix, elim in MATCH_RES:
        m = rx.search(text)
        if m:
            if elim and m.lastindex and m.lastindex > 1 and m.group(2):
                return f"{prefix}{m.group(1)}-{m.group(2)}"
            return f"{prefix}{m.group(1)}"
    return None


@dataclass
class Regions:
    """Pixel rects (x0, y0, x1, y1) learned by the locate pass."""
    timer: tuple[int, int, int, int]
    label: tuple[int, int, int, int] | None = None
    panel: tuple[int, int, int, int] | None = None


@dataclass
class Sample:
    video_ts: float
    timer: int | None = None
    phase_text: str | None = None       # "AUTONOMOUS" / "DRIVER CONTROL" if shown
    match_name: str | None = None
    red_teams: list[str] = field(default_factory=list)
    blue_teams: list[str] = field(default_factory=list)
    red_score: int | None = None
    blue_score: int | None = None


def _expand(box, w, h, fx=1.5, fy=1.5):
    x0, y0, x1, y1 = box
    bw, bh = (x1 - x0) * fx, (y1 - y0) * fy
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    return (int(max(0, cx - bw)), int(max(0, cy - bh)),
            int(min(w, cx + bw)), int(min(h, cy + bh)))


def locate(dets: list[Det], w: int, h: int) -> Regions | None:
    """Find the overlay regions from one full-frame OCR pass."""
    timer_box = label_box = None
    red_hdr = blue_hdr = None
    for text, _c, box in dets:
        t = text.strip()
        if timer_box is None and parse_timer(t) is not None:
            timer_box = box
        if label_box is None and parse_match_name(t):
            label_box = box
        if re.search(r"RED\s*ALLIANCE", t, re.I):
            red_hdr = box
        if re.search(r"BLUE\s*ALLIANCE", t, re.I):
            blue_hdr = box
    if timer_box is None:
        return None
    regions = Regions(timer=_expand(timer_box, w, h, 1.6, 2.2))
    if label_box is not None:
        regions.label = _expand(label_box, w, h, 1.8, 1.6)
    if red_hdr is not None and blue_hdr is not None:
        x0 = int(max(0, min(red_hdr[0], blue_hdr[0]) - 0.02 * w))
        y0 = int(max(0, min(red_hdr[1], blue_hdr[1]) - 0.02 * h))
        regions.panel = (x0, y0, w, h)
    return regions


def parse_sample(ts: float, timer_dets: list[Det], label_dets: list[Det],
                 panel_dets: list[Det]) -> Sample:
    s = Sample(video_ts=ts)
    for text, _c, _b in timer_dets:
        t = parse_timer(text)
        if t is not None and s.timer is None:
            s.timer = t
        if re.search(r"AUTON", text, re.I):
            s.phase_text = "auton"
        elif re.search(r"DRIVER", text, re.I):
            s.phase_text = "driver"
    for text, _c, _b in label_dets:
        name = parse_match_name(text)
        if name:
            s.match_name = name
            break

    # Panel: rows grouped by alliance header y-position. Team rows carry a
    # small rank box; the alliance's live score is the remaining standalone
    # number below its last team row.
    red_y = blue_y = None
    for text, _c, (x0, y0, x1, y1) in panel_dets:
        if re.search(r"RED\s*ALLIANCE", text, re.I):
            red_y = y0
        elif re.search(r"BLUE\s*ALLIANCE", text, re.I):
            blue_y = y0
    if red_y is None and blue_y is None:
        return s

    def section(y: float) -> str | None:
        if red_y is not None and blue_y is not None:
            if red_y < blue_y:
                return "red" if red_y <= y < blue_y else ("blue" if y >= blue_y else None)
            return "blue" if blue_y <= y < red_y else ("red" if y >= red_y else None)
        return None

    team_rows: dict[str, list[float]] = {"red": [], "blue": []}
    numbers: list[tuple[str, float, float, float]] = []  # (val, x0, y0, height)
    for text, _c, (x0, y0, x1, y1) in panel_dets:
        t = text.strip()
        sec = section(y0)
        if sec is None:
            continue
        if TEAM_RE.match(t):
            getattr(s, f"{sec}_teams").append(t)
            team_rows[sec].append(y0)
        elif INT_RE.match(t):
            numbers.append((t, x0, y0, y1 - y0))
    for sec in ("red", "blue"):
        rows = team_rows[sec]
        if not rows:
            continue  # no team rows visible = score-entry/idle panel state;
                      # standalone numbers there are ranks, not live scores
        cands = []
        for val, x0, y0, hh in numbers:
            if section(y0) != sec:
                continue
            near_team = any(abs(y0 - ty) < hh * 1.2 for ty in rows)
            if not near_team:      # rank boxes sit on team rows; scores don't
                cands.append((hh, int(val)))
        if cands:
            cands.sort(reverse=True)  # live score is the biggest text
            setattr(s, f"{sec}_score", cands[0][1])
    return s


@dataclass
class ResultCard:
    match_name: str
    red_score: int
    blue_score: int
    breakdown: dict[str, tuple[int, int]] = field(default_factory=dict)
    red_teams: list[str] = field(default_factory=list)
    blue_teams: list[str] = field(default_factory=list)


def parse_result_card(dets: list[Det], w: int, h: int) -> ResultCard | None:
    """Parse a full-screen result card if this frame is one."""
    name = None
    rows: dict[str, tuple[float, float]] = {}
    for text, _c, (x0, y0, x1, y1) in dets:
        t = text.strip().lower()
        if name is None:
            name = parse_match_name(text)
        for key, slug in BREAKDOWN_ROWS.items():
            if key in t:
                rows[slug] = (y0, y1)
    if name is None or "blocks" not in rows:
        return None
    card = ResultCard(match_name=name, red_score=-1, blue_score=-1)
    nums: list[tuple[float, float, float, int]] = []  # (height, x0, yc, val)
    for text, _c, (x0, y0, x1, y1) in dets:
        t = text.strip()
        if INT_RE.match(t):
            nums.append((y1 - y0, x0, (y0 + y1) / 2, int(t)))
        elif TEAM_RE.match(t.split(" ")[0]):
            (card.red_teams if x0 < w / 2 else card.blue_teams).append(t.split(" ")[0])
    if not nums:
        return None
    # Final scores first: the two tallest numbers (they dwarf table text and
    # rank boxes), one per half. Everything matched here is excluded from the
    # breakdown-table pass so a big score overlapping a row's y-band doesn't
    # get misfiled.
    med_h = sorted(n[0] for n in nums)[len(nums) // 2]
    used: set[int] = set()
    for i, (hh, x, _yc, v) in sorted(enumerate(nums), key=lambda kv: -kv[1][0]):
        if hh < 1.5 * med_h:
            break
        if x < w / 3 and card.red_score < 0:
            card.red_score, _ = v, used.add(i)
        elif x > 2 * w / 3 and card.blue_score < 0:
            card.blue_score, _ = v, used.add(i)
    # Breakdown values live in the center table, flanking the row label.
    for i, (hh, x, yc, v) in enumerate(nums):
        if i in used or not (0.2 * w < x < 0.8 * w):
            continue
        for slug, (ry0, ry1) in rows.items():
            if ry0 - 5 <= yc <= ry1 + 5:
                cur = card.breakdown.get(slug, (None, None))
                card.breakdown[slug] = ((v, cur[1]) if x < w / 2 else (cur[0], v))
                break
    if card.red_score < 0 or card.blue_score < 0:
        return None
    return card

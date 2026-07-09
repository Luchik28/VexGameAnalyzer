"""Split a tournament VOD into matches by OCR'ing the stream overlay.

Pipeline per video:
1. Locate pass: full-frame OCR every few seconds until the overlay's regions
   (timer, match label, alliance panel) are found -> overlay.Regions.
2. Sample pass: every `period` seconds OCR only those small crops; parse into
   overlay.Sample. When idle (no timer), occasionally re-OCR the full frame
   to catch full-screen result cards and camera/layout changes.
3. MatchStateMachine turns timer/phase readings into MatchSpans:

    IDLE --(timer <= 0:15 decreasing, or phase AUTONOMOUS)--> AUTON
    AUTON --(timer jumps up toward driver length)-----------> DRIVER
    DRIVER --(0:00 / vanished / new label)-------------------> IDLE

Result cards (final score + scoring breakdown) are matched to spans by
match name in ingest.
"""

from dataclasses import dataclass, field

import cv2

from vexga.segment.overlay import (Regions, ResultCard, Sample, locate,
                                   parse_result_card, parse_sample)


@dataclass
class MatchSpan:
    name: str | None
    start_ts: float                 # video ts of auton start (t=0)
    auton_end_ts: float | None
    end_ts: float | None
    red_teams: list[str] = field(default_factory=list)
    blue_teams: list[str] = field(default_factory=list)
    scores: list[tuple[float, int | None, int | None]] = field(default_factory=list)
    result: ResultCard | None = None
    notes: str = ""


class MatchStateMachine:
    IDLE, AUTON, DRIVER = "idle", "auton", "driver"

    def __init__(self, auton_seconds: int = 15, driver_seconds: int = 105) -> None:
        self.auton_s = auton_seconds
        self.driver_s = driver_seconds
        self.state = self.IDLE
        self.current: MatchSpan | None = None
        self.done: list[MatchSpan] = []
        self._prev: Sample | None = None

    def _finish(self, end_ts: float | None) -> None:
        if self.current is not None:
            self.current.end_ts = end_ts
            self.done.append(self.current)
        self.current = None
        self.state = self.IDLE

    def feed(self, s: Sample) -> None:
        prev = self._prev
        self._prev = s
        if s.timer is None:
            if self.state == self.DRIVER and prev is not None and prev.timer is not None:
                projected_end = prev.video_ts + prev.timer
                if s.video_ts > projected_end + 10:
                    self._finish(projected_end)
            elif (self.state == self.AUTON and self.current is not None
                    and s.video_ts - self.current.start_ts > self.auton_s + 15):
                self.current = None  # overlay vanished mid-auton: stale start
                self.state = self.IDLE
            return

        decreasing = prev is not None and prev.timer is not None and s.timer < prev.timer

        # A different match label while one is running: missed the boundary
        # (stream cut) - close at projected end, reprocess from IDLE.
        if (self.current is not None and s.match_name and self.current.name
                and s.match_name != self.current.name):
            projected = (prev.video_ts + prev.timer
                         if prev is not None and prev.timer is not None else s.video_ts)
            self._finish(min(projected, s.video_ts))

        if self.state == self.IDLE:
            in_auton = s.phase_text == "auton" or s.timer <= self.auton_s
            in_driver = s.phase_text == "driver" or s.timer > self.auton_s
            # timer >= 5: the overlay ticks 3-2-1 during post-match score
            # finalization, which would otherwise look like an auton start.
            if decreasing and in_auton and 5 <= s.timer <= self.auton_s:
                start = s.video_ts - (self.auton_s - s.timer)
                self.current = MatchSpan(s.match_name, start, None, None)
                self.state = self.AUTON
            elif decreasing and in_driver and s.timer > self.auton_s:
                # Joined mid-driver (missed auton or driver-only replay).
                driver_elapsed = self.driver_s - s.timer
                start = s.video_ts - driver_elapsed - self.auton_s
                self.current = MatchSpan(s.match_name, start, s.video_ts - driver_elapsed, None)
                self.current.notes = "auton-missed"
                self.state = self.DRIVER

        elif self.state == self.AUTON:
            assert self.current is not None
            if s.timer > self.auton_s:  # jumped to the driver countdown
                auton_end = s.video_ts - (self.driver_s - s.timer)
                self.current.auton_end_ts = auton_end
                # Re-anchor: the driver jump pins t=0 far more reliably than
                # the (noise-prone) auton entry did.
                expected_start = auton_end - self.auton_s
                delta = expected_start - self.current.start_ts
                if abs(delta) > 3:
                    self.current.start_ts = expected_start
                    self.current.scores = [(t - delta, r, b)
                                           for t, r, b in self.current.scores if t - delta >= 0]
                self.state = self.DRIVER
            elif s.video_ts - self.current.start_ts > self.auton_s + 15:
                self.current = None  # stuck: never reached driver
                self.state = self.IDLE
            elif prev is not None and prev.timer is not None and s.timer > prev.timer:
                self.current = None  # aborted / reset before driver
                self.state = self.IDLE

        elif self.state == self.DRIVER:
            assert self.current is not None
            if s.timer == 0:
                self._absorb(s)  # keep the final score reading
                self._finish(s.video_ts)
            elif prev is not None and prev.timer is not None and s.timer > prev.timer + 2:
                self._finish(prev.video_ts + prev.timer)
                self.feed(s)
                return

        if self.current is not None:
            self._absorb(s)

    def _absorb(self, s: Sample) -> None:
        assert self.current is not None
        if s.match_name and not self.current.name:
            self.current.name = s.match_name
        for team in s.red_teams:
            if team not in self.current.red_teams:
                self.current.red_teams.append(team)
        for team in s.blue_teams:
            if team not in self.current.blue_teams:
                self.current.blue_teams.append(team)
        if s.red_score is not None or s.blue_score is not None:
            self.current.scores.append(
                (s.video_ts - self.current.start_ts, s.red_score, s.blue_score))


def _crop_dets(ocr, frame, region):
    x0, y0, x1, y1 = region
    dets = ocr.read(frame[y0:y1, x0:x1])
    return [(t, c, (bx0 + x0, by0 + y0, bx1 + x0, by1 + y0))
            for t, c, (bx0, by0, bx1, by1) in dets]


def scan_video(path: str, period: float = 1.0, ocr=None,
               idle_full_every: float = 5.0,
               progress_every: float = 600.0) -> tuple[list[MatchSpan], list[ResultCard]]:
    from vexga.segment.ocr import get_ocr

    ocr = ocr or get_ocr()
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    duration = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / fps
    h, w = None, None

    regions: Regions | None = None
    sm = MatchStateMachine()
    cards: list[ResultCard] = []
    seen_cards: set[str] = set()
    ts, last_full = 0.0, -1e9
    next_report = progress_every
    while ts < duration:
        cap.set(cv2.CAP_PROP_POS_MSEC, ts * 1000)
        ok, frame = cap.read()
        if not ok:
            break
        if h is None:
            h, w = frame.shape[:2]

        sample = None
        if regions is not None:
            timer_dets = _crop_dets(ocr, frame, regions.timer)
            label_dets = _crop_dets(ocr, frame, regions.label) if regions.label else []
            panel_dets = _crop_dets(ocr, frame, regions.panel) if regions.panel else []
            sample = parse_sample(ts, timer_dets, label_dets, panel_dets)

        need_full = regions is None or (
            (sample is None or sample.timer is None) and ts - last_full >= idle_full_every)
        if need_full:
            last_full = ts
            full = ocr.read(frame)
            card = parse_result_card(full, w, h)
            if card is not None and card.match_name not in seen_cards:
                seen_cards.add(card.match_name)
                cards.append(card)
            found = locate(full, w, h)
            if found is not None:
                # Keep previously learned auxiliary regions if the new pass
                # found fewer (label/panel hide between matches).
                if regions is not None:
                    found.label = found.label or regions.label
                    found.panel = found.panel or regions.panel
                regions = found
            if sample is None:
                sample = parse_sample(ts, full, full, full)
        sm.feed(sample)

        ts += period
        if ts >= next_report:
            print(f"  scanned {ts/60:.0f}/{duration/60:.0f} min: "
                  f"{len(sm.done)} matches, {len(cards)} result cards")
            next_report += progress_every
    cap.release()
    if sm.state == MatchStateMachine.DRIVER and sm.current is not None:
        sm._finish(None)
    return sm.done, cards

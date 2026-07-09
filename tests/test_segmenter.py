"""State machine + overlay parsing unit tests (no video needed)."""

from vexga.segment.overlay import Sample, parse_match_name, parse_timer
from vexga.segment.segmenter import MatchStateMachine


def feed_seq(sm, seq, t0=0.0):
    ts = t0
    for kw in seq:
        sm.feed(Sample(video_ts=ts, **kw))
        ts += 1.0
    return ts


def test_timer_parsing():
    assert parse_timer("1:45") == 105
    assert parse_timer("0:15") == 15
    assert parse_timer("o:00") == 0          # OCR confusion
    assert parse_timer("O;O5") == 5
    assert parse_timer("12:45") is None      # stream clock, not match timer
    assert parse_timer("row 1") is None


def test_match_name_parsing():
    assert parse_match_name("QUAL 118") == "Q118"
    assert parse_match_name("Qualification 12") == "Q12"
    assert parse_match_name("SF 1-2") == "SF1-2"
    assert parse_match_name("FINAL 1") == "F1"
    assert parse_match_name("UP NEXT: QUAL 119") is None


def test_full_match():
    sm = MatchStateMachine()
    seq = ([{"timer": 15, "match_name": "Q3"}] * 5
           + [{"timer": t, "match_name": "Q3", "phase_text": "auton"} for t in range(14, -1, -1)]
           + [{"timer": t, "match_name": "Q3", "phase_text": "driver",
               "red_teams": ["1234A"], "blue_teams": ["5678B"],
               "red_score": 2 * (105 - t) // 10, "blue_score": (105 - t) // 10}
              for t in range(105, -1, -1)]
           + [{}] * 5)
    feed_seq(sm, seq)
    assert len(sm.done) == 1
    m = sm.done[0]
    assert m.name == "Q3"
    assert m.top_teams("red") == ["1234A"] and m.top_teams("blue") == ["5678B"]
    # A rare misread loses the vote to the dominant correct read.
    m.red_teams.extend(["12344A", "1234A", "1234A"])
    assert m.top_teams("red")[0] == "1234A"
    assert abs((m.auton_end_ts - m.start_ts) - 15) <= 2
    assert abs((m.end_ts - m.start_ts) - 120) <= 2
    assert m.scores[-1][1] == 21 and m.scores[-1][2] == 10


def test_stream_cut_mid_driver():
    sm = MatchStateMachine()
    seq = ([{"timer": t, "match_name": "Q5", "phase_text": "driver"} for t in range(100, 40, -1)]
           + [{"timer": t, "match_name": "Q6", "phase_text": "auton"} for t in range(14, 5, -1)])
    feed_seq(sm, seq)
    assert sm.done and sm.done[0].name == "Q5" and sm.done[0].notes == "auton-missed"
    assert sm.current is not None and sm.current.name == "Q6" and sm.state == "auton"


def test_static_timer_never_starts():
    sm = MatchStateMachine()
    feed_seq(sm, [{"timer": 15, "match_name": "Q1"}] * 30)
    assert sm.state == "idle" and not sm.done and sm.current is None


def test_finalization_ticks_do_not_start_match():
    # After a match the overlay ticks 3-2-1-0 while scores are finalized.
    sm = MatchStateMachine()
    feed_seq(sm, [{"timer": t, "match_name": "Q9"} for t in (3, 2, 1, 0)] + [{}] * 10)
    assert sm.state == "idle" and not sm.done and sm.current is None


def test_start_reanchored_on_driver_jump():
    # Noisy auton entry at a wrong time gets re-anchored by the driver jump.
    sm = MatchStateMachine()
    seq = ([{"timer": 14, "match_name": "Q8"}, {"timer": 12, "match_name": "Q8"}]
           + [{"timer": 12, "match_name": "Q8"}] * 20          # stalled reads
           + [{"timer": t, "match_name": "Q8", "phase_text": "driver"}
              for t in range(105, 90, -1)])
    feed_seq(sm, seq)
    # entered at ts=1 (timer 14 -> start=0); driver 105 first seen at ts=22
    # -> auton_end=22, re-anchored start = 7.
    assert sm.state == "driver" and sm.current is not None
    assert abs(sm.current.start_ts - (22 - 15)) <= 1


def test_stuck_auton_discarded():
    sm = MatchStateMachine()
    seq = ([{"timer": 14, "match_name": "Q9"}, {"timer": 13, "match_name": "Q9"}]
           + [{"timer": None}] * 40)
    feed_seq(sm, seq)
    assert not sm.done  # never reached driver; nothing recorded



if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"{name} OK")

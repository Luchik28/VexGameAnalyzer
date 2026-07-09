"""V5RC Override (2026-27) game configuration - SKELETON.

Fill in from the Override Game Manual Appendix A once field drawings are
published, following the same canonical frame convention as pushback.py
(origin bottom-left of audience view, x right, y up, inches). Everything the
pipeline needs for a new season lives in this one file plus a relabeled
detection dataset (see docs/new-season-runbook.md).
"""

from vexga.games.base import GameConfig, register

FIELD = 140.4

OVERRIDE = register(
    GameConfig(
        name="override",
        season="2026-27",
        field_size=FIELD,
        auton_seconds=15,     # TODO: confirm from Override manual
        driver_seconds=105,   # TODO: confirm from Override manual
        detect_classes=("robot_red", "robot_blue"),  # TODO: add game objects
        zones=(),             # TODO: goal/park/loader footprints from Appendix A
        landmarks=(),         # TODO: calibration landmarks
        start_positions={},   # TODO: legal starting tiles per slot
        archetypes=(),        # TODO: fill in as the meta develops
    )
)

"""V5RC Push Back (2025-26) game configuration.

Geometry from the Game Manual v0.1 field reference drawing (Appendix A, page
A-5, "Field Reference Specifications"). Canonical frame = that drawing's
audience view: origin at the bottom-left interior corner of the field floor,
x to the right (red wall -> blue wall), y away from the bottom wall, inches.

Key facts (manual v0.1):
- Interior floor 140.4" x 140.4" (6x6 tiles inside the perimeter).
- Red alliance station + park zone on the LEFT wall, blue on the RIGHT wall;
  the vertical autonomous line sits at x = 70.2.
- Two Long Goals (48.8" long, 13.33" enclosed center section, 15 blocks max)
  run horizontally, centered on x = 70.2, at y = 23.44 (near) and y = 116.97
  (far). Control zones: two open ends + enclosed center per goal.
- Center Goals (22.6" each, 7 blocks max) form an X at field center, one
  Upper and one Lower, on the two 45-degree diagonals.
- Park zones (18.87" wide along wall x 16.86" deep) centered vertically on
  each side wall: y from 60.77 to 79.64, extending 16.88" into the field.
- Four loaders (21.34" tall) on the top and bottom walls at x = 16.88 (red's)
  and x = 123.57 (blue's).
- Match: 15 s autonomous + 105 s driver control.
- Scoring: block scored 3; long-goal control zone 10 each; center upper 8,
  lower 6; 1 parked robot 8, 2 parked 30; auton bonus 10 (tie 5+5).
"""

from vexga.games.base import GameConfig, Landmark, Zone, register

FIELD = 140.4
CX = FIELD / 2  # 70.2

_LG_X0, _LG_X1 = 45.83, 94.62  # long goal x span
_LG_Y_NEAR, _LG_Y_FAR = 23.44, 116.97  # long goal centerlines
_LG_HALF_W = 2.5  # goal footprint half-width (approx)

_PZ_Y0, _PZ_Y1 = 60.77, 79.64  # park zone extent along side walls
_PZ_DEPTH = 16.88

_CG_HALF = 22.6 / 2  # center goal half-length
_CG_HALF_W = 2.5


def _rect(x0: float, y0: float, x1: float, y1: float) -> tuple[tuple[float, float], ...]:
    return ((x0, y0), (x1, y0), (x1, y1), (x0, y1))


def _diag_goal(sign: int) -> tuple[tuple[float, float], ...]:
    """Footprint of one center goal: a thin rectangle along a 45-deg diagonal
    through field center. sign=+1 for the /-diagonal, -1 for the \\-diagonal."""
    import math

    ux, uy = math.sqrt(0.5), sign * math.sqrt(0.5)  # unit vector along goal
    px, py = -uy, ux  # unit normal
    l, w = _CG_HALF, _CG_HALF_W
    return (
        (CX - l * ux - w * px, CX - l * uy - w * py),
        (CX + l * ux - w * px, CX + l * uy - w * py),
        (CX + l * ux + w * px, CX + l * uy + w * py),
        (CX - l * ux + w * px, CX - l * uy + w * py),
    )


ZONES = (
    # Long goals: three control sections each (red-side open, enclosed center,
    # blue-side open). Section splits at the 13.33" enclosed center.
    Zone("long_near_red", _rect(_LG_X0, _LG_Y_NEAR - _LG_HALF_W, CX - 13.33 / 2, _LG_Y_NEAR + _LG_HALF_W), "goal"),
    Zone("long_near_center", _rect(CX - 13.33 / 2, _LG_Y_NEAR - _LG_HALF_W, CX + 13.33 / 2, _LG_Y_NEAR + _LG_HALF_W), "goal"),
    Zone("long_near_blue", _rect(CX + 13.33 / 2, _LG_Y_NEAR - _LG_HALF_W, _LG_X1, _LG_Y_NEAR + _LG_HALF_W), "goal"),
    Zone("long_far_red", _rect(_LG_X0, _LG_Y_FAR - _LG_HALF_W, CX - 13.33 / 2, _LG_Y_FAR + _LG_HALF_W), "goal"),
    Zone("long_far_center", _rect(CX - 13.33 / 2, _LG_Y_FAR - _LG_HALF_W, CX + 13.33 / 2, _LG_Y_FAR + _LG_HALF_W), "goal"),
    Zone("long_far_blue", _rect(CX + 13.33 / 2, _LG_Y_FAR - _LG_HALF_W, _LG_X1, _LG_Y_FAR + _LG_HALF_W), "goal"),
    # Center goals share one X footprint on the floor; upper/lower can't be
    # separated in floor coordinates, so the pipeline treats proximity to the
    # X as "at center goals" and relies on the score timeline for upper/lower.
    Zone("center_upper", _diag_goal(+1), "goal"),
    Zone("center_lower", _diag_goal(-1), "goal"),
    Zone("park_red", _rect(0, _PZ_Y0, _PZ_DEPTH, _PZ_Y1), "park"),
    Zone("park_blue", _rect(FIELD - _PZ_DEPTH, _PZ_Y0, FIELD, _PZ_Y1), "park"),
    # Loader approach areas: one tile-ish square in front of each loader.
    Zone("loader_red_near", _rect(4.0, 0.0, 30.0, 14.0), "loader"),
    Zone("loader_red_far", _rect(4.0, FIELD - 14.0, 30.0, FIELD), "loader"),
    Zone("loader_blue_near", _rect(FIELD - 30.0, 0.0, FIELD - 4.0, 14.0), "loader"),
    Zone("loader_blue_far", _rect(FIELD - 30.0, FIELD - 14.0, FIELD, FIELD), "loader"),
)

LANDMARKS = (
    Landmark("corner_bl", 0, 0),
    Landmark("corner_br", FIELD, 0),
    Landmark("corner_tr", FIELD, FIELD),
    Landmark("corner_tl", 0, FIELD),
    Landmark("long_near_left_end", _LG_X0, _LG_Y_NEAR),
    Landmark("long_near_right_end", _LG_X1, _LG_Y_NEAR),
    Landmark("long_far_left_end", _LG_X0, _LG_Y_FAR),
    Landmark("long_far_right_end", _LG_X1, _LG_Y_FAR),
    Landmark("field_center", CX, CX),
    Landmark("auton_line_near", CX, 0),
    Landmark("auton_line_far", CX, FIELD),
    Landmark("park_red_outer_near", _PZ_DEPTH, _PZ_Y0),
    Landmark("park_red_outer_far", _PZ_DEPTH, _PZ_Y1),
    Landmark("park_blue_outer_near", FIELD - _PZ_DEPTH, _PZ_Y0),
    Landmark("park_blue_outer_far", FIELD - _PZ_DEPTH, _PZ_Y1),
)

PUSHBACK = register(
    GameConfig(
        name="pushback",
        season="2025-26",
        field_size=FIELD,
        auton_seconds=15,
        driver_seconds=105,
        detect_classes=("robot_red", "robot_blue", "block_red", "block_blue"),
        zones=ZONES,
        landmarks=LANDMARKS,
        # Robots start contacting their park zone barrier; slot 1 = the robot
        # nearer the far wall. Approximate seeds for identity assignment.
        start_positions={
            "red1": (10.0, 88.0),
            "red2": (10.0, 52.0),
            "blue1": (FIELD - 10.0, 88.0),
            "blue2": (FIELD - 10.0, 52.0),
        },
        # Community archetype taxonomy (editable; seed-labeling UI reads this).
        archetypes=(
            "ruigan",       # continuous flywheel/roller shooter
            "s-bot",        # S-shaped conveyor path bot
            "basket-bot",   # accumulates blocks in a basket, dumps at goals
            "pusher",       # plow/defense oriented
            "hybrid",
            "other",
        ),
    )
)

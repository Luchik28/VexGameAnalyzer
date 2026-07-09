"""Game-agnostic model of a VEX field and game.

All game-specific knowledge (field geometry, object classes, zones, scoring,
match timing, robot archetypes) lives in a GameConfig instance, one per season
(games/pushback.py, games/override.py). Pipeline code must only depend on this
interface so a new season is a new config + relabeled dataset.

Coordinate convention: field-plane coordinates in inches. Origin at the
red-side left corner when viewed from the audience camera, x to the right,
y away from the camera. The calibration tool maps every video into this frame,
so all stored tracks are comparable across events.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Zone:
    """A named region of the field floor (goal footprint, park zone, loader
    approach, etc.). Polygon vertices in field inches, closed implicitly."""

    name: str
    polygon: tuple[tuple[float, float], ...]
    kind: str  # "goal" | "park" | "loader" | "sector"

    def contains(self, x: float, y: float) -> bool:
        # Ray-casting point-in-polygon.
        inside = False
        pts = self.polygon
        j = len(pts) - 1
        for i in range(len(pts)):
            xi, yi = pts[i]
            xj, yj = pts[j]
            if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi) + xi:
                inside = not inside
            j = i
        return inside


@dataclass(frozen=True)
class Landmark:
    """A visually identifiable field point used to validate calibration."""

    name: str
    x: float
    y: float


@dataclass(frozen=True)
class GameConfig:
    name: str
    season: str
    field_size: float  # inches, square field
    auton_seconds: int
    driver_seconds: int
    # Detector classes, index order == YOLO class ids.
    detect_classes: tuple[str, ...]
    zones: tuple[Zone, ...]
    landmarks: tuple[Landmark, ...]
    # Robot starting tiles per slot at auton start, slot -> (x, y) tile center.
    start_positions: dict[str, tuple[float, float]] = field(default_factory=dict)
    # Known robot archetypes for this game (used by the classifier seed UI).
    archetypes: tuple[str, ...] = ()

    @property
    def match_seconds(self) -> int:
        return self.auton_seconds + self.driver_seconds

    def zones_of_kind(self, kind: str) -> list[Zone]:
        return [z for z in self.zones if z.kind == kind]

    def zone_at(self, x: float, y: float) -> Zone | None:
        for z in self.zones:
            if z.contains(x, y):
                return z
        return None


_REGISTRY: dict[str, "GameConfig"] = {}


def register(cfg: GameConfig) -> GameConfig:
    _REGISTRY[cfg.name] = cfg
    return cfg


def get_game(name: str) -> GameConfig:
    if name not in _REGISTRY:
        # Import side-effect registration.
        from vexga.games import pushback, override  # noqa: F401
    return _REGISTRY[name]

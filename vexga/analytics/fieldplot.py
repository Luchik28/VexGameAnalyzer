"""Shared matplotlib field rendering + palette for analytics figures.

Colors follow the dataviz method: alliance identity is semantic (red/blue
alliances), heatmaps are single-hue sequential ramps, chart chrome stays
recessive. Palette values from the validated reference palette.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Polygon as MplPolygon

from vexga.games.base import GameConfig

INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"
RED = "#e34948"    # categorical slot 6 - red alliance
BLUE = "#2a78d6"   # categorical slot 1 - blue alliance

# Single-hue sequential ramps (surface -> saturated), per alliance.
CMAP_BLUE = LinearSegmentedColormap.from_list(
    "vexga_blue", [SURFACE, "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
CMAP_RED = LinearSegmentedColormap.from_list(
    "vexga_red", [SURFACE, "#fbdad9", "#f2a09e", "#e34948", "#b02a29", "#7c1b1a"])

plt.rcParams.update({
    "font.family": "sans-serif",
    "text.color": INK, "axes.edgecolor": GRID, "axes.labelcolor": INK_2,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.dpi": 150,
})


def draw_field(ax, game: GameConfig, zone_labels: bool = False) -> None:
    """Recessive field outline: tiles, auton line, zone footprints."""
    F = game.field_size
    ax.set_xlim(0, F)
    ax.set_ylim(0, F)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for i in range(7):
        c = i * F / 6
        ax.axvline(c, color=GRID, lw=0.6, zorder=1)
        ax.axhline(c, color=GRID, lw=0.6, zorder=1)
    ax.axvline(F / 2, color=MUTED, lw=1.0, ls=(0, (5, 4)), zorder=2)
    for z in game.zones:
        face = {"park": ("#e34948" if "red" in z.name else "#2a78d6", 0.10),
                "goal": (MUTED, 0.18), "loader": (MUTED, 0.08)}.get(z.kind, (MUTED, 0.05))
        ax.add_patch(MplPolygon(z.polygon, closed=True, facecolor=face[0],
                                alpha=face[1], edgecolor=MUTED, lw=0.7, zorder=2))
        if zone_labels:
            cx = np.mean([p[0] for p in z.polygon])
            cy = np.mean([p[1] for p in z.polygon])
            ax.text(cx, cy, z.name, fontsize=5, color=INK_2, ha="center", va="center")
    for s in ax.spines.values():
        s.set_color(MUTED)


def position_heatmap(ax, xy: np.ndarray, game: GameConfig, alliance: str,
                     bins: int = 36) -> None:
    """Density of positions on the field, single-hue ramp per alliance."""
    F = game.field_size
    h, _, _ = np.histogram2d(xy[:, 0], xy[:, 1], bins=bins, range=[[0, F], [0, F]])
    h = h.T  # histogram2d returns x-major
    if h.max() > 0:
        h = h / h.max()
    cmap = CMAP_RED if alliance == "red" else CMAP_BLUE
    ax.imshow(h, origin="lower", extent=(0, F, 0, F), cmap=cmap,
              interpolation="bilinear", zorder=0, vmin=0, vmax=1)
    draw_field(ax, game)

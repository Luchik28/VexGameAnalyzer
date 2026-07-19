"""Per-team scouting reports: heatmaps + auton routes + tendencies.

    .venv/bin/python -m vexga.analytics.scout --team 1234A
    .venv/bin/python -m vexga.analytics.scout --all   # every tracked team

Writes data/exports/scout/<team>.html with embedded PNGs.
"""

import argparse
import base64
import io
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from vexga.analytics.features import robot_match_features
from vexga.analytics.fieldplot import BLUE, INK, INK_2, MUTED, RED, SURFACE, draw_field, position_heatmap
from vexga.config import EXPORTS
from vexga.games.base import get_game
from vexga.store.db import connect


def _png(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def team_positions(con, team: str) -> dict[int, np.ndarray]:
    """match_id -> (n,3) array of [t, x, y] for one team."""
    out: dict[int, list] = {}
    for r in con.execute(
        "SELECT match_id, t, x_in, y_in FROM robot_tracks WHERE team=? AND conf>0 ORDER BY match_id, t",
        (team,),
    ):
        out.setdefault(r["match_id"], []).append((r["t"], r["x_in"], r["y_in"]))
    return {k: np.array(v) for k, v in out.items()}


def render_team(team: str, game_name: str = "pushback", rm=None) -> Path | None:
    game = get_game(game_name)
    con = connect()
    pos = team_positions(con, team)
    if not pos:
        return None
    all_xy = np.vstack([p[:, 1:] for p in pos.values()])
    alliances = {r["match_id"]: ("red" if r["slot"].startswith("red") else "blue")
                 for r in con.execute(
                     "SELECT DISTINCT match_id, slot FROM robot_tracks WHERE team=?", (team,))}
    main_alliance = max(("red", "blue"),
                        key=lambda a: sum(1 for v in alliances.values() if v == a))

    # Heatmap (all matches) + auton routes overlay.
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.6))
    position_heatmap(axes[0], all_xy, game, main_alliance)
    axes[0].set_title(f"{team} — position density ({len(pos)} matches)",
                      fontsize=9, color=INK)
    draw_field(axes[1], game)
    for mid, p in pos.items():
        auton = p[p[:, 0] <= game.auton_seconds]
        if len(auton) > 1:
            c = RED if alliances.get(mid) == "red" else BLUE
            axes[1].plot(auton[:, 1], auton[:, 2], color=c, lw=1.4, alpha=0.7, zorder=3)
            axes[1].plot(auton[0, 1], auton[0, 2], "o", color=c, ms=4, zorder=4)
    axes[1].set_title("autonomous routes (dot = start)", fontsize=9, color=INK)
    heat_png = _png(fig)

    if rm is None:
        rm = robot_match_features(game)  # callers doing many teams pass it in
    mine = rm.filter(rm["team"] == team) if not rm.is_empty() else rm
    arch = con.execute("SELECT archetype, archetype_conf FROM team_robots WHERE team=?",
                       (team,)).fetchone()
    decided = mine.drop_nulls(subset=["won"]) if not mine.is_empty() else mine
    stats_rows = ""
    if not mine.is_empty():
        wr = f"{decided['won'].mean():.0%} ({len(decided)})" if len(decided) else "n/a"
        stats_rows = "".join(
            f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in {
                "matches tracked": len(mine),
                "win rate (decided)": wr,
                "mean speed (in/s)": f"{mine['speed_mean'].mean():.1f}",
                "time in offensive half": f"{mine['frac_offensive_half'].mean():.0%}",
                "time near loaders": f"{mine['frac_near_loader'].mean():.0%}",
                "time near goals": f"{mine['frac_near_goal'].mean():.0%}",
                "endgame in own park zone": f"{mine['endgame_in_own_park'].mean():.0%}",
            }.items())

    out_dir = EXPORTS / "scout"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{team.replace('/', '_')}.html"
    out.write_text(f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>{team} scouting report</title>
<style>body{{font:14px/1.5 -apple-system,sans-serif;background:{SURFACE};color:{INK};
max-width:960px;margin:24px auto;padding:0 16px}}
h1{{font-size:20px}} .arch{{color:{INK_2}}} img{{max-width:100%}}
table{{border-collapse:collapse}} td{{padding:3px 10px;border-bottom:1px solid #e1e0d9}}
td:first-child{{color:{MUTED}}}</style></head><body>
<h1>{team} — scouting report</h1>
<p class="arch">robot archetype: <b>{arch['archetype'] if arch else 'unclassified'}</b>
{f"({arch['archetype_conf']:.0%} conf)" if arch else ""}</p>
<img src="data:image/png;base64,{heat_png}">
<h2>Tendencies</h2><table>{stats_rows}</table>
</body></html>""")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--team")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--game", default="pushback")
    args = ap.parse_args()
    con = connect()
    teams = ([args.team] if args.team else
             [r["team"] for r in con.execute(
                 "SELECT DISTINCT team FROM robot_tracks WHERE team IS NOT NULL")])
    rm = robot_match_features(get_game(args.game)) if len(teams) > 1 else None
    for t in teams:
        p = render_team(t, args.game, rm=rm)
        print(f"{t}: {p if p else 'no data'}")


if __name__ == "__main__":
    main()

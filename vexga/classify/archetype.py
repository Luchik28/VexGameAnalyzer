"""Robot archetype classification per (event, team).

Features per team:
- Visual: mean-pooled ImageNet ResNet-50 embedding over all the team's crops
  (robust to single bad crops; torchvision ships with ultralytics' deps).
- Behavioral: aggregates from robot_tracks/zone_states over the team's
  matches - fraction of time near loaders / goals / own half, mean speed,
  travel range. These separate playstyles even when crops are blurry.

Training: user labels a seed set of teams (label_tool.py -> team_labels.json),
then LogisticRegression on [visual | behavioral]; per-team predictions with
confidence go to the team_robots table.
"""

import json
from pathlib import Path

import numpy as np

from vexga.config import FRAMES, MODELS
from vexga.games.base import get_game
from vexga.store.db import connect

LABELS_PATH = MODELS / "team_labels.json"


def _embedder():
    import torch
    import torchvision.models as tvm
    import torchvision.transforms as T

    model = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V2)
    model.fc = torch.nn.Identity()
    model.eval()
    tf = T.Compose([T.ToTensor(), T.Resize((224, 224), antialias=True),
                    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    def embed(imgs_bgr: list[np.ndarray]) -> np.ndarray:
        import torch as _t
        with _t.no_grad():
            batch = _t.stack([tf(im[:, :, ::-1].copy()) for im in imgs_bgr])
            return model(batch).numpy()
    return embed


def visual_features(event_id: int, team: str, max_crops: int = 40) -> np.ndarray | None:
    import cv2

    d = FRAMES / "crops" / str(event_id) / team.replace("/", "_")
    paths = sorted(d.glob("*.jpg"))[:max_crops]
    imgs = [im for p in paths if (im := cv2.imread(str(p))) is not None]
    if not imgs:
        return None
    embed = _embedder()
    return embed(imgs).mean(axis=0)


def behavioral_features(con, event_id: int, team: str, game_name: str = "pushback") -> np.ndarray:
    game = get_game(game_name)
    rows = con.execute(
        "SELECT rt.t, rt.x_in, rt.y_in, rt.slot, rt.match_id FROM robot_tracks rt"
        " JOIN matches m ON m.id = rt.match_id"
        " WHERE m.event_id = ? AND rt.team = ? AND rt.conf > 0 ORDER BY rt.match_id, rt.t",
        (event_id, team),
    ).fetchall()
    if not rows:
        return np.zeros(8)
    xy = np.array([(r["x_in"], r["y_in"]) for r in rows])
    t = np.array([r["t"] for r in rows])
    mid = np.array([r["match_id"] for r in rows])
    d = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    dt = np.diff(t)
    same = (mid[1:] == mid[:-1]) & (dt > 0) & (dt < 2)
    speed = d[same] / dt[same] if same.any() else np.array([0.0])
    on_red_side = (xy[:, 0] < game.field_size / 2).mean()
    frac = {"loader": 0.0, "goal": 0.0, "park": 0.0}
    for z in game.zones:
        inside = np.array([z.contains(x, y) for x, y in xy])
        frac[z.kind] = frac.get(z.kind, 0) + inside.mean()
    return np.array([
        speed.mean(), np.percentile(speed, 90), xy[:, 0].std(), xy[:, 1].std(),
        on_red_side, frac["loader"], frac["goal"], frac["park"],
    ])


def team_feature_matrix(event_id: int, teams: list[str]) -> tuple[np.ndarray, list[str]]:
    con = connect()
    feats, kept = [], []
    for team in teams:
        v = visual_features(event_id, team)
        if v is None:
            continue
        b = behavioral_features(con, event_id, team)
        # scale-balance: visual embeddings are ~2048-dim, behavior 8-dim
        feats.append(np.concatenate([v / np.linalg.norm(v), b / (np.abs(b).max() + 1e-6) * 3]))
        kept.append(team)
    return np.array(feats), kept


def train_and_classify(event_id: int, game_name: str = "pushback") -> None:
    """Fit on user-labeled seed teams, classify all teams with crops."""
    from sklearn.linear_model import LogisticRegression

    labels: dict[str, str] = json.loads(LABELS_PATH.read_text()) if LABELS_PATH.exists() else {}
    if len(set(labels.values())) < 2:
        raise RuntimeError(f"need seed labels for >=2 archetypes in {LABELS_PATH} - run label_tool first")
    crops_root = FRAMES / "crops" / str(event_id)
    all_teams = sorted(p.name for p in crops_root.iterdir() if p.is_dir())
    X, kept = team_feature_matrix(event_id, all_teams)
    seed_idx = [i for i, t in enumerate(kept) if t in labels]
    clf = LogisticRegression(max_iter=2000, C=1.0)
    clf.fit(X[seed_idx], [labels[kept[i]] for i in seed_idx])
    proba = clf.predict_proba(X)
    con = connect()
    for i, team in enumerate(kept):
        j = int(np.argmax(proba[i]))
        con.execute(
            "INSERT OR REPLACE INTO team_robots (event_id, team, archetype, archetype_conf)"
            " VALUES (?,?,?,?)",
            (event_id, team, str(clf.classes_[j]), float(proba[i, j])),
        )
    con.commit()
    print(f"classified {len(kept)} teams ({len(seed_idx)} seed-labeled)")

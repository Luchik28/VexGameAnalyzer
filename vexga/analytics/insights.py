"""Win-condition analysis and strategy clustering over extracted features."""

import numpy as np
import polars as pl

from vexga.analytics.features import match_features, robot_match_features

WIN_FEATURES = ["speed_mean", "frac_offensive_half", "frac_near_loader",
                "frac_near_goal", "auton_dist", "robots_parked_endgame"]
CLUSTER_FEATURES = ["speed_mean", "speed_p90", "range_x", "range_y",
                    "frac_offensive_half", "frac_near_loader", "frac_near_goal",
                    "auton_dist", "endgame_in_own_park"]


def win_condition_analysis() -> pl.DataFrame:
    """Standardized logistic regression of win on alliance behavior features.
    Coefficients are directly comparable (features are z-scored)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    mf = match_features().drop_nulls(subset=["won"])
    if len(mf) < 30:
        raise RuntimeError(f"only {len(mf)} decided alliance-matches - need more tracked data")
    X = StandardScaler().fit_transform(mf.select(WIN_FEATURES).to_numpy())
    y = mf["won"].to_numpy().astype(int)
    clf = LogisticRegression(max_iter=1000).fit(X, y)
    acc = float(clf.score(X, y))
    out = pl.DataFrame({
        "feature": WIN_FEATURES,
        "coef": clf.coef_[0],
        "abs_coef": np.abs(clf.coef_[0]),
    }).sort("abs_coef", descending=True)
    print(f"n={len(mf)} alliance-matches, in-sample accuracy {acc:.2f}")
    return out


def strategy_clusters(k: int = 5) -> pl.DataFrame:
    """KMeans playstyle clusters over robot-match features; returns the
    feature table with a cluster column + per-cluster win rates."""
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    rm = robot_match_features()
    if rm.is_empty():
        raise RuntimeError("no robot-match features - run tracking first")
    X = StandardScaler().fit_transform(rm.select(CLUSTER_FEATURES).to_numpy())
    km = KMeans(n_clusters=k, n_init=10, random_state=7).fit(X)
    rm = rm.with_columns(pl.Series("cluster", km.labels_))
    summary = (
        rm.drop_nulls(subset=["won"])
        .group_by("cluster")
        .agg([
            pl.len().alias("n"),
            pl.col("won").mean().alias("win_rate"),
            *[pl.col(f).mean().round(2) for f in CLUSTER_FEATURES],
        ])
        .sort("cluster")
    )
    print(summary)
    return rm

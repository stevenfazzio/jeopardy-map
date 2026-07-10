"""Characterize the EVoC ambient-space regions: inventory, composition, drift.

Consumes data/analysis/evoc_labels.parquet (from 10_evoc_toponymy.py). Per the
Toponymy convention, layer 0 is the FINEST; "Unlabelled" rows are that layer's
unclustered points — a real feature of the space, kept and reported, not noise
to filter.

  1. Layer inventory: region counts, Unlabelled share, size distribution.
  2. On a "story layer" (region count closest to STORY_LAYER_TARGET):
       - Daily-double lift, placement-adjusted (expected DDs per region =
         sum of members' P(DD | round x board_row))
       - Double-Jeopardy share among main-round clues (curriculum gradient)
       - Final Jeopardy enrichment
       - Season trend: change in each region's share of a season's clues,
         expressed as percentage points per decade
  3. Cross-walk to the 2D map's regions (AMI on doubly-labeled rows) — how
     much of the map's region structure survives in the honest space.

Output: printed tables + data/analysis/evoc_region_stats.parquet.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    ANALYSIS_DIR,
    EVOC_LABELS_PARQUET,
    atomic_write_parquet,
    load_clues,
    load_embeddings,
    load_map_labels,
)

STORY_LAYER_TARGET = 60  # pick the layer with about this many regions for tables
MIN_EXPECTED_DD = 20
REGION_STATS_PARQUET = ANALYSIS_DIR / "evoc_region_stats.parquet"


def season_trend_pp_per_decade(sub: pd.DataFrame, region_col: str, region: str) -> tuple[float, float]:
    """OLS slope of region share-of-season over seasons, scaled to pp/decade."""
    per_season = sub.groupby("season", observed=True)[region_col].apply(lambda s: (s == region).mean())
    x = per_season.index.to_numpy(float)
    y = per_season.to_numpy(float)
    x = x - x.mean()
    slope = float((x * (y - y.mean())).sum() / (x**2).sum())
    resid = y - y.mean() - slope * x
    se = float(np.sqrt((resid**2).sum() / max(len(x) - 2, 1) / (x**2).sum()))
    return slope * 10 * 100, (slope / se if se > 0 else np.nan)


def main():
    if not EVOC_LABELS_PARQUET.exists():
        sys.exit(f"{EVOC_LABELS_PARQUET} missing - run analysis/10_evoc_toponymy.py first")
    ids, _ = load_embeddings(normalize=False)
    ev = pd.read_parquet(EVOC_LABELS_PARQUET)
    pos = pd.Index(ev["clue_id"]).get_indexer(ids)
    assert (pos >= 0).all()
    ev = ev.iloc[pos].reset_index(drop=True)

    df = load_clues(ids, columns=["round", "board_row", "daily_double", "season"])
    df = pd.concat([df, ev.drop(columns=["clue_id"])], axis=1)
    label_cols = sorted(c for c in ev.columns if c.startswith("evoc_label_"))

    # --- 1. inventory ---
    print("=== EVoC layer inventory (layer 0 = finest) ===")
    for c in label_cols:
        named = df[c][df[c] != "Unlabelled"]
        sizes = named.value_counts()
        print(
            f"  {c}: {sizes.size:,} regions, {1 - len(named) / len(df):.1%} Unlabelled, "
            f"median size {int(sizes.median()):,}, largest '{sizes.index[0]}' ({sizes.iloc[0]:,})"
        )

    # --- 2. story layer composition ---
    n_regions = {c: df[c].nunique() for c in label_cols}
    story = min(label_cols, key=lambda c: abs(n_regions[c] - STORY_LAYER_TARGET))
    print(f"\n=== story layer: {story} ({n_regions[story]} regions incl. Unlabelled) ===")

    main_rounds = df["round"].isin(["jeopardy", "double_jeopardy"])
    p_cell = (
        df.groupby(["round", df["board_row"].fillna(-1)], observed=True)["daily_double"]
        .transform("mean")
        .fillna(0.0)
        .to_numpy()
    )
    p_dj = (df.loc[main_rounds, "round"] == "double_jeopardy").mean()
    p_fj = (df["round"] == "final_jeopardy").mean()

    g = df.groupby(story, observed=True)
    stats = pd.DataFrame({"n": g.size(), "dd_obs": g["daily_double"].sum().astype(int)})
    stats["dd_exp"] = pd.Series(p_cell, index=df.index).groupby(df[story], observed=True).sum()
    dd_var = pd.Series(p_cell * (1 - p_cell), index=df.index).groupby(df[story], observed=True).sum()
    stats["dd_lift"] = stats["dd_obs"] / stats["dd_exp"]
    stats["dd_z"] = (stats["dd_obs"] - stats["dd_exp"]) / np.sqrt(dd_var)

    mr = df[main_rounds]
    stats["dj_share"] = mr.groupby(story, observed=True)["round"].apply(lambda s: (s == "double_jeopardy").mean())
    n_main = mr.groupby(story, observed=True).size()
    stats["dj_z"] = (stats["dj_share"] - p_dj) * np.sqrt(n_main) / np.sqrt(p_dj * (1 - p_dj))
    stats["fj_lift"] = g["round"].apply(lambda s: (s == "final_jeopardy").mean()) / p_fj

    trends = {r: season_trend_pp_per_decade(df, story, r) for r in stats.index}
    stats["trend_pp_decade"] = {r: t[0] for r, t in trends.items()}
    stats["trend_t"] = {r: t[1] for r, t in trends.items()}

    stats = stats.sort_values("n", ascending=False)
    atomic_write_parquet(stats.reset_index(names="region"), REGION_STATS_PARQUET)

    def show(title, frame, cols):
        print(f"\n--- {title} ---")
        print(frame[cols].round(2).to_string())

    solid = stats[stats["dd_exp"] >= MIN_EXPECTED_DD]
    show("largest regions", stats.head(12), ["n", "dd_lift", "dj_share", "fj_lift", "trend_pp_decade"])
    dd_cols = ["n", "dd_obs", "dd_exp", "dd_lift", "dd_z"]
    show("daily-double hot spots (placement-adjusted)", solid.nlargest(8, "dd_z"), dd_cols)
    show("daily-double dead zones", solid.nsmallest(8, "dd_z"), ["n", "dd_obs", "dd_exp", "dd_lift", "dd_z"])
    show("most Double-Jeopardy-skewed", stats[n_main >= 400].nlargest(8, "dj_z"), ["n", "dj_share", "dj_z", "fj_lift"])
    show("most Jeopardy-round-skewed", stats[n_main >= 400].nsmallest(8, "dj_z"), ["n", "dj_share", "dj_z", "fj_lift"])
    big = stats[stats["n"] >= 800]
    show(
        "fastest-growing regions (pp of a season's clues per decade)",
        big.nlargest(8, "trend_pp_decade"),
        ["n", "trend_pp_decade", "trend_t"],
    )
    show("fastest-shrinking regions", big.nsmallest(8, "trend_pp_decade"), ["n", "trend_pp_decade", "trend_t"])

    # --- 3. cross-walk to the 2D map's named regions ---
    try:
        from sklearn.metrics import adjusted_mutual_info_score
    except ImportError:
        print("sklearn unavailable; skipping AMI crosswalk")
        return
    map_lab = load_map_labels(ids)
    map_cols = sorted(c for c in map_lab.columns if c.startswith("label_layer_"))
    print("\n=== AMI crosswalk: EVoC layers vs 2D-map layers (doubly-labeled rows only) ===")
    for ec in label_cols:
        e = df[ec].to_numpy()
        best = None
        for mc in map_cols:
            m = map_lab[mc].to_numpy()
            both = (e != "Unlabelled") & (m != "Unlabelled")
            ami = adjusted_mutual_info_score(e[both], m[both])
            if best is None or ami > best[1]:
                best = (mc, ami, float(both.mean()))
        print(f"  {ec} ({n_regions[ec]:>5,} regions) <-> best {best[0]}: AMI={best[1]:.3f} (coverage {best[2]:.0%})")


if __name__ == "__main__":
    main()

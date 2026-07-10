"""Metadata x geometry sweep: what does the embedding space know about each field?

For every (non-tag-derived) metadata field, measures association between the
field and position in the ORIGINAL 1024-d space via the cached ambient kNN
graph (k=25), with permutation nulls:

  categorical -> per-class same-label neighbor rate vs shuffle (lift + z)
  binary      -> neighbor rate around positives vs stratified shuffle
  continuous  -> rank assortativity r (Moran's-I flavor) vs shuffle

daily_double gets two variants: shuffles stratified by round (raw) and by
round x board_row (placement-adjusted; the honest one). A synthetic random
binary field is included as a calibration control (should sit at z ~ 0).

Effect sizes (lift, r) are the headline; with n=135k nearly everything clears
significance. Output: printed tables + data/analysis/metadata_sweep.parquet.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    SWEEP_PARQUET,
    ambient_knn,
    atomic_write_parquet,
    categorical_neighbor_test,
    continuous_neighbor_test,
    load_clues,
    load_embeddings,
    neighbor_rate_test,
)

COLS = [
    "round",
    "board_row",
    "daily_double",
    "game_type",
    "delivery",
    "visual_clue",
    "is_repeat_clue",
    "value",
    "clue_order",
    "clue_len_words",
    "answer_len_chars",
    "answer_word_count",
    "answer_freq",
    "category_recurrence",
    "air_date",
    "season",
]


def main():
    ids, _ = load_embeddings(normalize=False)
    df = load_clues(ids, columns=COLS)
    nbr = ambient_knn()
    print(f"{len(df):,} clues, kNN graph {nbr.shape}", flush=True)
    rows = []

    def add(field, kind, variant, res):
        rows.append({"field": field, "kind": kind, "variant": variant, **res})
        print(
            f"  {field:<22} {variant:<28} z={res['z']:+7.1f}  effect={res.get('lift', res['stat_obs']):.3f}", flush=True
        )

    # --- categorical ---
    t0 = time.time()
    main_rounds = df["round"].isin(["jeopardy", "double_jeopardy"]).to_numpy()
    cat_fields = {
        "round": df["round"].to_numpy(),
        "game_type": df["game_type"].fillna("Regular").to_numpy(),
        "delivery": df["delivery"].fillna("Standard").to_numpy(),
        "season": df["season"].astype(int).astype(str).to_numpy(),
    }
    for field, labels in cat_fields.items():
        for res in categorical_neighbor_test(nbr, labels):
            add(field, "categorical", res.pop("class"), res)

    # --- binary ---
    dd = df["daily_double"].fillna(False).to_numpy(bool)
    strata_round = df["round"].to_numpy()
    strata_cell = (df["round"].astype(str) + "|" + df["board_row"].fillna(-1).astype(int).astype(str)).to_numpy()
    add("daily_double", "binary", "strat: round", neighbor_rate_test(nbr, dd, strata=strata_round))
    add("daily_double", "binary", "strat: round x row", neighbor_rate_test(nbr, dd, strata=strata_cell))
    add("visual_clue", "binary", "raw", neighbor_rate_test(nbr, df["visual_clue"].fillna(False).to_numpy(bool)))
    add("is_repeat_clue", "binary", "raw", neighbor_rate_test(nbr, df["is_repeat_clue"].fillna(False).to_numpy(bool)))
    rng = np.random.default_rng(7)
    add("random_control", "binary", "raw (calibration)", neighbor_rate_test(nbr, rng.random(len(df)) < 0.05))

    # --- continuous (rank assortativity) ---
    air_year = df["air_date"].dt.year + df["air_date"].dt.dayofyear / 365.25
    cont_fields = {
        "value": (df["value"].to_numpy(float), main_rounds),
        "board_row": (df["board_row"].to_numpy(float), main_rounds),
        "clue_len_words": (df["clue_len_words"].to_numpy(float), None),
        "answer_len_chars": (df["answer_len_chars"].to_numpy(float), None),
        "answer_word_count": (df["answer_word_count"].to_numpy(float), None),
        "answer_freq": (df["answer_freq"].to_numpy(float), None),
        "category_recurrence": (df["category_recurrence"].to_numpy(float), None),
        "clue_order": (df["clue_order"].to_numpy(float), None),
        "air_year": (air_year.to_numpy(float), None),
    }
    for field, (x, mask) in cont_fields.items():
        add(field, "continuous", "rank assortativity", continuous_neighbor_test(nbr, x, mask=mask))

    out = pd.DataFrame(rows)
    atomic_write_parquet(out, SWEEP_PARQUET)
    print(f"\nWrote {SWEEP_PARQUET} ({len(out)} rows) in {time.time() - t0:.0f}s")
    with pd.option_context("display.width", 200):
        print("\n=== ranked by |z| ===")
        show = out.assign(abs_z=out["z"].abs()).sort_values("abs_z", ascending=False)
        cols = ["field", "kind", "variant", "n_pos", "base_rate", "stat_obs", "null_mean", "z", "lift"]
        print(show[[c for c in cols if c in show.columns]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()

"""Temporal drift of the clue corpus in embedding space, season by season.

Two views over seasons 32-41 (2016-2025):
  1. Season centroid trajectory: cosine distance between consecutive season
     centroids (and each season vs season 32), against a shuffled-season null.
     Rerun on Regular-play-only rows, since tournament formats (Masters,
     Champions Wildcard, ...) concentrate in recent seasons and could
     masquerade as content drift.
  2. Per-season same-season kNN neighbor rate (local view; the sweep
     establishes global significance, this shows the per-season profile),
     with a repeats-excluded sensitivity check.

Output: printed tables + data/analysis/season_drift.parquet (+ centroid npz
for later figures).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    ANALYSIS_DIR,
    DRIFT_PARQUET,
    ambient_knn,
    atomic_save_npz,
    atomic_write_parquet,
    load_clues,
    load_embeddings,
)

N_PERM = 200
SEED = 42


def season_centroids(emb, seasons, season_list):
    cent = np.stack([emb[seasons == s].mean(axis=0) for s in season_list])
    cent /= np.linalg.norm(cent, axis=1, keepdims=True)
    return cent


def consecutive_cos_dist(cent):
    return 1.0 - np.einsum("ij,ij->i", cent[:-1], cent[1:])


def trajectory_test(emb, seasons, season_list, n_perm=N_PERM, seed=SEED):
    rng = np.random.default_rng(seed)
    cent = season_centroids(emb, seasons, season_list)
    obs = consecutive_cos_dist(cent)
    null = np.empty((n_perm, len(obs)))
    for i in range(n_perm):
        null[i] = consecutive_cos_dist(season_centroids(emb, rng.permutation(seasons), season_list))
    z = (obs - null.mean(axis=0)) / null.std(axis=0)
    return cent, obs, null.mean(axis=0), z


def main():
    ids, emb = load_embeddings()
    df = load_clues(ids, columns=["season", "game_type", "is_repeat_clue"])
    seasons = df["season"].astype(int).to_numpy()
    season_list = np.sort(np.unique(seasons))
    rows = []

    for variant, mask in [
        ("all clues", np.ones(len(df), bool)),
        ("Regular play only", (df["game_type"].fillna("Regular") == "Regular").to_numpy()),
    ]:
        cent, obs, null_mu, z = trajectory_test(emb[mask], seasons[mask], season_list)
        print(f"\n=== centroid drift, {variant} ===")
        for i in range(len(obs)):
            pair = f"S{season_list[i]}->S{season_list[i + 1]}"
            print(f"  {pair:<10} cos_dist={obs[i]:.5f}  null={null_mu[i]:.5f}  z={z[i]:+.1f}")
            rows.append(
                {
                    "analysis": "centroid_drift",
                    "variant": variant,
                    "unit": pair,
                    "stat": float(obs[i]),
                    "null_mean": float(null_mu[i]),
                    "z": float(z[i]),
                }
            )
        if variant == "all clues":
            atomic_save_npz(ANALYSIS_DIR / "season_centroids.npz", centroids=cent, seasons=season_list)
        # cumulative distance from the first season, no null needed (monotone read)
        cum = 1.0 - cent[0] @ cent.T
        print("  cumulative vs first season:", " ".join(f"S{s}:{d:.5f}" for s, d in zip(season_list, cum)))

    # --- local view: same-season neighbor rate per season ---
    nbr = ambient_knn()
    fresh = ~df["is_repeat_clue"].fillna(False).to_numpy(bool)
    print("\n=== same-season kNN neighbor rate (k=25) ===")
    for s in season_list:
        src = seasons == s
        base = float(src.mean())
        rate = float(np.mean(seasons[nbr[src]] == s))
        rate_fresh = float(np.mean(seasons[nbr[src & fresh]] == s))
        print(f"  S{s}: base {base:.3f}  nbr rate {rate:.3f} (lift {rate / base:.2f})  excl. repeats {rate_fresh:.3f}")
        rows.append(
            {
                "analysis": "same_season_knn",
                "variant": "all clues",
                "unit": f"S{s}",
                "stat": rate,
                "null_mean": base,
                "z": np.nan,
                "lift": rate / base,
                "lift_no_repeats": rate_fresh / base,
            }
        )

    atomic_write_parquet(pd.DataFrame(rows), DRIFT_PARQUET)
    print(f"\nWrote {DRIFT_PARQUET}")


if __name__ == "__main__":
    main()

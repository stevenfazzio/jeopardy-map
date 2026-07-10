"""Batch-level robustness: which same-label clustering survives the fact that
Jeopardy content arrives in category/episode batches?

The sweep's clue-level shuffles answer "are same-label clues neighbors more
than a random scatter?" — but a category is ~5 clues on one topic, so any
label assigned at category or episode granularity (a tournament's episodes,
a Clue Crew category) clusters trivially. Here the null permutes labels at
the field's natural assignment level instead:

  game_type, season      -> permute across EPISODES
  delivery, visual_clue  -> permute across (episode, category) batches

Mechanics: groups of equal size trade their internal label PATTERNS wholesale
(size-matched pattern permutation), so each group keeps its structure and the
same-batch mechanical component moves INTO the null. Surviving z means genuine
cross-batch topical signature; z ~ 0 means the sweep's lift was batching.

Output: printed table + data/analysis/batch_robustness.parquet.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    ANALYSIS_DIR,
    ambient_knn,
    atomic_write_parquet,
    load_clues,
    load_embeddings,
)

N_PERM = 300
SEED = 42
MIN_CLASS_N = 300
OUT_PARQUET = ANALYSIS_DIR / "batch_robustness.parquet"


class PatternPermuter:
    """Size-matched group-pattern permutation of a label vector.

    Precomputes, per group size s, the (m, s) matrix of member positions and
    of original labels; a draw scatters row-permuted label patterns back into
    position. Groups only ever swap patterns with same-size groups.
    """

    def __init__(self, labels: np.ndarray, group_codes: np.ndarray):
        self.labels = labels
        order = np.argsort(group_codes, kind="stable")
        sorted_codes = group_codes[order]
        starts = np.flatnonzero(np.r_[True, sorted_codes[1:] != sorted_codes[:-1]])
        bounds = np.r_[starts, len(order)]
        by_size: dict[int, list[np.ndarray]] = {}
        for a, b in zip(bounds[:-1], bounds[1:]):
            by_size.setdefault(b - a, []).append(order[a:b])
        self.blocks = []
        for size, groups in by_size.items():
            pos = np.vstack(groups)  # (m, size)
            self.blocks.append((pos, labels[pos]))

    def draw(self, rng: np.random.Generator) -> np.ndarray:
        out = np.empty_like(self.labels)
        for pos, lab in self.blocks:
            out[pos] = lab[rng.permutation(len(lab))]
        return out


class LabelSwapPermuter:
    """Permute a group-constant label across ALL groups (no size matching).

    Correct null for episode-level fields: size-matched pattern swaps are
    confounded when group size correlates with the label (primetime formats
    have ~120-180-clue episode_ids vs ~60 for regular play, so their patterns
    could only trade places with each other and the null collapses onto the
    observed value).
    """

    def __init__(self, labels: np.ndarray, group_codes: np.ndarray):
        n_groups = group_codes.max() + 1
        self.group_label = np.empty(n_groups, dtype=object)
        self.group_label[group_codes] = labels  # group-constant by construction
        self.group_codes = group_codes
        self.n_groups = n_groups

    def draw(self, rng: np.random.Generator) -> np.ndarray:
        return self.group_label[rng.permutation(self.n_groups)][self.group_codes]


def group_permutation_test(nbr_idx, labels, group_codes, mode, n_perm=N_PERM, seed=SEED, min_class_n=MIN_CLASS_N):
    rng = np.random.default_rng(seed)
    if mode == "swap":
        per_group = pd.DataFrame({"g": group_codes, "v": labels}).groupby("g")["v"].nunique()
        assert (per_group == 1).all(), "swap mode requires group-constant labels"
        permuter = LabelSwapPermuter(labels, group_codes)
    else:
        permuter = PatternPermuter(labels, group_codes)
    classes = [c for c, n in zip(*np.unique(labels, return_counts=True)) if n >= min_class_n]

    def stats(lab):
        return [float(np.mean(lab[nbr_idx[lab == c]] == c)) for c in classes]

    obs = stats(labels)
    null = np.empty((n_perm, len(classes)))
    for i in range(n_perm):
        null[i] = stats(permuter.draw(rng))
    rows = []
    for j, c in enumerate(classes):
        mu, sd = null[:, j].mean(), null[:, j].std()
        rows.append(
            {
                "class": str(c),
                "stat_obs": obs[j],
                "null_mean": float(mu),
                "null_sd": float(sd),
                "z": float((obs[j] - mu) / sd) if sd > 0 else np.nan,
                "lift_vs_batch_null": float(obs[j] / mu) if mu > 0 else np.nan,
                "n_pos": int(np.sum(labels == c)),
            }
        )
    return rows


def main():
    ids, _ = load_embeddings(normalize=False)
    df = load_clues(ids, columns=["episode_id", "category", "game_type", "season", "delivery", "visual_clue"])
    nbr = ambient_knn()

    ep_key = df["episode_id"].astype(str)
    episode_codes = pd.factorize(ep_key)[0]
    batch_codes = pd.factorize(ep_key + "|" + df["category"].astype(str))[0]
    print(f"{len(df):,} clues, {episode_codes.max() + 1:,} episodes, {batch_codes.max() + 1:,} category batches")

    # game_type is not quite episode-constant (77 episodes mix types, e.g.
    # Power Players weeks); use the episode's modal label for the episode test
    gt_episode = df["game_type"].fillna("Regular").groupby(ep_key).transform(lambda s: s.mode().iloc[0])
    rows = []
    specs = [
        ("game_type (episode-modal)", gt_episode.to_numpy(), episode_codes, "episode", "swap"),
        ("season", df["season"].astype(int).astype(str).to_numpy(), episode_codes, "episode", "swap"),
        ("delivery", df["delivery"].fillna("Standard").to_numpy(), batch_codes, "category batch", "pattern"),
        (
            "visual_clue",
            np.where(df["visual_clue"].fillna(False), "visual", "not visual"),
            batch_codes,
            "category batch",
            "pattern",
        ),
    ]
    for field, labels, codes, level, mode in specs:
        print(f"\n=== {field} (null: {mode} across {level}s) ===", flush=True)
        for res in group_permutation_test(nbr, labels, codes, mode):
            c = res.pop("class")
            rows.append({"field": field, "class": c, "null_level": level, **res})
            print(
                f"  {c:<32} obs={res['stat_obs']:.4f}  batch-null={res['null_mean']:.4f}"
                f"  z={res['z']:+6.1f}  lift={res['lift_vs_batch_null']:.2f}",
                flush=True,
            )

    atomic_write_parquet(pd.DataFrame(rows), OUT_PARQUET)
    print(f"\nWrote {OUT_PARQUET}")


if __name__ == "__main__":
    main()

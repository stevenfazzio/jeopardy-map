"""Round gradient decomposition: scheduling vs writing, within category titles.

Slate (2011) showed the J -> DJ "curriculum gradient" at the category-TITLE
level: Double Jeopardy schedules more opera/physics categories. Category
titles are all that counting can see. Embeddings can ask the finer question:
for the SAME recurring category title (e.g. "science", "world capitals"),
are the DJ clues written measurably differently from the J clues — i.e. does
part of the gradient live in the writing, not just the board lineup?

Design (2016+ corpus, J/DJ rounds only):
  * "Dual titles" = category_normalized values fielded in BOTH rounds at
    least MIN_BATCHES times each (a batch = one episode x title x round
    category of ~5 clues). All tests run on dual-title clues only, so the
    title itself is controlled by construction.
  * Probe triple (episode-grouped StratifiedGroupKFold, like 30_probes):
      raw        AUC of round from the embedding (title identity + writing)
      title-only AUC after replacing each clue with its title centroid
                 (identity signal alone)
      demeaned   AUC after subtracting each title's centroid (writing signal
                 alone — the number that answers the question)
  * Batch-level permutation test: project title-centered batch centroids
    onto the J->DJ axis; stat = mean projection of DJ batches minus J
    batches, with the axis RE-ESTIMATED inside every permutation (round
    labels shuffled across batches within title — the suite's batch-honest
    null; clue-level shuffles would flatter, see CLAUDE.md).
  * Headline ratio: observed within-title shift as a fraction of the global
    J->DJ centroid displacement over the same rows — "X% of the round
    gradient survives inside identical category titles".

Output: data/analysis/round_within_title.parquet — summary rows plus
per-title projection stats (for a titles dot-plot figure) — and a printed
summary. Read-only over pipeline artifacts.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import ANALYSIS_DIR, atomic_write_parquet, load_clues, load_embeddings  # noqa: E402

OUT_PARQUET = ANALYSIS_DIR / "round_within_title.parquet"

MIN_BATCHES = 6  # per round, for a title to count as dual
FOLDS = 3
TRAIN_CAP = 50_000
N_PERM = 1000
SEED = 42


def grouped_auc(emb: np.ndarray, y: np.ndarray, groups: np.ndarray, name: str, rng) -> float:
    skf = StratifiedGroupKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
    aucs = []
    for tr, te in skf.split(emb, y, groups):
        if len(tr) > TRAIN_CAP:
            tr = rng.choice(tr, size=TRAIN_CAP, replace=False)
        clf = LogisticRegression(max_iter=1000, C=1.0)
        clf.fit(emb[tr], y[tr])
        aucs.append(roc_auc_score(y[te], clf.decision_function(emb[te])))
    print(f"  {name:<34} AUC {np.mean(aucs):.3f} +/- {np.std(aucs):.3f}", flush=True)
    return float(np.mean(aucs))


def batch_projection_test(cent: np.ndarray, title_idx: np.ndarray, is_dj: np.ndarray, rng) -> dict:
    """Mean title-centered projection difference (DJ - J batches), axis re-fit per permutation."""
    # title means of batch centroids are label-free: compute once
    n_titles = title_idx.max() + 1
    sums = np.zeros((n_titles, cent.shape[1]), dtype=np.float64)
    np.add.at(sums, title_idx, cent)
    counts = np.bincount(title_idx, minlength=n_titles).astype(np.float64)
    centered = cent - sums[title_idx] / counts[title_idx, None]

    def stat(labels: np.ndarray) -> float:
        w = cent[labels].mean(axis=0) - cent[~labels].mean(axis=0)
        w /= np.linalg.norm(w)
        proj = centered @ w
        return float(proj[labels].mean() - proj[~labels].mean())

    obs = stat(is_dj)
    # permute round labels across batches WITHIN each title (batch-honest null)
    order = np.argsort(title_idx, kind="stable")
    bounds = np.searchsorted(title_idx[order], np.arange(n_titles + 1))
    null = np.empty(N_PERM)
    labels = is_dj.copy()
    for i in range(N_PERM):
        perm = labels.copy()
        for a, b in zip(bounds[:-1], bounds[1:]):
            seg = perm[order[a:b]]
            rng.shuffle(seg)
            perm[order[a:b]] = seg
        null[i] = stat(perm)
    return {
        "stat_obs": obs,
        "null_mean": float(null.mean()),
        "null_sd": float(null.std()),
        "z": float((obs - null.mean()) / null.std()),
        "p_emp": float((np.sum(np.abs(null) >= abs(obs)) + 1) / (N_PERM + 1)),
    }


def main() -> None:
    rng = np.random.default_rng(SEED)
    t0 = time.time()
    ids, emb = load_embeddings()
    df = load_clues(ids, columns=["round", "category_normalized", "episode_id"])
    main_mask = df["round"].isin(["jeopardy", "double_jeopardy"]).to_numpy()

    d = df[main_mask].reset_index(drop=True)
    e = emb[main_mask]
    is_dj_clue = (d["round"] == "double_jeopardy").to_numpy()
    batch_key = d["episode_id"].astype(str) + "|" + d["category_normalized"] + "|" + d["round"]

    # dual titles: >= MIN_BATCHES batches in each round
    batches = pd.DataFrame(
        {"title": d["category_normalized"], "round": d["round"], "batch": batch_key}
    ).drop_duplicates("batch")
    per_round = batches.groupby(["title", "round"]).size().unstack(fill_value=0)
    enough = (per_round.get("jeopardy", 0) >= MIN_BATCHES) & (per_round.get("double_jeopardy", 0) >= MIN_BATCHES)
    dual = per_round[enough]
    dual_titles = set(dual.index)
    in_dual = d["category_normalized"].isin(dual_titles).to_numpy()
    print(
        f"dual titles: {len(dual_titles):,} (of {per_round.shape[0]:,} titles) covering "
        f"{in_dual.sum():,}/{len(d):,} main-round clues ({in_dual.mean():.1%}) "
        f"[{time.time() - t0:.0f}s]"
    )

    dd = d[in_dual].reset_index(drop=True)
    ee = e[in_dual]
    y = is_dj_clue[in_dual]
    groups = dd["episode_id"].astype(str).to_numpy()
    titles, title_idx = np.unique(dd["category_normalized"].to_numpy(), return_inverse=True)

    # title centroids over clues (for the probe variants)
    t_sums = np.zeros((len(titles), ee.shape[1]), dtype=np.float64)
    np.add.at(t_sums, title_idx, ee)
    t_counts = np.bincount(title_idx).astype(np.float64)
    t_cent = (t_sums / t_counts[:, None]).astype(np.float32)

    rows = []
    print("probe triple (episode-grouped CV, dual-title clues):", flush=True)
    for name, mat in [
        ("raw (identity + writing)", ee),
        ("title-only (identity)", t_cent[title_idx]),
        ("demeaned (writing)", ee - t_cent[title_idx]),
    ]:
        auc = grouped_auc(mat, y, groups, name, rng)
        rows.append({"analysis": "probe", "unit": name, "stat": auc, "null_mean": 0.5, "z": np.nan, "p_emp": np.nan})

    # batch-centroid permutation test
    print("batch-level projection test (axis re-fit per permutation):", flush=True)
    bkey = batch_key[in_dual.nonzero()[0]].to_numpy() if hasattr(batch_key, "to_numpy") else batch_key[in_dual]
    b_ids, b_idx = np.unique(bkey, return_inverse=True)
    b_sums = np.zeros((len(b_ids), ee.shape[1]), dtype=np.float64)
    np.add.at(b_sums, b_idx, ee)
    b_counts = np.bincount(b_idx).astype(np.float64)
    b_cent = b_sums / b_counts[:, None]
    first_of_batch = np.full(len(b_ids), -1, dtype=int)
    seen = set()
    for i, b in enumerate(b_idx):
        if b not in seen:
            first_of_batch[b] = i
            seen.add(b)
    b_title = title_idx[first_of_batch]
    b_is_dj = y[first_of_batch]
    res = batch_projection_test(b_cent, b_title, b_is_dj, rng)
    print(
        f"  within-title DJ-J batch shift {res['stat_obs']:.5f} "
        f"(null {res['null_mean']:.5f} +/- {res['null_sd']:.5f}, z={res['z']:+.1f}, p={res['p_emp']:.4f})"
    )
    rows.append({"analysis": "batch_projection", "unit": "within_title_shift", **res})

    # headline ratio: within-title shift as fraction of the global J->DJ displacement
    global_shift = np.linalg.norm(ee[y].mean(axis=0) - ee[~y].mean(axis=0))
    frac = res["stat_obs"] / global_shift
    print(f"  global J->DJ displacement {global_shift:.5f}; within-title fraction {frac:.1%}")
    rows.append(
        {"analysis": "decomposition", "unit": "within_title_fraction", "stat": float(frac), "null_mean": np.nan}
    )

    # per-title stats for the dot-plot figure (observed axis)
    w = (b_cent[b_is_dj].mean(axis=0) - b_cent[~b_is_dj].mean(axis=0)).astype(np.float64)
    w /= np.linalg.norm(w)
    n_titles = title_idx.max() + 1
    t_b_sums = np.zeros((n_titles, b_cent.shape[1]))
    np.add.at(t_b_sums, b_title, b_cent)
    t_b_counts = np.bincount(b_title, minlength=n_titles).astype(np.float64)
    proj = (b_cent - t_b_sums[b_title] / t_b_counts[b_title, None]) @ w
    per_title = pd.DataFrame({"title": titles[b_title], "is_dj": b_is_dj, "proj": proj})
    tstats = (
        per_title.groupby(["title", "is_dj"])["proj"]
        .mean()
        .unstack()
        .rename(columns={False: "proj_j", True: "proj_dj"})
        .assign(shift=lambda x: x["proj_dj"] - x["proj_j"])
        .join(dual[["jeopardy", "double_jeopardy"]])
        .reset_index()
    )
    pos_share = float((tstats["shift"] > 0).mean())
    print(f"  {pos_share:.1%} of {len(tstats)} dual titles shift toward the DJ direction")
    rows.append({"analysis": "decomposition", "unit": "share_titles_positive", "stat": pos_share})

    out = pd.concat([pd.DataFrame(rows).assign(kind="summary"), tstats.assign(kind="per_title")], ignore_index=True)
    atomic_write_parquet(out, OUT_PARQUET)
    print(f"\nwrote {OUT_PARQUET} ({len(out)} rows) in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()

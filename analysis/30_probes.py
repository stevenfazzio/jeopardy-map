"""Linear probes: how well does a linear readout of the raw 1024-d embedding
predict each metadata field?

Complements the kNN sweep: the sweep measures LOCAL structure (are same-label
clues neighbors?), probes measure GLOBAL linear structure, and give the
blog-friendly framing "a linear model predicts X from the embedding at AUC/R2
= y".

CV folds are GROUPED BY EPISODE: Jeopardy content arrives in category batches
of ~5 near-identical-topic clues inside an episode, so ungrouped folds leak
same-category clues across the train/test split and flatter every
episode-level field (game_type, season). Training folds are subsampled (test
folds never are) to keep fits fast. A label-shuffled daily_double control
calibrates the pipeline (should give ~0.5).

Output: printed table + data/analysis/probe_results.parquet.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import PROBES_PARQUET, atomic_write_parquet, load_clues, load_embeddings  # noqa: E402

TRAIN_CAP = 50_000
FOLDS = 3
SEED = 42

COLS = [
    "episode_id",
    "round",
    "board_row",
    "daily_double",
    "game_type",
    "visual_clue",
    "is_repeat_clue",
    "value",
    "clue_len_words",
    "answer_len_chars",
    "answer_freq",
    "category_recurrence",
    "air_date",
]


def _subsample_train(tr_idx: np.ndarray, y: np.ndarray | None, rng) -> np.ndarray:
    if len(tr_idx) <= TRAIN_CAP:
        return tr_idx
    sub = tr_idx[rng.choice(len(tr_idx), size=TRAIN_CAP, replace=False)]
    if y is not None and y[sub].sum() < 50:  # rare positives: keep every one
        pos = tr_idx[y[tr_idx]]
        neg = rng.choice(tr_idx[~y[tr_idx]], size=TRAIN_CAP - len(pos), replace=False)
        sub = np.concatenate([pos, neg])
    return sub


def probe_binary(emb, y, mask, groups, name, rows, rng):
    y = y.astype(bool)
    idx_all = np.flatnonzero(mask)
    skf = StratifiedGroupKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
    aucs = []
    for tr, te in skf.split(idx_all, y[idx_all], groups[idx_all]):
        tr_idx = _subsample_train(idx_all[tr], y, rng)
        clf = LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced")
        clf.fit(emb[tr_idx], y[tr_idx])
        aucs.append(roc_auc_score(y[idx_all[te]], clf.decision_function(emb[idx_all[te]])))
    rows.append(
        {
            "field": name,
            "metric": "AUC",
            "value": float(np.mean(aucs)),
            "sd": float(np.std(aucs)),
            "n": int(mask.sum()),
            "n_pos": int(y[mask].sum()),
        }
    )
    print(f"  {name:<28} AUC {np.mean(aucs):.3f} +/- {np.std(aucs):.3f}", flush=True)


def probe_multiclass(emb, labels, groups, name, rows, rng):
    idx_all = np.arange(len(labels))
    skf = StratifiedGroupKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
    aucs, accs = [], []
    for tr, te in skf.split(idx_all, labels, groups):
        tr_idx = _subsample_train(idx_all[tr], None, rng)
        clf = LogisticRegression(max_iter=1000, C=1.0)
        clf.fit(emb[tr_idx], labels[tr_idx])
        proba = clf.predict_proba(emb[te])
        aucs.append(roc_auc_score(labels[te], proba, multi_class="ovr", average="macro"))
        accs.append((clf.classes_[proba.argmax(1)] == labels[te]).mean())
    for metric, vals in [("macro-AUC (ovr)", aucs), ("accuracy", accs)]:
        rows.append(
            {
                "field": name,
                "metric": metric,
                "value": float(np.mean(vals)),
                "sd": float(np.std(vals)),
                "n": len(labels),
                "n_pos": None,
            }
        )
    print(f"  {name:<28} macro-AUC {np.mean(aucs):.3f}, acc {np.mean(accs):.3f}", flush=True)


def probe_continuous(emb, x, mask, groups, name, rows, rng):
    mask = mask & np.isfinite(x)
    idx_all = np.flatnonzero(mask)
    gkf = GroupKFold(n_splits=FOLDS)
    r2s = []
    for tr, te in gkf.split(idx_all, groups=groups[idx_all]):
        tr_idx = _subsample_train(idx_all[tr], None, rng)
        reg = Ridge(alpha=10.0)
        reg.fit(emb[tr_idx], x[tr_idx])
        r2s.append(r2_score(x[idx_all[te]], reg.predict(emb[idx_all[te]])))
    rows.append(
        {
            "field": name,
            "metric": "R2",
            "value": float(np.mean(r2s)),
            "sd": float(np.std(r2s)),
            "n": int(mask.sum()),
            "n_pos": None,
        }
    )
    print(f"  {name:<28} R2 {np.mean(r2s):.3f} +/- {np.std(r2s):.3f}", flush=True)


def main():
    rng = np.random.default_rng(SEED)
    ids, emb = load_embeddings()
    df = load_clues(ids, columns=COLS)
    groups = df["episode_id"].astype(str).to_numpy()
    all_mask = np.ones(len(df), dtype=bool)
    main_rounds = df["round"].isin(["jeopardy", "double_jeopardy"]).to_numpy()
    rows = []
    t0 = time.time()

    print("binary probes (episode-grouped CV):", flush=True)
    dd = df["daily_double"].fillna(False).to_numpy(bool)
    probe_binary(emb, dd, main_rounds, groups, "daily_double (main rounds)", rows, rng)
    dd_ctrl = dd.copy()
    dd_ctrl[main_rounds] = rng.permutation(dd_ctrl[main_rounds])
    probe_binary(emb, dd_ctrl, main_rounds, groups, "dd_shuffled (control)", rows, rng)
    probe_binary(
        emb,
        (df["game_type"].fillna("Regular") != "Regular").to_numpy(),
        all_mask,
        groups,
        "special_game (vs Regular)",
        rows,
        rng,
    )
    probe_binary(emb, df["visual_clue"].fillna(False).to_numpy(bool), all_mask, groups, "visual_clue", rows, rng)
    probe_binary(emb, df["is_repeat_clue"].fillna(False).to_numpy(bool), all_mask, groups, "is_repeat_clue", rows, rng)

    print("multiclass probes:", flush=True)
    probe_multiclass(emb, df["round"].to_numpy(), groups, "round (3-class)", rows, rng)

    print("continuous probes:", flush=True)
    air_year = (df["air_date"].dt.year + df["air_date"].dt.dayofyear / 365.25).to_numpy()
    probe_continuous(emb, df["value"].to_numpy(float), main_rounds, groups, "value (main rounds)", rows, rng)
    probe_continuous(emb, df["board_row"].to_numpy(float), main_rounds, groups, "board_row (main rounds)", rows, rng)
    probe_continuous(emb, df["clue_len_words"].to_numpy(float), all_mask, groups, "clue_len_words", rows, rng)
    probe_continuous(emb, df["answer_len_chars"].to_numpy(float), all_mask, groups, "answer_len_chars", rows, rng)
    probe_continuous(emb, np.log10(df["answer_freq"].to_numpy(float)), all_mask, groups, "log10 answer_freq", rows, rng)
    probe_continuous(
        emb,
        np.log10(df["category_recurrence"].to_numpy(float)),
        all_mask,
        groups,
        "log10 category_recurrence",
        rows,
        rng,
    )
    probe_continuous(emb, air_year, all_mask, groups, "air_year", rows, rng)

    out = pd.DataFrame(rows)
    atomic_write_parquet(out, PROBES_PARQUET)
    print(f"\nWrote {PROBES_PARQUET} ({len(out)} rows) in {time.time() - t0:.0f}s")
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()

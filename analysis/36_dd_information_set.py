"""Daily Double predictability by information set: what can a PLAYER see?

The DD analyses in 30/50 read the full embed_text (category + clue + answer),
but a player choosing a cell sees only the category title, the row, and the
round — the clue text is exactly what a pick reveals. This script decomposes
DD predictability into nested information sets, all probed with the suite's
episode-grouped CV (see CLAUDE.md on batch effects):

  position            round x row one-hot — the folklore baseline
  title               Cohere embedding of the category title ALONE
  title + position    everything on the board before a pick
  content             the pipeline's full clue embeddings (30_probes' ~0.69)
  content + position  ceiling

plus a within-category leg: among complete 5-clue categories that contain a
DD, can you tell WHICH clue hides it before anything is revealed? Models:
row-only, row + clue length, row + batch-demeaned content embedding (the
"writing tell"); metrics = clue-level AUC and within-batch top-1 accuracy
(chance is exactly 20%);

plus a board backtest that turns the ladder into first-pick probabilities:
reconstruct every complete board (episode x round, 6 categories x 5 rows,
exactly 1 DD in J / 2 in DJ), score each cell with OUT-OF-FOLD model scores,
and let each strategy pick its argmax cell. Position scores are constant
within a row and title scores within a column, so the strategies reduce to
"best row, random column" / "best category, random row" / "best cell" /
"read every clue first (impossible)"; ties are handled as exact expectations
(mean DD rate over tied cells), so there is no Monte Carlo noise. Reported
per round vs the analytic random baselines 1/30 and 2/30.

Title embeddings are the script's only API cost (~22k unique titles, ~$0.01),
cached incrementally + resumably in data/analysis/title_embeddings.npz with
the same model/input_type as the clue embeddings. Needs stages 00-02.

Output: data/analysis/dd_information_set.parquet + printed table.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))

from common import ANALYSIS_DIR, atomic_write_parquet, load_clues, load_embeddings  # noqa: E402
from config import (  # noqa: E402
    CO_API_KEY,
    COHERE_EMBED_MODEL,
    COHERE_INPUT_TYPE,
    COHERE_OUTPUT_DIM,
    EMBED_BATCH,
)

TITLE_EMB_NPZ = ANALYSIS_DIR / "title_embeddings.npz"
OUT_PARQUET = ANALYSIS_DIR / "dd_information_set.parquet"

FOLDS = 3
TRAIN_CAP = 50_000
SEED = 42
CHECKPOINT_BATCHES = 50  # title-embedding cache flush cadence
MAX_TITLE_COST_USD = 0.50  # sanity guard; expected ~$0.01


# ------------------------------------------------------------- title vectors
def _embed_with_retry(client, chunk: list[str], max_retries: int = 5) -> np.ndarray:
    for attempt in range(max_retries):
        try:
            resp = client.embed(
                model=COHERE_EMBED_MODEL,
                input_type=COHERE_INPUT_TYPE,
                texts=chunk,
                output_dimension=COHERE_OUTPUT_DIM,
                embedding_types=["float"],
            )
            return np.asarray(resp.embeddings.float_, dtype=np.float32)
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            wait = min(2**attempt * 5, 60)
            print(f"  embed attempt {attempt + 1} failed ({type(e).__name__}: {e}); retry in {wait}s", flush=True)
            time.sleep(wait)


def _save_title_cache(sig: str, titles: np.ndarray, emb: np.ndarray) -> None:
    tmp = str(TITLE_EMB_NPZ) + ".tmp.npz"
    np.savez(tmp, sig=sig, titles=titles, emb=emb)
    os.replace(tmp, TITLE_EMB_NPZ)


def title_embeddings(titles: np.ndarray) -> np.ndarray:
    """[len(titles), dim] float32, unit-normalized; incremental resumable cache."""
    sig = f"{COHERE_EMBED_MODEL}_{COHERE_INPUT_TYPE}_{COHERE_OUTPUT_DIM}"
    cached_titles = np.array([], dtype=object)
    cached_emb = np.zeros((0, COHERE_OUTPUT_DIM), dtype=np.float32)
    if TITLE_EMB_NPZ.exists():
        npz = np.load(TITLE_EMB_NPZ, allow_pickle=True)
        if str(npz["sig"]) == sig:
            cached_titles, cached_emb = npz["titles"], npz["emb"]

    have = set(cached_titles)
    missing = sorted({t for t in titles if t not in have})  # deterministic order for resume
    if missing:
        est_cost = sum(len(t.split()) for t in missing) * 1.4 * 0.12 / 1e6
        assert est_cost < MAX_TITLE_COST_USD, f"title embed cost ~${est_cost:.2f} exceeds guard; check inputs"
        if not CO_API_KEY:
            raise RuntimeError("CO_API_KEY not set; add it to .env (see .env.example)")
        import cohere

        # explicit timeout: a hung connection otherwise blocks forever (no
        # exception, so the retry loop never fires)
        client = cohere.ClientV2(api_key=CO_API_KEY, timeout=120)
        print(f"embedding {len(missing):,} new titles (~${est_cost:.3f}, cached at {TITLE_EMB_NPZ.name})")
        new = np.zeros((len(missing), COHERE_OUTPUT_DIM), dtype=np.float32)
        for bi, s in enumerate(range(0, len(missing), EMBED_BATCH)):
            chunk = missing[s : s + EMBED_BATCH]
            new[s : s + len(chunk)] = _embed_with_retry(client, chunk)
            done = s + len(chunk)
            if (bi + 1) % CHECKPOINT_BATCHES == 0 or done == len(missing):
                _save_title_cache(
                    sig,
                    np.concatenate([cached_titles, np.array(missing[:done], dtype=object)]),
                    np.vstack([cached_emb, new[:done]]),
                )
                print(f"  embedded {done:,}/{len(missing):,}", flush=True)
        cached_titles = np.concatenate([cached_titles, np.array(missing, dtype=object)])
        cached_emb = np.vstack([cached_emb, new])

    pos = {t: i for i, t in enumerate(cached_titles)}
    out = cached_emb[[pos[t] for t in titles]]
    out = out / np.linalg.norm(out, axis=1, keepdims=True)
    return out.astype(np.float32)


# ------------------------------------------------------------------- probes
def _subsample_train(tr_idx: np.ndarray, rng) -> np.ndarray:
    if len(tr_idx) <= TRAIN_CAP:
        return tr_idx
    return tr_idx[rng.choice(len(tr_idx), size=TRAIN_CAP, replace=False)]


def probe_auc(X: np.ndarray, y: np.ndarray, groups: np.ndarray, name: str, rows: list, rng) -> float:
    """Binary probe, episode-grouped folds; identical splits for every feature set."""
    skf = StratifiedGroupKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
    idx = np.arange(len(y))
    aucs = []
    for tr, te in skf.split(idx, y, groups):
        tr = _subsample_train(tr, rng)
        clf = LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced")
        clf.fit(X[tr], y[tr])
        aucs.append(roc_auc_score(y[te], clf.decision_function(X[te])))
    rows.append(
        {
            "leg": "board_visible",
            "features": name,
            "metric": "AUC",
            "value": float(np.mean(aucs)),
            "sd": float(np.std(aucs)),
            "n": len(y),
            "n_pos": int(y.sum()),
        }
    )
    print(f"  {name:<22} AUC {np.mean(aucs):.3f} +/- {np.std(aucs):.3f}", flush=True)
    return float(np.mean(aucs))


def within_category_leg(emb: np.ndarray, d: pd.DataFrame, rows: list, rng) -> None:
    """Among complete 5-clue DD categories: which clue is it? Row prior vs writing tell."""
    batch = (d["episode_id"].astype(str) + "|" + d["category_normalized"] + "|" + d["round"]).to_numpy()
    y_all = d["daily_double"].fillna(False).to_numpy(bool)
    b_ids, b_idx = np.unique(batch, return_inverse=True)
    n_per = np.bincount(b_idx)
    dd_per = np.bincount(b_idx, weights=y_all).astype(int)
    keep_b = (n_per == 5) & (dd_per == 1)
    mask = keep_b[b_idx]
    print(f"within-category leg: {keep_b.sum():,} complete 5-clue DD categories ({mask.sum():,} clues)")

    y = y_all[mask]
    groups = d["episode_id"].astype(str).to_numpy()[mask]
    bi = b_idx[mask]
    row_oh = pd.get_dummies(d["board_row"].astype(int)[mask]).to_numpy(np.float32)
    length = d["clue_len_words"].to_numpy(float)[mask]
    length = np.nan_to_num((length - np.nanmean(length)) / np.nanstd(length)).reshape(-1, 1).astype(np.float32)
    e = emb[mask]
    sums = np.zeros((len(b_ids), e.shape[1]), dtype=np.float64)
    np.add.at(sums, bi, e)
    counts = np.bincount(bi, minlength=len(b_ids)).astype(np.float64)
    demeaned = (e - (sums / np.where(counts == 0, 1, counts)[:, None])[bi]).astype(np.float32)

    skf = StratifiedGroupKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
    idx = np.arange(len(y))
    for name, X in [
        ("row only", row_oh),
        ("row + clue length", np.hstack([row_oh, length])),
        ("row + demeaned content", np.hstack([row_oh, demeaned])),
    ]:
        aucs, top1s = [], []
        for tr, te in skf.split(idx, y, groups):
            tr = _subsample_train(tr, rng)
            clf = LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced")
            clf.fit(X[tr], y[tr])
            score = clf.decision_function(X[te])
            aucs.append(roc_auc_score(y[te], score))
            te_df = pd.DataFrame({"b": bi[te], "score": score, "y": y[te]})
            hits = te_df.loc[te_df.groupby("b")["score"].idxmax(), "y"]
            top1s.append(float(hits.mean()))
        for metric, vals in [("AUC", aucs), ("top1_acc", top1s)]:
            rows.append(
                {
                    "leg": "within_category",
                    "features": name,
                    "metric": metric,
                    "value": float(np.mean(vals)),
                    "sd": float(np.std(vals)),
                    "n": len(y),
                    "n_pos": int(y.sum()),
                }
            )
        print(
            f"  {name:<22} AUC {np.mean(aucs):.3f} +/- {np.std(aucs):.3f}   top-1 {np.mean(top1s):.3f} (chance 0.200)",
            flush=True,
        )


def board_backtest(
    d: pd.DataFrame,
    e_content: np.ndarray,
    e_title: np.ndarray,
    x_pos: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    rows: list,
    rng,
) -> None:
    """First-pick DD hit rate on complete boards, one row per strategy x round."""
    board = (d["episode_id"].astype(str) + "|" + d["round"]).to_numpy()
    cat = (board + "|" + d["category_normalized"].to_numpy()).astype(object)
    br = d["board_row"].astype(int).to_numpy()

    bdf = pd.DataFrame({"board": board, "cat": cat, "row": br, "dd": y, "round": d["round"].to_numpy()})
    per_cat = bdf.groupby(["board", "cat"], sort=False).agg(
        n=("row", "size"), rows_ok=("row", lambda s: sorted(s) == [1, 2, 3, 4, 5])
    )
    per_board = per_cat.groupby("board").agg(n_cats=("n", "size"), cells=("n", "sum"), all_ok=("rows_ok", "all"))
    per_board = per_board.join(bdf.groupby("board").agg(dds=("dd", "sum"), round=("round", "first")))
    req = np.where(per_board["round"] == "jeopardy", 1, 2)
    complete = per_board[
        (per_board["cells"] == 30) & (per_board["n_cats"] == 6) & per_board["all_ok"] & (per_board["dds"] == req)
    ]
    dropped = len(per_board) - len(complete)
    print(
        f"board backtest: {len(complete):,} complete boards "
        f"({(complete['round'] == 'jeopardy').sum():,} J / {(complete['round'] == 'double_jeopardy').sum():,} DJ; "
        f"{dropped:,} incomplete boards dropped)",
        flush=True,
    )

    strategies = {
        "position (best row, random column)": lambda: x_pos,
        "title (best category, random row)": lambda: e_title,
        "title + position": lambda: np.hstack([e_title, x_pos]),
        "content + position (impossible)": lambda: np.hstack([e_content, x_pos]),
    }
    skf = StratifiedGroupKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
    splits = list(skf.split(np.arange(len(y)), y, groups))
    ev = pd.DataFrame({"board": board, "row": br, "dd": y})
    for name, feat in strategies.items():
        X = feat()
        oof = np.full(len(y), np.nan)
        for tr, te in splits:
            tr = _subsample_train(tr, rng)
            clf = LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced")
            clf.fit(X[tr], y[tr])
            oof[te] = clf.decision_function(X[te])
        ev[name] = oof

    ev = ev[ev["board"].isin(set(complete.index))].reset_index(drop=True)
    ev_g = ev.groupby("board")
    result = complete[["round", "dds"]].copy()
    for name in strategies:
        # identical features -> bitwise-identical scores, so ties are exact; the
        # per-board hit probability is the mean DD rate over argmax-tied cells
        tied = ev[ev[name] == ev_g[name].transform("max")]
        result[name] = tied.groupby("board")["dd"].mean()

    pos_name = next(iter(strategies))
    tied_pos = ev[ev[pos_name] == ev_g[pos_name].transform("max")].drop_duplicates("board")
    row_pick = tied_pos.merge(complete[["round"]], left_on="board", right_index=True)

    for r_name, r_lab, r_req in [("jeopardy", "J", 1), ("double_jeopardy", "DJ", 2)]:
        sub = result[result["round"] == r_name]
        picks = row_pick[row_pick["round"] == r_name]["row"].value_counts(normalize=True)
        print(f"  {r_lab}: n={len(sub):,} boards, random baseline {r_req / 30:.3f}; position strategy picks row(s):")
        print(f"     {dict((int(k), round(float(v), 2)) for k, v in picks.items())}")
        rows.append(
            {
                "leg": "board_backtest",
                "features": f"random ({r_lab})",
                "metric": "p_first_pick",
                "value": r_req / 30,
                "sd": np.nan,
                "n": len(sub),
                "n_pos": int(sub["dds"].sum()),
            }
        )
        for name in strategies:
            v = sub[name]
            rows.append(
                {
                    "leg": "board_backtest",
                    "features": f"{name} ({r_lab})",
                    "metric": "p_first_pick",
                    "value": float(v.mean()),
                    "sd": float(v.std()),
                    "n": len(sub),
                    "n_pos": int(sub["dds"].sum()),
                }
            )
            print(f"     {name:<38} p(first pick hits DD) {v.mean():.3f}  ({v.mean() / (r_req / 30):.2f}x random)")


def main() -> None:
    rng = np.random.default_rng(SEED)
    t0 = time.time()
    ids, emb = load_embeddings()
    cols = ["episode_id", "round", "board_row", "daily_double", "category", "category_normalized", "clue_len_words"]
    df = load_clues(ids, columns=cols)
    main_mask = (df["round"].isin(["jeopardy", "double_jeopardy"]) & df["board_row"].notna()).to_numpy()
    d = df[main_mask].reset_index(drop=True)
    e_content = emb[main_mask]
    del emb

    y = d["daily_double"].fillna(False).to_numpy(bool)
    groups = d["episode_id"].astype(str).to_numpy()
    print(f"main-round clues: {len(d):,}; DD: {y.sum():,} ({y.mean():.2%}) [{time.time() - t0:.0f}s]")

    title_text = d["category"].fillna("").str.strip().str.upper().replace("", "UNTITLED").to_numpy()
    uniq = np.unique(title_text)
    t_emb_uniq = title_embeddings(uniq)
    t_pos = {t: i for i, t in enumerate(uniq)}
    e_title = t_emb_uniq[[t_pos[t] for t in title_text]]

    cell = d["round"].astype(str) + "|" + d["board_row"].astype(int).astype(str)
    x_pos = pd.get_dummies(cell).to_numpy(np.float32)

    rows: list[dict] = []
    print("board-visible probes (episode-grouped CV, identical folds):", flush=True)
    probe_auc(x_pos, y, groups, "position", rows, rng)
    probe_auc(e_title, y, groups, "title", rows, rng)
    x_tp = np.hstack([e_title, x_pos])
    probe_auc(x_tp, y, groups, "title + position", rows, rng)
    y_ctrl = rng.permutation(y)
    probe_auc(x_tp, y_ctrl, groups, "shuffled control (t+p)", rows, rng)
    del x_tp
    probe_auc(e_content, y, groups, "content", rows, rng)
    x_cp = np.hstack([e_content, x_pos])
    probe_auc(x_cp, y, groups, "content + position", rows, rng)
    del x_cp

    within_category_leg(e_content, d, rows, rng)
    board_backtest(d, e_content, e_title, x_pos, y, groups, rows, rng)

    out = pd.DataFrame(rows)
    atomic_write_parquet(out, OUT_PARQUET)
    print(f"\nwrote {OUT_PARQUET} ({len(out)} rows) in {time.time() - t0:.0f}s")
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()

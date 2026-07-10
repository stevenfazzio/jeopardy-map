"""Shared loaders and spatial statistics for the ambient-space analysis suite.

This package characterizes the clue corpus in the ORIGINAL 1024-d embedding
space (the 2D UMAP layout is a visualization artifact; see analysis scripts for
what each test does). Everything here is read-only over the pipeline artifacts
in data/; the suite caches its own intermediates under data/analysis/.

Canonical row order everywhere = the embedding npz clue_id order. Every loader
aligns to it and asserts a perfect 1:1 match, so row i means the same clue in
the embedding matrix, the kNN graph, the metadata frame, and the EVoC labels.

Run as a script to (re)build the ambient kNN cache:
    uv run python analysis/common.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))

from config import (  # noqa: E402
    CLUE_EMB_NPZ,
    CLUE_ROWS_PARQUET,
    DATA_DIR,
    TOPONYMY_LABELS_PARQUET,
)

ANALYSIS_DIR = DATA_DIR / "analysis"
ANALYSIS_DIR.mkdir(exist_ok=True)

K_NEIGHBORS = 25
AMBIENT_KNN_NPZ = ANALYSIS_DIR / "ambient_knn.npz"
TOP1_SIM_NPZ = ANALYSIS_DIR / "top1_sim.npz"
EVOC_CLUSTERS_NPZ = ANALYSIS_DIR / "evoc_cluster_layers.npz"
EVOC_LABELS_PARQUET = ANALYSIS_DIR / "evoc_labels.parquet"
SWEEP_PARQUET = ANALYSIS_DIR / "metadata_sweep.parquet"
PROBES_PARQUET = ANALYSIS_DIR / "probe_results.parquet"
DRIFT_PARQUET = ANALYSIS_DIR / "season_drift.parquet"


# ---------------------------------------------------------------- io helpers
def atomic_save_npz(path: Path, **arrays) -> None:
    tmp = str(path) + ".tmp.npz"
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


def atomic_write_parquet(df: pd.DataFrame, path: Path) -> None:
    tmp = str(path) + ".tmp"
    df.to_parquet(tmp, index=False)
    check = pd.read_parquet(tmp)
    assert len(check) == len(df), f"row count mismatch writing {path}"
    os.replace(tmp, path)


def load_embeddings(normalize: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Return (clue_id, emb float32). clue_id order here is THE canonical order."""
    npz = np.load(CLUE_EMB_NPZ, allow_pickle=True)
    ids = npz["clue_id"]
    emb = npz["emb"].astype(np.float32)
    if normalize:
        emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    return ids, emb


def load_clue_ids() -> np.ndarray:
    """Canonical clue_id order WITHOUT touching the 531MB embedding matrix
    (npz members load lazily, so this is milliseconds, not tens of seconds)."""
    with np.load(CLUE_EMB_NPZ, allow_pickle=True) as npz:
        return npz["clue_id"]


def load_clues(ids: np.ndarray, columns: list[str] | None = None) -> pd.DataFrame:
    """clue_rows.parquet aligned 1:1 to the canonical clue_id order."""
    if columns is not None and "clue_id" not in columns:
        columns = ["clue_id", *columns]
    df = pd.read_parquet(CLUE_ROWS_PARQUET, columns=columns)
    pos = pd.Index(df["clue_id"]).get_indexer(ids)
    assert (pos >= 0).all(), "clue_id mismatch between embeddings and clue_rows"
    return df.iloc[pos].reset_index(drop=True)


def load_map_labels(ids: np.ndarray) -> pd.DataFrame:
    """The 2D-substrate Toponymy labels (stage 04), aligned to canonical order."""
    lab = pd.read_parquet(TOPONYMY_LABELS_PARQUET)
    pos = pd.Index(lab["clue_id"]).get_indexer(ids)
    assert (pos >= 0).all(), "clue_id mismatch between embeddings and toponymy_labels"
    return lab.iloc[pos].reset_index(drop=True)


# ---------------------------------------------------------------- kNN graph
def _knn_signature(ids: np.ndarray, k: int) -> np.ndarray:
    return np.array([str(len(ids)), str(ids[0]), str(ids[-1]), str(k)])


def build_ambient_knn(k: int = K_NEIGHBORS, chunk: int = 2048) -> np.ndarray:
    """Exact cosine kNN over the full corpus via chunked matmul (~3 min)."""
    ids, emb = load_embeddings()
    n = len(ids)
    idx = np.empty((n, k), dtype=np.int32)
    t0 = time.time()
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        sims = emb[s:e] @ emb.T
        part = np.argpartition(-sims, k + 1, axis=1)[:, : k + 1]
        for r in range(e - s):
            cand = part[r]
            cand = cand[cand != s + r]
            idx[s + r] = cand[np.argsort(-sims[r, cand])][:k]
        if (s // chunk) % 15 == 0:
            print(f"  kNN {e:,}/{n:,} ({time.time() - t0:.0f}s)", flush=True)
    atomic_save_npz(AMBIENT_KNN_NPZ, idx=idx, clue_id=ids, signature=_knn_signature(ids, k))
    print(f"kNN cache written: {AMBIENT_KNN_NPZ} ({time.time() - t0:.0f}s)")
    return idx


def ambient_knn(k: int = K_NEIGHBORS) -> np.ndarray:
    """Cached ambient kNN indices [n, k], canonical row order; builds if stale."""
    if AMBIENT_KNN_NPZ.exists():
        npz = np.load(AMBIENT_KNN_NPZ, allow_pickle=True)
        if np.array_equal(npz["signature"], _knn_signature(load_clue_ids(), k)):
            return npz["idx"]
        print("kNN cache stale (signature mismatch); rebuilding")
    return build_ambient_knn(k)


def top1_neighbor_sim() -> np.ndarray:
    """Cosine similarity of each clue to its nearest ambient neighbor.

    Cached: the one-time compute is the only reason a downstream consumer
    (e.g. the report) would need the embedding matrix at all."""
    sig = _knn_signature(load_clue_ids(), K_NEIGHBORS)
    if TOP1_SIM_NPZ.exists():
        npz = np.load(TOP1_SIM_NPZ, allow_pickle=True)
        if np.array_equal(npz["signature"], sig):
            return npz["sim"]
    _, emb = load_embeddings()
    nbr = ambient_knn()
    sim = np.einsum("ij,ij->i", emb, emb[nbr[:, 0]]).astype(np.float32)
    atomic_save_npz(TOP1_SIM_NPZ, sim=sim, signature=sig)
    return sim


# ------------------------------------------------------- permutation machinery
def _permute_within_strata(
    values: np.ndarray, strata_order: np.ndarray, bounds: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Shuffle values within contiguous strata segments given by (order, bounds)."""
    out = values.copy()
    vo = values[strata_order]
    for a, b in zip(bounds[:-1], bounds[1:]):
        seg = vo[a:b].copy()
        rng.shuffle(seg)
        out[strata_order[a:b]] = seg
    return out


def _strata_layout(strata: np.ndarray | None, n: int) -> tuple[np.ndarray, np.ndarray]:
    if strata is None:
        strata = np.zeros(n, dtype=np.int8)
    order = np.argsort(strata, kind="stable")
    _, starts = np.unique(strata[order], return_index=True)
    return order, np.append(starts, n)


def neighbor_rate_test(
    nbr_idx: np.ndarray, y: np.ndarray, n_perm: int = 500, strata: np.ndarray | None = None, seed: int = 42
) -> dict:
    """Binary y: rate of y=1 among neighbors of y=1 points, vs (stratified) shuffle.

    The neighbor pool is the full graph; only the LABELS are permuted, so any
    subgroup restriction is expressed through `strata` (a stratum where y is
    constant is never mixed with the rest).
    """
    rng = np.random.default_rng(seed)
    y = y.astype(bool)

    def stat(labels):
        return float(labels[nbr_idx[labels]].mean())

    obs = stat(y)
    order, bounds = _strata_layout(strata, len(y))
    null = np.empty(n_perm)
    for i in range(n_perm):
        null[i] = stat(_permute_within_strata(y, order, bounds, rng))
    base = float(y.mean())
    return {
        "stat_obs": obs,
        "null_mean": float(null.mean()),
        "null_sd": float(null.std()),
        "z": float((obs - null.mean()) / null.std()),
        "lift": float(obs / null.mean()),
        "base_rate": base,
        "p_emp": float((np.sum(null >= obs) + 1) / (n_perm + 1)),
        "n_pos": int(y.sum()),
    }


def categorical_neighbor_test(
    nbr_idx: np.ndarray, labels: np.ndarray, n_perm: int = 300, min_class_n: int = 300, seed: int = 42
) -> list[dict]:
    """Per-class same-label neighbor rate vs raw shuffle; one result row per class."""
    rng = np.random.default_rng(seed)
    classes = [c for c, n in zip(*np.unique(labels, return_counts=True)) if n >= min_class_n]

    def stats(lab):
        return [float(np.mean(lab[nbr_idx[lab == c]] == c)) for c in classes]

    obs = stats(labels)
    null = np.empty((n_perm, len(classes)))
    for i in range(n_perm):
        null[i] = stats(rng.permutation(labels))
    rows = []
    for j, c in enumerate(classes):
        mu, sd = null[:, j].mean(), null[:, j].std()
        rows.append(
            {
                "class": str(c),
                "stat_obs": obs[j],
                "null_mean": float(mu),
                "null_sd": float(sd),
                "z": float((obs[j] - mu) / sd),
                "lift": float(obs[j] / mu),
                "base_rate": float(np.mean(labels == c)),
                "n_pos": int(np.sum(labels == c)),
            }
        )
    return rows


def continuous_neighbor_test(
    nbr_idx: np.ndarray,
    x: np.ndarray,
    n_perm: int = 200,
    mask: np.ndarray | None = None,
    rank: bool = True,
    seed: int = 42,
) -> dict:
    """Assortativity of a continuous field on the kNN graph (Moran's-I flavored).

    Correlates each point's value with its neighbors' mean value; permutation
    null re-gathers neighbor means from shuffled values. `mask` restricts which
    points count as sources (their neighbors may be anywhere); masked-out points
    still receive permuted values so the graph stays intact. `rank=True` uses
    rank-transformed values (Spearman flavor) for skew robustness.
    """
    rng = np.random.default_rng(seed)
    if mask is None:
        mask = np.ones(len(x), dtype=bool)
    valid = mask & np.isfinite(x)
    v = pd.Series(x).rank().to_numpy(np.float64) if rank else x.astype(np.float64)
    v_fill = np.where(np.isfinite(v), v, np.nanmean(v[valid]))

    def stat(vals):
        nbr_mean = vals[nbr_idx[valid]].mean(axis=1)
        return float(np.corrcoef(vals[valid], nbr_mean)[0, 1])

    obs = stat(v_fill)
    null = np.empty(n_perm)
    for i in range(n_perm):
        null[i] = stat(rng.permutation(v_fill))
    return {
        "stat_obs": obs,
        "null_mean": float(null.mean()),
        "null_sd": float(null.std()),
        "z": float((obs - null.mean()) / null.std()),
        "n_pos": int(valid.sum()),
    }


# ------------------------------------------------------- region hierarchy
def dmp_region_tree() -> pd.DataFrame:
    """The region hierarchy over EVoC layers 5 -> 4 -> 3 as a tidy node table.

    Built the way DataMapPlot builds its topic tree (interactive_helpers.
    _find_parent_id): a cluster's parent is its int-cast MEDIAN provenance
    through the coarser layers' point-level labels, with noise (-1) as a
    first-class value — so "Minor subtopics" nodes are real branches holding
    the named finer regions in a parent's unclustered space, bottoming out in
    residual leaves. Clue placement is honor-then-residual (ramify): each clue
    counts at its finest named cluster's node, so leaves partition all clues
    and branchvalues="total" areas are exact. Single-child chains are
    collapsed (DataMapPlot's remove_duplicate_chains: EVoC keeps clusters
    stable across scales — 13 of 27 layer-4 clusters are point-identical to a
    layer-3 cluster — and Toponymy's duplicate-name disambiguation is
    within-layer only, so persistent clusters otherwise render as "X > X");
    never across the named/Minor boundary.

    Rows are DFS-ordered with siblings sorted named-by-size and Minor last —
    feed straight into go.Treemap with sort=False. Columns: id, label, parent
    ("" for the root), n, dj_share, dd_lift (placement-adjusted), fj_lift.
    """
    ids = load_clue_ids()
    clues = load_clues(ids, columns=["round", "board_row", "daily_double"])
    evoc = pd.read_parquet(EVOC_LABELS_PARQUET)
    assert (evoc["clue_id"].to_numpy() == ids).all()
    with np.load(EVOC_CLUSTERS_NPZ, allow_pickle=True) as cl:
        assert np.array_equal(cl["clue_id"], ids)
        l3, l4, l5 = (cl[f"layer_{i}"].astype(int) for i in (3, 4, 5))
    names3, names4, names5 = (evoc[f"evoc_label_{i}"].to_numpy() for i in (3, 4, 5))

    root = "b"
    minor = "Minor subtopics"
    node_name = {root: "All clues"}
    node_parent: dict[str, str] = {}

    def ensure(nid, parent, name):
        if nid not in node_name:
            node_name[nid] = name
            node_parent[nid] = parent
        return nid

    node5 = {int(g): ensure(f"{root}_{g}", root, names5[l5 == g][0]) for g in np.unique(l5[l5 >= 0])}
    node5[-1] = ensure(f"{root}_-1", root, minor)  # the coarse-scale noise container

    node4 = {}
    for p in np.unique(l4[l4 >= 0]):
        p5 = int(np.median(l5[l4 == p]))  # DataMapPlot's estimator, int-cast median
        node4[int(p)] = ensure(f"{node5[p5]}_{p}", node5[p5], names4[l4 == p][0])

    def minor4(p5):
        return ensure(f"{node5[p5]}_-1", node5[p5], minor)

    node3 = {}
    for c in np.unique(l3[l3 >= 0]):
        members = l3 == c
        p4 = int(np.median(l4[members]))
        # snap to the layer-4 cluster's canonical node (DataMapPlot joins raw
        # id strings, which can dangle when the two layers' medians disagree)
        parent = node4[p4] if p4 >= 0 else minor4(int(np.median(l5[members])))
        node3[int(c)] = ensure(f"{parent}_{c}", parent, names3[members][0])

    def residual_leaf(p4, p5):
        parent = node4[p4] if p4 >= 0 else minor4(p5)
        return ensure(f"{parent}_-1", parent, minor)

    leaf = [node3[a] if a >= 0 else residual_leaf(b, g) for a, b, g in zip(l3, l4, l5)]

    remap: dict[str, str] = {}
    while True:
        kids_tmp: dict[str, list[str]] = {}
        for n_, p_ in node_parent.items():
            kids_tmp.setdefault(p_, []).append(n_)
        collapsed = False
        for n_ in list(node_name):
            if n_ not in node_name or n_ == root:
                continue
            ch = kids_tmp.get(n_, [])
            if len(ch) != 1:
                continue
            child = ch[0]
            if (node_name[n_] == minor) != (node_name[child] == minor):
                continue
            for grand in kids_tmp.get(child, []):
                node_parent[grand] = n_
            remap[child] = n_
            del node_name[child], node_parent[child]
            collapsed = True
        if not collapsed:
            break

    def resolve(x):
        while x in remap:
            x = remap[x]
        return x

    leaf = [resolve(x) for x in leaf]
    p_cell = clues.groupby(["round", clues["board_row"].fillna(-1)], observed=True)["daily_double"].transform("mean")
    p_fj_all = float((clues["round"] == "final_jeopardy").mean())
    leaf_stats = (
        pd.DataFrame(
            {
                "leaf": leaf,
                "dd": clues["daily_double"].fillna(False).astype(bool),
                "dd_exp": p_cell.fillna(0.0),
                "dj": clues["round"] == "double_jeopardy",
                "main": clues["round"].isin(["jeopardy", "double_jeopardy"]),
                "fj": clues["round"] == "final_jeopardy",
            }
        )
        .groupby("leaf")
        .agg(
            n=("dd", "size"),
            dd=("dd", "sum"),
            dd_exp=("dd_exp", "sum"),
            dj=("dj", "sum"),
            main=("main", "sum"),
            fj=("fj", "sum"),
        )
    )
    totals = {nid: np.zeros(6) for nid in node_name}
    for leaf_id, row in leaf_stats.iterrows():
        vec = row.to_numpy(float)
        nid = leaf_id
        while True:
            totals[nid] += vec
            if nid == root:
                break
            nid = node_parent.get(nid, root)

    kids: dict[str, list[str]] = {}
    for nid, par in node_parent.items():
        kids.setdefault(par, []).append(nid)

    rows: list[dict] = []

    def emit(nid):
        n, dd, dd_exp, dj, main_n, fj = totals[nid]
        rows.append(
            {
                "id": nid,
                "label": node_name[nid],
                "parent": "" if nid == root else node_parent[nid],
                "n": n,
                "dj_share": dj / main_n if main_n else 0.5,
                "dd_lift": dd / dd_exp if dd_exp else np.nan,
                "fj_lift": (fj / n) / p_fj_all if n else np.nan,
            }
        )
        # named children by size, "Minor subtopics" pinned last (ramify's ordering)
        for kid in sorted(kids.get(nid, []), key=lambda k: (node_name[k] == minor, -totals[k][0])):
            emit(kid)

    emit(root)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    build_ambient_knn()

"""Static datamap of unique category titles, colored by Daily Double rate.

One node = one unique main-round category title (36's exact title recipe, so the
cached title embeddings are reused verbatim; run 36 first). Node size = how many
times the title aired (an appearance = one episode x round x title). Color =
appearance-weighted DD rate over the title's 25 nearest neighbors in embedding
space, on a diverging scale centered at the pooled base rate (~25%): red =
DD-cold neighborhood, blue = DD-hot, near-average recedes. Region labels are
word-cloud style (over the points, size ~ cluster size, tempered to 9-28pt);
the subtitle spells out the red/blue encoding, so the figure is
self-explanatory without its blog caption (a board-avg marker on the colorbar
was tried and cut as redundant with the subtitle). Design decisions, settled
by comparing rendered alternatives (2026-07-14):

  - kNN-smoothed rate, NOT the title's own raw DD fraction: 88.9% of titles air
    exactly once, so the raw fraction is 0/1 speckle with no visible spatial
    story. The smoothed color also literally IS the blog claim ("embedding
    neighborhoods predict DDs").
  - Diverging around base, NOT a sequential ramp: both tried; sequential left
    the mid-heavy map mushy, and either a light or dark sequential endpoint
    disappears against the white page.
  - Rates are pooled across rounds; a DJ-leaning title picks up ~2x
    mechanically -- caption it, don't adjust it (per-round rigor lives in 36).
  - Points colored by non-cluster data follow the datamapplot colour_controls
    recipe: color_label_text=False, add_glow=False (the glow is cluster-palette
    colored and paints fake regional signal over a value colormap).

Phases, so the free parts are never hostage to the paid part (mirrors 10):
  1. per-title stats + cached title embeddings (free)
  2. UMAP of the 21.6k titles, stage-03 params      -> data/analysis/title_umap.npz
  3. ToponymyClusterer on the 2D coords (free), cost guard, then Toponymy/Haiku
     region naming (~$1.60, the only paid step)     -> data/analysis/title_region_labels.parquet
  4. datamapplot static render, one label layer     -> data/analysis/blog/fig7_title_map.png

Delete the npz/parquet caches to recompute; the render always reruns (~10s).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import nest_asyncio
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import ANALYSIS_DIR, atomic_save_npz, atomic_write_parquet  # noqa: E402
from config import (  # noqa: E402  (pipeline dir put on sys.path by common)
    ANTHROPIC_API_KEY,
    ANTHROPIC_MAX_CONCURRENCY,
    ANTHROPIC_MODEL_NAMING,
    CLUE_ROWS_PARQUET,
    CO_API_KEY,
    COHERE_EMBED_MODEL,
    COHERE_INPUT_TYPE,
    COHERE_OUTPUT_DIM,
    UMAP_MIN_DIST,
    UMAP_N_NEIGHBORS,
    UMAP_RANDOM_STATE,
)

nest_asyncio.apply()

TITLE_EMB_NPZ = ANALYSIS_DIR / "title_embeddings.npz"  # written by 36
TITLE_UMAP_NPZ = ANALYSIS_DIR / "title_umap.npz"
TITLE_LABELS_PARQUET = ANALYSIS_DIR / "title_region_labels.parquet"
BLOG_DIR = ANALYSIS_DIR / "blog"

SEED = 42
KNN_K = 25  # neighborhood for the smoothed DD rate (self included)
TARGET_STATIC_LABELS = 20  # render the label layer closest to this many regions
DPI = 200
# Naming cost anchor from 10: ~$2.15 per 1k regions on Haiku. Titles are far
# shorter documents than clues, so this over- rather than under-estimates.
EST_USD_PER_1K_REGIONS = 2.15
ABORT_ABOVE_USD = 5.0


# -------------------------------------------------------------- title stats
def title_stats() -> pd.DataFrame:
    """Per-title appearance counts and raw DD share, 36's exact title recipe."""
    cols = ["episode_id", "round", "board_row", "daily_double", "category"]
    df = pd.read_parquet(CLUE_ROWS_PARQUET, columns=cols)
    d = df[df["round"].isin(["jeopardy", "double_jeopardy"]) & df["board_row"].notna()].copy()
    d["title"] = d["category"].fillna("").str.strip().str.upper().replace("", "UNTITLED")
    d["dd"] = d["daily_double"].eq(True)
    app = d.groupby(["episode_id", "round", "title"], observed=True)["dd"].max().reset_index()
    stats = (
        app.groupby("title", observed=True)
        .agg(n=("dd", "size"), dd_n=("dd", "sum"), dd_frac=("dd", "mean"))
        .reset_index()
        .sort_values("title", kind="stable")  # canonical order = sorted titles, matching 36's np.unique
        .reset_index(drop=True)
    )
    dj = app[app["round"] == "double_jeopardy"].groupby("title", observed=True).size()
    stats["dj_share"] = (dj.reindex(stats["title"]).fillna(0).to_numpy() / stats["n"]).astype(np.float32)
    return stats


def load_title_embeddings(titles: np.ndarray) -> np.ndarray:
    """Cache-only read of 36's incremental store; unit-normalized, aligned to titles."""
    sig = f"{COHERE_EMBED_MODEL}_{COHERE_INPUT_TYPE}_{COHERE_OUTPUT_DIM}"
    if not TITLE_EMB_NPZ.exists():
        raise RuntimeError(f"{TITLE_EMB_NPZ} missing; run analysis/36_dd_information_set.py first")
    npz = np.load(TITLE_EMB_NPZ, allow_pickle=True)
    if str(npz["sig"]) != sig:
        raise RuntimeError(f"{TITLE_EMB_NPZ} signature mismatch ({npz['sig']} != {sig}); re-run 36")
    pos = {t: i for i, t in enumerate(npz["titles"])}
    missing = [t for t in titles if t not in pos]
    if missing:
        raise RuntimeError(f"{len(missing)} titles absent from the embedding cache (e.g. {missing[:3]}); re-run 36")
    emb = npz["emb"][[pos[t] for t in titles]].astype(np.float32)
    return emb / np.linalg.norm(emb, axis=1, keepdims=True)


# --------------------------------------------------------------------- umap
def title_umap(emb: np.ndarray, titles: np.ndarray) -> np.ndarray:
    sig = f"{len(titles)}_{titles[0]}_{titles[-1]}_{UMAP_N_NEIGHBORS}_{UMAP_MIN_DIST}"
    if TITLE_UMAP_NPZ.exists():
        npz = np.load(TITLE_UMAP_NPZ, allow_pickle=True)
        if str(npz["sig"]) == sig:
            print("UMAP: cache hit", flush=True)
            return npz["coords"].astype(np.float32)
    import umap

    t0 = time.time()
    reducer = umap.UMAP(
        metric="cosine",
        n_neighbors=UMAP_N_NEIGHBORS,
        min_dist=UMAP_MIN_DIST,
        random_state=UMAP_RANDOM_STATE,
    )
    coords = reducer.fit_transform(emb).astype(np.float32)
    atomic_save_npz(TITLE_UMAP_NPZ, sig=sig, coords=coords, titles=titles)
    print(f"UMAP done in {time.time() - t0:.0f}s -> {TITLE_UMAP_NPZ.name}", flush=True)
    return coords


# ------------------------------------------------------------ region naming
def region_labels(titles: np.ndarray, emb: np.ndarray, coords: np.ndarray) -> pd.DataFrame:
    """Toponymy region names per title (label_layer_0 = finest), cached parquet."""
    if TITLE_LABELS_PARQUET.exists():
        df = pd.read_parquet(TITLE_LABELS_PARQUET)
        if len(df) == len(titles) and (df["title"].to_numpy() == titles).all():
            print("region names: cache hit", flush=True)
            return df
        print("region names: cache stale (title set changed), recomputing", flush=True)

    from toponymy import Toponymy, ToponymyClusterer
    from toponymy.cluster_layer import ClusterLayerText
    from toponymy.embedding_wrappers import CohereEmbedder
    from toponymy.llm_wrappers import AsyncAnthropicNamer

    # Pre-fit the clusterer ourselves (free) so the cost guard can see the
    # region count before any API call; Toponymy's documented reuse branch then
    # picks up cluster_layers_/cluster_tree_ as-is (same path 10 uses).
    t0 = time.time()
    clusterer = ToponymyClusterer(min_clusters=6)
    clusterer.fit(coords, emb, layer_class=ClusterLayerText)
    counts = [int(layer.cluster_labels.max()) + 1 for layer in clusterer.cluster_layers_]
    n_regions = sum(counts)
    print(f"clustering done in {time.time() - t0:.0f}s; layers (finest first): {counts}", flush=True)

    est = n_regions / 1_000 * EST_USD_PER_1K_REGIONS
    print(f"{n_regions:,} regions -> estimated naming cost ~${est:.2f} on {ANTHROPIC_MODEL_NAMING}", flush=True)
    if est > ABORT_ABOVE_USD:
        raise RuntimeError(f"naming estimate exceeds ${ABORT_ABOVE_USD:.0f} guard; nothing spent")
    if not (CO_API_KEY and ANTHROPIC_API_KEY):
        raise RuntimeError("CO_API_KEY and ANTHROPIC_API_KEY required for region naming; add to .env")

    llm = AsyncAnthropicNamer(
        api_key=ANTHROPIC_API_KEY,
        model=ANTHROPIC_MODEL_NAMING,
        max_concurrent_requests=ANTHROPIC_MAX_CONCURRENCY,
    )
    embedder = CohereEmbedder(api_key=CO_API_KEY, model=COHERE_EMBED_MODEL)
    topic_model = Toponymy(
        llm_wrapper=llm,
        text_embedding_model=embedder,
        clusterer=clusterer,  # pre-fitted; Toponymy.fit skips reclustering
        object_description="Jeopardy category titles",
        corpus_description="Jeopardy category titles (a few words each, often puns or wordplay)",
        lowest_detail_level=0.5,
        highest_detail_level=1.0,
    )
    np.random.seed(SEED)
    t1 = time.time()
    topic_model.fit(objects=list(titles), embedding_vectors=emb, clusterable_vectors=coords)
    print(f"naming done in {time.time() - t1:.0f}s", flush=True)

    out = {"title": titles}
    for i, names in enumerate(topic_model.topic_name_vectors_):
        out[f"label_layer_{i}"] = names
    df = pd.DataFrame(out)
    atomic_write_parquet(df, TITLE_LABELS_PARQUET)
    print(f"wrote {TITLE_LABELS_PARQUET} ({len(counts)} layers)", flush=True)
    return df


# ------------------------------------------------------------ knn smoothing
def knn_smoothed_rate(emb: np.ndarray, n_app: np.ndarray, dd_n: np.ndarray) -> np.ndarray:
    """Appearance-weighted DD rate over each title's KNN_K nearest titles (cosine, self included)."""
    n = len(emb)
    out = np.empty(n, dtype=np.float32)
    for s in range(0, n, 2048):
        sims = emb[s : s + 2048] @ emb.T
        idx = np.argpartition(-sims, KNN_K, axis=1)[:, : KNN_K + 1]
        out[s : s + 2048] = dd_n[idx].sum(axis=1) / n_app[idx].sum(axis=1)
    return out


# ------------------------------------------------------------------- render
def diverging_cmap():
    """House diverging palette (75's POLE_LO/POLE_HI) with a light-but-visible
    neutral: red-orange = below the base DD rate, blue = above, and only the
    near-average middle recedes -- both extremes stay saturated on white."""
    from matplotlib.colors import LinearSegmentedColormap

    stops = ["#c73635", "#e2a89e", "#e8e6df", "#a9c3e2", "#1c5cab"]
    return LinearSegmentedColormap.from_list("dd_div", stops)


def render(
    coords: np.ndarray,
    names: np.ndarray,
    sizes: np.ndarray,
    values: np.ndarray,
    cmap,
    norm,
    cbar_label: str,
    title: str,
    sub_title: str,
    out_png: Path,
    label_style: dict | None = None,
) -> None:
    import datamapplot
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_hex
    from matplotlib.ticker import PercentFormatter

    point_colors = np.array([to_hex(c) for c in cmap(norm(values))])
    fig, ax = datamapplot.create_plot(
        coords,
        names,
        noise_label="Unlabelled",
        noise_color="#c9c8c1",
        marker_color_array=point_colors,
        marker_size_array=sizes,
        # the docs' recipe for points colored by non-cluster data: neutral label
        # text and no glow, since both otherwise use the cluster palette and
        # fight the value colormap (datamapplot colour_controls docs)
        color_label_text=False,
        add_glow=False,
        force_matplotlib=True,  # datashader path ignores per-point sizes
        figsize=(12, 12),
        dpi=DPI,
        title=title,
        sub_title=sub_title,
        **(label_style or {}),
    )
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.01, shrink=0.55)
    cbar.set_label(cbar_label, fontsize=9)
    cbar.ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    cbar.ax.tick_params(labelsize=8)
    fig.savefig(out_png, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_png}", flush=True)


def main() -> None:
    t0 = time.time()
    BLOG_DIR.mkdir(exist_ok=True)
    stats = title_stats()
    titles = stats["title"].to_numpy()
    n_app = stats["n"].to_numpy(np.float32)
    dd_n = stats["dd_n"].to_numpy(np.float32)
    base_rate = dd_n.sum() / n_app.sum()
    print(f"{len(titles):,} unique titles; {(n_app == 1).mean():.1%} air once; base DD share {base_rate:.1%}")

    emb = load_title_embeddings(titles)
    coords = title_umap(emb, titles)
    labels = region_labels(titles, emb, coords)

    # label layer for the static render: closest named-region count to target
    layer_cols = [c for c in labels.columns if c.startswith("label_layer_")]
    k_named = {c: labels.loc[labels[c] != "Unlabelled", c].nunique() for c in layer_cols}
    layer = min(layer_cols, key=lambda c: abs(k_named[c] - TARGET_STATIC_LABELS))
    names = labels[layer].to_numpy()
    print(f"label layers {k_named} -> rendering {layer} ({k_named[layer]} regions)")

    smoothed = knn_smoothed_rate(emb, n_app, dd_n)
    sizes = 2.0 + 3.0 * np.sqrt(n_app)

    # crop layout outliers so a handful of far-flung titles don't waste half the
    # canvas (never silent: the count is printed)
    lo = np.percentile(coords, 0.1, axis=0)
    hi = np.percentile(coords, 99.9, axis=0)
    pad = 0.03 * (hi - lo)
    keep = ((coords >= lo - pad) & (coords <= hi + pad)).all(axis=1)
    print(f"render crop: dropping {(~keep).sum()} outlier titles of {len(keep):,}")
    coords, names, sizes = coords[keep], names[keep], sizes[keep]
    n_app, dd_n = n_app[keep], dd_n[keep]
    smoothed = smoothed[keep]
    # size (= times aired) stays deliberately undescribed: "bigger = airs more"
    # is the intuitive read, and the subtitle was carrying too much
    sub = (
        f"one dot = one unique category title, 2016{chr(0x2013)}2025; "
        f"red = fewer Daily Doubles than the {base_rate:.0%} average, blue = more"
    )
    from matplotlib.colors import TwoSlopeNorm

    lo, hi = (float(p) for p in np.percentile(smoothed, [2, 98]))
    # word-cloud style labels over the points (datamapplot label_over_points
    # docs recipe), sized by cluster size. NOTE: sizes are MinMax-scaled onto
    # [min_font_size, max_font_size], so the biggest/smallest clusters ALWAYS
    # get the range endpoints; the scaling factor only bends the curve for
    # mid-sized clusters -- tempering the spread means narrowing the range,
    # not just lowering the factor.
    label_style = dict(
        label_over_points=True,
        label_wrap_width=12,
        font_family="Roboto Condensed",
        dynamic_label_size=True,
        dynamic_label_size_scaling_factor=0.5,  # default 0.75
        max_font_size=28,
        min_font_size=9,
        min_font_weight=100,
        max_font_weight=1000,
    )
    render(
        coords,
        names,
        sizes,
        smoothed,
        diverging_cmap(),
        TwoSlopeNorm(vmin=lo, vcenter=base_rate, vmax=hi),
        cbar_label="share of airings with a Daily Double (title neighborhood)",
        title="Where the Daily Doubles are, by category-title neighborhood",
        sub_title=sub,
        out_png=BLOG_DIR / "fig7_title_map.png",
        label_style=label_style,
    )

    # console summary for the claim inventory: extremes by neighborhood rate
    named = pd.DataFrame({"region": names, "napp": n_app, "dd": dd_n}).query("region != 'Unlabelled'")
    reg = named.groupby("region").agg(rate=("dd", "sum"), n=("napp", "sum"))
    reg["rate"] /= reg["n"]
    reg = reg[reg["n"] >= 200].sort_values("rate")
    print("\nregion DD share extremes (>=200 appearances):")
    print(pd.concat([reg.head(5), reg.tail(5)]).round(3).to_string())
    print(f"\ndone in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()

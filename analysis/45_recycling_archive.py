"""Clue-recycling cadence over the FULL 1983-2025 archive (censoring fix).

The report's recycling figure (70_report.py F8) measures season gaps between
embedding near-duplicates *within the 2016+ window*, so gaps are right-censored
at ~9 seasons — a "peak at 7-8" sits at the censoring edge and could be an
artifact of it. This script recomputes the cadence over all ~568k raw clues,
where gaps up to 40 seasons are observable. No embeddings, no API calls: it
reads only jeopardy_raw.parquet (stage 00) and uses two independent
near-duplicate definitions:

  A. The dataset's own repeat linkage (`repeat_clue_ids`: TF-IDF cosine >= 0.85
     against earlier clues, computed archive-wide by the dataset authors).
     Headline variant: gap to the MOST RECENT earlier source (the writers'-room
     quantity: "how long since this material last aired"); all (repeat, source)
     edges kept as a secondary variant.
  B. Lexical near-dupes computed here, blocked by normalized answer (recycled
     clues keep their answer): within an answer block, token-set Jaccard on
     normalized clue text >= JACCARD_MIN. Exact normalized-text duplicates are
     a by-product (jaccard == 1). Blocks over BLOCK_CAP unique texts are
     LOGGED and subsampled, never silently truncated.

Gap-0 forensics (2026-07): raw near-dupe pairs at gap 0 are overwhelmingly
duplicate RECORDS of the same broadcast clue — author-linkage gap-0 pairs are
81% same episode_id / 88% same air_date / 100% same category (escaped-quote
text variants that survived the dataset's cross-source dedup), plus a few
±7-day date-noise twins. Pairs with the same episode_id or air dates closer
than MIN_DAYS_APART are therefore excluded (counted and printed, never
silent); what remains at gap 0 is genuine same-season reuse across episodes.

Expected-gap null (per method): recycling that ignores time. Each repeat in
season t draws its source uniformly from all archive clues in seasons <= t;
the expected gap distribution is the mixture of earlier-season sizes over the
observed repeat seasons. Observed share / expected share per gap is the
cadence profile. (One-sided by construction — cleaner than the report's
symmetric si*sj null, which this script also reproduces for the window
comparison figure.)

The recycling process is NON-STATIONARY, so pooled-archive cadence is also
decomposed by repeat era (scopes "s<32", "s32-39", "s40-41", each against its
own era-conditioned null): seasons 40-41 (2023-2025, the WGA-strike and
post-strike era, when the show said on the record it would redeploy
previously-written material) recycle at a far higher rate and reach much
deeper into the back catalog than any earlier era. The era × era pair matrix
also shows the dataset's own repeat linkage (method A) under-detects
old-era repeats (its S40-41 rows dwarf everything; the lexical method finds
comparable recycling in every era), so METHOD B (LEXICAL) IS THE PRIMARY
ESTIMATE here and method A is corroboration for recent eras only.

Outputs (data/analysis/, all atomic):
  recycling_pairs.parquet    one row per near-dupe pair (method, ids, seasons,
                             gap, jaccard where applicable) — the figure cache
  recycling_archive.parquet  tidy per-gap rows: (method, scope, gap, n_pairs,
                             observed_share, expected_share, ratio) where
                             scope is "archive" or "window2016" (both ends in
                             the 2016+ window, i.e. the censored view)
Printed: peak gap, dead-zone depletion, and the share of recycling pairs the
2016+ window could not see (the censored mass).
"""

from __future__ import annotations

import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import ANALYSIS_DIR, atomic_write_parquet  # noqa: E402
from config import JEOPARDY_RAW_PARQUET  # noqa: E402  (pipeline dir on path via common)

PAIRS_PARQUET = ANALYSIS_DIR / "recycling_pairs.parquet"
CADENCE_PARQUET = ANALYSIS_DIR / "recycling_archive.parquet"
STRIKES_PARQUET = ANALYSIS_DIR / "recycling_strike_months.parquet"

JACCARD_MIN = 0.6  # near-dupe threshold for method B (pairs stored from 0.5)
JACCARD_STORE = 0.5
BLOCK_CAP = 2000  # max unique texts per answer block before subsampling (logged)
WINDOW_START_SEASON = 32  # season 32 began Sept 2015; 2016+ window ~ S32-41
MIN_DAYS_APART = 90  # below this (or same episode), a "pair" is duplicate records, not recycling
MAX_GAP = 41
SEED = 42

_norm_re = re.compile(r"[^a-z0-9 ]+")


def _norm(s: pd.Series) -> pd.Series:
    return s.fillna("").str.lower().str.replace(_norm_re, " ", regex=True).str.split().str.join(" ")


# --------------------------------------------------------------- method A
def author_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """(repeat, source) edges from repeat_clue_ids, with seasons and gaps."""
    rep = df.loc[df["repeat_clue_ids"].str.len() > 0, ["clue_id", "season", "repeat_clue_ids"]]
    edges = rep.explode("repeat_clue_ids").rename(columns={"repeat_clue_ids": "src_id"})
    season_of = df.set_index("clue_id")["season"]
    edges["src_season"] = edges["src_id"].map(season_of)
    n_missing = int(edges["src_season"].isna().sum())
    if n_missing:
        print(f"  [A] {n_missing} source ids not found in archive (dropped)")
    edges = edges.dropna(subset=["src_season"])
    edges = edges[edges["clue_id"] != edges["src_id"]]
    edges["gap"] = edges["season"].astype(int) - edges["src_season"].astype(int)
    n_neg = int((edges["gap"] < 0).sum())
    if n_neg:
        # "earlier" per the dataset card; violations are cross-source dedup noise
        print(f"  [A] {n_neg} edges with source AFTER repeat ({n_neg / len(edges):.1%}) (dropped)")
        edges = edges[edges["gap"] >= 0]
    return edges.rename(columns={"clue_id": "id", "src_id": "id_src", "season": "s", "src_season": "s_src"})


# --------------------------------------------------------------- method B
def lexical_pairs(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Same-answer token-Jaccard near-dupe pairs (>= JACCARD_STORE)."""
    t0 = time.time()
    work = pd.DataFrame(
        {
            "clue_id": df["clue_id"],
            "season": df["season"].astype(int),
            "ans": _norm(df["answer"]),
            "txt": _norm(df["clue_text"]),
        }
    )
    work = work[(work["ans"] != "") & (work["txt"] != "")]
    rows: list[tuple] = []
    n_blocks = n_capped = n_pairs_checked = 0
    for _, grp in work.groupby("ans", sort=False):
        if len(grp) < 2:
            continue
        # identical normalized texts: link every later occurrence to each earlier one
        uniq = grp.drop_duplicates("txt")
        if grp["txt"].duplicated().any():
            for _, dgrp in grp[grp["txt"].duplicated(keep=False)].groupby("txt", sort=False):
                recs = dgrp.sort_values("season").to_records(index=False)
                for i in range(1, len(recs)):
                    for j in range(i):
                        rows.append((recs[i]["clue_id"], recs[j]["clue_id"], recs[i]["season"], recs[j]["season"], 1.0))
        if len(uniq) < 2:
            continue
        n_blocks += 1
        if len(uniq) > BLOCK_CAP:
            n_capped += 1
            print(f"  [B] block '{grp['ans'].iloc[0][:40]}' has {len(uniq)} unique texts; sampling {BLOCK_CAP}")
            uniq = uniq.sample(BLOCK_CAP, random_state=rng.integers(2**31))
        toks = [frozenset(t.split()) for t in uniq["txt"]]
        ids = uniq["clue_id"].to_numpy()
        seas = uniq["season"].to_numpy()
        for i in range(1, len(toks)):
            ti = toks[i]
            for j in range(i):
                n_pairs_checked += 1
                inter = len(ti & toks[j])
                if not inter:
                    continue
                jac = inter / (len(ti) + len(toks[j]) - inter)
                if jac >= JACCARD_STORE:
                    a, b = (i, j) if seas[i] >= seas[j] else (j, i)
                    rows.append((ids[a], ids[b], seas[a], seas[b], jac))
    out = pd.DataFrame(rows, columns=["id", "id_src", "s", "s_src", "jaccard"])
    out["gap"] = out["s"] - out["s_src"]
    print(
        f"  [B] {n_blocks:,} multi-text answer blocks ({n_capped} capped), "
        f"{n_pairs_checked / 1e6:.1f}M pairs checked, {len(out):,} pairs >= {JACCARD_STORE} "
        f"({time.time() - t0:.0f}s)"
    )
    return out


# --------------------------------------------------------------- cadence math
def uniform_source_expectation(repeat_seasons: np.ndarray, season_sizes: pd.Series) -> np.ndarray:
    """Expected gap shares if each repeat drew its source uniformly from seasons <= its own."""
    sizes = season_sizes.reindex(range(1, int(season_sizes.index.max()) + 1), fill_value=0).to_numpy(float)
    exp = np.zeros(MAX_GAP + 1)
    for t, n_t in zip(*np.unique(repeat_seasons, return_counts=True)):
        pool = sizes[: int(t)]  # seasons 1..t (index s-1)
        if pool.sum() == 0:
            continue
        p = pool / pool.sum()
        gaps = int(t) - np.arange(1, int(t) + 1)
        exp[gaps] += n_t * p
    return exp / exp.sum()


def cadence_rows(pairs: pd.DataFrame, method: str, scope: str, season_sizes: pd.Series) -> list[dict]:
    gaps = pairs["gap"].to_numpy(int)
    obs = np.bincount(gaps, minlength=MAX_GAP + 1)[: MAX_GAP + 1]
    obs_share = obs / obs.sum()
    exp_share = uniform_source_expectation(pairs["s"].to_numpy(int), season_sizes)
    return [
        {
            "method": method,
            "scope": scope,
            "gap": g,
            "n_pairs": int(obs[g]),
            "observed_share": float(obs_share[g]),
            "expected_share": float(exp_share[g]),
            "ratio": float(obs_share[g] / exp_share[g]) if exp_share[g] > 0 else np.nan,
        }
        for g in range(MAX_GAP + 1)
    ]


def summarize(pairs: pd.DataFrame, method: str) -> None:
    gaps = pairs["gap"].to_numpy(int)
    obs = np.bincount(gaps, minlength=MAX_GAP + 1)
    peak = int(np.argmax(obs[1:]) + 1)  # ignore gap 0 for the peak (same-season)
    beyond = float((gaps >= 10).mean())
    print(
        f"  [{method}] {len(pairs):,} pairs | median gap {np.median(gaps):.0f} | modal gap>0 {peak} | "
        f"gap>=10 share {beyond:.1%} (invisible to the 2016+ window)"
    )


def drop_duplicate_records(pairs: pd.DataFrame, meta: pd.DataFrame, tag: str) -> pd.DataFrame:
    """Remove pairs that are duplicate records of one broadcast clue, not recycling."""
    ep = meta["episode_id"]
    dt = meta["air_date"]
    same_ep = ep.loc[pairs["id"]].to_numpy() == ep.loc[pairs["id_src"]].to_numpy()
    days = np.abs((dt.loc[pairs["id"]].to_numpy() - dt.loc[pairs["id_src"]].to_numpy())).astype("timedelta64[D]")
    days = days.astype(float)
    dup = same_ep | (days < MIN_DAYS_APART) | ~np.isfinite(days)
    print(
        f"  [{tag}] dropped {int(dup.sum()):,}/{len(pairs):,} pairs as duplicate records "
        f"(same episode / <{MIN_DAYS_APART}d)"
    )
    out = pairs[~dup].copy()
    out["days_apart"] = days[~dup]
    return out


def main() -> None:
    rng = np.random.default_rng(SEED)
    print(f"loading {JEOPARDY_RAW_PARQUET.name} ...", flush=True)
    df = pd.read_parquet(
        JEOPARDY_RAW_PARQUET,
        columns=[
            "clue_id",
            "season",
            "clue_text",
            "answer",
            "repeat_clue_ids",
            "is_repeat_clue",
            "episode_id",
            "air_date",
        ],
    )
    season_sizes = df["season"].value_counts().sort_index()
    print(f"  {len(df):,} clues, seasons {season_sizes.index.min()}-{season_sizes.index.max()}")
    meta = df.set_index("clue_id")[["episode_id", "air_date"]].copy()
    meta["air_date"] = pd.to_datetime(meta["air_date"], errors="coerce")

    print("method A: dataset repeat linkage (TF-IDF >= 0.85, archive-wide)", flush=True)
    a_all = drop_duplicate_records(author_pairs(df), meta, "A")
    # headline: gap to the MOST RECENT earlier source per repeat clue
    a_nearest = a_all.loc[a_all.groupby("id")["gap"].idxmin()]
    summarize(a_nearest, "A nearest-source")

    print("method B: lexical near-dupes (same answer, token Jaccard)", flush=True)
    b_all = drop_duplicate_records(lexical_pairs(df, rng), meta, "B")
    b_used = b_all[b_all["jaccard"] >= JACCARD_MIN]
    b_nearest = b_used.loc[b_used.groupby("id")["gap"].idxmin()]
    summarize(b_nearest, f"B nearest-source (J>={JACCARD_MIN})")

    pairs_cache = pd.concat(
        [a_all.assign(method="author", jaccard=np.nan), b_all.assign(method="lexical")], ignore_index=True
    )[["method", "id", "id_src", "s", "s_src", "gap", "days_apart", "jaccard"]]
    atomic_write_parquet(pairs_cache, PAIRS_PARQUET)

    era_bands = [("s<32", 1, 31), ("s32-39", 32, 39), ("s40-41", 40, 41)]
    rows: list[dict] = []
    for method, nearest in [
        ("author_nearest", a_nearest),
        ("author_all_edges", a_all),
        ("lexical_nearest", b_nearest),
    ]:
        rows += cadence_rows(nearest, method, "archive", season_sizes)
        window = nearest[(nearest["s"] >= WINDOW_START_SEASON) & (nearest["s_src"] >= WINDOW_START_SEASON)]
        if len(window):
            win_sizes = season_sizes[season_sizes.index >= WINDOW_START_SEASON]
            rows += cadence_rows(window, method, "window2016", win_sizes)
        for band, lo, hi in era_bands:
            sub = nearest[(nearest["s"] >= lo) & (nearest["s"] <= hi)]
            if len(sub):
                rows += cadence_rows(sub, method, band, season_sizes)

    # per-season nearest-source repeat rate (the surge series; lexical = primary)
    rate = (b_nearest.groupby("s").size().reindex(season_sizes.index, fill_value=0) / season_sizes * 1000).rename(
        "repeats_per_1k"
    )
    print("\nlexical nearest-source repeats per 1,000 clues aired, by repeat season:")
    print("  " + " ".join(f"S{s}:{v:.0f}" for s, v in rate.items() if s >= 20))
    rows += [
        {
            "method": "lexical_rate",
            "scope": "per_season",
            "gap": int(s),  # season number, reusing the gap column
            "n_pairs": int(b_nearest[b_nearest["s"] == s].shape[0]),
            "observed_share": float(v / 1000),
            "expected_share": np.nan,
            "ratio": np.nan,
        }
        for s, v in rate.items()
    ]

    # WGA-strike forensics: monthly recycled-clue rate through both strike eras.
    # Strikes: Nov 2007 - Feb 2008 (mid-S24) and May - Sep 2023 (S39/S40 boundary).
    # Jeopardy tapes ~2-3 months ahead, so the recycled material AIRS with a lag:
    # S24's surge lands Feb-Jun 2008, S40's fills most of 2023-24.
    month_all = pd.to_datetime(df["air_date"], errors="coerce").dt.to_period("M")
    month_rep = meta["air_date"].loc[b_nearest["id"]].dt.to_period("M")
    strike_windows = [("2007-01", "2009-06"), ("2022-07", "2025-07")]
    strike_rows = []
    for start, end in strike_windows:
        sel = (month_all >= start) & (month_all <= end)
        totals = month_all[sel].value_counts().sort_index()
        reps = month_rep[(month_rep >= start) & (month_rep <= end)].value_counts().reindex(totals.index, fill_value=0)
        for mth, tot in totals.items():
            strike_rows.append(
                {
                    "air_month": str(mth),
                    "n_clues": int(tot),
                    "n_repeats": int(reps[mth]),
                    "repeats_per_1k": float(reps[mth] / tot * 1000),
                }
            )
    strikes = pd.DataFrame(strike_rows)
    atomic_write_parquet(strikes, STRIKES_PARQUET)
    peak = strikes.loc[strikes["repeats_per_1k"].idxmax()]
    print(
        f"\nstrike-window monthly rates written; peak month {peak['air_month']} "
        f"at {peak['repeats_per_1k']:.0f} recycled/1k (vs ~20/1k baseline)"
    )

    atomic_write_parquet(pd.DataFrame(rows), CADENCE_PARQUET)
    print(
        f"\nwrote {PAIRS_PARQUET.name} ({len(pairs_cache):,} rows), {CADENCE_PARQUET.name} ({len(rows)} rows), "
        f"{STRIKES_PARQUET.name} ({len(strikes)} rows)"
    )

    # the money tables: pooled + era-decomposed lexical cadence, author corroboration
    tidy = pd.DataFrame(rows)
    for method, scope in [
        ("lexical_nearest", "archive"),
        ("lexical_nearest", "s<32"),
        ("lexical_nearest", "s32-39"),
        ("lexical_nearest", "s40-41"),
        ("author_nearest", "s40-41"),
    ]:
        t = tidy.query("method == @method and scope == @scope and gap <= 24")
        if not len(t):
            continue
        print(f"\n{method} [{scope}] (gaps 0-24, n={int(t['n_pairs'].sum())}):")
        print("  gap: " + " ".join(f"{g:>5d}" for g in t["gap"]))
        print("  n  : " + " ".join(f"{n:>5d}" for n in t["n_pairs"]))
        print("  o/e: " + " ".join(f"{r:>5.2f}" for r in t["ratio"]))


if __name__ == "__main__":
    main()

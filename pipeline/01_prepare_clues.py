"""Prepare clues for embedding: window, clean, derive difficulty, build embed input.

Keeps clues in the JEOPARDY_START_DATE window (default last decade); within it we
keep ALL rows including repeat clues (1 row = 1 node, as designed). The robworks
dump is already clean (no HTML); we just normalize whitespace and drop the tiny
fraction with empty clue or answer. `difficulty` is pulled out of the topic_tags
array (a 'difficulty:N' tag); the remaining tags are joined into topic_tags_str for
hover. embed_text is the exact string stage 02 embeds.

Input:  data/jeopardy_raw.parquet   (~568k, full archive)
Output: data/clue_rows.parquet      (~135k in the default window; fewer if MAX_CLUES set)
"""

from __future__ import annotations

import os
import re
import tempfile

import numpy as np
import pandas as pd
from config import CLUE_ROWS_PARQUET, JEOPARDY_RAW_PARQUET, JEOPARDY_START_DATE, MAX_CLUES, SUBSET_SEED

KEEP_COLS = [
    "clue_id",
    "air_date",
    "season",
    "episode_id",
    "round",
    "category",
    "category_normalized",
    "value",
    "daily_double",
    "clue_text",
    "answer",
    "is_repeat_clue",
    "clue_word_count",
    "answer_word_count",
]
_WS = re.compile(r"\s+")
_DIFF = re.compile(r"^difficulty:(\d+)$")


def _norm(s: pd.Series) -> pd.Series:
    return s.fillna("").str.replace(_WS, " ", regex=True).str.strip()


def parse_topic_tags(tags):
    """Return (difficulty:int|nan, 'tag1, tag2' of the non-difficulty tags)."""
    diff = np.nan
    topical = []
    if tags is not None and not (np.isscalar(tags) and pd.isna(tags)):
        for t in tags:
            if not isinstance(t, str):
                continue
            m = _DIFF.match(t)
            if m:
                diff = int(m.group(1))
            else:
                topical.append(t)
    return diff, ", ".join(topical)


def main():
    print(f"Reading {JEOPARDY_RAW_PARQUET}")
    raw = pd.read_parquet(JEOPARDY_RAW_PARQUET)
    print(f"  raw: {len(raw):,} rows")

    df = raw[KEEP_COLS + ["topic_tags"]].copy()
    df["air_date"] = pd.to_datetime(df["air_date"], errors="coerce")

    if JEOPARDY_START_DATE is not None:
        before = len(df)
        df = df[df["air_date"] >= pd.Timestamp(JEOPARDY_START_DATE)].reset_index(drop=True)
        print(f"  window >= {JEOPARDY_START_DATE}: kept {len(df):,} of {before:,}")

    # Drop the tiny fraction with empty clue or answer (nothing to embed).
    blank = df["clue_text"].fillna("").str.strip().eq("") | df["answer"].fillna("").str.strip().eq("")
    n_blank = int(blank.sum())
    if n_blank:
        print(f"  dropping {n_blank} rows with blank clue or answer")
        df = df.loc[~blank].reset_index(drop=True)

    df["category"] = _norm(df["category"])
    df["clue_text"] = _norm(df["clue_text"])
    df["answer"] = _norm(df["answer"])

    parsed = df["topic_tags"].map(parse_topic_tags)
    df["difficulty"] = [p[0] for p in parsed]
    df["topic_tags_str"] = [p[1] for p in parsed]
    df = df.drop(columns=["topic_tags"])

    df["embed_text"] = "Category: " + df["category"] + "\nClue: " + df["clue_text"] + "\nAnswer: " + df["answer"]

    if MAX_CLUES is not None and len(df) > MAX_CLUES:
        df = df.sample(n=MAX_CLUES, random_state=SUBSET_SEED).reset_index(drop=True)
        print(f"  MAX_CLUES={MAX_CLUES}: subsampled to {len(df):,}")

    print(f"\nFinal: {len(df):,} clues, {df['answer'].nunique():,} unique answers")
    print(f"  date range: {df['air_date'].min()} -> {df['air_date'].max()}")
    print(f"  rounds: {df['round'].value_counts().to_dict()}")
    print(
        f"  difficulty present: {int(df['difficulty'].notna().sum()):,}"
        f"; daily_double: {int(df['daily_double'].sum()):,}"
        f"; repeats: {int(df['is_repeat_clue'].sum()):,}"
    )

    out = CLUE_ROWS_PARQUET
    tmp_fd, tmp_path = tempfile.mkstemp(dir=os.path.dirname(out), suffix=".parquet.tmp")
    os.close(tmp_fd)
    try:
        df.to_parquet(tmp_path, index=False)
        verify = pd.read_parquet(tmp_path)
        assert len(verify) == len(df), f"row count mismatch: {len(verify)} vs {len(df)}"
        os.replace(tmp_path, out)
    except Exception:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise
    print(f"\nWrote {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()

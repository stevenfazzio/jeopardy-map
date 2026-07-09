"""Prepare clues for embedding: window, clean, derive difficulty, build embed input.

Keeps clues in the JEOPARDY_START_DATE window (default last decade); within it we
keep ALL rows including repeat clues (1 row = 1 node, as designed). The robworks
dump is already clean (no HTML); we just normalize whitespace and drop the tiny
fraction with empty clue or answer.

Derived per-clue fields for colormaps/hover (all cheap, none touch embed_text):
  difficulty           difficulty:N tag parsed out of topic_tags
  subject              first non-difficulty topic tag (else "Untagged")
  game_type            tournament/special bucket parsed from `notes` (else "Regular")
  host_aside           host commentary / stage directions: the (...)/[...] spans in `notes`
  delivery/presenter/visual_clue   from the clue's leading "(X presents...)" parenthetical
  board_row            `value` as a fraction of the round's top value (per season) -> 1..5
  clue_len_words       clue word count, parenthetical asides removed
  answer_len_chars     answer char count, parenthetical asides removed
  repeat_count         number of other clue_ids this one repeats (len of repeat_clue_ids)
  answer_freq          archive-wide count of this answer ("stock answers")
  category_recurrence  archive-wide airings of this category (raw category_frequency)

answer_freq and category_recurrence are computed over the FULL archive before the
date window, so they stay truthful regardless of JEOPARDY_START_DATE / MAX_CLUES.

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
    "category_frequency",
    "value",
    "daily_double",
    "clue_text",
    "answer",
    "is_repeat_clue",
    "repeat_clue_ids",
    "clue_order",
    "answer_word_count",
    "notes",
]
_WS = re.compile(r"\s+")
_DIFF = re.compile(r"^difficulty:(\d+)$")
# A leading "(...)" parenthetical on a clue (who delivers it).
_LEAD_PAREN = re.compile(r"^\s*\((.*?)\)")
# Any parenthetical span — stripped before measuring clue/answer length so delivery
# boilerplate and "(N letters)" hints don't inflate the count.
_PARENS = re.compile(r"\([^)]*\)")
# Parenthetical / bracketed asides inside `notes` (host commentary, stage directions).
_ASIDE = re.compile(r"\(([^)]*)\)|\[([^\]]*)\]")
_SELF_INTRO = re.compile(r"^\s*(?:hi[,!]?\s*)?i'?m\s+[A-Z]", re.I)
_PRESENTS = re.compile(r"\b(presents|delivers|gives|reads|reports|performs|sings|shows|demonstrates|explains)\b", re.I)
_MEDIA = re.compile(r"monitor|screen|display|shows?|pictured|diagram|\bmap\b|image|photo|painting|video", re.I)
_LOCATION = re.compile(r"\b(?:from|in|at|outside)\b\s+[A-Z]")
# Clue Crew first names: a bare "(Sarah presents the clue from...)" omits "Clue Crew",
# so catch the roster by name. Exact single-name match only, so "Jimmy Kimmel" (a
# celebrity guest) is NOT swept in.
CLUE_CREW_NAMES = {"Sarah", "Jimmy", "Kelly", "Sofia", "Cheryl", "Jon"}

# `notes` event fragment -> canonical game_type. Checked in order; first hit wins.
# Substring matches collapse sub-brackets (e.g. all "Champions Wildcard Group/Hearts/..."
# -> "Champions Wildcard") and merge variants (National College Championship ->
# College Championship; Masters knockout -> Jeopardy! Masters).
GAME_TYPE_RULES = [
    ("tournament of champions", "Tournament of Champions"),
    ("teachers tournament", "Teachers Tournament"),
    ("masters", "Jeopardy! Masters"),
    ("second chance", "Second Chance"),
    ("college championship", "College Championship"),
    ("celebrity", "Primetime Celebrity Jeopardy!"),
    ("teen tournament", "Teen Tournament"),
    ("invitational", "Jeopardy! Invitational"),
    ("champions wildcard", "Champions Wildcard"),
    ("high school reunion", "High School Reunion"),
    ("professors tournament", "Professors Tournament"),
    ("greatest of all time", "Greatest of All Time"),
    ("all-star games", "All-Star Games"),
    ("all star games", "All-Star Games"),
    ("power players", "Power Players"),
]


def _norm(s: pd.Series) -> pd.Series:
    return s.fillna("").str.replace(_WS, " ", regex=True).str.strip()


def _strip_parens(s: str) -> str:
    """Drop parenthetical spans and collapse the leftover whitespace."""
    return _WS.sub(" ", _PARENS.sub(" ", s)).strip()


def parse_topic_tags(tags):
    """Return (difficulty:int|nan, [subject tags in original order])."""
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
    return diff, topical


def parse_game_type(note):
    """Map a clue's `notes` to a canonical game/tournament bucket; default 'Regular'."""
    if not isinstance(note, str) or not note.strip():
        return "Regular"
    stripped = _ASIDE.sub(" ", note).lower()  # drop "(Alex: ...)" / "[Laughter]" first
    for frag, label in GAME_TYPE_RULES:
        if frag in stripped:
            return label
    # leftover event-ish text we don't recognize -> Other special; pure aside -> Regular
    return "Other special" if stripped.strip(" .,;:-") else "Regular"


def parse_host_aside(note):
    """Extract host commentary / stage directions: the (...) and [...] spans in `notes`."""
    if not isinstance(note, str) or not note:
        return ""
    parts = [a or b for a, b in _ASIDE.findall(note)]
    return " ".join(p.strip() for p in parts if p.strip())


def extract_presenter(inner: str) -> str:
    """Pull the presenter's name out of a leading clue parenthetical, else ''."""
    s = inner.strip()
    m = re.match(r"(?:hi[,!]?\s*)?i'?m\s+([A-Z][\w.'-]+(?:\s+[A-Z][\w.'-]+){0,3})", s, re.I)
    if m:
        return m.group(1)
    m = re.match(r"([A-Z][\w.'-]+)\s+of the [Cc]lue [Cc]rew", s)
    if m:
        return m.group(1)
    m = re.match(
        r"([A-Z][\w.'-]+(?:\s+[A-Z][\w.'-]+){0,3})\s+(?:presents|delivers|gives|reads|reports|performs|sings)",
        s,
    )
    if m:
        return m.group(1)
    return ""


def parse_delivery(clue):
    """From a clue's leading parenthetical: (delivery bucket, presenter name, visual?)."""
    m = _LEAD_PAREN.match(clue)
    if not m:
        return "Standard", "", False
    inner = m.group(1).strip()
    presenter = extract_presenter(inner)
    if "clue crew" in inner.lower() or presenter in CLUE_CREW_NAMES:
        deliv = "Clue Crew"
    elif _SELF_INTRO.match(inner) or _PRESENTS.search(inner):
        deliv = "Celebrity"
    else:
        deliv = "Other"
    visual = deliv == "Clue Crew" and bool(_MEDIA.search(inner) or _LOCATION.search(inner))
    return deliv, presenter, visual


def main():
    print(f"Reading {JEOPARDY_RAW_PARQUET}")
    raw = pd.read_parquet(JEOPARDY_RAW_PARQUET)
    print(f"  raw: {len(raw):,} rows")

    # Archive-wide answer frequency ("stock answers"), computed BEFORE the date window
    # so counts are over the whole 1983-2025 archive regardless of window / MAX_CLUES.
    # (category_recurrence reuses the raw category_frequency column, already archive-wide.)
    answer_freq_map = _norm(raw["answer"]).str.lower().value_counts()

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

    # difficulty + primary subject from topic_tags
    parsed = df["topic_tags"].map(parse_topic_tags)
    df["difficulty"] = [p[0] for p in parsed]
    subjects = [p[1] for p in parsed]
    df["subject"] = [s[0] if s else "Untagged" for s in subjects]
    df["topic_tags_str"] = [", ".join(s) for s in subjects]

    # game type + host aside, both parsed from `notes`
    df["game_type"] = df["notes"].map(parse_game_type)
    df["host_aside"] = df["notes"].map(parse_host_aside)

    # who delivers the clue, from its leading parenthetical
    deliv = df["clue_text"].map(parse_delivery)
    df["delivery"] = [d[0] for d in deliv]
    df["presenter"] = [d[1] for d in deliv]
    df["visual_clue"] = [d[2] for d in deliv]

    # era-stable board row 1..5: value as a fraction of the round's top value (per
    # season, so the 2001 dollar-doubling cancels), snapped to the 5-row board. Robust
    # to the ~1% off-grid values (100/300/500) that dense-ranking would split into 6-8.
    # Final Jeopardy has no value -> NaN.
    grp_max = df.groupby(["season", "round"])["value"].transform("max")
    df["board_row"] = (df["value"] / grp_max * 5).round().clip(1, 5)

    # lengths & counts. Lengths exclude parenthetical asides (the delivery boilerplate
    # "(Sarah of the Clue Crew presents...)", "(N letters)" hints, alt-answer notes), so
    # they measure the actual clue/answer rather than the staging.
    df["clue_len_words"] = df["clue_text"].map(lambda s: len(_strip_parens(s).split())).astype(int)
    df["answer_len_chars"] = df["answer"].map(lambda s: len(_strip_parens(s))).astype(int)
    df["repeat_count"] = df["repeat_clue_ids"].map(lambda x: len(x) if x is not None and not np.isscalar(x) else 0)

    # archive-wide frequencies
    df["answer_freq"] = df["answer"].str.lower().map(answer_freq_map).fillna(1).astype(int)
    df = df.rename(columns={"category_frequency": "category_recurrence"})

    df["embed_text"] = "Category: " + df["category"] + "\nClue: " + df["clue_text"] + "\nAnswer: " + df["answer"]

    df = df.drop(columns=["topic_tags", "repeat_clue_ids", "notes"])

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
    print(f"  subject: {df['subject'].nunique()} values, {(df['subject'] != 'Untagged').mean() * 100:.0f}% tagged")
    print(f"  game_type (top 6): {dict(df['game_type'].value_counts().head(6))}")
    print(f"  delivery: {df['delivery'].value_counts().to_dict()}")
    print(
        f"  presenters named: {int((df['presenter'] != '').sum()):,}"
        f"; host asides: {int((df['host_aside'] != '').sum()):,}"
        f"; visual clues: {int(df['visual_clue'].sum()):,}"
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

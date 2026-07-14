"""DD category-picking heuristics a HUMAN could use: an LLM backtest.

36's board backtest shows the category strip carries real Daily Double signal,
but its title strategy is a logistic regression over 1024-d embeddings — no
player can run that at the podium. This script tests human-realizable versions:
give an LLM (Haiku 4.5, temperature 0) the six category titles of a real board
plus a short best-practices heuristic, ask it to pick the single most DD-likely
category, and score the pick with the same exact-expectation first-pick metrics
as 36 (random baselines: 1/30 J and 2/30 DJ at the cell level).

Arms (one prompt variant each, run over every complete board):
  control        no heuristic — measures Claude's prior about Jeopardy
  one_liner      the single sentence a blog reader would retain
  paragraph      a faithful prose distillation of the measured region lifts
  quantitative   the paragraph with the actual lift numbers attached
  paragraph_cot  the paragraph + brief reasoning before the answer

Boards are rebuilt from clue_rows.parquet with 36's completeness criteria and
asserted to reproduce the same 991 J / 874 DJ boards, so numbers are directly
comparable to dd_information_set.parquet. Category order is shuffled
deterministically per board (md5-seeded) to kill position-in-list bias.

Honesty caveats (also in the claim inventory): the heuristics are distilled
from the same 2016+ window they are evaluated on (a one-paragraph rule can
barely overfit, but strictly it is in-sample), and an LLM applying a rule with
perfect consistency is an upper bound on a prepared player, not a median
contestant.

Cost: ~9.3k Haiku 4.5 calls, ~$3-4 total, cost-guarded; picks are cached per
(arm, board) in data/analysis/llm_picks.parquet (atomic, resumable), so reruns
and new arms only pay for what's missing. Needs stages 00-01 only (no
embeddings). Env: ANTHROPIC_API_KEY.

Output: data/analysis/dd_llm_heuristic.parquet + printed comparison table.
"""

from __future__ import annotations

import asyncio
import hashlib
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))

from common import ANALYSIS_DIR, atomic_write_parquet  # noqa: E402
from config import ANTHROPIC_API_KEY, ANTHROPIC_MAX_CONCURRENCY, CLUE_ROWS_PARQUET  # noqa: E402

PICKS_PARQUET = ANALYSIS_DIR / "llm_picks.parquet"
OUT_PARQUET = ANALYSIS_DIR / "dd_llm_heuristic.parquet"
INFO_SET_PARQUET = ANALYSIS_DIR / "dd_information_set.parquet"  # 36's results, for the comparison print

MODEL = "claude-haiku-4-5-20251001"  # same model as the project's region naming
MAX_TOKENS = 16
MAX_TOKENS_COT = 300
CHECKPOINT_EVERY = 200
MAX_COST_USD = 10.0  # guard; expected ~$3-4
MAX_BOARDS = None  # smoke-test knob: e.g. 12 -> first 12 boards per round-agnostic sort; None = all
EXPECTED_BOARDS = (991, 874)  # (J, DJ) — must match 36's board backtest exactly

ONE_LINER = (
    "Best practice: at any given board position, Daily Doubles favor academic-sounding "
    "categories (geography, royalty, Shakespeare, mythology) and avoid pop-culture ones "
    "(TV, rock music, sports, food, movies)."
)

PARAGRAPH = (
    "Best practices for spotting Daily Double categories on Jeopardy!: Daily Doubles are "
    "not spread evenly across topics. Net of board position, they are markedly MORE likely "
    "in categories about British royalty and monarchy, Shakespeare, European geography and "
    "capitals, geography of the Americas, classical mythology, and similar classic academic "
    "subjects. They are markedly LESS likely in categories about rock and pop music, TV "
    "shows, professional sports, food and drink, and movies. Wordplay and pun categories "
    "are roughly average. When titles are ambiguous, prefer the one that sounds most like a "
    "school subject and least like entertainment or leisure."
)

QUANTITATIVE = PARAGRAPH + (
    " Measured rates relative to position-expected: British monarchy 2.03x, Shakespeare "
    "2.02x, European geography 1.63x, classical mythology 1.57x, Americas geography 1.52x; "
    "rock music 0.16x, TV shows 0.18x, pro sports 0.18x, food 0.23x, movies 0.34x."
)

ARMS: dict[str, tuple[str | None, bool]] = {  # arm -> (heuristic text, chain_of_thought)
    "control": (None, False),
    "one_liner": (ONE_LINER, False),
    "paragraph": (PARAGRAPH, False),
    "quantitative": (QUANTITATIVE, False),
    "paragraph_cot": (PARAGRAPH, True),
}

ANSWER_RE = re.compile(r"ANSWER:\s*([1-6])")


# ------------------------------------------------------------------- boards
def build_boards() -> pd.DataFrame:
    """One row per (board, category): display title, dd count, dd-at-row-4 flag.

    Completeness criteria are identical to 36's board backtest; asserted below.
    """
    cols = ["episode_id", "round", "board_row", "daily_double", "category", "category_normalized"]
    d = pd.read_parquet(CLUE_ROWS_PARQUET, columns=cols)
    d = d[d["round"].isin(["jeopardy", "double_jeopardy"]) & d["board_row"].notna()].copy()
    d["board"] = d["episode_id"].astype(str) + "|" + d["round"]
    d["row"] = d["board_row"].astype(int)
    d["dd"] = d["daily_double"].fillna(False).astype(bool)

    per_cat = d.groupby(["board", "category_normalized"]).agg(
        n=("row", "size"), rows_ok=("row", lambda s: sorted(s) == [1, 2, 3, 4, 5])
    )
    per_board = per_cat.groupby("board").agg(n_cats=("n", "size"), cells=("n", "sum"), all_ok=("rows_ok", "all"))
    per_board = per_board.join(d.groupby("board").agg(dds=("dd", "sum"), round=("round", "first")))
    req = np.where(per_board["round"] == "jeopardy", 1, 2)
    complete = per_board[
        (per_board["cells"] == 30) & (per_board["n_cats"] == 6) & per_board["all_ok"] & (per_board["dds"] == req)
    ]
    n_j = int((complete["round"] == "jeopardy").sum())
    n_dj = int((complete["round"] == "double_jeopardy").sum())
    assert (n_j, n_dj) == EXPECTED_BOARDS, f"board reconstruction drifted from 36: got ({n_j}, {n_dj})"

    dd_cells = d[d["dd"]]
    cats = (
        d[d["board"].isin(complete.index)]
        .groupby(["board", "category_normalized"])
        .agg(title=("category", "first"))
        .reset_index()
    )
    cat_dd = dd_cells.groupby(["board", "category_normalized"]).agg(
        dd_count=("dd", "sum"), dd_row4=("row", lambda s: bool((s == 4).any()))
    )
    cats = cats.merge(cat_dd, how="left", on=["board", "category_normalized"])
    cats["dd_count"] = cats["dd_count"].fillna(0).astype(int)
    cats["dd_row4"] = cats["dd_row4"].eq(True)
    cats["title"] = cats["title"].fillna("").str.strip().str.upper().replace("", "UNTITLED")
    cats["round"] = cats["board"].str.split("|").str[1]
    print(f"boards: {n_j:,} J / {n_dj:,} DJ complete (matches 36)")
    return cats


def presentation_order(board: str, cat_norms: list[str]) -> list[str]:
    """Deterministic per-board shuffle of the six categories (kills list-position bias)."""
    seed = int(hashlib.md5(board.encode()).hexdigest()[:8], 16)
    perm = np.random.default_rng(seed).permutation(len(cat_norms))
    ordered = sorted(cat_norms)
    return [ordered[i] for i in perm]


def build_prompt(round_name: str, titles: list[str], heuristic: str | None, cot: bool) -> str:
    nice_round = "Jeopardy" if round_name == "jeopardy" else "Double Jeopardy"
    lines = [f"You are looking at a Jeopardy! board from the {nice_round} round. The six categories are:", ""]
    lines += [f"{i + 1}. {t}" for i, t in enumerate(titles)]
    lines += ["", "Which ONE category is most likely to contain a Daily Double?"]
    if heuristic:
        lines += ["", heuristic]
    if cot:
        lines += ["", 'Reason briefly (one or two sentences), then end with a final line "ANSWER: <number 1-6>".']
    else:
        lines += ["", 'Reply with only "ANSWER: <number 1-6>".']
    return "\n".join(lines)


# --------------------------------------------------------------------- calls
def load_picks() -> pd.DataFrame:
    if PICKS_PARQUET.exists():
        return pd.read_parquet(PICKS_PARQUET)
    return pd.DataFrame(columns=["arm", "board", "pick", "raw"]).astype({"pick": int})


async def collect_picks(cats: pd.DataFrame) -> pd.DataFrame:
    import anthropic

    boards = sorted(cats["board"].unique())
    if MAX_BOARDS is not None:
        boards = boards[:MAX_BOARDS]
        print(f"smoke run: restricted to {len(boards)} boards")
    board_set = set(boards)
    board_round = cats.drop_duplicates("board").set_index("board")["round"]
    board_titles = {
        b: dict(zip(g["category_normalized"], g["title"])) for b, g in cats.groupby("board") if b in board_set
    }

    picks = load_picks()
    have = set(zip(picks["arm"], picks["board"]))
    todo = [(arm, b) for arm in ARMS for b in boards if (arm, b) not in have]
    if not todo:
        print("all picks cached; no API calls needed")
        return picks

    # crude cost guard: ~250 tokens in, ~10 out (CoT ~130 out) per call
    n_cot = sum(1 for arm, _ in todo if ARMS[arm][1])
    est = (len(todo) * 250 * 1.0 + (len(todo) * 10 + n_cot * 120) * 5.0) / 1e6
    assert est < MAX_COST_USD, f"estimated ${est:.2f} exceeds ${MAX_COST_USD} guard"
    print(f"{len(todo):,} calls to make ({len(have):,} cached); est ~${est:.2f} with {MODEL}")
    if not ANTHROPIC_API_KEY:
        raise RuntimeError("ANTHROPIC_API_KEY not set; add it to .env (see .env.example)")

    # explicit timeout: a hung socket otherwise blocks forever (see 36's postmortem);
    # the SDK retries 429/5xx/timeouts itself
    client = anthropic.AsyncAnthropic(api_key=ANTHROPIC_API_KEY, timeout=60.0, max_retries=5)
    sem = asyncio.Semaphore(ANTHROPIC_MAX_CONCURRENCY)

    async def one(arm: str, board: str) -> dict:
        heuristic, cot = ARMS[arm]
        order = presentation_order(board, list(board_titles[board]))
        prompt = build_prompt(board_round[board], [board_titles[board][c] for c in order], heuristic, cot)
        async with sem:
            resp = await client.messages.create(
                model=MODEL,
                max_tokens=MAX_TOKENS_COT if cot else MAX_TOKENS,
                temperature=0.0,
                messages=[{"role": "user", "content": prompt}],
            )
        raw = "".join(b.text for b in resp.content if b.type == "text")
        m = ANSWER_RE.findall(raw)
        return {"arm": arm, "board": board, "pick": int(m[-1]) if m else -1, "raw": raw}

    new_rows: list[dict] = []
    t0 = time.time()
    tasks = [one(arm, b) for arm, b in todo]
    for i, fut in enumerate(asyncio.as_completed(tasks), 1):
        new_rows.append(await fut)
        if (i % CHECKPOINT_EVERY == 0 or i == len(tasks)) and new_rows:
            picks = pd.concat([picks, pd.DataFrame(new_rows)], ignore_index=True)
            new_rows = []
            atomic_write_parquet(picks, PICKS_PARQUET)
            print(f"  {i:,}/{len(tasks):,} picks ({time.time() - t0:.0f}s)", flush=True)
    return picks


# ---------------------------------------------------------------- evaluation
def evaluate(cats: pd.DataFrame, picks: pd.DataFrame) -> pd.DataFrame:
    cat_lookup = cats.set_index(["board", "category_normalized"])[["dd_count", "dd_row4"]]
    board_round = cats.drop_duplicates("board").set_index("board")["round"]
    board_cats = {b: list(g["category_normalized"]) for b, g in cats.groupby("board")}

    rows = []
    for arm in ARMS:
        sub = picks[picks["arm"] == arm]
        n_invalid = int((sub["pick"] < 1).sum())
        sub = sub[sub["pick"] >= 1]
        chosen = [presentation_order(b, board_cats[b])[p - 1] for b, p in zip(sub["board"], sub["pick"])]
        info = cat_lookup.loc[list(zip(sub["board"], chosen))]
        ev = pd.DataFrame(
            {
                "round": board_round[sub["board"]].to_numpy(),
                "p_cat_random_row": info["dd_count"].to_numpy() / 5.0,
                "hit_row4": info["dd_row4"].to_numpy().astype(float),
            }
        )
        for r_name, r_lab in [("jeopardy", "J"), ("double_jeopardy", "DJ")]:
            g = ev[ev["round"] == r_name]
            for strategy, col in [("category, random row", "p_cat_random_row"), ("category + row 4", "hit_row4")]:
                rows.append(
                    {
                        "arm": arm,
                        "round": r_lab,
                        "strategy": strategy,
                        "p_first_pick": float(g[col].mean()),
                        "sd": float(g[col].std()),
                        "n_boards": len(g),
                        "n_invalid": n_invalid,
                    }
                )
    return pd.DataFrame(rows)


def print_comparison(out: pd.DataFrame) -> None:
    """Side-by-side with 36's LR strategies and the analytic random baselines."""
    lr = pd.read_parquet(INFO_SET_PARQUET) if INFO_SET_PARQUET.exists() else pd.DataFrame()
    for r_lab, req in [("J", 1), ("DJ", 2)]:
        print(f"\n{r_lab}: random baseline {req / 30:.3f}")
        if len(lr):
            for feat in [f"title (best category, random row) ({r_lab})", f"title + position ({r_lab})"]:
                row = lr[(lr["leg"] == "board_backtest") & (lr["features"] == feat)]
                if len(row):
                    print(f"  {'[36 LR] ' + feat:<58} {float(row['value'].iloc[0]):.3f}")
        for _, r in out[out["round"] == r_lab].iterrows():
            label = f"[LLM] {r['arm']} · {r['strategy']}"
            extra = f"  (invalid: {r['n_invalid']})" if r["n_invalid"] else ""
            print(f"  {label:<58} {r['p_first_pick']:.3f}  ({r['p_first_pick'] / (req / 30):.2f}x random){extra}")


def main() -> None:
    t0 = time.time()
    cats = build_boards()
    picks = asyncio.run(collect_picks(cats))
    out = evaluate(cats, picks)
    atomic_write_parquet(out, OUT_PARQUET)
    print_comparison(out)
    print(f"\nwrote {OUT_PARQUET} ({len(out)} rows) in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A Python data pipeline that produces an interactive 2D semantic map of Jeopardy
clues. **One node = one clue row** from `robworks-software/jeopardy-clues`; there
is no Wikipedia data. Each clue is embedded from its own `category` + `clue_text`
+ `answer`, laid out with UMAP, and the regions are named by Toponymy. Final
artifact: `docs/index.html` (deployable to GitHub Pages); local copy at
`data/clue_map.html`.

This is the lean sibling of `../jeopardy-wikipedia-map/` (which matches clues to
Wikipedia articles and colors by over/under-representation). This project keeps
only the fetch → embed → reduce → label → visualize spine; the fetch stage and
the DataMapPlot stage are ported from there.

A companion **`analysis/` suite** (own section below) characterizes the corpus in
the raw 1024-d embedding space for the blog post — statistics, an HTML report,
and a standalone region treemap — treating the 2D layout as a visualization
artifact rather than the object of study.

## Commands

```bash
make install      # uv sync --extra dev
make lint         # ruff check + ruff format --check  (line-length 120, rules E/F/I)
make format       # ruff format
make test         # pytest (no tests authored yet)

# Run a stage (run in numeric order 00 -> 05; each is standalone):
uv run python pipeline/02_embed_clues.py
```

There is no `make pipeline` target; run stages by number. **Smoke-test by editing
constants in `pipeline/config.py`, not via CLI args** (no stage takes args). Key
knobs: `JEOPARDY_START_DATE` (analysis window; default `"2016-01-01"` ≈ last decade
~135k clues, `None` = full ~568k archive) and `MAX_CLUES` (set e.g. `3000` to
dry-run all six stages cheaply on a deterministic random subset; `None` = the whole
window). A 3k smoke run costs cents; the full archive is the expensive one (below).

## Pipeline architecture

`pipeline/config.py` is the central hub: every stage does `from config import ...`
(the stage's own directory is on `sys.path` when run as `python pipeline/XX.py`, so
it's a bare import, not `from pipeline.config`). All paths and tunable constants
live there; it also loads `.env`.

Six sequential stages, each writing to `data/` (gitignored):

```
00 fetch_jeopardy   robworks-software/jeopardy-clues (HF), union splits   -> jeopardy_raw.parquet (~568k, 1983-2025)
01 prepare_clues    window, clean, difficulty + colormap/hover fields, embed_text -> clue_rows.parquet (~135k default; MAX_CLUES subsets too)
02 embed_clues      Cohere embed-v4.0, input_type=clustering, float 1024  -> clue_embeddings.npz (emb + clue_id)
03 reduce_umap      UMAP cosine, n_neighbors=15, min_dist=0.05            -> umap_coords.npz (coords + clue_id)
04 label_topics     Toponymy regions (Cohere terms + Claude naming)       -> toponymy_labels.parquet (label_layer_*)
05 visualize        DataMapPlot interactive HTML                          -> data/clue_map.html + docs/index.html
```

**`clue_id` is the alignment key** across every stage. Stages 02/03/04 each carry
`clue_id` alongside their arrays, and stage 05 merges everything back on it. The
embeddings npz, the coords npz, and the labels parquet are all one-row-per-clue.

## Cross-cutting decisions (the non-obvious stuff)

- **Embeddings use `input_type="clustering"` (embed-v4.0), not `search_document`.**
  The only downstream use of these vectors is grouping/visualization (UMAP + the
  Toponymy clusterer), which is exactly what Cohere tunes the `clustering` mode for.
  We are NOT doing retrieval. `output_dimension=1024`, `embedding_types=["float"]`,
  unit-normalized by the model.

- **Toponymy uses TWO independent embedding spaces; ours and its are unrelated.**
  This is the subtle one. Toponymy's internal `CohereEmbedder` embeds candidate
  region-name keyphrases as `search_query` and ranks them *against a centroid of
  other keyphrase vectors* (`keyphrases.py`), never against our document vectors.
  Our `embedding_vectors` are used only for clustering, centroids, and exemplar
  selection (doc-vs-doc). The two spaces are bridged solely by **cluster membership**
  (which clue is in which cluster), never by a cross-space dot product. Consequences:
  our embed model / `input_type` / dimension are a *free choice* w.r.t. Toponymy, and
  Toponymy's term embeddings staying at the v4 default dim (it doesn't pass
  `output_dimension`) is harmless. Do not "fix" an input_type/dim mismatch between the
  two — there is nothing to match.

- **Toponymy names *places in the space*, not documents** (it's a place-naming, not
  a topic-modeling, library). `cluster_layers_` is **finest-first** (index 0 = most,
  narrowest regions). Toponymy and DataMapPlot are by the same authors and agree on
  this order, so we pass it through unchanged end-to-end: parquet `label_layer_0` =
  finest, `label_layer_N` = coarsest, and DataMapPlot's `*label_layers` wants the
  same (first arg = finest, last = coarsest at zoom-out). Stage 05 just spreads the
  columns directly. Points in unnamed space come back **`Unlabelled`** — that is
  signal (a gap on the map at that zoom), not a labeling failure; the map keeps
  them. Expect a high `Unlabelled` fraction at small `MAX_CLUES` (sparse density);
  it shrinks at full scale.

- **`difficulty` comes from `topic_tags`, not `value`.** Each clue's `topic_tags`
  array carries a `difficulty:N` tag (present on ~100% of rows); stage 01 parses it
  out. Prefer it over raw `value` for a "hardness" colormap: `value` spans the 2001
  dollar-doubling and is NaN for Final Jeopardy, so across all 41 seasons it has an
  era discontinuity. `value` is still offered as a colormap, just de-emphasized.

- **Colormap/hover metadata is derived in stage 01 and is layout-free.** Beyond
  `difficulty`, stage 01 derives per-clue fields used only for the stage-05 colormap
  dropdown and hover, never embedded: `subject` (first non-difficulty topic tag, else
  `Untagged`; ~36% tagged in the window), `game_type` (tournament/special bucket
  parsed from `notes`, else `Regular`), `host_aside` (the `(...)`/`[...]` spans in
  `notes`), `delivery`/`presenter`/`visual_clue` (parsed from a clue's leading
  `(X presents…)` parenthetical — Clue Crew vs celebrity guest), `board_row` (value as
  a fraction of the round's top value per season → era-stable 1–5, robust to off-grid
  $100/300/500 clues), `clue_len_words`/`answer_len_chars` (length with parenthetical
  asides stripped, so delivery boilerplate doesn't inflate it), `repeat_count`, and the archive-wide counts
  `answer_freq` ("stock answers") and `category_recurrence` (computed on the full raw
  **before** the date window so they stay truthful under windowing/`MAX_CLUES`). Because
  none of these touch `embed_text`, adding or changing one is cheap: **re-run 01 then 05
  only** (seconds, no API) — the `clue_id` set is unchanged, so the existing embeddings/
  UMAP/Toponymy artifacts stay aligned.

- **The map is dark mode; every color choice in stage 05 assumes a near-black
  background.** The hover tooltip is styled as a Jeopardy clue card (`#060ce9`,
  caps category strip, gold answer) in show-style Google-Font stand-ins — Oswald ≈
  Swiss 911 for the category, Bellota Text ≈ ITC Korinna for clue/answer (the real
  faces are commercial), loaded via `custom_html` since datamapplot only embeds
  `font_family`/`tooltip_font_family` itself — but tuned for rapid hover-scanning
  over show fidelity: fixed card width (identical geometry every hover), clue in
  sentence case not board-caps, opaque card with no backdrop blur (compositing
  cost on the hover hot path). Sequential colormaps are truncated (and
  reversed where needed) so they run dim-but-visible → bright — a cmap's near-black
  end renders tiny points invisible on the dark ground, the mirror of near-white on
  white. Right-skewed continuous fields (lengths, value, log counts) are winsorized
  at p99 so outliers don't crush the bulk into the dim end. Categoricals use a
  glasbey palette with a lightness floor (`lightness_bounds=(40, 90)`), and the
  dominant defaults (`Untagged`/`Regular`) are pinned to receding greys.

- **Category wordplay shapes the layout.** Jeopardy categories are often puns
  ("RHYME TIME", "POTENT POTABLES"). Because `category` is in the embed text, a
  large wordplay/vocabulary region (e.g. "Word Puzzles", ~1/3 of the map at the
  coarsest scale) reliably emerges, organized by the gimmick rather than subject
  matter. This is expected. If you ever want a purely topical map, drop `category`
  from `embed_text` in stage 01 and re-run 02 onward.

- **Repeats are kept (1 row = 1 node).** `is_repeat_clue` rows are not collapsed;
  near-identical repeat clues land as overlapping points by design.

## Analysis suite (`analysis/`)

A second, self-contained package that analyzes the corpus in the **raw 1024-d
embedding space**. Read-only over the pipeline's `data/` artifacts; all of its
own outputs and caches live in `data/analysis/` (gitignored). Scripts are
numbered and standalone like pipeline stages (no CLI args); they need stages
00–02 to have run (`50`/`70` also read stage 04's labels; `45` needs only
stage 00). Blog-facing numbers live in `data/claim_inventory.md` (gitignored,
with `data/prior_art_research.md`) — update it when a headline stat changes.

```bash
uv run python analysis/common.py             # (re)build the cached exact kNN graph (~3 min)
uv run python analysis/20_metadata_sweep.py  # then any numbered script, in any order after 10
```

- **`common.py` is the hub** (the `config.py` analogue): aligned loaders —
  canonical row order everywhere = the embedding npz `clue_id` order — the cached
  exact cosine k=25 kNN graph, the permutation-test machinery, and
  `dmp_region_tree()`, the region hierarchy shared by `70` and `80`.
- **Scripts:** `10` EVoC ambient clustering + Toponymy/Haiku region naming (the
  main paid step, ~$8.40, cost-guarded; the clustering itself is free and takes
  ~20s) · `20` metadata × geometry kNN sweep vs permutation nulls · `30` linear
  probes (episode-grouped CV; incl. the 5-class board_row probe that mirrors
  Boettcher 2016) · `35` round-gradient decomposition within recurring category
  titles (scheduling vs writing; result: ~all scheduling, writing ≈3%, p=.11 —
  never quote its per-title/fraction columns without the axis-refit null) ·
  `36` DD information-set ladder — what a PLAYER sees before picking: title-only
  Cohere embeddings (~$0.01, cached/resumable `data/analysis/title_embeddings.npz`;
  ClientV2 gets an explicit `timeout=` — a hung socket raises nothing and stalls
  retry loops forever) vs round×row position vs full content, plus a
  within-category which-clue-is-the-DD leg, and a board backtest on complete
  fully-revealed boards (first-pick P(DD): J 3.3% random → 12.0% title+position,
  DJ 6.7% → 15.8%; position strategy = always row 4; the two heuristics stack
  ~multiplicatively); headline inversion: position alone (AUC 0.715) beats full
  clue text (0.690) ·
  `37` LLM heuristic backtest — is the title signal HUMAN-usable? Haiku 4.5
  (temp 0) picks the DD category from the six titles on 36's exact boards under
  5 prompt arms; ONE SENTENCE of best practices ≈ the full LR title model
  (J 5.9% vs 6.0% category-only), while the no-heuristic control lands BELOW
  random (z≈−5, picks skew wordplay) — the pattern isn't in the prior; needs
  only stages 00–01; picks cached `data/analysis/llm_picks.parquet` (~$4 paid
  artifact, resumable) ·
  `40` season drift · `45` full-archive recycling cadence + WGA-strike months
  (reads only stage 00's raw parquet, no embeddings; lexical near-dupes are the
  PRIMARY method — the dataset's own repeat links under-detect pre-2016; pairs
  from the same episode or <90 days are duplicate records, not recycling) ·
  `50` region composition + AMI crosswalk vs the 2D map's regions · `60`
  batch-level robustness nulls · `70` → `report.html` (the working report;
  its time section keeps the old windowed cadence figure as a deliberate
  censoring exhibit next to the archive truth from `45`) · `75` →
  `data/analysis/blog/fig*.html` (standalone blog figures, CDN plotly,
  numbers pinned to `data/claim_inventory.md`; regen ~0.3s) · `80` →
  `treemap.html` (standalone full-page treemap with a DJ-share / FJ-lift /
  DD-lift color dropdown).
- **toponymy 0.5.0's `EVoCClusterer` wrapper is incompatible with evoc 0.3.1**
  (it passes since-dropped kwargs), so `10` drives `evoc.EVoC` directly and hands
  `Toponymy` a duck-typed pre-fitted clusterer via its documented reuse branch.
- **Jeopardy content arrives in batches** — a category is ~5 same-topic clues in
  one episode — so clue-level nulls wildly flatter episode/category-level fields
  (game_type, season, delivery). Trust `60`'s episode/category-swap nulls and the
  probes' episode-grouped CV, never raw clue-level lifts for those fields.
- **Tautology channels:** `embed_text` contains the delivery parenthetical and
  the category name, so delivery/visual_clue/category_recurrence "signals" are
  partly circular. `subject`/`difficulty` (dataset-author `topic_tags`) are
  excluded from all analyses by decision.
- **The region treemap follows DataMapPlot's topic-tree algorithm faithfully:**
  parent = int-cast median provenance through coarser layers' point-level labels,
  noise (−1) as a first-class parent (so "Minor subtopics" are real branches),
  ramify-style honor-then-residual clue placement, and single-child chains
  collapsed (EVoC keeps clusters stable across scales — 13/27 layer-4 clusters
  are point-identical to a layer-3 cluster — and Toponymy's duplicate-name
  disambiguation is within-layer only, so persistent clusters otherwise render
  as "X > X"). Fidelity to that reference is deliberate; don't re-improvise it.
- **Keep report regen fast (~0.5s warm).** Heavy derivations are cached with
  signature checks (kNN graph, top-1 neighbor sims; AMI crosswalk — delete
  `data/analysis/ami_crosswalk.parquet` to recompute), and `load_clue_ids()`
  reads ids lazily instead of loading the 531MB embedding matrix. Memoize any
  new heavy computation the moment a figure becomes something to iterate on.

## Conventions & gotchas

- **Atomic writes / resumability.** Every stage writes via tmp + `os.replace`.
  Stage 02 (Cohere) checkpoints every `EMBED_CHECKPOINT_EVERY` batches to
  `clue_embeddings.npz.progress.npz` and resumes from it; the npz also stores a
  signature (first/last clue_id + n + model + dim) so a stale cache is ignored.
  Treat `data/*.parquet` and `*.npz` as expensive to regenerate.
- **`transformers` is a required dep even though we never run a HF model.**
  Toponymy's `llm_wrappers.py` does a top-level `import transformers`; without it,
  `import toponymy` fails. We pull `transformers` (no torch needed — the
  "PyTorch not found" notice on import is expected and fine).
- **`tee | tail` masks the Python exit code** for long background runs — use
  `set -o pipefail` if you pipe a stage's output and care about success.
- **The live embeddings predate the escape-stripping in `_norm`** (stage 01 now
  removes the dump's literal `\"` `\'` artifacts). `clue_embeddings.npz` was
  computed on the escaped text and deliberately not regenerated — semantically
  negligible, and stage 02's signature is clue_id-based so nothing invalidates.
  A future full re-run embeds the clean text.
- `experiments/` (if added) is for one-off diagnostics, not part of the pipeline.

## Cost / scale

Default run = last decade (`JEOPARDY_START_DATE="2016-01-01"`, ~135k clues) + Haiku
4.5 naming. Measured: **~21 min, ~$10** — embed ~8m / $0.61 (5.1M tokens @ 37.7
tok/clue), UMAP ~1.7m, Toponymy ~11m with `ANTHROPIC_MAX_CONCURRENCY=24` (~4,200
regions named, 6 layers, the bulk of the ~$9 Claude cost), render ~15s. Use a
PRODUCTION Cohere key (trial keys throttle hard).

Full archive (`JEOPARDY_START_DATE=None`, ~568k) + Sonnet 4.6 naming: not yet run.
Extrapolating, ~$85 (Sonnet) and likely ~1-2 hr — embed ~35m, UMAP ~10m, Toponymy
the main cost/time (~12,500 regions; count scales ~N^0.9). Push
`ANTHROPIC_MAX_CONCURRENCY` higher if your Anthropic tier allows.

Resumability: stage 02 (embeddings) checkpoints and resumes; UMAP and Toponymy do
NOT, so run them in a stable/background session rather than a fragile shell.

## Environment

`.env` (loaded by `config.py` via python-dotenv) or shell env vars:

| Variable | Used by | Purpose |
|---|---|---|
| `CO_API_KEY` | stages 02, 04; analysis/10 | Cohere `embed-v4.0` (clue embeddings) + Toponymy keyphrase embeddings |
| `ANTHROPIC_API_KEY` | stage 04; analysis/10 | Claude region naming (default `claude-haiku-4-5-20251001`; via `ANTHROPIC_MODEL_NAMING`) |

Stages 00, 01, 03, 05 need no external auth (Hugging Face is public), and
neither does any analysis script other than `10`.

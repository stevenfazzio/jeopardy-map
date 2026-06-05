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
01 prepare_clues    window (JEOPARDY_START_DATE), clean, difficulty, embed_text -> clue_rows.parquet (~135k default; MAX_CLUES subsets too)
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

- **Category wordplay shapes the layout.** Jeopardy categories are often puns
  ("RHYME TIME", "POTENT POTABLES"). Because `category` is in the embed text, a
  large wordplay/vocabulary region (e.g. "Word Puzzles", ~1/3 of the map at the
  coarsest scale) reliably emerges, organized by the gimmick rather than subject
  matter. This is expected. If you ever want a purely topical map, drop `category`
  from `embed_text` in stage 01 and re-run 02 onward.

- **Repeats are kept (1 row = 1 node).** `is_repeat_clue` rows are not collapsed;
  near-identical repeat clues land as overlapping points by design.

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
| `CO_API_KEY` | stages 02, 04 | Cohere `embed-v4.0` (clue embeddings) + Toponymy keyphrase embeddings |
| `ANTHROPIC_API_KEY` | stage 04 | Claude region naming (default `claude-haiku-4-5-20251001`; via `ANTHROPIC_MODEL_NAMING`) |

Stages 00, 01, 03, 05 need no external auth (Hugging Face is public).

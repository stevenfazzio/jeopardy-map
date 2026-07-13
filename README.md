# jeopardy-map

An interactive semantic map of ~135,000 *Jeopardy!* clues (2016–2025), plus a
statistical analysis of the full 1984–2025 archive.

**Live map:** <https://stevenfazzio.com/jeopardy-map/>

Every clue is embedded from its category + clue + answer text, laid out in 2D
with UMAP, and organized into LLM-named regions at every zoom level. One dot =
one clue; hover shows the clue as a Jeopardy-style card. A companion analysis
suite works in the raw 1024-dimensional embedding space, treating the 2D
layout as a visualization artifact rather than the object of study.

## Some things the data shows

- **Daily Doubles are placed by topic, not just position.** The famous
  rows-3-and-4 pattern is only half the story: net of board position, DDs are
  ~2× enriched in regions like British monarchy, Shakespeare, and European
  geography, and ~0.2× depleted in rock music, TV, and sports.
- **The rounds form a curriculum** — food/sports/pop culture skew toward the
  Jeopardy round; composers, mythology, and chemistry toward Double Jeopardy;
  Final Jeopardy triples down on prestige topics and (2016–2025) has never
  been culinary. Within a recurring category title, though, the writing is
  round-agnostic: the curriculum lives in the scheduling, not the clues.
- **Both WGA strikes are visible in the clue-recycling record.** Writers
  normally avoid material from the last ~2 seasons and reuse the rest of the
  back catalog for decades. During the 2023 strike the show ran on reruns —
  47% of clues airing in November 2023 recycled older material (baseline:
  ~2%) — and the 2007–08 strike shows the same signature at 20%.
- **Tournament "topical identities" are mostly an illusion of batching.**
  Naive statistics say tournaments are up to 17× topically distinctive;
  swapping whole episodes under the null collapses nearly all of it.

Working figures live in the analysis report; the numbers above are
window-qualified and sourced in `data/claim_inventory.md` (untracked).

## How it works

A six-stage Python pipeline (`pipeline/`), each stage a standalone script:

| Stage | What it does | Output |
|---|---|---|
| 00 | Fetch [robworks-software/jeopardy-clues](https://huggingface.co/datasets/robworks-software/jeopardy-clues) (~568k clues, 1984–2025) | `jeopardy_raw.parquet` |
| 01 | Window (2016+), clean, derive hover/colormap metadata | `clue_rows.parquet` |
| 02 | Embed with Cohere `embed-v4.0` (`input_type="clustering"`, 1024-d) | `clue_embeddings.npz` |
| 03 | UMAP layout (cosine, n_neighbors=15) | `umap_coords.npz` |
| 04 | Region naming with [Toponymy](https://github.com/TutteInstitute/toponymy) (Claude Haiku 4.5) | `toponymy_labels.parquet` |
| 05 | Render with [DataMapPlot](https://github.com/TutteInstitute/datamapplot) | `docs/index.html` |

The `analysis/` suite (scripts `10`–`80`) characterizes the corpus in ambient
embedding space: EVoC clustering, metadata × geometry permutation tests,
linear probes with episode-grouped CV, full-archive recycling detection, and
an HTML report. Jeopardy content arrives in ~5-clue category batches inside
episodes, so the suite leans on episode-swap nulls and grouped
cross-validation throughout — naive clue-level nulls wildly overstate
significance for anything episode-structured.

## Running it

```bash
make install                          # uv sync --extra dev
cp .env.example .env                  # or export CO_API_KEY / ANTHROPIC_API_KEY
uv run python pipeline/00_fetch_jeopardy.py   # then 01..05 in order
```

Measured cost for the shipped scope (~135k clues): **~$10 and ~21 minutes**
(embedding ~$0.61, region naming ~$9 via Claude Haiku 4.5). Stages 00, 01,
03, 05 need no API keys. Set `MAX_CLUES` in `pipeline/config.py` for a
cents-scale smoke run. See `CLAUDE.md` for the full architecture notes,
gotchas, and cost details.

## Prior work

Content analyses of the Jeopardy corpus have a history worth crediting:
[Slate (2011)](https://www.slate.com/articles/arts/culturebox/2011/02/ill_take_jeopardy_trivia_for_200_alex.html)
found the round-curriculum direction from category titles; Ben Schmidt
published an unlabeled [Nomic Atlas embedding map](https://atlas.nomic.ai/data/bmschmidt/jeopardy-questions-full/map)
of the older Kaggle corpus in 2023; IBM's Watson papers characterized the
question domain at the answer-type level; and
[Emma Boettcher's 2016 master's paper](https://cdr.lib.unc.edu/downloads/bz60d095h?locale=en)
studied clue difficulty from text features. The Daily Double *position*
pattern is long-established (Slate 2011, FlowingData 2015, FiveThirtyEight
2019); the topical dimension here is, as far as we can tell, new.

## Data, license, disclaimer

Code is MIT-licensed. Clue text, answers, and categories are the property of
Jeopardy Productions, Inc. and are **not** covered by the code license — see
the scope note in [LICENSE](LICENSE). This is an unofficial fan/research
project, not affiliated with or endorsed by Jeopardy Productions or Sony
Pictures Television. Game content reaches this project via community-
maintained datasets derived from fan transcriptions; rights holders can open
an issue for prompt takedown.

"""Central config for the jeopardy-map pipeline. Every stage does `from config
import ...` (the stage's own dir is on sys.path when run as `python pipeline/XX.py`).
Edit constants here for smoke tests rather than adding CLI args."""

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

load_dotenv(PROJECT_ROOT / ".env")
CO_API_KEY = os.environ.get("CO_API_KEY")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")

PROJECT_NAME = "Jeopardy Clue Map"
PROJECT_TAGLINE = "Every Jeopardy! clue, laid out by what it's about"

# --- Stage 00: fetch Jeopardy clues ---
# robworks-software/jeopardy-clues: ~568k clues, seasons 1-41 (1985-2025), split
# into arbitrary train/validation/test partitions which we union back together.
HF_JEOPARDY_DATASET = "robworks-software/jeopardy-clues"
JEOPARDY_RAW_PARQUET = DATA_DIR / "jeopardy_raw.parquet"

# --- Stage 01: prepare ---
# Analysis window: keep clues on/after this date. "2016-01-01" = the last ~decade
# (~135k clues), matching the sibling project's window; set to None for the full
# 1983-2025 archive (~568k). Repeats are kept within the window (1 row = 1 node).
JEOPARDY_START_DATE = "2016-01-01"
CLUE_ROWS_PARQUET = DATA_DIR / "clue_rows.parquet"
# Smoke-test knob: cap to a random subset for a fast end-to-end dry run.
# None = use the whole window. Set e.g. 2000 to exercise all six stages cheaply.
MAX_CLUES = None
SUBSET_SEED = 42  # deterministic subset when MAX_CLUES is set

# --- Stage 02: embed clues (Cohere embed-v4.0) ---
# One vector per clue from "Category / Clue / Answer" (built in stage 01 as
# embed_text). input_type="clustering" because the only downstream use is
# grouping/visualization (UMAP + Toponymy's clusterer), which is exactly what
# Cohere tunes that mode for. Toponymy's INTERNAL keyphrase embedder lives in its
# own (search_query) space and is never compared against these vectors, so our
# input_type and output_dimension are free choices. See CLAUDE.md.
COHERE_EMBED_MODEL = "embed-v4.0"
COHERE_INPUT_TYPE = "clustering"
COHERE_OUTPUT_DIM = 1024  # Matryoshka dim; 256/512/1024/1536 allowed. Lever for npz size.
EMBED_BATCH = 96  # Cohere embed max texts per call
EMBED_CHECKPOINT_EVERY = 50  # batches between progress checkpoints
CLUE_EMB_NPZ = DATA_DIR / "clue_embeddings.npz"  # float32 [N x dim] + aligned clue_id

# --- Stage 03: UMAP layout ---
# Same params as the sibling map projects for shape consistency. random_state
# fixed for reproducibility (disables UMAP parallelism, but this is a one-shot run).
UMAP_COORDS_NPZ = DATA_DIR / "umap_coords.npz"
UMAP_N_NEIGHBORS = 15
UMAP_MIN_DIST = 0.05
UMAP_RANDOM_STATE = 42

# --- Stage 04: Toponymy region labels (optional; the costliest stage at scale) ---
ANTHROPIC_MODEL_NAMING = "claude-haiku-4-5-20251001"
# AsyncAnthropicNamer fires this many naming calls concurrently (Toponymy's default
# is 10, enforced via an asyncio.Semaphore). Raise if your Anthropic tier allows —
# observed runs show no 429s at 10, so there's headroom. The per-layer Cohere
# keyphrase embedding is sequential and is NOT sped up by this.
ANTHROPIC_MAX_CONCURRENCY = 24
TOPONYMY_LABELS_PARQUET = DATA_DIR / "toponymy_labels.parquet"

# --- Stage 05: DataMapPlot visualization ---
MAP_HTML = DATA_DIR / "clue_map.html"
DOCS_HTML = PROJECT_ROOT / "docs" / "index.html"

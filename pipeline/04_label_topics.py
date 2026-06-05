"""Hierarchical region labels for the map via Toponymy + Claude.

Toponymy names *regions of the embedding space* (place-naming), not individual
clues. The 2D UMAP coords are the substrate the named regions sit on
(clusterable_vectors); the clue embeddings carry the semantic content used while
clustering (embedding_vectors). Toponymy's own keyphrase/term embedder
(CohereEmbedder) is INDEPENDENT of these vectors — it embeds candidate names in
its own (search_query) space and ranks them against each other, never against our
doc vectors — so its model and output dim need not match ours. Output is a
per-clue region name at several zoom levels, fed to DataMapPlot's label layers
(finest first, matching Toponymy's `cluster_layers_` and DataMapPlot's
`*label_layers` convention). Clues sitting in unnamed space come back
"Unlabelled"; that is a gap in the map (signal), not a labeling failure.

Inputs:  data/clue_embeddings.npz, data/umap_coords.npz, data/clue_rows.parquet
Output:  data/toponymy_labels.parquet  (clue_id + label_layer_0..k, finest first)
"""

from __future__ import annotations

import os

import nest_asyncio
import numpy as np
import pandas as pd
from config import (
    ANTHROPIC_API_KEY,
    ANTHROPIC_MAX_CONCURRENCY,
    ANTHROPIC_MODEL_NAMING,
    CLUE_EMB_NPZ,
    CLUE_ROWS_PARQUET,
    CO_API_KEY,
    COHERE_EMBED_MODEL,
    TOPONYMY_LABELS_PARQUET,
    UMAP_COORDS_NPZ,
)

nest_asyncio.apply()

MAX_DOC_CHARS = 2_000


def main():
    from toponymy import Toponymy, ToponymyClusterer
    from toponymy.embedding_wrappers import CohereEmbedder
    from toponymy.llm_wrappers import AsyncAnthropicNamer

    # The coords define point order; align the embedding matrix to the same
    # clue_id order so clusterable_vectors and embedding_vectors line up row-wise.
    crd = np.load(UMAP_COORDS_NPZ, allow_pickle=True)
    coords = crd["coords"].astype(np.float32)
    clue_id = crd["clue_id"]

    ed = np.load(CLUE_EMB_NPZ, allow_pickle=True)
    row = {c: i for i, c in enumerate(ed["clue_id"])}
    idx = np.array([row[c] for c in clue_id], dtype=np.int64)
    embeddings = ed["emb"][idx].astype(np.float32)

    # Documents = the same "Category / Clue / Answer" text we embedded.
    text_by_id = pd.read_parquet(CLUE_ROWS_PARQUET, columns=["clue_id", "embed_text"]).set_index("clue_id")[
        "embed_text"
    ]
    documents = text_by_id.reindex(clue_id).fillna("").str.slice(0, MAX_DOC_CHARS).tolist()
    print(f"Loaded {len(documents):,} clues; embeddings {embeddings.shape}")

    llm = AsyncAnthropicNamer(
        api_key=ANTHROPIC_API_KEY,
        model=ANTHROPIC_MODEL_NAMING,
        max_concurrent_requests=ANTHROPIC_MAX_CONCURRENCY,
    )
    embedder = CohereEmbedder(api_key=CO_API_KEY, model=COHERE_EMBED_MODEL)
    clusterer = ToponymyClusterer(min_clusters=6)

    topic_model = Toponymy(
        llm_wrapper=llm,
        text_embedding_model=embedder,
        clusterer=clusterer,
        object_description="Jeopardy clues",
        corpus_description="Jeopardy clues, each given as its category, clue text, and answer",
        lowest_detail_level=0.5,
        highest_detail_level=1.0,
    )
    np.random.seed(42)
    topic_model.fit(objects=documents, embedding_vectors=embeddings, clusterable_vectors=coords)

    n_layers = len(topic_model.topic_name_vectors_)
    if n_layers == 0:
        raise ValueError("Toponymy produced 0 cluster layers")
    print(f"Toponymy produced {n_layers} cluster layer(s)")

    out = {"clue_id": clue_id}
    for i, names in enumerate(topic_model.topic_name_vectors_):
        out[f"label_layer_{i}"] = names

    df = pd.DataFrame(out)
    tmp = str(TOPONYMY_LABELS_PARQUET) + ".tmp"
    df.to_parquet(tmp, index=False)
    os.replace(tmp, TOPONYMY_LABELS_PARQUET)
    print(f"Wrote {TOPONYMY_LABELS_PARQUET} ({n_layers} layers)")


if __name__ == "__main__":
    main()

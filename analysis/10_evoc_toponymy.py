"""EVoC ambient-space clustering + Toponymy Haiku naming (analysis twin of stage 04).

Where stage 04 draws regions on the 2D UMAP substrate (clusterable_vectors=
coords), EVoC clusters directly on the 1024-d embedding vectors, giving a
region inventory of the HONEST space, independent of projection distortions.

Note: toponymy 0.5.0's EVoCClusterer wrapper targets an older evoc API (it
passes min_num_clusters / next_cluster_size_quantile, which evoc 0.3.1 dropped),
so we drive evoc.EVoC directly and hand Toponymy a duck-typed pre-fitted
clusterer. Toponymy.fit only reads .cluster_layers_ / .cluster_tree_ from it
(the documented "already been fit" branch) and then syncs runtime config onto
the layers, so this is the supported reuse path, minus the broken wrapper.

Two phases, so the free part is never hostage to the paid part:
  1. EVoC clustering (no API calls)  -> data/analysis/evoc_cluster_layers.npz
  2. Cost check, then Toponymy naming (Claude Haiku + Cohere keyphrases)
     -> data/analysis/evoc_labels.parquet (per-clue cluster ids + region names,
        finest layer first, "Unlabelled" = that layer's unclustered points)
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import nest_asyncio
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    EVOC_CLUSTERS_NPZ,
    EVOC_LABELS_PARQUET,
    atomic_save_npz,
    atomic_write_parquet,
    load_clues,
    load_embeddings,
)
from config import (  # noqa: E402  (pipeline dir put on sys.path by common)
    ANTHROPIC_API_KEY,
    ANTHROPIC_MAX_CONCURRENCY,
    ANTHROPIC_MODEL_NAMING,
    CO_API_KEY,
    COHERE_EMBED_MODEL,
)

nest_asyncio.apply()

MAX_DOC_CHARS = 2_000  # mirror stage 04
SEED = 42
# Naming cost scales with region count. Anchor: the 2D run named ~4,200 regions
# for ~$9 on Haiku. Abort (keeping the free clustering artifact) if the estimate
# blows past the authorized ballpark.
EST_USD_PER_1K_REGIONS = 2.15
ABORT_ABOVE_USD = 15.0


class PrefitClusterer:
    """Duck-typed stand-in Toponymy.fit accepts as an already-fitted clusterer."""

    def __init__(self, cluster_layers, cluster_tree):
        self.cluster_layers_ = cluster_layers
        self.cluster_tree_ = cluster_tree


def main():
    import evoc
    from toponymy import Toponymy
    from toponymy.cluster_layer import ClusterLayerText
    from toponymy.clustering import build_cluster_tree, centroids_from_labels
    from toponymy.embedding_wrappers import CohereEmbedder
    from toponymy.llm_wrappers import AsyncAnthropicNamer

    ids, emb = load_embeddings()
    print(f"Loaded embeddings {emb.shape}", flush=True)

    # --- phase 1: cluster (free) ---
    t0 = time.time()
    model = evoc.EVoC(random_state=SEED)
    model.fit(emb)
    label_layers = [np.asarray(v, dtype=np.int32) for v in model.cluster_layers_]
    print(f"EVoC clustering done in {time.time() - t0:.0f}s; {len(label_layers)} layers", flush=True)

    counts = []
    cluster_arrays = {}
    for i, labels in enumerate(label_layers):
        k = int(labels.max()) + 1
        counts.append(k)
        cluster_arrays[f"layer_{i}"] = labels
        print(f"  layer {i}: {k:,} clusters, {np.mean(labels < 0):.1%} unclustered", flush=True)
    if counts != sorted(counts, reverse=True):
        print("WARNING: layer cluster counts are not monotone finest->coarsest", flush=True)
    atomic_save_npz(EVOC_CLUSTERS_NPZ, clue_id=ids, **cluster_arrays)
    print(f"Wrote {EVOC_CLUSTERS_NPZ} (free artifact)", flush=True)

    # --- phase 2: name (paid) ---
    n_regions = sum(counts)
    est = n_regions / 1_000 * EST_USD_PER_1K_REGIONS
    print(f"{n_regions:,} regions total -> estimated naming cost ~${est:.2f} on Haiku", flush=True)
    if est > ABORT_ABOVE_USD:
        print(f"ABORT: estimate exceeds ${ABORT_ABOVE_USD:.0f} guard; clustering artifact kept.", flush=True)
        sys.exit(3)

    layers = [
        ClusterLayerText(labels, centroids_from_labels(labels, emb), layer_id=i)
        for i, labels in enumerate(label_layers)
    ]
    clusterer = PrefitClusterer(layers, build_cluster_tree(label_layers))

    documents = load_clues(ids, columns=["embed_text"])["embed_text"].fillna("").str.slice(0, MAX_DOC_CHARS).tolist()
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
        object_description="Jeopardy clues",
        corpus_description="Jeopardy clues, each given as its category, clue text, and answer",
        lowest_detail_level=0.5,
        highest_detail_level=1.0,
    )
    t1 = time.time()
    topic_model.fit(objects=documents, embedding_vectors=emb, clusterable_vectors=emb)
    print(f"Naming done in {time.time() - t1:.0f}s", flush=True)

    out = {"clue_id": ids}
    for i, labels in enumerate(label_layers):
        out[f"evoc_cluster_{i}"] = labels
    for i, names in enumerate(topic_model.topic_name_vectors_):
        out[f"evoc_label_{i}"] = names
    atomic_write_parquet(pd.DataFrame(out), EVOC_LABELS_PARQUET)
    print(f"Wrote {EVOC_LABELS_PARQUET} ({len(label_layers)} layers)", flush=True)


if __name__ == "__main__":
    main()

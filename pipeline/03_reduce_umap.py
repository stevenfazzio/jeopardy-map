"""Reduce the clue embeddings to 2D with UMAP for the map layout.

Same params as the sibling map projects (cosine, n_neighbors=15, min_dist=0.05)
for shape consistency. random_state fixed for reproducibility.

Input:  data/clue_embeddings.npz  (emb [N x dim] float32, aligned clue_id)
Output: data/umap_coords.npz       (coords [N x 2] float32, aligned clue_id)
"""

from __future__ import annotations

import os

import numpy as np
import umap
from config import CLUE_EMB_NPZ, UMAP_COORDS_NPZ, UMAP_MIN_DIST, UMAP_N_NEIGHBORS, UMAP_RANDOM_STATE


def main():
    d = np.load(CLUE_EMB_NPZ, allow_pickle=True)
    emb = d["emb"].astype(np.float32)
    clue_id = d["clue_id"]
    print(f"Loaded {emb.shape[0]:,} embeddings x {emb.shape[1]}")

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=UMAP_N_NEIGHBORS,
        min_dist=UMAP_MIN_DIST,
        metric="cosine",
        random_state=UMAP_RANDOM_STATE,
        verbose=True,
    )
    coords = reducer.fit_transform(emb).astype(np.float32)
    print(f"UMAP coords: {coords.shape}")

    tmp = str(UMAP_COORDS_NPZ) + ".tmp.npz"
    np.savez(tmp, coords=coords, clue_id=clue_id)
    os.replace(tmp, UMAP_COORDS_NPZ)
    print(f"Wrote {UMAP_COORDS_NPZ} ({UMAP_COORDS_NPZ.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()

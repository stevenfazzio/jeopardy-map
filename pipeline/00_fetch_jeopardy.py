"""Download the full Jeopardy clue archive from Hugging Face.

Source: https://huggingface.co/datasets/robworks-software/jeopardy-clues
~568k clues across seasons 1-41 (1985-2025), partitioned into arbitrary
train/validation/test splits which we union back together.

Output: data/jeopardy_raw.parquet (one row per clue, all seasons).
Cleaning + the embedding input string are built in stage 01.
"""

import os
import tempfile

import pandas as pd
from config import HF_JEOPARDY_DATASET, JEOPARDY_RAW_PARQUET
from datasets import load_dataset


def main():
    print(f"Downloading {HF_JEOPARDY_DATASET} ...")
    ds = load_dataset(HF_JEOPARDY_DATASET)
    print(f"Splits: {list(ds.keys())}")

    frames = []
    for split_name in ds.keys():
        df = ds[split_name].to_pandas()
        df["_hf_split"] = split_name
        frames.append(df)
        print(f"  {split_name}: {len(df):,} rows")

    full = pd.concat(frames, ignore_index=True)
    print(f"\nUnioned: {len(full):,} rows x {len(full.columns)} cols")

    if "clue_id" in full.columns:
        n_dupes = full["clue_id"].duplicated().sum()
        if n_dupes:
            print(f"WARNING: {n_dupes:,} duplicate clue_ids across splits; keeping first.")
            full = full.drop_duplicates(subset="clue_id", keep="first").reset_index(drop=True)

    if "air_date" in full.columns:
        dates = pd.to_datetime(full["air_date"], errors="coerce")
        print(f"air_date range: {dates.min()} -> {dates.max()}")

    print(f"\nColumns: {list(full.columns)}")

    out = JEOPARDY_RAW_PARQUET
    tmp_fd, tmp_path = tempfile.mkstemp(dir=os.path.dirname(out), suffix=".parquet.tmp")
    os.close(tmp_fd)
    try:
        full.to_parquet(tmp_path, index=False)
        verify = pd.read_parquet(tmp_path)
        assert len(verify) == len(full), f"row count mismatch: {len(verify)} vs {len(full)}"
        assert set(verify.columns) == set(full.columns), "column mismatch on round-trip"
        os.replace(tmp_path, out)
    except Exception:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise

    print(f"\nWrote {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()

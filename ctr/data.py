"""Load the Parquet splits written by spark_prep into numpy arrays."""
import json
import os

import numpy as np
import pandas as pd

from ctr.config import CAT_COLS, INT_COLS, LABEL


def load_split(processed_dir: str, name: str, max_rows: int | None = None) -> dict:
    """Return {"y": [n], "dense": float32 [n,13], "cats": int32 [n,26]}.

    NOTE: this reads the whole split into memory. That is fine up to a few million
    rows; for the full 45M-row Criteo set stream Parquet batches instead
    (see README, "Scaling to full Criteo").
    """
    df = pd.read_parquet(os.path.join(processed_dir, name))
    if max_rows is not None:
        df = df.iloc[:max_rows]
    return {
        "y": df[LABEL].to_numpy(dtype=np.float32),
        "dense": df[INT_COLS].to_numpy(dtype=np.float32),
        "cats": df[CAT_COLS].to_numpy(dtype=np.int32),
    }


def load_metadata(processed_dir: str) -> dict:
    with open(os.path.join(processed_dir, "metadata.json")) as f:
        return json.load(f)

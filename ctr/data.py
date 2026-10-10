"""Load the Parquet splits written by spark_prep.

Two ways to read them:
  * load_split    - whole split into numpy arrays (simple, but RAM = rows x ~156 bytes x a few copies)
  * stream_split  - generator of mini-batches read from disk, so peak RAM stays roughly constant
                    no matter how many rows the split has (this is what makes full Criteo feasible on a laptop)
"""
import json
import os

import numpy as np
import pandas as pd
import pyarrow.dataset as pads

from ctr.config import CAT_COLS, INT_COLS, LABEL

_COLS = [LABEL] + INT_COLS + CAT_COLS


def load_split(processed_dir: str, name: str, max_rows: int | None = None) -> dict:
    """Return {"y": [n], "dense": float32 [n,13], "cats": int32 [n,26]}, reading the whole split into memory."""
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


def count_rows(processed_dir: str, name: str) -> int:
    """Row count from Parquet metadata only (no data is read)."""
    return pads.dataset(os.path.join(processed_dir, name), format="parquet").count_rows()


def load_labels(processed_dir: str, name: str, n: int | None = None) -> np.ndarray:
    """Read only the label column (first n rows, in file order)."""
    scanner = pads.dataset(os.path.join(processed_dir, name), format="parquet").scanner(
        columns=[LABEL], use_threads=False)
    tbl = scanner.head(n) if n is not None else scanner.to_table()
    return tbl.column(LABEL).to_numpy().astype(np.float32)


def _to_arrays(rb):
    dense = np.stack([rb.column(c).to_numpy(zero_copy_only=False) for c in INT_COLS], axis=1).astype(np.float32, copy=False)
    cats = np.stack([rb.column(c).to_numpy(zero_copy_only=False) for c in CAT_COLS], axis=1).astype(np.int32, copy=False)
    y = rb.column(LABEL).to_numpy(zero_copy_only=False).astype(np.float32, copy=False)
    return dense, cats, y


def stream_split(processed_dir: str, name: str, batch_size: int = 1024, shuffle: bool = False,
                 shuffle_rows: int = 500_000, seed: int = 0, max_rows: int | None = None):
    """Yield (dense, cats, y) mini-batches read from disk, in file order.

    Memory is bounded by `shuffle_rows` (the in-memory chunk) rather than by the split size:
    shuffle_rows=500k rows is about 80 MB. With shuffle=True, rows are shuffled WITHIN each
    chunk (a local shuffle: neighbouring ~500k rows are mixed, the chunk order stays chronological).
    That is an approximation of a full shuffle, which would need the whole split in RAM.
    `max_rows` stops after the first max_rows rows. use_threads=False keeps the row order deterministic.
    """
    chunk_target = shuffle_rows if shuffle else 131_072
    scanner = pads.dataset(os.path.join(processed_dir, name), format="parquet").scanner(
        columns=_COLS, batch_size=min(chunk_target, 131_072), use_threads=False)
    rng = np.random.default_rng(seed)
    remaining = max_rows
    pending, pending_n = [], 0

    def minibatches(chunks):
        dense = np.concatenate([c[0] for c in chunks])
        cats = np.concatenate([c[1] for c in chunks])
        y = np.concatenate([c[2] for c in chunks])
        if shuffle:
            p = rng.permutation(len(y))
            dense, cats, y = dense[p], cats[p], y[p]
        for i in range(0, len(y), batch_size):
            yield dense[i:i + batch_size], cats[i:i + batch_size], y[i:i + batch_size]

    for rb in scanner.to_batches():
        if remaining is not None:
            if remaining <= 0:
                break
            if rb.num_rows > remaining:
                rb = rb.slice(0, remaining)
            remaining -= rb.num_rows
        pending.append(_to_arrays(rb))
        pending_n += rb.num_rows
        if pending_n >= chunk_target:
            yield from minibatches(pending)
            pending, pending_n = [], 0
    if pending:
        yield from minibatches(pending)

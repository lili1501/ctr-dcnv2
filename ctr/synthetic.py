"""Synthetic data in the exact Criteo Display Advertising format.

NOT real data. It exists so the pipeline can be run and tested end to end where the
real dataset cannot be downloaded. The label is generated from a known logit that
includes main effects AND a categorical x categorical interaction plus an
integer x categorical interaction, so models that learn feature crosses have real
signal to find. Any metric computed on this data validates code, not a research claim.

Format per line (tab-separated, no header):
    label, I1..I13 (ints, may be empty), C1..C26 (8-hex-char strings, may be empty)

Usage:
    python -m ctr.synthetic --rows 200000 --out data/raw/train.txt --seed 0
"""
import argparse
import os

import numpy as np

from ctr.config import CAT_COLS, INT_COLS

# Cardinalities loosely mimic Criteo: a few tiny, many medium, some huge.
_CARDS = [1400, 550, 200_000, 100_000, 300, 20, 12_000, 600, 3, 40_000, 5_000,
          150_000, 3_000, 25, 9_000, 120_000, 10, 4_000, 2_000, 4, 150_000, 15,
          80, 30_000, 70, 20_000]
if len(_CARDS) != len(CAT_COLS):
    raise ImportError('ctr.synthetic only supports the criteo_kaggle schema (26 categorical columns); unset CTR_SCHEMA')


def generate(n_rows: int, seed: int = 0, target_ctr: float = 0.25) -> dict:
    rng = np.random.default_rng(seed)

    # --- integer features: heavy-tailed counts, ~10% missing -------------------
    ints = np.floor(rng.lognormal(mean=1.5, sigma=1.6, size=(n_rows, len(INT_COLS)))).astype(np.int64)
    ints[:, 1] -= 3                                             # I2 can be slightly negative in Criteo
    int_missing = rng.random(ints.shape) < 0.10

    # --- categorical features: Zipf-distributed ids, ~5% missing ---------------
    cats = np.empty((n_rows, len(CAT_COLS)), dtype=np.int64)
    for j, card in enumerate(_CARDS):
        z = rng.zipf(1.3, size=n_rows) - 1
        cats[:, j] = z % card
    cat_missing = rng.random(cats.shape) < 0.05

    # --- ground-truth logit ----------------------------------------------------
    # Per-category-value random effects (stable across rows via a seeded table).
    eff_rng = np.random.default_rng(12345)
    tables = [eff_rng.normal(0, 0.6, size=card) for card in _CARDS]

    z = np.zeros(n_rows)
    z += 0.35 * np.log1p(np.clip(ints[:, 0], 0, None))           # I1 main effect
    z -= 0.25 * np.log1p(np.clip(ints[:, 3], 0, None))           # I4 main effect
    for j in (0, 1, 5, 8, 13):                                    # a few cat main effects
        z += tables[j][cats[:, j]]
    z += 1.2 * tables[0][cats[:, 0]] * tables[1][cats[:, 1]]      # cat x cat interaction (C1 x C2)
    z += 0.30 * np.log1p(np.clip(ints[:, 6], 0, None)) * tables[5][cats[:, 5]]  # int x cat interaction

    # Calibrate the intercept so the base rate is ~target_ctr, then sample labels.
    lo, hi = -10.0, 10.0
    for _ in range(50):
        mid = (lo + hi) / 2
        if (1 / (1 + np.exp(-(z + mid)))).mean() > target_ctr:
            hi = mid
        else:
            lo = mid
    p = 1 / (1 + np.exp(-(z + (lo + hi) / 2)))
    y = (rng.random(n_rows) < p).astype(np.int8)
    return {"y": y, "ints": ints, "int_missing": int_missing, "cats": cats,
            "cat_missing": cat_missing, "true_p": p}


def _hex(v: int, col: int) -> str:
    # deterministic 8-hex-char token per (column, id), like Criteo's anonymised hashes
    return format((v * 2654435761 + col * 40503 + 0x9E3779B1) & 0xFFFFFFFF, "08x")


def write_tsv(path: str, n_rows: int, seed: int = 0, chunk: int = 100_000) -> float:
    """Write n_rows lines in Criteo format. Returns the empirical click rate."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    clicks, written, part = 0, 0, 0
    with open(path, "w") as f:
        while written < n_rows:
            m = min(chunk, n_rows - written)
            d = generate(m, seed=seed * 1_000_003 + part)
            for i in range(m):
                int_f = ["" if d["int_missing"][i, k] else str(d["ints"][i, k]) for k in range(len(INT_COLS))]
                cat_f = ["" if d["cat_missing"][i, k] else _hex(int(d["cats"][i, k]), k) for k in range(len(CAT_COLS))]
                f.write("\t".join([str(int(d["y"][i]))] + int_f + cat_f) + "\n")
            clicks += int(d["y"].sum())
            written += m
            part += 1
    return clicks / n_rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=200_000)
    ap.add_argument("--out", default="data/raw/train.txt")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    ctr = write_tsv(a.out, a.rows, a.seed)
    print(f"wrote {a.rows} rows to {a.out}; empirical CTR={ctr:.4f}")

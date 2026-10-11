"""Shared constants: dataset schema and default hyperparameters.

The schema is chosen by the CTR_SCHEMA environment variable (default "criteo_kaggle"):
  criteo_kaggle      Criteo Display Advertising Challenge: label, I1..I13, C1..C26
  criteo_attribution Criteo Attribution Modeling dataset (click label): label, I1, C1..C11
                     (see ctr/prepare_attribution.py for which source columns map where)
Every module reads its column lists from here, so a pipeline run uses one schema end to end.
"""
import os

SCHEMAS = {
    "criteo_kaggle": (13, 26),
    "criteo_attribution": (1, 11),
}
SCHEMA = os.environ.get("CTR_SCHEMA", "criteo_kaggle")
if SCHEMA not in SCHEMAS:
    raise ValueError(f"CTR_SCHEMA must be one of {sorted(SCHEMAS)}, got {SCHEMA!r}")
N_INT, N_CAT = SCHEMAS[SCHEMA]

LABEL = "label"
INT_COLS = [f"I{i}" for i in range(1, N_INT + 1)]   # integer (count-like) features
CAT_COLS = [f"C{i}" for i in range(1, N_CAT + 1)]   # hashed categorical features
RAW_COLUMNS = [LABEL] + INT_COLS + CAT_COLS         # raw file column order (tab-separated, no header)

# Categorical values are hashed into [1, NUM_BUCKETS-1]; index 0 is reserved for "missing".
NUM_BUCKETS = 50_000

# Split fractions by row order. Both datasets are sorted by time, so a positional
# split is a time-based split (train on the past, evaluate on the future).
TRAIN_FRAC, VAL_FRAC = 0.8, 0.1   # remainder (0.1) is test

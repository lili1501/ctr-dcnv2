"""Shared constants: Criteo schema and default hyperparameters."""

LABEL = "label"
INT_COLS = [f"I{i}" for i in range(1, 14)]   # 13 integer (count-like) features
CAT_COLS = [f"C{i}" for i in range(1, 27)]   # 26 hashed categorical features
RAW_COLUMNS = [LABEL] + INT_COLS + CAT_COLS  # Criteo train.txt column order (tab-separated, no header)

# Categorical values are hashed into [1, NUM_BUCKETS-1]; index 0 is reserved for "missing".
NUM_BUCKETS = 50_000

# Split fractions by row order. Criteo's train file is chronological, so a positional
# split is a time-based split (train on the past, evaluate on the future).
TRAIN_FRAC, VAL_FRAC = 0.8, 0.1   # remainder (0.1) is test

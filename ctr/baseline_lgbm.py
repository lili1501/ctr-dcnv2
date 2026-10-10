"""LightGBM baseline on the preprocessed Criteo features.

Integer features are used as numeric; the 26 hashed categorical indices are passed to
LightGBM as native categorical features (with regularisation, because each has up to
NUM_BUCKETS levels). Early stopping watches validation log loss.

Usage:
    python -m ctr.baseline_lgbm --processed data/processed --out artifacts/lgbm
"""
import argparse
import json
import os
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from ctr.config import CAT_COLS, INT_COLS
from ctr.data import load_split
from ctr.metrics import evaluate


def to_frame(split: dict) -> pd.DataFrame:
    df = pd.DataFrame(split["dense"], columns=INT_COLS)
    for j, c in enumerate(CAT_COLS):
        df[c] = pd.Categorical(split["cats"][:, j])
    return df


def train(processed_dir: str, out_dir: str, max_train_rows: int | None = None, seed: int = 0) -> dict:
    tr = load_split(processed_dir, "train", max_train_rows)
    va = load_split(processed_dir, "val")
    te = load_split(processed_dir, "test")

    # Make category dtypes identical across splits (union of all levels seen).
    frames = {k: to_frame(v) for k, v in (("train", tr), ("val", va), ("test", te))}
    for c in CAT_COLS:
        levels = pd.api.types.union_categoricals([frames[k][c] for k in frames]).categories
        for k in frames:
            frames[k][c] = frames[k][c].cat.set_categories(levels)

    params = dict(
        objective="binary", metric="binary_logloss",
        learning_rate=0.05, num_leaves=63, min_data_in_leaf=100,
        feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
        lambda_l2=10.0,
        cat_smooth=50, cat_l2=20, min_data_per_group=200, max_cat_threshold=64,
        seed=seed, num_threads=os.cpu_count(), verbose=-1,
    )
    dtrain = lgb.Dataset(frames["train"], tr["y"], categorical_feature=CAT_COLS, free_raw_data=False)
    dval = lgb.Dataset(frames["val"], va["y"], reference=dtrain)

    t0 = time.time()
    booster = lgb.train(params, dtrain, num_boost_round=2000, valid_sets=[dval],
                        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
    fit_seconds = time.time() - t0

    preds = {k: booster.predict(frames[k], num_iteration=booster.best_iteration) for k in ("val", "test")}
    results = {
        "best_iteration": int(booster.best_iteration),
        "fit_seconds": round(fit_seconds, 1),
        "base_rate_logloss_test": None,
        "val": evaluate(va["y"], preds["val"]),
        "test": evaluate(te["y"], preds["test"]),
    }
    # reference: constant prediction of the train click rate
    from ctr.metrics import log_loss
    results["base_rate_logloss_test"] = log_loss(te["y"], np.full(len(te["y"]), tr["y"].mean()))

    os.makedirs(out_dir, exist_ok=True)
    booster.save_model(os.path.join(out_dir, "lgbm.txt"))
    np.save(os.path.join(out_dir, "pred_val.npy"), preds["val"])
    np.save(os.path.join(out_dir, "pred_test.npy"), preds["test"])
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", default="data/processed")
    ap.add_argument("--out", default="artifacts/lgbm")
    ap.add_argument("--max-train-rows", type=int, default=None)
    a = ap.parse_args()
    print(json.dumps(train(a.processed, a.out, a.max_train_rows), indent=2))

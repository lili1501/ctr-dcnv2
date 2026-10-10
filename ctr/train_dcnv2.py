"""Train and evaluate DCN-V2 on the preprocessed splits.

Early stopping watches VALIDATION log loss; the test split is scored once at the end.

Usage:
    python -m ctr.train_dcnv2 --processed data/processed --out artifacts/dcnv2
"""
import argparse
import json
import os
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import keras
import numpy as np
import tensorflow as tf

from ctr.config import NUM_BUCKETS
from ctr.data import load_metadata, load_split
from ctr.dcnv2 import build_dcnv2
from ctr.metrics import evaluate


def make_ds(split: dict, batch_size: int, shuffle: bool, seed: int) -> tf.data.Dataset:
    ds = tf.data.Dataset.from_tensor_slices(({"dense": split["dense"], "cats": split["cats"]}, split["y"]))
    if shuffle:
        ds = ds.shuffle(min(len(split["y"]), 200_000), seed=seed, reshuffle_each_iteration=True)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def train(processed_dir: str, out_dir: str, epochs: int = 10, batch_size: int = 1024, lr: float = 1e-3,
          seed: int = 0, max_train_rows: int | None = None, **model_kwargs) -> dict:
    keras.utils.set_random_seed(seed)
    meta = load_metadata(processed_dir)
    num_buckets = meta.get("num_buckets", NUM_BUCKETS)

    tr = load_split(processed_dir, "train", max_train_rows)
    va = load_split(processed_dir, "val")
    te = load_split(processed_dir, "test")

    model = build_dcnv2(num_buckets=num_buckets, **model_kwargs)
    model.compile(optimizer=keras.optimizers.Adam(lr), loss=keras.losses.BinaryCrossentropy(),
                  metrics=[keras.metrics.AUC(name="auc")])

    t0 = time.time()
    hist = model.fit(
        make_ds(tr, batch_size, True, seed), validation_data=make_ds(va, 4096, False, seed),
        epochs=epochs, verbose=2, shuffle=False,  # shuffling is done in the tf.data pipeline
        callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)])
    fit_seconds = time.time() - t0

    def predict(split):
        return model.predict({"dense": split["dense"], "cats": split["cats"]}, batch_size=4096, verbose=0).ravel()

    p_val, p_test = predict(va), predict(te)
    results = {
        "params": int(model.count_params()),
        "epochs_run": len(hist.history["loss"]),
        "best_epoch": int(np.argmin(hist.history["val_loss"])) + 1,
        "fit_seconds": round(fit_seconds, 1),
        "model_kwargs": {k: (list(v) if isinstance(v, tuple) else v) for k, v in model_kwargs.items()},
        "val": evaluate(va["y"], p_val),
        "test": evaluate(te["y"], p_test),
    }

    os.makedirs(out_dir, exist_ok=True)
    model.save(os.path.join(out_dir, "dcnv2.keras"))
    np.save(os.path.join(out_dir, "pred_val.npy"), p_val)
    np.save(os.path.join(out_dir, "pred_test.npy"), p_test)
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)
    with open(os.path.join(out_dir, "history.json"), "w") as f:
        json.dump({k: [float(x) for x in v] for k, v in hist.history.items()}, f, indent=2)
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", default="data/processed")
    ap.add_argument("--out", default="artifacts/dcnv2")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--max-train-rows", type=int, default=None)
    ap.add_argument("--emb-dim", type=int, default=16)
    ap.add_argument("--n-cross", type=int, default=3)
    ap.add_argument("--cross-rank", type=int, default=None)
    ap.add_argument("--structure", default="parallel", choices=["parallel", "stacked"])
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--emb-l2", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    r = train(a.processed, a.out, a.epochs, a.batch_size, a.lr, a.seed, a.max_train_rows,
              emb_dim=a.emb_dim, n_cross=a.n_cross, cross_rank=a.cross_rank,
              structure=a.structure, dropout=a.dropout, emb_l2=a.emb_l2)
    print(json.dumps(r, indent=2))

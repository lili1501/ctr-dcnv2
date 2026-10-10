"""Train and evaluate DCN-V2 on the preprocessed splits.

Early stopping watches VALIDATION log loss; the test split is scored once at the end.

Two data modes:
  default      the splits are loaded into RAM (fast, simple; fine up to a few million rows)
  --streaming  mini-batches are read from the Parquet files on disk, so RAM stays small and
               roughly constant regardless of dataset size (use this for the full Criteo set)

Usage:
    python -m ctr.train_dcnv2 --processed data/processed --out artifacts/dcnv2
    python -m ctr.train_dcnv2 --processed data/processed --out artifacts/dcnv2 --streaming --eval-max-rows 500000
"""
import argparse
import json
import os
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import keras
import numpy as np
import tensorflow as tf

from ctr.config import CAT_COLS, INT_COLS, NUM_BUCKETS
from ctr.data import count_rows, load_metadata, load_split, stream_split
from ctr.dcnv2 import build_dcnv2
from ctr.metrics import evaluate
from ctr.resources import configure_tf_threads


def make_ds(split: dict, batch_size: int, shuffle: bool, seed: int) -> tf.data.Dataset:
    ds = tf.data.Dataset.from_tensor_slices(({"dense": split["dense"], "cats": split["cats"]}, split["y"]))
    if shuffle:
        ds = ds.shuffle(min(len(split["y"]), 200_000), seed=seed, reshuffle_each_iteration=True)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def make_stream_ds(processed_dir: str, name: str, batch_size: int, shuffle: bool, seed: int,
                   max_rows: int | None, shuffle_rows: int) -> tf.data.Dataset:
    """tf.data pipeline fed by a generator that streams Parquet batches from disk."""
    epoch = {"n": 0}

    def gen():
        s = seed + epoch["n"]          # fresh local shuffle each epoch
        epoch["n"] += 1
        for dense, cats, y in stream_split(processed_dir, name, batch_size, shuffle, shuffle_rows, s, max_rows):
            yield {"dense": dense, "cats": cats}, y

    sig = ({"dense": tf.TensorSpec((None, len(INT_COLS)), tf.float32),
            "cats": tf.TensorSpec((None, len(CAT_COLS)), tf.int32)}, tf.TensorSpec((None,), tf.float32))
    return tf.data.Dataset.from_generator(gen, output_signature=sig).prefetch(tf.data.AUTOTUNE)


def predict_stream(model, processed_dir: str, name: str, max_rows: int | None, batch_size: int = 4096):
    """Score a split batch by batch; only the (small) label and prediction vectors are kept in RAM."""
    ys, ps = [], []
    for dense, cats, y in stream_split(processed_dir, name, batch_size, max_rows=max_rows):
        ps.append(np.asarray(model.predict_on_batch({"dense": dense, "cats": cats})).ravel())
        ys.append(y)
    return np.concatenate(ys), np.concatenate(ps)


def train(processed_dir: str, out_dir: str, epochs: int = 10, batch_size: int = 1024, lr: float = 1e-3,
          seed: int = 0, max_train_rows: int | None = None, streaming: bool = False,
          eval_max_rows: int | None = None, shuffle_rows: int = 500_000, **model_kwargs) -> dict:
    keras.utils.set_random_seed(seed)
    configure_tf_threads(tf)
    meta = load_metadata(processed_dir)
    num_buckets = meta.get("num_buckets", NUM_BUCKETS)

    model = build_dcnv2(num_buckets=num_buckets, **model_kwargs)
    model.compile(optimizer=keras.optimizers.Adam(lr), loss=keras.losses.BinaryCrossentropy(),
                  metrics=[keras.metrics.AUC(name="auc")])
    stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)

    t0 = time.time()
    if streaming:
        train_ds = make_stream_ds(processed_dir, "train", batch_size, True, seed, max_train_rows, shuffle_rows)
        val_ds = make_stream_ds(processed_dir, "val", 4096, False, seed, eval_max_rows, shuffle_rows)
        hist = model.fit(train_ds, validation_data=val_ds, epochs=epochs, verbose=2, callbacks=[stop])
        fit_seconds = time.time() - t0
        y_val, p_val = predict_stream(model, processed_dir, "val", eval_max_rows)
        y_test, p_test = predict_stream(model, processed_dir, "test", eval_max_rows)
        train_rows = min(count_rows(processed_dir, "train"), max_train_rows or 10 ** 12)
    else:
        tr = load_split(processed_dir, "train", max_train_rows)
        va = load_split(processed_dir, "val", eval_max_rows)
        te = load_split(processed_dir, "test", eval_max_rows)
        hist = model.fit(
            make_ds(tr, batch_size, True, seed), validation_data=make_ds(va, 4096, False, seed),
            epochs=epochs, verbose=2, shuffle=False,  # shuffling is done in the tf.data pipeline
            callbacks=[stop])
        fit_seconds = time.time() - t0
        predict = lambda s: model.predict({"dense": s["dense"], "cats": s["cats"]}, batch_size=4096, verbose=0).ravel()
        y_val, p_val, y_test, p_test = va["y"], predict(va), te["y"], predict(te)
        train_rows = len(tr["y"])

    results = {
        "params": int(model.count_params()),
        "streaming": streaming,
        "train_rows": int(train_rows),
        "eval_max_rows": eval_max_rows,
        "epochs_run": len(hist.history["loss"]),
        "best_epoch": int(np.argmin(hist.history["val_loss"])) + 1,
        "fit_seconds": round(fit_seconds, 1),
        "model_kwargs": {k: (list(v) if isinstance(v, tuple) else v) for k, v in model_kwargs.items()},
        "val": evaluate(y_val, p_val),
        "test": evaluate(y_test, p_test),
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
    ap.add_argument("--streaming", action="store_true", help="stream batches from disk (low, constant RAM)")
    ap.add_argument("--eval-max-rows", type=int, default=None, help="score only the first N val/test rows")
    ap.add_argument("--shuffle-rows", type=int, default=500_000, help="streaming: rows held in RAM for the local shuffle")
    ap.add_argument("--emb-dim", type=int, default=16)
    ap.add_argument("--n-cross", type=int, default=3)
    ap.add_argument("--cross-rank", type=int, default=None)
    ap.add_argument("--structure", default="parallel", choices=["parallel", "stacked"])
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--emb-l2", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    r = train(a.processed, a.out, a.epochs, a.batch_size, a.lr, a.seed, a.max_train_rows,
              streaming=a.streaming, eval_max_rows=a.eval_max_rows, shuffle_rows=a.shuffle_rows,
              emb_dim=a.emb_dim, n_cross=a.n_cross, cross_rank=a.cross_rank,
              structure=a.structure, dropout=a.dropout, emb_l2=a.emb_l2)
    print(json.dumps(r, indent=2))

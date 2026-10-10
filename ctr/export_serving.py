"""Export the trained DCN-V2 as a TensorFlow SavedModel for TF Serving, and verify it.

TF Serving expects  <model_base_path>/<integer version>/saved_model.pb  (+ variables/).

Usage:
    python -m ctr.export_serving --model artifacts/dcnv2/dcnv2.keras --out artifacts/serving/ctr --version 1
"""
import argparse
import json
import os
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import keras
import numpy as np
import tensorflow as tf

from ctr.data import load_split
import ctr.dcnv2  # noqa: F401  (registers the custom layers so load_model works)


def export(model_path: str, out_dir: str, version: int = 1) -> str:
    model = keras.saving.load_model(model_path)
    path = os.path.join(out_dir, str(version))
    model.export(path)  # SavedModel with a `serving_default` signature
    return path


def describe_signature(saved_model_dir: str) -> dict:
    loaded = tf.saved_model.load(saved_model_dir)
    fn = loaded.signatures["serving_default"]
    _, kwargs = fn.structured_input_signature
    return {
        "inputs": {k: {"shape": v.shape.as_list(), "dtype": v.dtype.name} for k, v in kwargs.items()},
        "outputs": {k: {"shape": v.shape.as_list(), "dtype": v.dtype.name} for k, v in fn.structured_outputs.items()},
    }


def verify(saved_model_dir: str, model_path: str, processed_dir: str, n: int = 2000) -> dict:
    """Exported model must reproduce the Keras model's predictions on real test rows."""
    keras_model = keras.saving.load_model(model_path)
    te = load_split(processed_dir, "test", max_rows=n)
    ref = keras_model.predict({"dense": te["dense"], "cats": te["cats"]}, batch_size=1024, verbose=0).ravel()

    fn = tf.saved_model.load(saved_model_dir).signatures["serving_default"]
    out_key = next(iter(fn.structured_outputs))
    got = np.concatenate([
        fn(dense=tf.constant(te["dense"][i:i + 1024]), cats=tf.constant(te["cats"][i:i + 1024]))[out_key].numpy().ravel()
        for i in range(0, len(ref), 1024)])
    return {"rows": int(len(ref)), "max_abs_diff": float(np.max(np.abs(ref - got))),
            "output_key": out_key, "all_in_0_1": bool(np.all((got > 0) & (got < 1)))}


def inprocess_latency(saved_model_dir: str, processed_dir: str, batch_size: int = 1, n_requests: int = 300) -> dict:
    """Wall-clock latency of the SavedModel signature called in-process (NO network, NO TF Serving).

    This is only a lower bound / sanity check. Real TF Serving p95 must be measured against a
    running server with scripts/benchmark_serving.py.
    """
    te = load_split(processed_dir, "test", max_rows=max(n_requests * batch_size, 1000))
    fn = tf.saved_model.load(saved_model_dir).signatures["serving_default"]
    dense, cats = tf.constant(te["dense"][:batch_size]), tf.constant(te["cats"][:batch_size])
    for _ in range(20):                      # warm-up
        fn(dense=dense, cats=cats)
    lat = []
    for _ in range(n_requests):
        t0 = time.perf_counter()
        fn(dense=dense, cats=cats)
        lat.append((time.perf_counter() - t0) * 1000)
    return {"batch_size": batch_size, "p50_ms": float(np.percentile(lat, 50)),
            "p95_ms": float(np.percentile(lat, 95)), "p99_ms": float(np.percentile(lat, 99))}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="artifacts/dcnv2/dcnv2.keras")
    ap.add_argument("--out", default="artifacts/serving/ctr")
    ap.add_argument("--version", type=int, default=1)
    ap.add_argument("--processed", default="data/processed")
    ap.add_argument("--reports", default="reports")
    a = ap.parse_args()
    path = export(a.model, a.out, a.version)
    report = {"saved_model": path, "signature": describe_signature(path),
              "verification": verify(path, a.model, a.processed),
              "inprocess_latency": [inprocess_latency(path, a.processed, b) for b in (1, 32)]}
    os.makedirs(a.reports, exist_ok=True)
    with open(os.path.join(a.reports, "serving_check.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))

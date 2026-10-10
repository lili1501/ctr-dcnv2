"""Latency benchmark client for a running TF Serving instance (REST API).

Start the server first (see README), then:
    python -m ctr.benchmark_serving --url http://localhost:8501/v1/models/ctr:predict \
        --processed data/processed --batch-size 1 --requests 2000

Reports client-observed latency (includes network + JSON encode/decode), which is the number
to quote as "p95 latency" for a served model. It sends the TF Serving REST "instances" format:
    {"instances": [{"dense": [...13 floats], "cats": [...26 ints]}, ...]}
and expects {"predictions": [[p], ...]} back.
"""
import argparse
import json
import time

import numpy as np
import requests

from ctr.data import load_split


def build_payload(dense: np.ndarray, cats: np.ndarray) -> dict:
    return {"instances": [{"dense": d.tolist(), "cats": c.tolist()} for d, c in zip(dense, cats)]}


def run_benchmark(url: str, dense: np.ndarray, cats: np.ndarray, batch_size: int = 1,
                  n_requests: int = 1000, warmup: int = 50, timeout: float = 5.0) -> dict:
    sess = requests.Session()  # keep-alive, like a real client
    n = len(dense)
    payloads = [build_payload(dense[(i * batch_size) % n:(i * batch_size) % n + batch_size],
                              cats[(i * batch_size) % n:(i * batch_size) % n + batch_size])
                for i in range(min(n_requests + warmup, 500))]  # pre-build so encoding cost isn't timed twice
    lat, errors, bad = [], 0, 0
    for i in range(n_requests + warmup):
        body = json.dumps(payloads[i % len(payloads)])
        t0 = time.perf_counter()
        try:
            r = sess.post(url, data=body, timeout=timeout)
            ok = r.status_code == 200
            preds = r.json().get("predictions") if ok else None
        except requests.RequestException:
            ok, preds = False, None
        dt = (time.perf_counter() - t0) * 1000
        if i < warmup:
            continue
        if not ok:
            errors += 1
            continue
        flat = np.asarray(preds, dtype=float).ravel()
        if flat.size != batch_size or not np.all((flat > 0) & (flat < 1)):
            bad += 1
        lat.append(dt)
    out = {"url": url, "batch_size": batch_size, "ok_requests": len(lat), "errors": errors, "bad_responses": bad}
    if lat:
        out.update({"p50_ms": float(np.percentile(lat, 50)), "p95_ms": float(np.percentile(lat, 95)),
                    "p99_ms": float(np.percentile(lat, 99)), "mean_ms": float(np.mean(lat))})
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8501/v1/models/ctr:predict")
    ap.add_argument("--processed", default="data/processed")
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--requests", type=int, default=1000)
    a = ap.parse_args()
    te = load_split(a.processed, "test", max_rows=10_000)
    print(json.dumps(run_benchmark(a.url, te["dense"], te["cats"], a.batch_size, a.requests), indent=2))

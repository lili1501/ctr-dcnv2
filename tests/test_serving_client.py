"""Checks the export + benchmark client against a LOCAL STAND-IN for TF Serving's REST API.

The stand-in runs the real exported SavedModel and speaks the same JSON format
("instances" in, "predictions" out). It validates our payload format, SavedModel export and
client logic. It does NOT validate the real TF Serving binary or its latency.
"""
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import keras
import numpy as np
import pytest
import tensorflow as tf

from ctr.benchmark_serving import build_payload, run_benchmark
from ctr.dcnv2 import build_dcnv2


@pytest.fixture(scope="module")
def served(tmp_path_factory):
    model = build_dcnv2(num_buckets=60, emb_dim=4, n_cross=2, mlp_units=(8,))
    path = str(tmp_path_factory.mktemp("sm") / "1")
    model.export(path)
    fn = tf.saved_model.load(path).signatures["serving_default"]

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):  # silence
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            inst = body["instances"]
            dense = tf.constant([i["dense"] for i in inst], tf.float32)
            cats = tf.constant([i["cats"] for i in inst], tf.int32)
            out = next(iter(fn(dense=dense, cats=cats).values())).numpy().tolist()
            data = json.dumps({"predictions": out}).encode()
            self.send_response(200); self.send_header("Content-Length", str(len(data))); self.end_headers()
            self.wfile.write(data)

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}/v1/models/ctr:predict", model
    srv.shutdown()


def test_exported_savedmodel_matches_keras(served):
    url, model = served
    rng = np.random.default_rng(0)
    dense, cats = rng.normal(size=(8, 13)).astype("float32"), rng.integers(0, 60, (8, 26)).astype("int32")
    import requests
    got = np.array(requests.post(url, json=build_payload(dense, cats)).json()["predictions"]).ravel()
    ref = model.predict({"dense": dense, "cats": cats}, verbose=0).ravel()
    np.testing.assert_allclose(got, ref, atol=1e-6)


@pytest.mark.parametrize("batch", [1, 16])
def test_benchmark_client_reports_sane_latency_and_no_errors(served, batch):
    url, _ = served
    rng = np.random.default_rng(1)
    dense, cats = rng.normal(size=(400, 13)).astype("float32"), rng.integers(0, 60, (400, 26)).astype("int32")
    r = run_benchmark(url, dense, cats, batch_size=batch, n_requests=40, warmup=5)
    assert r["errors"] == 0 and r["bad_responses"] == 0 and r["ok_requests"] == 40
    assert 0 < r["p50_ms"] <= r["p95_ms"] <= r["p99_ms"]


def test_benchmark_client_counts_errors_when_server_unreachable():
    rng = np.random.default_rng(2)
    dense, cats = rng.normal(size=(10, 13)).astype("float32"), rng.integers(0, 60, (10, 26)).astype("int32")
    r = run_benchmark("http://127.0.0.1:9/v1/models/x:predict", dense, cats, n_requests=3, warmup=0, timeout=0.5)
    assert r["errors"] == 3 and r["ok_requests"] == 0

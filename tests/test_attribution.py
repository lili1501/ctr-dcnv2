"""Criteo Attribution dataset support: schema switch, converter, truncated downloads, and a real end-to-end run."""
import gzip
import json
import os
import subprocess
import sys

import numpy as np
import pytest

from ctr.prepare_attribution import convert

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COLS = ("timestamp uid campaign conversion conversion_timestamp conversion_id attribution click click_pos click_nb cost cpo "
        "time_since_last_click cat1 cat2 cat3 cat4 cat5 cat6 cat7 cat8 cat9").split()


def make_file(path, n, seed=0, unsorted=False):
    rng = np.random.default_rng(seed)
    ts = np.sort(rng.integers(0, 10**6, n))
    if unsorted:
        ts = ts[::-1].copy()
    camp, uid = rng.integers(0, 30, n), rng.integers(0, 2000, n)
    cats = rng.integers(0, 100, (n, 9))
    tsl = np.where(rng.random(n) < .5, -1, rng.integers(0, 90000, n))
    z = -1.2 + 1.0 * ((camp % 5) == 0) + 0.8 * ((cats[:, 0] % 4) == 1) * ((cats[:, 1] % 3) == 0)
    click = (rng.random(n) < 1 / (1 + np.exp(-z))).astype(int)
    conv = ((rng.random(n) < .05) & (click == 1)).astype(int)
    with gzip.open(path, "wt") as f:
        f.write("\t".join(COLS) + "\n")
        for i in range(n):
            f.write("\t".join(map(str, [ts[i], uid[i], camp[i], conv[i], 5 if conv[i] else -1, 7 if conv[i] else -1, conv[i],
                                        click[i], 0 if click[i] else -1, 1 if click[i] else -1, 0.5, 0.1 if conv[i] else -1,
                                        tsl[i], *cats[i]])) + "\n")
    return click, camp, tsl


def test_schema_switch_changes_columns_in_a_fresh_process():
    code = "from ctr.config import *; print(len(INT_COLS), len(CAT_COLS), RAW_COLUMNS[-1])"
    run = lambda env: subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True,
                                     env={**os.environ, **env}).stdout.split()
    assert run({"CTR_SCHEMA": "criteo_attribution"}) == ["1", "11", "C11"]
    env = {k: v for k, v in os.environ.items() if k != "CTR_SCHEMA"}
    assert subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, env=env).stdout.split() == ["13", "26", "C26"]
    bad = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, env={**os.environ, "CTR_SCHEMA": "nope"})
    assert bad.returncode != 0 and "CTR_SCHEMA" in bad.stderr


def test_converter_maps_columns_and_excludes_outcomes(tmp_path):
    src, dst = str(tmp_path / "a.tsv.gz"), str(tmp_path / "a.txt")
    click, camp, tsl = make_file(src, 3000)
    a = convert(src, dst, chunksize=700)
    lines = [l.split("\t") for l in open(dst).read().splitlines()]
    assert len(lines) == a["rows"] == 3000 and all(len(l) == 13 for l in lines)
    assert [int(l[0]) for l in lines] == click.tolist()                 # label is click
    assert [int(l[2]) for l in lines] == camp.tolist() and [int(l[1]) for l in lines] == tsl.tolist()
    assert a["sorted_by_timestamp"] and not a["truncated_input"] and abs(a["click_rate"] - click.mean()) < 1e-9
    assert {"conversion", "attribution", "click_pos", "click_nb", "cost", "cpo", "timestamp"} <= set(a["excluded_columns"])
    # the audit shows why click_pos is excluded: it is defined exactly when the impression was clicked
    cp = a["leakage_check"]["click_pos"]
    assert cp["click_rate_when_present"] == 1.0 and cp["click_rate_when_absent"] == 0.0


def test_max_rows_and_truncated_download(tmp_path):
    src = str(tmp_path / "a.tsv.gz")
    make_file(src, 6000)
    full = convert(src, str(tmp_path / "full.txt"))
    cap = convert(src, str(tmp_path / "cap.txt"), max_rows=1234, chunksize=500)
    assert cap["rows"] == 1234 and open(tmp_path / "cap.txt").read().count("\n") == 1234
    raw = open(src, "rb").read()
    cut = str(tmp_path / "cut.tsv.gz"); open(cut, "wb").write(raw[: len(raw) // 2])
    c = convert(cut, str(tmp_path / "cut.txt"))
    assert c["truncated_input"] and 0 < c["rows"] < full["rows"]
    got = open(tmp_path / "cut.txt").read().splitlines()
    assert len(got) == c["rows"] and all(len(l.split("\t")) == 13 for l in got)       # no half-written last row
    assert got == open(tmp_path / "full.txt").read().splitlines()[: c["rows"]]       # exact prefix of the full file


def test_converter_rejects_bad_input(tmp_path):
    src = str(tmp_path / "x.tsv.gz")
    with gzip.open(src, "wt") as f:
        f.write("a\tb\n1\t2\n")
    with pytest.raises(ValueError, match="missing expected columns"):
        convert(src, str(tmp_path / "x.txt"))


def test_pipeline_end_to_end_on_attribution_file(tmp_path):
    """Run `python pipeline.py --dataset attribution` for real (Spark, LightGBM, DCN-V2, export, report)."""
    src = str(tmp_path / "attr.tsv.gz")
    make_file(src, 20000)
    env = {**os.environ, "SPARK_LOCAL_DIRS": str(tmp_path / "spark")}
    r = subprocess.run([sys.executable, "pipeline.py", "--dataset", "attribution", "--raw", src, "--max-rows", "15000",
                        "--seeds", "0", "--num-buckets", "2000", "--processed", str(tmp_path / "proc"),
                        "--artifacts", str(tmp_path / "art"), "--reports", str(tmp_path / "rep")],
                       cwd=ROOT, env=env, capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    res = json.load(open(tmp_path / "rep" / "results.json"))
    meta = json.load(open(tmp_path / "proc" / "metadata.json"))
    info = json.load(open(tmp_path / "rep" / "run_info.json"))
    assert meta["total_rows"] == 15000 and info["dataset"] == "attribution" and res["data"] == "real"
    assert 0.5 < res["lightgbm"]["auc"] < 1 and 0.5 < res["dcnv2"]["auc"]["mean"] < 1
    md = open(tmp_path / "rep" / "resume_numbers.md").read()
    assert "Criteo Attribution dataset" in md and "CC BY-NC-SA" in md and "45.8M-row Criteo Kaggle" in md
    assert "click_pos" in md and "Draft resume bullet" in md


def test_attribution_with_missing_file_and_synthetic_flag(tmp_path):
    r = subprocess.run([sys.executable, "pipeline.py", "--dataset", "attribution", "--raw", str(tmp_path / "nope.gz")],
                       cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 2 and "huggingface.co/datasets/criteo" in r.stderr
    r = subprocess.run([sys.executable, "pipeline.py", "--dataset", "attribution", "--synthetic-rows", "100"],
                       cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 2

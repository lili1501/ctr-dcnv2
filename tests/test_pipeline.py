"""End-to-end smoke test on a tiny synthetic file: generator -> Spark -> LightGBM -> DCN-V2."""
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import pandas as pd
import pytest

from ctr.config import CAT_COLS, INT_COLS, RAW_COLUMNS
from ctr.synthetic import write_tsv

N, BUCKETS = 6000, 200


@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    from ctr.spark_prep import run
    d = tmp_path_factory.mktemp("pipe")
    raw, out = str(d / "train.txt"), str(d / "processed")
    write_tsv(raw, N, seed=7)
    meta = run(raw, out, num_buckets=BUCKETS)
    return raw, out, meta


def test_generator_format(prepared):
    raw, _, _ = prepared
    with open(raw) as f:
        lines = f.read().splitlines()
    assert len(lines) == N
    assert all(len(l.split("\t")) == len(RAW_COLUMNS) for l in lines)
    assert {l.split("\t")[0] for l in lines} == {"0", "1"}


def test_spark_splits_are_ordered_complete_and_roughly_80_10_10(prepared):
    raw, out, meta = prepared
    tr, va, te = (pd.read_parquet(os.path.join(out, s)) for s in ("train", "val", "test"))
    assert len(tr) + len(va) + len(te) == N
    assert abs(len(tr) / N - 0.8) < 0.02 and abs(len(va) / N - 0.1) < 0.02
    raw_labels = pd.read_csv(raw, sep="\t", header=None, usecols=[0])[0].to_numpy()
    seq = np.concatenate([tr.label.to_numpy(), va.label.to_numpy(), te.label.to_numpy()])
    assert np.array_equal(seq, raw_labels)           # chronological order preserved, nothing reshuffled


def test_spark_hashing_and_scaling(prepared):
    raw, out, meta = prepared
    tr = pd.read_parquet(os.path.join(out, "train"))
    allp = pd.concat([pd.read_parquet(os.path.join(out, s)) for s in ("train", "val", "test")], ignore_index=True)
    assert allp[CAT_COLS].to_numpy().min() >= 0 and allp[CAT_COLS].to_numpy().max() < BUCKETS
    raw_df = pd.read_csv(raw, sep="\t", header=None, keep_default_na=False, dtype=str)
    assert np.array_equal(allp[CAT_COLS].to_numpy() == 0, (raw_df.iloc[:, 14:] == "").to_numpy())  # 0 <=> missing
    np.testing.assert_allclose(tr[INT_COLS].mean().to_numpy(), 0, atol=1e-4)  # train-only standardisation
    np.testing.assert_allclose(tr[INT_COLS].std().to_numpy(), 1, atol=1e-3)
    assert not np.isnan(allp[INT_COLS].to_numpy()).any()


def test_lightgbm_and_dcnv2_beat_the_base_rate(prepared, tmp_path):
    from ctr.baseline_lgbm import train as train_lgbm
    from ctr.train_dcnv2 import train as train_dcn
    _, out, _ = prepared
    lg = train_lgbm(out, str(tmp_path / "lgbm"))
    dc = train_dcn(out, str(tmp_path / "dcn"), epochs=3, batch_size=256, emb_dim=4, n_cross=2, mlp_units=(16,))
    for r in (lg, dc):
        assert np.isfinite(r["test"]["log_loss"]) and 0.5 < r["test"]["auc"] <= 1.0
    assert dc["val"]["log_loss"] < lg["base_rate_logloss_test"]      # learned something beyond the prior
    assert os.path.exists(tmp_path / "dcn" / "dcnv2.keras")

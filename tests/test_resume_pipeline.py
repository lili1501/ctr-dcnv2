import copy
import json
import os
import sys

import pytest

import pipeline
from ctr import compare, resume_summary as rs

BASE = {
    "data": "real", "n_test": 4_600_000,
    "lightgbm": {"auc": 0.7900, "log_loss": 0.4600, "ece": 0.0040},
    "dcnv2": {"auc": {"mean": 0.7960, "runs": [0.796, 0.7958, 0.7962]}, "log_loss": {"mean": 0.4540, "runs": [0.454] * 3},
              "ece": {"mean": 0.0030, "runs": [0.003] * 3}},
    "deep_only": {"auc": {"mean": 0.7930, "runs": [0.793] * 3}, "log_loss": {"mean": 0.4570, "runs": [0.457] * 3},
                  "ece": {"mean": 0.0035, "runs": [0.0035] * 3}},
    "base_rate_log_loss": 0.56, "logloss_improvement_pct_vs_lgbm": 100 * (0.46 - 0.454) / 0.46,
    "auc_gain_vs_lgbm": 0.006,
    "bootstrap_dcnv2_seed0_minus_lgbm": {"auc_diff_ci95": [0.0050, 0.0070], "logloss_diff_ci95": [-0.007, -0.005]},
}
META = {"num_buckets": 200000, "total_rows": 11_460_154,
        "splits": {"train": {"rows": 9_168_123, "ctr": 0.2562}, "val": {"rows": 1_146_015, "ctr": 0.25},
                   "test": {"rows": 1_146_016, "ctr": 0.2551}}}
SERVING = {"verification": {"rows": 2000, "max_abs_diff": 0.0}}


def make(**over):
    r = copy.deepcopy(BASE)
    r.update(over)
    return r


def test_verdict_boundaries():
    assert rs.verdict([0.001, 0.01]) == "improved" and rs.verdict([-0.01, -0.001]) == "worse"
    assert rs.verdict([-0.001, 0.002]) == "matched" and rs.verdict([0.0, 0.01]) == "matched"


def test_synthetic_results_never_produce_a_bullet():
    for r in (make(data="synthetic"), {k: v for k, v in BASE.items() if k != "data"}):
        out = rs.render(r, META)
        assert out.startswith("# NOT FOR A RESUME") and "Draft resume bullet" not in out and "0.7960" not in out


def test_improved_bullet_has_exact_numbers():
    out = rs.render(make(), META, {"subsample_every": 4, "elapsed_seconds": 7200}, SERVING,
                    {"params": 21_496_944, "best_epoch": 2, "fit_seconds": 3600.4})
    bullet = next(l for l in out.splitlines() if l.startswith("> "))
    for s in ("11.5M", "every 4th row", "0.7960", "0.7900", "+0.0060", "[+0.0050, +0.0070]", "1.30% lower log loss",
              "max abs difference 0"):
        assert s in bullet, (s, bullet)
    assert "improved" in out and "2.00 h" in out and "21,496,944" in out and "+0.0030 vs the same network without cross layers" in out
    assert "latency" in out.lower() and "p95" in out           # the 'what not to claim' warning about latency is present
    assert "p95" not in bullet and "TF Serving" not in bullet  # nothing unmeasured in the bullet itself


def test_matched_and_worse_wording_is_honest():
    tie = make(); tie["dcnv2"]["auc"]["mean"] = 0.7902; tie["bootstrap_dcnv2_seed0_minus_lgbm"]["auc_diff_ci95"] = [-0.0005, 0.0009]
    out = rs.render(tie, META, {"subsample_every": 1}, None, None)
    assert "matched a LightGBM baseline" in out and "not statistically significant" in out and "full ~45.8M-row" in out

    bad = make(logloss_improvement_pct_vs_lgbm=-0.5); bad["dcnv2"]["auc"]["mean"] = 0.7850
    bad["bootstrap_dcnv2_seed0_minus_lgbm"]["auc_diff_ci95"] = [-0.006, -0.004]
    out = rs.render(bad, META, {"subsample_every": 4}, None, None)
    bullet = next(l for l in out.splitlines() if l.startswith("> "))
    assert "do not claim an improvement" in out and "0.5" in out and "higher than LightGBM" in out
    assert "improv" not in bullet.lower() and "outperform" not in bullet.lower()


def test_compare_labels_follow_the_data():
    r = make()
    assert "SYNTHETIC" not in compare.render_md(r, 3, 3) and "real data" in compare.render_md(r, 3, 3)
    syn = compare.render_md(make(data="synthetic"), 3, 3)
    assert "SYNTHETIC DATA" in syn
    assert "SYNTHETIC" in compare.render_md({k: v for k, v in BASE.items() if k != "data"}, 3, 3)   # missing field -> cautious


def test_pipeline_refuses_missing_data(capsys, tmp_path):
    rc = pipeline.main(["--raw", str(tmp_path / "nope.txt")])
    assert rc == 2 and "not found" in capsys.readouterr().err


def test_write_summary_requires_results(tmp_path):
    with pytest.raises(FileNotFoundError):
        rs.write_summary(str(tmp_path), str(tmp_path), str(tmp_path))


def test_spark_defaults_follow_the_allocation_and_respect_user_settings(tmp_path):
    d = pipeline.spark_defaults({}, 64.0, 8, "/repo")
    assert d == {"SPARK_DRIVER_MEMORY": "38g", "SPARK_CORES": "8", "SPARK_LOCAL_DIRS": "/repo/data/spark_tmp"}
    assert pipeline.spark_defaults({}, 2.0, 0, "/r")["SPARK_DRIVER_MEMORY"] == "4g"           # floor, and at least 1 core
    assert pipeline.spark_defaults({}, 2.0, 0, "/r")["SPARK_CORES"] == "1"
    mine = {"SPARK_DRIVER_MEMORY": "10g", "SPARK_CORES": "3", "SPARK_LOCAL_DIRS": "/blue/x"}
    assert pipeline.spark_defaults(mine, 64.0, 8, "/repo") == {}                              # never override the user
    assert "SPARK_LOCAL_DIRS" not in pipeline.spark_defaults({"SPARK_LOCAL_DIR": "/x"}, 64.0, 8, "/r")
    assert "SPARK_DRIVER_MEMORY" not in pipeline.spark_defaults({}, None, 8, "/r")            # unknown memory -> leave Spark's default


def test_available_memory_takes_the_smallest_limit(monkeypatch, tmp_path):
    info = tmp_path / "meminfo"; info.write_text("MemTotal: 999999999 kB\nMemAvailable: %d kB\n" % (200 * 1024 * 1024))   # 200 GiB free
    monkeypatch.setattr(pipeline, "_cgroup_limit_gb", lambda: 16.0)
    env = {"SLURM_MEM_PER_NODE": str(8 * 1024)}
    assert pipeline.available_memory_gb(env, str(info)) == pytest.approx(8.0)          # SLURM allocation is the smallest
    monkeypatch.setattr(pipeline, "_cgroup_limit_gb", lambda: 2.0)
    assert pipeline.available_memory_gb(env, str(info)) == pytest.approx(2.0)          # cgroup limit is smaller still
    monkeypatch.setattr(pipeline, "_cgroup_limit_gb", lambda: None)
    assert pipeline.available_memory_gb({}, str(info)) == pytest.approx(200.0)         # falls back to MemAvailable
    assert pipeline.available_memory_gb({}, str(tmp_path / "missing")) is None

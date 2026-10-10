import numpy as np
import pytest

from ctr.metrics import auc, calibration_table, evaluate, expected_calibration_error, log_loss


def test_log_loss_known_value():
    # y=1,p=0.8 -> -ln(0.8); y=0,p=0.1 -> -ln(0.9)
    got = log_loss([1, 0], [0.8, 0.1])
    assert got == pytest.approx((-np.log(0.8) - np.log(0.9)) / 2)


def test_log_loss_is_finite_for_extreme_probabilities():
    assert np.isfinite(log_loss([1, 0], [0.0, 1.0]))  # clipped, not inf


def test_constant_base_rate_logloss_equals_entropy():
    rng = np.random.default_rng(0)
    y = (rng.random(200_000) < 0.25).astype(float)
    entropy = -(0.25 * np.log(0.25) + 0.75 * np.log(0.75))
    assert log_loss(y, np.full_like(y, y.mean())) == pytest.approx(entropy, abs=2e-3)


def test_auc_perfect_random_and_inverted():
    y = np.array([0, 0, 1, 1])
    assert auc(y, [0.1, 0.2, 0.8, 0.9]) == 1.0
    assert auc(y, [0.9, 0.8, 0.2, 0.1]) == 0.0
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 50_000)
    assert auc(y, rng.random(50_000)) == pytest.approx(0.5, abs=0.01)


def test_ece_small_when_calibrated_large_when_not():
    rng = np.random.default_rng(2)
    p = rng.beta(1, 6, 200_000)                      # skewed like CTR predictions
    y = (rng.random(p.size) < p).astype(float)       # labels drawn FROM p -> calibrated
    assert expected_calibration_error(y, p) < 0.005
    assert expected_calibration_error(y, np.clip(p * 2, 0, 1)) > 0.05   # systematically over-predicting


def test_calibration_table_bins_cover_all_rows_and_are_sorted():
    rng = np.random.default_rng(3)
    p = rng.random(1003)                              # not divisible by bins
    y = (rng.random(1003) < p).astype(float)
    mp, ob, cnt = calibration_table(y, p, n_bins=10)
    assert cnt.sum() == 1003 and len(mp) == 10
    assert np.all(np.diff(mp) >= 0)


def test_evaluate_reports_mean_pred_and_observed_ctr():
    r = evaluate(np.array([1, 0, 0, 0]), np.array([0.5, 0.2, 0.2, 0.1]))
    assert r["observed_ctr"] == 0.25 and r["mean_pred"] == pytest.approx(0.25) and r["n"] == 4

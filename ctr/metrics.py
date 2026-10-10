"""Evaluation metrics for CTR prediction: AUC, log loss, and calibration."""
import numpy as np
from sklearn.metrics import roc_auc_score

_EPS = 1e-7


def log_loss(y, p) -> float:
    y = np.asarray(y, dtype=np.float64)
    p = np.clip(np.asarray(p, dtype=np.float64), _EPS, 1 - _EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def auc(y, p) -> float:
    return float(roc_auc_score(y, p))


def calibration_table(y, p, n_bins: int = 10):
    """Equal-count (quantile) bins of predicted probability.

    Returns arrays (mean_predicted, observed_rate, count) per bin. Quantile bins are used
    because CTR predictions are heavily skewed toward small values; equal-width bins would
    leave most bins empty.
    """
    y, p = np.asarray(y, dtype=np.float64), np.asarray(p, dtype=np.float64)
    order = np.argsort(p, kind="stable")
    chunks = np.array_split(order, n_bins)
    chunks = [c for c in chunks if len(c) > 0]
    mean_pred = np.array([p[c].mean() for c in chunks])
    obs = np.array([y[c].mean() for c in chunks])
    cnt = np.array([len(c) for c in chunks])
    return mean_pred, obs, cnt


def expected_calibration_error(y, p, n_bins: int = 15) -> float:
    """Count-weighted mean |observed - predicted| over equal-count bins."""
    mp, ob, cnt = calibration_table(y, p, n_bins)
    return float(np.sum(cnt * np.abs(ob - mp)) / cnt.sum())


def evaluate(y, p) -> dict:
    """AUC, log loss, ECE, plus the mean prediction vs the observed click rate."""
    return {
        "auc": auc(y, p),
        "log_loss": log_loss(y, p),
        "ece": expected_calibration_error(y, p),
        "mean_pred": float(np.mean(p)),
        "observed_ctr": float(np.mean(y)),
        "n": int(len(y)),
    }

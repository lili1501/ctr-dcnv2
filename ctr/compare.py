"""Compare LightGBM vs DCN-V2 (and the deep-only ablation) on the held-out test split.

Produces reports/results.json, reports/results.md and reports/calibration.png.

* Paired bootstrap over test rows gives 95% CIs for the AUC / log-loss differences.
* If --synthetic-rows is given, the generator's true click probabilities are recomputed
  to report an ORACLE ceiling (best possible AUC / log loss). That only exists for
  synthetic data; with real Criteo, omit the flag.

Usage:
    python -m ctr.compare --processed data/processed --synthetic-rows 300000
"""
import argparse
import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ctr.data import load_split
from ctr.metrics import auc, calibration_table, evaluate, log_loss
from ctr.synthetic import generate

BLUE, ORANGE, GRAY, INK, MUTED = "#2a78d6", "#eb6834", "#8a8984", "#0b0b0b", "#52514e"  # validated pair (light)


def oracle_probs(total_rows: int, n_test: int, seed: int = 0, chunk: int = 100_000):
    """Re-create the generator's true p for the LAST n_test rows (the test split), plus labels."""
    ps, ys, part, done = [], [], 0, 0
    while done < total_rows:
        m = min(chunk, total_rows - done)
        d = generate(m, seed=seed * 1_000_003 + part)
        ps.append(d["true_p"]); ys.append(d["y"]); done += m; part += 1
    return np.concatenate(ps)[-n_test:], np.concatenate(ys)[-n_test:]


def paired_bootstrap(y, p_a, p_b, n_boot=500, seed=0):
    """Return mean and 95% CI of (metric(a) - metric(b)) for AUC and log loss."""
    rng = np.random.default_rng(seed)
    d_auc, d_ll = [], []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        d_auc.append(auc(y[i], p_a[i]) - auc(y[i], p_b[i]))
        d_ll.append(log_loss(y[i], p_a[i]) - log_loss(y[i], p_b[i]))
    ci = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    return {"auc_diff_mean": float(np.mean(d_auc)), "auc_diff_ci95": ci(d_auc),
            "logloss_diff_mean": float(np.mean(d_ll)), "logloss_diff_ci95": ci(d_ll)}


def plot_calibration(y, preds: dict, path: str, n_bins: int = 10):
    fig, ax = plt.subplots(figsize=(6.2, 5.2), dpi=150)
    ax.plot([0, 1], [0, 1], ls="--", lw=1, color=GRAY, zorder=1)
    ax.text(0.86, 0.80, "perfect calibration", color=MUTED, fontsize=8.5, rotation=45, ha="center", va="center")
    spec = {"LightGBM": (BLUE, "o"), "DCN-V2": (ORANGE, "s")}
    for name, p in preds.items():
        mp, ob, _ = calibration_table(y, p, n_bins)
        color, marker = spec[name]
        ax.plot(mp, ob, color=color, lw=2, marker=marker, ms=6, mec="#fcfcfb", mew=1.5, label=name, zorder=3)
        # labels sit in empty space beside the line ends, offset apart so they never collide
        xy_text = (24, -28) if name == "LightGBM" else (10, 10)
        ax.annotate(name, (mp[-1], ob[-1]), xytext=xy_text, textcoords="offset points", color=INK, fontsize=9,
                    arrowprops=dict(arrowstyle="-", color="#c9c8c3", lw=0.8, shrinkA=0, shrinkB=3))
    ax.set_xlabel("Mean predicted click probability (equal-count bins)", color=MUTED, fontsize=9)
    ax.set_ylabel("Observed click rate", color=MUTED, fontsize=9)
    ax.set_title("Calibration on held-out test split (synthetic data)", loc="left", color=INK, fontsize=11)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.grid(True, color="#e6e5e1", lw=0.6); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c9c8c3")
    ax.tick_params(colors=MUTED, labelsize=8.5)
    ax.legend(frameon=False, loc="upper left", fontsize=9, labelcolor=INK)
    fig.patch.set_facecolor("#fcfcfb"); ax.set_facecolor("#fcfcfb")
    fig.tight_layout()
    fig.savefig(path, facecolor=fig.get_facecolor())
    plt.close(fig)


def run(processed: str, artifacts: str, reports: str, synthetic_rows: int | None) -> dict:
    y = load_split(processed, "test")["y"].astype(np.float64)
    lgbm = np.load(os.path.join(artifacts, "lgbm", "pred_test.npy"))
    dcn_dirs = sorted(glob.glob(os.path.join(artifacts, "dcnv2*")))
    deep_dirs = sorted(glob.glob(os.path.join(artifacts, "deep_only*")))
    dcn = [np.load(os.path.join(d, "pred_test.npy")) for d in dcn_dirs]
    deep = [np.load(os.path.join(d, "pred_test.npy")) for d in deep_dirs]

    def summarize(preds):
        ms = [evaluate(y, p) for p in preds]
        return {k: {"mean": float(np.mean([m[k] for m in ms])), "std": float(np.std([m[k] for m in ms])),
                    "runs": [m[k] for m in ms]} for k in ("auc", "log_loss", "ece")}

    res = {"n_test": int(len(y)), "lightgbm": evaluate(y, lgbm),
           "dcnv2": summarize(dcn), "deep_only": summarize(deep) if deep else None,
           "base_rate_log_loss": log_loss(y, np.full_like(y, y.mean()))}
    ll_l, ll_d = res["lightgbm"]["log_loss"], res["dcnv2"]["log_loss"]["mean"]
    res["logloss_improvement_pct_vs_lgbm"] = 100 * (ll_l - ll_d) / ll_l
    res["auc_gain_vs_lgbm"] = res["dcnv2"]["auc"]["mean"] - res["lightgbm"]["auc"]
    res["bootstrap_dcnv2_seed0_minus_lgbm"] = paired_bootstrap(y, dcn[0], lgbm)

    if synthetic_rows:
        p_true, y_true = oracle_probs(synthetic_rows, len(y))
        assert np.array_equal(y_true, y), "oracle labels do not align with the test split"
        res["oracle"] = {"auc": auc(y, p_true), "log_loss": log_loss(y, p_true)}

    os.makedirs(reports, exist_ok=True)
    plot_calibration(y, {"LightGBM": lgbm, "DCN-V2": dcn[0]}, os.path.join(reports, "calibration.png"))
    with open(os.path.join(reports, "results.json"), "w") as f:
        json.dump(res, f, indent=2)
    with open(os.path.join(reports, "results.md"), "w") as f:
        f.write(render_md(res, len(dcn), len(deep)))
    return res


def render_md(r: dict, n_dcn: int, n_deep: int) -> str:
    b = r["bootstrap_dcnv2_seed0_minus_lgbm"]
    rows = [("Constant base rate", None, r["base_rate_log_loss"], None),
            ("LightGBM", r["lightgbm"]["auc"], r["lightgbm"]["log_loss"], r["lightgbm"]["ece"])]
    if r.get("deep_only"):
        d = r["deep_only"]
        rows.append((f"Deep-only (no cross layers), {n_deep} seed(s)", d["auc"]["mean"], d["log_loss"]["mean"], d["ece"]["mean"]))
    d = r["dcnv2"]
    rows.append((f"DCN-V2, {n_dcn} seed(s)", d["auc"]["mean"], d["log_loss"]["mean"], d["ece"]["mean"]))
    if "oracle" in r:
        rows.append(("Oracle (true probabilities)", r["oracle"]["auc"], r["oracle"]["log_loss"], None))
    f = lambda v, n=4: "-" if v is None else f"{v:.{n}f}"
    lines = ["# Results (SYNTHETIC DATA - validates the code, not a research claim)", "",
             f"Held-out test split, {r['n_test']:,} rows. Replace with numbers from a real Criteo run before citing.", "",
             "| Model | AUC | Log loss | ECE |", "|---|---|---|---|"]
    lines += [f"| {n} | {f(a)} | {f(l)} | {f(e)} |" for n, a, l, e in rows]
    lines += ["", f"* DCN-V2 vs LightGBM: log loss {r['logloss_improvement_pct_vs_lgbm']:+.2f}% better, "
                  f"AUC {r['auc_gain_vs_lgbm']:+.4f}.",
              f"* Paired bootstrap (DCN-V2 seed 0 minus LightGBM), 95% CI: AUC diff {b['auc_diff_ci95'][0]:+.4f} to "
              f"{b['auc_diff_ci95'][1]:+.4f}; log-loss diff {b['logloss_diff_ci95'][0]:+.4f} to {b['logloss_diff_ci95'][1]:+.4f}.",
              "* ECE = expected calibration error over 15 equal-count bins (lower is better).", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", default="data/processed")
    ap.add_argument("--artifacts", default="artifacts")
    ap.add_argument("--reports", default="reports")
    ap.add_argument("--synthetic-rows", type=int, default=None, help="total rows of the synthetic file (enables oracle)")
    a = ap.parse_args()
    out = run(a.processed, a.artifacts, a.reports, a.synthetic_rows)
    print(open(os.path.join(a.reports, "results.md")).read())

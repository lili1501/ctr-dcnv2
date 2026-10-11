"""Turn a finished run into resume-ready facts, using ONLY numbers the run measured.

Writes reports/resume_numbers.md: the measured table, the facts you can state, and a draft bullet whose wording
follows what the results actually show (improved / matched / worse than LightGBM). It refuses to produce a
bullet for synthetic data.

    python -m ctr.resume_summary --reports reports --processed data/processed --artifacts artifacts
"""
from __future__ import annotations

import argparse
import json
import os

FULL_CRITEO_ROWS = "45.8M"      # rows in the Criteo Kaggle train.txt (Criteo_x4 dataset card: 45,840,617)
FULL_ATTRIBUTION_ROWS = "16.5M"  # rows in criteo_attribution_dataset.tsv.gz (16,468,027 per the dataset card)


def _load(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def verdict(ci: list[float]) -> str:
    lo, hi = ci
    return "improved" if lo > 0 else "worse" if hi < 0 else "matched"


def millions(n: int) -> str:
    return f"{n / 1e6:.1f}M"


def render(res: dict, meta: dict, run_info: dict | None = None, serving: dict | None = None,
           dcn_metrics: dict | None = None) -> str:
    if res.get("data", "synthetic") == "synthetic":
        return ("# NOT FOR A RESUME\n\nThis run used synthetic data (or an old results file with no `data` field). "
                "Run `python pipeline.py` on the real Criteo file to get citable numbers.\n")

    L, D = res["lightgbm"], res["dcnv2"]
    auc_l, auc_d = L["auc"], D["auc"]["mean"]
    ll_l, ll_d = L["log_loss"], D["log_loss"]["mean"]
    boot = res["bootstrap_dcnv2_seed0_minus_lgbm"]
    lo, hi = boot["auc_diff_ci95"]
    v = verdict(boot["auc_diff_ci95"])
    pct = res["logloss_improvement_pct_vs_lgbm"]
    n_seeds = len(D["auc"]["runs"])
    rows = meta["total_rows"]
    sub = (run_info or {}).get("subsample_every")
    attribution = (run_info or {}).get("dataset") == "attribution"
    full_rows, what = ((FULL_ATTRIBUTION_ROWS, "Criteo Attribution dataset") if attribution
                       else (FULL_CRITEO_ROWS, "Criteo training file"))
    sub_clause = (f" (every {sub}th row of the ~{full_rows}-row {what}, in time order)"
                  if sub and int(sub) > 1 else f" (the full ~{full_rows}-row {what})")
    split = meta["splits"]

    out = ["# Resume numbers (measured on this run)", "",
           "## Run details",
           f"- Rows processed: {rows:,} ({millions(rows)}){sub_clause.replace(' (', ' - ', 1).rstrip(')')}",
           f"- Chronological split: train {split['train']['rows']:,} / val {split['val']['rows']:,} / test {split['test']['rows']:,} "
           f"(click rate {split['test']['ctr']:.2%} on test)",
           f"- Hash buckets per categorical feature: {meta['num_buckets']:,}",
           f"- DCN-V2 runs: {n_seeds} seed(s); test rows scored: {res['n_test']:,}"]
    if dcn_metrics:
        out.append(f"- DCN-V2: {dcn_metrics['params']:,} parameters, best epoch {dcn_metrics['best_epoch']}, "
                   f"fit time (seed 0) {dcn_metrics['fit_seconds']:.0f}s")
    if run_info and run_info.get("elapsed_seconds"):
        out.append(f"- Whole pipeline wall time: {run_info['elapsed_seconds'] / 3600:.2f} h")

    out += ["", "## Results (held-out test split)", "",
            "| Model | AUC | Log loss | ECE |", "|---|---|---|---|",
            f"| LightGBM | {auc_l:.4f} | {ll_l:.4f} | {L['ece']:.4f} |"]
    if res.get("deep_only"):
        d = res["deep_only"]
        out.append(f"| Deep-only (no cross layers), {len(d['auc']['runs'])} seed(s) | {d['auc']['mean']:.4f} | "
                   f"{d['log_loss']['mean']:.4f} | {d['ece']['mean']:.4f} |")
    out.append(f"| DCN-V2, {n_seeds} seed(s) | {auc_d:.4f} | {ll_d:.4f} | {D['ece']['mean']:.4f} |")
    out += ["", f"- AUC difference DCN-V2 - LightGBM: {auc_d - auc_l:+.4f}; paired-bootstrap 95% CI on seed 0: [{lo:+.4f}, {hi:+.4f}] -> **{v}**",
            f"- Log loss: {abs(pct):.2f}% {'lower' if pct > 0 else 'higher'} than LightGBM"]
    if res.get("deep_only"):
        gain = auc_d - res["deep_only"]["auc"]["mean"]
        out.append(f"- Cross-layer ablation: DCN-V2 AUC {gain:+.4f} vs the same network without cross layers")

    verified = ""
    if serving and "verification" in serving:
        x = serving["verification"]
        verified = (f"; SavedModel export checked against Keras on {x['rows']:,} rows (max abs difference {x['max_abs_diff']:g})")
    base = (f"Built a PySpark preprocessing pipeline and a DCN-V2 click-through-rate model (TensorFlow/Keras) on "
            f"{millions(rows)} Criteo ad impressions{sub_clause}, predicting clicks, with a chronological train/validation/test split")
    out += ["", "## Draft resume bullet (edit freely, but keep every number tied to the table above)", ""]
    if v == "improved":
        out.append(f"> {base}: test AUC {auc_d:.4f} vs {auc_l:.4f} for a LightGBM baseline "
                   f"(+{auc_d - auc_l:.4f}, paired-bootstrap 95% CI [{lo:+.4f}, {hi:+.4f}]) and "
                   f"{pct:.2f}% lower log loss{verified}.")
    elif v == "matched":
        out.append(f"> {base}: matched a LightGBM baseline (test AUC {auc_d:.4f} vs {auc_l:.4f}; the difference was not "
                   f"statistically significant, 95% CI [{lo:+.4f}, {hi:+.4f}]), with calibration checked (ECE {D['ece']['mean']:.4f})"
                   f"{verified}.")
    else:
        out.append(f"> {base}, benchmarked against a LightGBM baseline (test AUC {auc_d:.4f} vs {auc_l:.4f}) with "
                   f"calibration and a cross-layer ablation{verified}.")
        out += ["", "**DCN-V2 did not beat LightGBM on this run, so do not claim an improvement.** Things to try before "
                "re-running: `--emb-l2` on the embeddings, a smaller `--emb-dim`, more hash buckets, or fewer epochs."]
    if attribution:
        out += ["", "## Dataset caveats (Criteo Attribution Modeling, click prediction)",
                "- Licence CC BY-NC-SA 4.0: fine for a portfolio/research project, not for commercial use. Cite Diemert et al., "
                "\"Attribution Modeling Increases Efficiency of Bidding in Display Advertising\", AdKDD 2017 (arXiv:1707.06409) "
                "in the README.",
                "- Features used: time_since_last_click, campaign, uid and cat1-cat9. Conversion, attribution, click_pos, click_nb, "
                "cost and cpo are excluded because they are outcomes or post-impression values (see `data/raw/criteo_attribution.txt.audit.json`).",
                "- `uid` lets the model memorise users seen in the training period; the split is chronological, but say so if asked.",
                "- This is not the 45.8M-row Criteo Kaggle benchmark, so do not compare the numbers with published Kaggle results."]
    out += ["", "## What not to claim",
            "- Serving latency: the in-process number in `serving_check.json` is not TF Serving latency. Only quote p95 after "
            "running `python -m ctr.benchmark_serving` against a live TF Serving container.",
            "- Anything about hyperparameter tuning: none was done.", ""]
    return "\n".join(out)


def write_summary(reports: str, processed: str, artifacts: str) -> str:
    res = _load(os.path.join(reports, "results.json"))
    if res is None:
        raise FileNotFoundError(f"{reports}/results.json not found; the pipeline did not reach the compare stage")
    meta = _load(os.path.join(processed, "metadata.json"))
    text = render(res, meta, _load(os.path.join(reports, "run_info.json")), _load(os.path.join(reports, "serving_check.json")),
                  _load(os.path.join(artifacts, "dcnv2_s0", "metrics.json")))
    path = os.path.join(reports, "resume_numbers.md")
    with open(path, "w") as f:
        f.write(text)
    return path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports", default="reports")
    ap.add_argument("--processed", default="data/processed")
    ap.add_argument("--artifacts", default="artifacts")
    a = ap.parse_args()
    print(open(write_summary(a.reports, a.processed, a.artifacts)).read())

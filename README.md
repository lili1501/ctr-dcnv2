# Ad Click-Through Rate Prediction with DCN-V2

Deep ranking for online advertising: a **DCN-V2 (Deep & Cross Network)** in TensorFlow/Keras, compared against a
**LightGBM** baseline on Criteo-format click logs. Raw logs are processed with **PySpark**; the model is exported as a
SavedModel for **TF Serving**.

> **Status of the numbers.** The Criteo dataset could not be downloaded in the environment this was built in, so every
> result below is from a **synthetic dataset in Criteo's exact format** (`ctr/synthetic.py`) with planted feature
> interactions. Those numbers validate that the code works; they are **not** results to cite. Run the pipeline on real
> Criteo (see below) to get the numbers for a resume or write-up.

## Pipeline

```
raw Criteo TSV ──PySpark──▶ train/val/test Parquet ──┬─▶ LightGBM baseline ──┐
 (label, I1-I13, C1-C26)    hash cats, log+scale ints │                       ├─▶ compare (AUC, log loss,
                            positional (time) split   └─▶ DCN-V2 (Keras) ─────┘   bootstrap CI, calibration)
                                                              └─▶ SavedModel ─▶ TF Serving ─▶ latency benchmark
```

| Stage | File | What it does |
|---|---|---|
| Data | `ctr/synthetic.py` | Criteo-format generator (label, 13 ints, 26 hashed cats, missing values) |
| Preprocess | `ctr/spark_prep.py` | PySpark: signed-log1p + standardise ints (train-only stats), murmur3-hash cats into 50k buckets (0 = missing), 80/10/10 split by row order |
| Baseline | `ctr/baseline_lgbm.py` | LightGBM with native categoricals, early stopping on validation log loss |
| Model | `ctr/dcnv2.py` | Shared embedding table → cross tower (`x_{l+1} = x0 ⊙ (W x_l + b) + x_l`) ‖ deep MLP; full-rank or low-rank cross; parallel or stacked |
| Train | `ctr/train_dcnv2.py` | Adam, early stopping on validation log loss; test split scored once |
| Metrics | `ctr/metrics.py` | AUC, log loss, calibration table, expected calibration error |
| Report | `ctr/compare.py` | Seeds, deep-only ablation, paired bootstrap CIs, calibration plot (+ oracle bound on synthetic data) |
| Serving | `ctr/export_serving.py`, `ctr/benchmark_serving.py`, `scripts/serve_tfserving.sh` | SavedModel export + parity check; REST latency client; Docker launch script |

## Quick start

Needs Python 3.13 and a JDK (17 or 21) for PySpark. Tested on CPU only.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m pytest tests -q                          # 50 tests
SYNTHETIC_ROWS=300000 bash scripts/run_pipeline.sh # synthetic end-to-end run (roughly 10 min on 2 CPU cores; estimated from per-stage timings, the full 300k script run was not timed)
```

### One command: `python pipeline.py`

After downloading `data/raw/train.txt` (below), run everything from one Python entry point. No sbatch is needed, but use a
session with enough memory (16 GB or more; Spark gets about 60% of what the session has, and all its cores):

```bash
python pipeline.py --subsample-every 40 --seeds "0"   # quick check on ~1.1M rows; note fit_seconds to estimate the full run
python pipeline.py                                    # default: every 4th row (~11.5M rows), seeds 0 1 2
nohup python pipeline.py > logs/pipeline.log 2>&1 &   # long runs: survives a closed terminal; follow with tail -f logs/pipeline.log
```

It wraps `scripts/run_pipeline.sh` and writes, in `reports/`: `results.md` / `results.json` / `calibration.png` (the comparison),
`run_info.json` (what was run, wall time) and `resume_numbers.md` (the measured facts and a draft bullet built only from the
results; it refuses to write one for synthetic data and words the claim as improved / matched / worse according to the
bootstrap CI). Spark shuffle files go to `data/spark_tmp` unless `SPARK_LOCAL_DIRS` is set. It stops with a clear message if
`train.txt` is missing. Training on CPU is slow for 11M rows with an 84M-parameter model; use a GPU session with
`pip install "tensorflow[and-cuda]==2.21.0"` if you have one (untested here), and time the small run first.

### Running on real Criteo

1. **Download** the Criteo Kaggle Display Advertising Challenge data. Criteo AI Lab's
   [resources page](https://ailab.criteo.com/ressources/) lists a direct link
   (`https://go.criteo.net/criteo-research-kaggle-display-advertising-challenge-dataset.tar.gz`); the data is also
   described on the [Kaggle competition page](https://www.kaggle.com/c/criteo-display-ad-challenge/). Read Criteo's terms
   of use on that page before using it. The labelled file is `train.txt` (13 integer + 26 hashed categorical features,
   ~7 days of traffic; 45,840,617 rows per the [Criteo_x4 dataset card](https://www.huggingface.co/datasets/reczoo/Criteo_x4/blob/main/README.md)).
   `test.txt` has no labels and is not used. Make sure you have tens of GB of free disk (Spark spills to disk).
   ```bash
   mkdir -p data/raw && cd data/raw
   curl -L -o criteo.tar.gz https://go.criteo.net/criteo-research-kaggle-display-advertising-challenge-dataset.tar.gz
   tar -xzf criteo.tar.gz && cd ../..
   wc -l data/raw/train.txt                 # sanity check: should be about 45.8M lines
   ```
2. **Start with a subsample** (every 4th row keeps chronological order and covers all days, giving ~11.5M rows), then scale up:
   ```bash
   SUBSAMPLE_EVERY=4 NUM_BUCKETS=200000 SPARK_DRIVER_MEMORY=16g SPARK_CORES='*' \
     RAW=data/raw/train.txt bash scripts/run_pipeline.sh      # leave SYNTHETIC_ROWS unset
   ```
   For the full file drop `SUBSAMPLE_EVERY` (needs far more RAM, see scaling notes). The `[X]M impressions` figure on the
   resume is the `total_rows` you actually processed, so state the subsample honestly.
3. Serve and benchmark:
   ```bash
   bash scripts/serve_tfserving.sh                      # needs Docker + Docker Hub access
   python -m ctr.benchmark_serving --url http://localhost:8501/v1/models/ctr:predict --batch-size 1 --requests 2000
   ```

### Smaller option: the Criteo Attribution dataset (click prediction)
If the Kaggle file is unavailable, use [criteo/criteo-attribution-dataset](https://huggingface.co/datasets/criteo/criteo-attribution-dataset)
(16.5M impressions, time-sorted, `criteo_attribution_dataset.tsv.gz`, 623 MB). A prefix of the file is enough: a byte-range
download is cut mid-stream, and the converter reads up to the cut and drops the incomplete last row.
```bash
curl -L -r 0-199999999 -o data/raw/attr_part.tsv.gz \
  https://huggingface.co/datasets/criteo/criteo-attribution-dataset/resolve/main/criteo_attribution_dataset.tsv.gz   # first ~200 MB
python pipeline.py --dataset attribution --raw data/raw/attr_part.tsv.gz --max-rows 3000000 --seeds "0 1 2"
```
Label = `click`; features = `time_since_last_click`, `campaign`, `uid`, `cat1`-`cat9`. Excluded as outcomes or post-impression
values: conversion fields, `attribution`, `click_pos`, `click_nb`, `cost`, `cpo` (the audit in
`data/raw/criteo_attribution.txt.audit.json` shows e.g. that `click_pos` is defined exactly when the impression was clicked).
A prefix covers only the earliest part of the timeline, so the chronological split is within that window; say so when citing numbers.
License CC BY-NC-SA 4.0 (non-commercial); cite Diemert et al., AdKDD 2017 (arXiv:1707.06409). Tested here on a generated file in
the real format (tests/test_attribution.py); never run on the real download.

### Running on UF HiPerGator (SLURM)

Do not run the pipeline on a login node; UF's docs reserve login nodes for small quick tests. Submit a batch job instead.

```bash
ssh <gatorlink>@hpg.rc.ufl.edu
cd /blue/<your_group>/<your_user>                     # work on /blue, not /home (40 GB quota)
git clone https://github.com/lili1501/ctr-dcnv2.git && cd ctr-dcnv2
module spider java; module spider python             # find exact module names, then load them
module load <java 17 or 21> <python>
python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
# download the data (section above) into data/raw/, e.g. inside a short job or via Globus/scp from your laptop
sbatch scripts/hipergator.sbatch                      # edit the module lines and resources in the file first
squeue -u $USER; tail -f logs/ctr-dcnv2_<jobid>.out   # monitor
```

`scripts/hipergator.sbatch` defaults to a 12 h, 8-core, 64 GB job on a quarter of the rows with one seed. Its logic was tested
here with SLURM faked; it has **not** been run on HiPerGator. Whether compute nodes can reach the internet is unverified,
so download the data from a login/dev session or transfer it with Globus/scp if `curl` fails inside a job.

**Scaling notes for full Criteo (not tested at this scale).** `ctr/data.py` loads a whole split into RAM, which will not
fit for ~36M training rows: switch training to streamed Parquet batches (e.g. `tf.data` over Parquet shards). Raise
`--num-buckets` (50k is sized for the small synthetic set; hash collisions grow with cardinality) and set
`SPARK_DRIVER_MEMORY` / `SPARK_CORES` for Spark. Only the 300k-row synthetic run was ever executed; the real-data settings above are untested defaults. A GPU is strongly recommended for DCN-V2 training.

## Results on SYNTHETIC data (code validation only)

Held-out test split, 30,037 rows; 300k rows total; the best epoch is epoch 1 for every neural run.

| Model | AUC | Log loss | ECE |
|---|---|---|---|
| Constant base rate | - | 0.5629 | - |
| LightGBM | 0.7768 | 0.4636 | 0.0108 |
| Deep-only (no cross layers), 3 seeds | 0.7764 | 0.4638 | 0.0107 |
| DCN-V2, 3 seeds | 0.7812 | 0.4597 | 0.0098 |
| Oracle (true generating probabilities) | 0.8040 | 0.4399 | - |

DCN-V2 vs LightGBM: log loss 0.84% lower, AUC +0.0044 (paired bootstrap 95% CI for AUC diff: +0.0028 to +0.0064).
Removing the cross layers erases the gain, which is consistent with the cross-feature interactions planted in the generator.
Both models are well calibrated (ECE ~0.01); see `reports/calibration.png`.

## What was verified, and what was not

**Verified by running it** (50 passing tests plus the runs above):
- Generator output matches the Criteo schema (40 fields/line, 25.07% CTR, 10% / 5% missing rates).
- Spark output checked against the raw file: row order and labels preserved; hash indices within range; index 0 exactly where the raw value was empty; integer transform matches an independent recomputation; train-only standardisation (mean ≈ 0, std = 1).
- Metrics against known answers; the cross layer against a NumPy reference (full-rank, low-rank, identity, degree-3 terms); model save/reload roundtrip.
- LightGBM and DCN-V2 train and evaluate end to end; results stable across 3 seeds; the deep-only ablation; bootstrap CIs.
- Exported SavedModel reproduces the Keras predictions exactly (max abs diff 0.0 on 2,000 test rows); signature is `dense` (float32 [N,13]) + `cats` (int32 [N,26]) → `output_0` (float32 [N,1]).
- The latency client and JSON payload format work against a **local stand-in** that mimics TF Serving's REST API.
- `scripts/run_pipeline.sh` runs cleanly end to end on a small synthetic configuration.

**Not verified:**
- **Real Criteo data.** Never run on it. All `[X]` values for impressions, AUC and log loss are still to be produced.
- **The real TF Serving binary and its latency.** Docker registries were unreachable, so the server was never started. `scripts/serve_tfserving.sh` and the REST path against a real server are untested. The in-process SavedModel latency in `reports/serving_check.json` (p95 ≈ 0.8 ms at batch 1) excludes network, JSON and the server and must not be quoted as TF Serving p95.
- gRPC serving, GPU training, and behaviour at 45M-row scale.
- Hyperparameters are untuned defaults (no search was run).

## Mapping to the resume bullets

| Bullet placeholder | Where the number comes from |
|---|---|
| `[X]M ad impressions` | `total_rows` printed by `ctr.spark_prep` (and `data/processed/metadata.json`) on real Criteo |
| `AUC to [X]` | `reports/results.md`, DCN-V2 row (real run) |
| `log loss by [X]%` | `reports/results.json` → `logloss_improvement_pct_vs_lgbm` (real run) |
| `[X] ms p95` | `ctr.benchmark_serving` output against a running TF Serving instance |

Wording note: Criteo's features are anonymised, so "ad and user features" is not literally accurate. Prefer
"anonymised high-cardinality categorical features" unless you can say which columns are which.

## Known limitations and next steps

- High-capacity embeddings overfit after one epoch (train AUC 0.89 vs val 0.73 by epoch 3). Next: embedding L2 (`--emb-l2`), smaller `emb_dim`, or frequency-based vocabulary thresholds instead of pure hashing.
- Hash collisions: e.g. column C3 had 21,022 distinct tokens mapped to 17,143 buckets (about 18% of tokens collide at 50k buckets).
- Missing-value indicators are not added; missing ints become 0 before scaling.
- The split is positional (Criteo's file is chronological), with cut points from approximate quantiles, so split fractions are within ~0.1% of 80/10/10.

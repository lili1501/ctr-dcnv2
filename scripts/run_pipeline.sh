#!/usr/bin/env bash
# Run the whole pipeline: (optional synthetic data) -> Spark -> LightGBM -> DCN-V2 (+ ablation) -> export -> report.
#
#   Synthetic smoke run :  SYNTHETIC_ROWS=300000 bash scripts/run_pipeline.sh
#   Real Criteo         :  RAW=/path/to/criteo/train.txt bash scripts/run_pipeline.sh
#
# Optional knobs (all unset by default):
#   SUBSAMPLE_EVERY=4   keep every 4th row (keeps chronological order, covers all days) -> ~11.5M rows of Criteo
#   NUM_BUCKETS=200000  hash buckets per categorical feature (default 50000; raise for real Criteo)
#   SPARK_DRIVER_MEMORY=24g SPARK_CORES='*'   Spark resources (defaults 4g / 2 cores)
set -euo pipefail
cd "$(dirname "$0")/.."

RAW=${RAW:-data/raw/train.txt}
PROCESSED=${PROCESSED:-data/processed}
ART=${ART:-artifacts}
REPORTS=${REPORTS:-reports}
SEEDS=${SEEDS:-"0 1 2"}
SYNTHETIC_ROWS=${SYNTHETIC_ROWS:-}   # leave empty when RAW is real Criteo data
SUBSAMPLE_EVERY=${SUBSAMPLE_EVERY:-}
NUM_BUCKETS=${NUM_BUCKETS:-}

if [[ -n "$SYNTHETIC_ROWS" ]]; then
  python -m ctr.synthetic --rows "$SYNTHETIC_ROWS" --out "$RAW"
fi

if [[ -n "$SUBSAMPLE_EVERY" ]]; then
  SUB="${RAW%.txt}_every${SUBSAMPLE_EVERY}.txt"
  awk -v k="$SUBSAMPLE_EVERY" 'NR % k == 0' "$RAW" > "$SUB"
  echo "subsampled every ${SUBSAMPLE_EVERY}th row -> $SUB ($(wc -l < "$SUB") rows)"
  RAW="$SUB"
fi

python -m ctr.spark_prep --input "$RAW" --output "$PROCESSED" ${NUM_BUCKETS:+--num-buckets "$NUM_BUCKETS"}
python -m ctr.baseline_lgbm --processed "$PROCESSED" --out "$ART/lgbm"

for s in $SEEDS; do
  python -m ctr.train_dcnv2 --processed "$PROCESSED" --out "$ART/dcnv2_s$s" --seed "$s"
  python -m ctr.train_dcnv2 --processed "$PROCESSED" --out "$ART/deep_only_s$s" --seed "$s" --n-cross 0
done

FIRST=${SEEDS%% *}
python -m ctr.export_serving --model "$ART/dcnv2_s$FIRST/dcnv2.keras" --out "$ART/serving/ctr" \
  --processed "$PROCESSED" --reports "$REPORTS"
# The oracle bound only exists for unsubsampled synthetic data (it recomputes the generator's true probabilities).
ORACLE=()
if [[ -n "$SYNTHETIC_ROWS" && -z "$SUBSAMPLE_EVERY" ]]; then ORACLE=(--synthetic-rows "$SYNTHETIC_ROWS"); fi
python -m ctr.compare --processed "$PROCESSED" --artifacts "$ART" --reports "$REPORTS" "${ORACLE[@]}"

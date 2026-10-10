#!/usr/bin/env bash
# Run the whole pipeline: (optional synthetic data) -> Spark -> LightGBM -> DCN-V2 (+ ablation) -> export -> report.
#
#   Synthetic smoke run :  SYNTHETIC_ROWS=300000 bash scripts/run_pipeline.sh
#   Real Criteo         :  RAW=/path/to/criteo/train.txt bash scripts/run_pipeline.sh
set -euo pipefail
cd "$(dirname "$0")/.."

RAW=${RAW:-data/raw/train.txt}
PROCESSED=${PROCESSED:-data/processed}
ART=${ART:-artifacts}
REPORTS=${REPORTS:-reports}
SEEDS=${SEEDS:-"0 1 2"}
SYNTHETIC_ROWS=${SYNTHETIC_ROWS:-}   # leave empty when RAW is real Criteo data

if [[ -n "$SYNTHETIC_ROWS" ]]; then
  python -m ctr.synthetic --rows "$SYNTHETIC_ROWS" --out "$RAW"
fi

python -m ctr.spark_prep --input "$RAW" --output "$PROCESSED"
python -m ctr.baseline_lgbm --processed "$PROCESSED" --out "$ART/lgbm"

for s in $SEEDS; do
  python -m ctr.train_dcnv2 --processed "$PROCESSED" --out "$ART/dcnv2_s$s" --seed "$s"
  python -m ctr.train_dcnv2 --processed "$PROCESSED" --out "$ART/deep_only_s$s" --seed "$s" --n-cross 0
done

FIRST=${SEEDS%% *}
python -m ctr.export_serving --model "$ART/dcnv2_s$FIRST/dcnv2.keras" --out "$ART/serving/ctr" \
  --processed "$PROCESSED" --reports "$REPORTS"
python -m ctr.compare --processed "$PROCESSED" --artifacts "$ART" --reports "$REPORTS" \
  ${SYNTHETIC_ROWS:+--synthetic-rows "$SYNTHETIC_ROWS"}

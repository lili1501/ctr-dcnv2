#!/bin/bash
# SLURM job script for UF HiPerGator: runs the whole CTR pipeline (Spark -> LightGBM -> DCN-V2 -> report).
#
# Submit from the repo directory, which must be on /blue:   sbatch scripts/hipergator.sbatch
# Check:  squeue -u $USER      Logs: logs/<jobname>_<jobid>.out      Cancel: scancel <jobid>
#
# Edit before submitting:
#   * the two `module load` lines (find exact names with `module spider java` and `module spider python`)
#   * --time / --mem / --cpus-per-task if needed. --mem MUST be set (UF's default is only 600 MB).
#   * add --account/--qos only if you use a secondary group's allocation.
#
#SBATCH --job-name=ctr-dcnv2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64gb
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
##SBATCH --account=YOUR_GROUP
##SBATCH --qos=YOUR_GROUP
## Optional GPU (UNTESTED here; needs a GPU partition name from UF docs and `pip install tensorflow[and-cuda]`):
##SBATCH --partition=<gpu-partition>
##SBATCH --gres=gpu:1

set -eo pipefail            # no -u: environment-module scripts reference unset variables
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs

# ---- environment (EDIT the module names) ------------------------------------------------------
if command -v module >/dev/null 2>&1; then
  module load java          # PySpark needs JDK 17 or 21; check `java -version` below
  module load python        # this project was tested only on Python 3.13
else
  echo "WARNING: 'module' not found; assuming java and python are already on PATH"
fi
source .venv/bin/activate

# ---- run settings (override by exporting before sbatch, e.g. SUBSAMPLE_EVERY= for the full dataset) --
export RAW=${RAW-data/raw/train.txt}
export SUBSAMPLE_EVERY=${SUBSAMPLE_EVERY-4}      # every 4th row ~= 11.5M rows; empty string = use all rows
export NUM_BUCKETS=${NUM_BUCKETS-200000}
export SEEDS=${SEEDS-0}                          # one seed first; use "0 1 2" for the final run
export SPARK_CORES=${SPARK_CORES-${SLURM_CPUS_PER_TASK:-2}}
export SPARK_DRIVER_MEMORY=${SPARK_DRIVER_MEMORY-24g}
# Spark spills shuffle data to disk; keep it on /blue (inside the repo dir), not a small node-local /tmp.
export SPARK_LOCAL_DIRS=${SPARK_LOCAL_DIRS-$PWD/data/spark_tmp}
mkdir -p "$SPARK_LOCAL_DIRS"

echo "== job ${SLURM_JOB_ID:-local} on $(hostname) | cpus=${SLURM_CPUS_PER_TASK:-?} mem=${SLURM_MEM_PER_NODE:-?}MB =="
java -version 2>&1 | head -1
python --version
echo "RAW=$RAW SUBSAMPLE_EVERY='$SUBSAMPLE_EVERY' NUM_BUCKETS=$NUM_BUCKETS SEEDS='$SEEDS'"
date

bash scripts/run_pipeline.sh

date
echo "== done. Results: reports/results.md  reports/results.json  reports/calibration.png =="

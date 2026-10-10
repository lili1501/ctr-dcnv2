#!/usr/bin/env bash
# Start TF Serving (official Docker image) on the exported SavedModel, REST on :8501, gRPC on :8500.
# Requires Docker and access to Docker Hub. Then benchmark with:
#   python -m ctr.benchmark_serving --url http://localhost:8501/v1/models/ctr:predict --batch-size 1 --requests 2000
set -euo pipefail
cd "$(dirname "$0")/.."
MODEL_DIR=${MODEL_DIR:-$(pwd)/artifacts/serving/ctr}   # must contain a numeric version dir, e.g. 1/
docker run --rm -p 8501:8501 -p 8500:8500 \
  --mount type=bind,source="$MODEL_DIR",target=/models/ctr \
  -e MODEL_NAME=ctr tensorflow/serving

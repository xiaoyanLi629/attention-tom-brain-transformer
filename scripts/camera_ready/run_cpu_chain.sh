#!/usr/bin/env bash
# Camera-ready re-analysis, CPU part. Run after Stage 2 features exist.
# Usage: CURRENT_RUN_TIMESTAMP=20260924_cameraready bash scripts/camera_ready/run_cpu_chain.sh
set -u
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
PY=/root/miniconda3/bin/python
step() { echo "=== $(date +%T) $*"; "$PY" -u "$@" || echo "!!! FAILED: $*"; }

step scripts/camera_ready/cr00_check_encoder.py
step scripts/s03_encoding_models.py
step scripts/s04_attention_analysis.py
step scripts/s06_probing.py --n-jobs 8
step scripts/camera_ready/cr02_encoding_significance.py
step scripts/camera_ready/cr07_sparsity_stats.py
step scripts/camera_ready/cr06_probing_bow.py
step scripts/camera_ready/cr05_gpt2_truncation.py
step scripts/camera_ready/cr03_lag_sweep.py
step scripts/camera_ready/cr04_nonlinear_encoding.py
step scripts/camera_ready/cr09_ablation_stability_crossref.py
step scripts/s07_visualization.py
echo "=== $(date +%T) CPU chain done"

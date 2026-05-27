#!/bin/bash
# Runs Phase 2 (GCN+PMA+Evidential) and Phase 3 (GPS+PMA+Evidential)
# using the best lambda already found in logs/best_evidential_coeff.txt.
# Skips Phase 1 (HP search) entirely.

set -eo pipefail
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate gnn

CONDA_LIB="$HOME/miniconda3/envs/gnn/lib/python3.10/site-packages"
export LD_LIBRARY_PATH="${CONDA_LIB}/nvidia/cusparse/lib:${CONDA_LIB}/nvidia/cublas/lib:${LD_LIBRARY_PATH}"
export PYTHONUNBUFFERED=1

PYTHON="$HOME/miniconda3/envs/gnn/bin/python"
mkdir -p logs save figures/gcn_evidential_tuned figures/gps_evidential_tuned figures/evidential_tuned_comparison

log() { echo ""; echo "===== [$(date '+%Y-%m-%d %H:%M:%S')] $* ====="; echo ""; }

BEST_COEFF=$(cat logs/best_evidential_coeff.txt | tr -d '[:space:]')
log "Using best evidential_coeff = ${BEST_COEFF}  (from logs/best_evidential_coeff.txt)"

GPU_ARGS="--use_gpu True --gpu_idx 0"
DATA_ARGS="--dataset_name Solubility --split_method scaffold --num_workers 0"
OPT_ARGS="--batch_size 64 --lr 1e-3 --weight_decay 1e-6 --num_epoches 150"
LOG_ARGS="--log_dir logs --patience 0"
ARCH_GCN="--model_type gcn --num_layers 4 --hidden_dim 128 --readout pma --dropout_prob 0.0"
ARCH_GPS="--num_layers 4 --hidden_dim 128 --readout pma --dropout_prob 0.0 \
          --num_heads 4 --local_mp_type gin --rwse_k 16"

# ---------------------------------------------------------------------------
log "PHASE 2: GCN + PMA + Evidential  coeff=${BEST_COEFF}"

for SEED in 999 888 777 666; do
    log "  gcn pma evidential  seed=${SEED}"
    $PYTHON gnn_regression_evidential.py \
        $GPU_ARGS $DATA_ARGS $OPT_ARGS $LOG_ARGS \
        $ARCH_GCN \
        --evidential_coeff "${BEST_COEFF}" \
        --seed "${SEED}" --data_seed "${SEED}" \
        --job_title "gcn_evidential_pma_tuned"
done

log "PHASE 2 DONE — plotting"
$PYTHON plot_training_curves.py \
    --log_dir logs \
    --pattern "gcn_evidential_pma_tuned_seed*" \
    --output_dir figures/gcn_evidential_tuned \
    --title "GCN + PMA + Evidential (tuned lambda=${BEST_COEFF})"

# ---------------------------------------------------------------------------
log "PHASE 3: GPS + PMA + Evidential  coeff=${BEST_COEFF}"

for SEED in 999 888 777 666; do
    log "  gps pma evidential  seed=${SEED}"
    $PYTHON gnn_regression_gps_evidential.py \
        $GPU_ARGS $DATA_ARGS $OPT_ARGS $LOG_ARGS \
        $ARCH_GPS \
        --evidential_coeff "${BEST_COEFF}" \
        --seed "${SEED}" --data_seed "${SEED}" \
        --job_title "gps_evidential_pma_tuned"
done

log "PHASE 3 DONE — plotting"
$PYTHON plot_training_curves.py \
    --log_dir logs \
    --pattern "gps_evidential_pma_tuned_seed*" \
    --output_dir figures/gps_evidential_tuned \
    --title "GPS + PMA + Evidential (tuned lambda=${BEST_COEFF})"

# ---------------------------------------------------------------------------
log "PHASE 4: cross-backend comparison plot"
$PYTHON plot_training_curves.py \
    --log_dir logs \
    --pattern "*_evidential_pma_tuned_seed*" \
    --output_dir figures/evidential_tuned_comparison \
    --title "GCN vs GPS + PMA + Evidential (tuned lambda=${BEST_COEFF})"

log "ALL DONE"
echo "  Best lambda : ${BEST_COEFF}"
echo "  GCN CSVs    : $(ls logs/gcn_evidential_pma_tuned_seed*.csv 2>/dev/null | wc -l) files"
echo "  GPS CSVs    : $(ls logs/gps_evidential_pma_tuned_seed*.csv 2>/dev/null | wc -l) files"

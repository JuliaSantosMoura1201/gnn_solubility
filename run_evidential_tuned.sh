#!/bin/bash
# Full pipeline: tune evidential lambda, then run GCN+PMA and GPS+PMA experiments.
#
# Phase 1 — Optuna HP search for the NIG regularisation coefficient (lambda).
#            Trials run with stdout suppressed and MedianPruner active.
#            Best lambda is written to logs/best_evidential_coeff.txt.
#
# Phase 2 — GCN + PMA + Evidential, 4 seeds (999 888 777 666).
# Phase 3 — GPS + PMA + Evidential, 4 seeds.
# Phase 4 — Training-curve plots for both backends.
#
# Launch:
#   nohup bash run_evidential_tuned.sh > logs/run_evidential_tuned_master.log 2>&1 &
#   tail -f logs/run_evidential_tuned_master.log
# ---------------------------------------------------------------------------

set -eo pipefail

# Log any signal that kills this script
trap 'echo "[$(date)] SCRIPT KILLED by signal $? (exit code $?)" >> logs/run_evidential_tuned_master.log' EXIT

# Pre-set LD_LIBRARY_PATH so conda activation scripts don't fail on unbound var
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate gnn

CONDA_LIB="$HOME/miniconda3/envs/gnn/lib/python3.10/site-packages"
export LD_LIBRARY_PATH="${CONDA_LIB}/nvidia/cusparse/lib:${CONDA_LIB}/nvidia/cublas/lib:${LD_LIBRARY_PATH}"

PYTHON="$HOME/miniconda3/envs/gnn/bin/python"
export PYTHONUNBUFFERED=1
mkdir -p logs save figures/hp_tune figures/gcn_evidential_tuned2 figures/gps_evidential_tuned2 figures/evidential_tuned_comparison2

log() { echo ""; echo "===== [$(date '+%Y-%m-%d %H:%M:%S')] $* ====="; echo ""; }

# ---------------------------------------------------------------------------
# Shared flags
# ---------------------------------------------------------------------------
GPU_ARGS="--use_gpu True --gpu_idx 0"
DATA_ARGS="--dataset_name Solubility --split_method scaffold --num_workers 0"
OPT_ARGS="--batch_size 64 --lr 1e-3 --weight_decay 1e-6 --num_epoches 150"
LOG_ARGS="--log_dir logs --patience 0"
ARCH_GCN="--model_type gcn --num_layers 4 --hidden_dim 128 --readout pma --dropout_prob 0.0"
ARCH_GPS="--num_layers 4 --hidden_dim 128 --readout pma --dropout_prob 0.0 \
          --num_heads 4 --local_mp_type gin --rwse_k 16"

# ---------------------------------------------------------------------------
# Phase 1 — HP tuning (lambda / evidential_coeff)
# ---------------------------------------------------------------------------
log "PHASE 1: Optuna HP search for evidential_coeff (lambda)"
log "  n_trials=20  tune_epochs=150  pruner_warmup=40  GPU=True"

$PYTHON hp_tune_evidential.py \
    $GPU_ARGS \
    --n_trials      20 \
    --tune_epochs   150 \
    --pruner_warmup 40 \
    --skip_final_run \
    --log_dir    logs \
    --output_dir figures/hp_tune

# Read the best coefficient found by Optuna
BEST_COEFF=$(cat logs/best_evidential_coeff.txt | tr -d '[:space:]')
log "PHASE 1 DONE  best evidential_coeff = ${BEST_COEFF}"

# ---------------------------------------------------------------------------
# Phase 2 — GCN + PMA + Evidential (4 seeds)
# ---------------------------------------------------------------------------
log "PHASE 2: GCN + PMA + Evidential  coeff=${BEST_COEFF}"

for SEED in 999 888 777 666; do
    log "  gcn pma evidential  seed=${SEED}  coeff=${BEST_COEFF}"
    $PYTHON gnn_regression_evidential.py \
        $GPU_ARGS $DATA_ARGS $OPT_ARGS $LOG_ARGS \
        $ARCH_GCN \
        --evidential_coeff "${BEST_COEFF}" \
        --seed "${SEED}" --data_seed "${SEED}" \
        --job_title "gcn_evidential_pma_tuned2"
done

log "PHASE 2 DONE — plotting GCN evidential curves"
$PYTHON plot_training_curves.py \
    --log_dir logs \
    --pattern "gcn_evidential_pma_tuned2_seed*" \
    --output_dir figures/gcn_evidential_tuned2 \
    --title "GCN + PMA + Evidential (tuned 150ep lambda=${BEST_COEFF})"

# ---------------------------------------------------------------------------
# Phase 3 — GPS + PMA + Evidential (4 seeds)
# ---------------------------------------------------------------------------
log "PHASE 3: GPS + PMA + Evidential  coeff=${BEST_COEFF}"

for SEED in 999 888 777 666; do
    log "  gps pma evidential  seed=${SEED}  coeff=${BEST_COEFF}"
    $PYTHON gnn_regression_gps_evidential.py \
        $GPU_ARGS $DATA_ARGS $OPT_ARGS $LOG_ARGS \
        $ARCH_GPS \
        --evidential_coeff "${BEST_COEFF}" \
        --seed "${SEED}" --data_seed "${SEED}" \
        --job_title "gps_evidential_pma_tuned2"
done

log "PHASE 3 DONE — plotting GPS evidential curves"
$PYTHON plot_training_curves.py \
    --log_dir logs \
    --pattern "gps_evidential_pma_tuned2_seed*" \
    --output_dir figures/gps_evidential_tuned2 \
    --title "GPS + PMA + Evidential (tuned 150ep lambda=${BEST_COEFF})"

# ---------------------------------------------------------------------------
# Phase 4 — Cross-backend summary
# ---------------------------------------------------------------------------
log "PHASE 4: cross-backend comparison plot"
$PYTHON plot_training_curves.py \
    --log_dir logs \
    --pattern "*_evidential_pma_tuned2_seed*" \
    --output_dir figures/evidential_tuned_comparison2 \
    --title "GCN vs GPS + PMA + Evidential (tuned 150ep lambda=${BEST_COEFF})"

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
log "ALL PHASES COMPLETE"
echo ""
echo "  Best lambda    : ${BEST_COEFF}"
echo "  Optuna plots   : figures/hp_tune/"
echo "  GCN CSVs       : $(ls logs/gcn_evidential_pma_tuned2_seed*.csv 2>/dev/null | wc -l) files"
echo "  GPS CSVs       : $(ls logs/gps_evidential_pma_tuned2_seed*.csv 2>/dev/null | wc -l) files"
echo "  Checkpoints    : $(ls save/best_gcn_evidential_pma_tuned2_*.pth save/best_gps_evidential_pma_tuned2_*.pth 2>/dev/null | wc -l) files"
echo ""

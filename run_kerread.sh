#!/bin/bash
# =============================================================================
# KerRead readout comparison experiment runner
#
# 8 runs total, 150 epochs each:
#   GCN  × KerRead × MCDO × 4 seeds
#   GPS  × KerRead × MCDO × 4 seeds
#
# Produces logs comparable to the existing PMA runs:
#   logs/arch_mcdo_gcn_pma_seed{N}.csv   ← existing GCN+PMA baseline
#   logs/gps_mcdo_pma_seed{N}.csv        ← existing GPS+PMA baseline
#   logs/gcn_mcdo_kerread_seed{N}.csv    ← new GCN+KerRead
#   logs/gps_mcdo_kerread_seed{N}.csv    ← new GPS+KerRead
#
# Launch:
#   nohup bash run_kerread.sh > logs/run_kerread_$(date +%Y%m%d_%H%M%S).log 2>&1 &
#   echo $!
# =============================================================================

set -eu

# --- environment -------------------------------------------------------------
CONDA_ENV_LIB="$HOME/miniconda3/envs/gnn/lib"
export LD_LIBRARY_PATH="${CONDA_ENV_LIB}:${LD_LIBRARY_PATH:-}"
PYTHON="$HOME/miniconda3/envs/gnn/bin/python"

mkdir -p logs save figures

# --- shared hyperparameters (identical to run_regression.sh) -----------------
GPU_ARGS="--use_gpu True --gpu_idx 0"
DATA_ARGS="--dataset_name Solubility --split_method scaffold --num_workers 0"
OPT_ARGS="--batch_size 64 --lr 1e-3 --weight_decay 1e-6 --num_epoches 150"
LOG_ARGS="--log_dir logs --patience 0"

GCN_KERREAD_ARGS="${GPU_ARGS} ${DATA_ARGS} ${OPT_ARGS} ${LOG_ARGS} \
    --num_layers 4 --hidden_dim 128 --readout kerread \
    --dropout_prob 0.2 --num_sampling 10"

GPS_KERREAD_ARGS="${GPU_ARGS} ${DATA_ARGS} ${OPT_ARGS} ${LOG_ARGS} \
    --num_layers 4 --hidden_dim 128 --readout kerread \
    --dropout_prob 0.2 --num_sampling 10 \
    --num_heads 4 --local_mp_type gin --rwse_k 16"

log() { echo ""; echo "===== [$(date '+%Y-%m-%d %H:%M:%S')] $* ====="; echo ""; }

# =============================================================================
# GROUP 1 — GCN + KerRead + MCDO
# =============================================================================
log "GROUP 1: GCN + KerRead + MCDO"

for SEED in 999 888 777 666; do
    log "  gcn kerread mcdo seed=${SEED}"
    ${PYTHON} gnn_regression_mcdo.py ${GCN_KERREAD_ARGS} \
        --model_type gcn \
        --seed ${SEED} --data_seed ${SEED} \
        --job_title "gcn_mcdo_kerread"
done

log "GROUP 1 done — plotting"
${PYTHON} plot_training_curves.py \
    --log_dir logs --pattern "gcn_mcdo_kerread_seed*" \
    --output_dir figures/gcn_kerread \
    --title "GCN + KerRead + MCDO" || true

# =============================================================================
# GROUP 2 — GPS + KerRead + MCDO
# =============================================================================
log "GROUP 2: GPS + KerRead + MCDO"

for SEED in 999 888 777 666; do
    log "  gps kerread mcdo seed=${SEED}"
    ${PYTHON} gnn_regression_mcdo_gps.py ${GPS_KERREAD_ARGS} \
        --seed ${SEED} --data_seed ${SEED} \
        --job_title "gps_mcdo_kerread"
done

log "GROUP 2 done — plotting"
${PYTHON} plot_training_curves.py \
    --log_dir logs --pattern "gps_mcdo_kerread_seed*" \
    --output_dir figures/gps_kerread \
    --title "GPS + KerRead + MCDO" || true

# =============================================================================
# Final comparison: KerRead vs PMA across both backbones
# =============================================================================
log "Plotting KerRead vs PMA comparison"
${PYTHON} plot_training_curves.py \
    --log_dir logs \
    --pattern "*mcdo*kerread*seed*" \
    --output_dir figures/kerread_vs_pma \
    --title "KerRead vs PMA: GCN and GPS backbones" || true

log "ALL DONE"
echo ""
echo "KerRead CSVs:  $(ls logs/*kerread*seed*.csv 2>/dev/null | wc -l) files"
echo "Best models:   $(ls save/best_*kerread*.pth 2>/dev/null | wc -l) checkpoints"

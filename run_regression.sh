#!/bin/bash
# =============================================================================
# Focused regression experiment runner
#
# 24 runs, GPU only, 150 epochs each:
#   GCN / GIN / GAT × PMA × MCDO × 4 seeds   (architecture comparison)
#   GPS             × PMA × MCDO × 4 seeds   (GPS backbone)
#   GCN             × PMA × Evidential × 4 seeds
#   GPS             × PMA × Evidential × 4 seeds
#
# Launch:
#   nohup bash run_regression.sh > logs/run_regression_$(date +%Y%m%d_%H%M%S).log 2>&1 &
#   echo $!
# =============================================================================

set -eu

# --- environment -------------------------------------------------------------
CONDA_ENV_LIB="$HOME/miniconda3/envs/gnn/lib"
export LD_LIBRARY_PATH="${CONDA_ENV_LIB}:${LD_LIBRARY_PATH:-}"
PYTHON="$HOME/miniconda3/envs/gnn/bin/python"

mkdir -p logs save figures

# --- shared hyperparameters (paper settings) ---------------------------------
GPU_ARGS="--use_gpu True --gpu_idx 0"
DATA_ARGS="--dataset_name Solubility --split_method scaffold --num_workers 0"
ARCH_ARGS="--num_layers 4 --hidden_dim 128 --readout pma"
OPT_ARGS="--batch_size 64 --lr 1e-3 --weight_decay 1e-6 --num_epoches 150"
LOG_ARGS="--log_dir logs --patience 0"

MCDO_ARGS="${GPU_ARGS} ${DATA_ARGS} ${ARCH_ARGS} ${OPT_ARGS} ${LOG_ARGS} \
           --dropout_prob 0.2 --num_sampling 10"

EVI_ARGS="${GPU_ARGS} ${DATA_ARGS} ${ARCH_ARGS} ${OPT_ARGS} ${LOG_ARGS} \
          --dropout_prob 0.0 --evidential_coeff 0.01"

GPS_MCDO_ARGS="${GPU_ARGS} ${DATA_ARGS} ${OPT_ARGS} ${LOG_ARGS} \
               --readout pma --dropout_prob 0.2 --num_sampling 10 \
               --num_heads 4 --local_mp_type gin --rwse_k 16"

GPS_EVI_ARGS="${GPU_ARGS} ${DATA_ARGS} ${OPT_ARGS} ${LOG_ARGS} \
              --readout pma --dropout_prob 0.0 --evidential_coeff 0.01 \
              --num_heads 4 --local_mp_type gin --rwse_k 16"

log() { echo ""; echo "===== [$(date '+%Y-%m-%d %H:%M:%S')] $* ====="; echo ""; }

# =============================================================================
# GROUP 1 — Architecture comparison: GCN / GIN / GAT × PMA × MCDO
# Replicates the node-update comparison from Figure 2 of 2210.07145
# =============================================================================
log "GROUP 1: Architecture comparison (GCN / GIN / GAT + PMA + MCDO)"

for MODEL in gcn gin gat; do
    for SEED in 999 888 777 666; do
        log "  ${MODEL} pma mcdo seed=${SEED}"
        ${PYTHON} gnn_regression_mcdo.py ${MCDO_ARGS} \
            --model_type ${MODEL} \
            --seed ${SEED} --data_seed ${SEED} \
            --job_title "arch_mcdo_${MODEL}_pma"
    done
done

log "GROUP 1 done — plotting"
${PYTHON} plot_training_curves.py \
    --log_dir logs --pattern "arch_mcdo_*" \
    --output_dir figures/arch_comparison \
    --title "Architecture comparison: GCN/GIN/GAT + PMA + MCDO"

# =============================================================================
# GROUP 2 — GPS + PMA + MCDO
# =============================================================================
log "GROUP 2: GPS + PMA + MCDO"

for SEED in 999 888 777 666; do
    log "  gps pma mcdo seed=${SEED}"
    ${PYTHON} gnn_regression_mcdo_gps.py ${GPS_MCDO_ARGS} \
        --seed ${SEED} --data_seed ${SEED} \
        --job_title "gps_mcdo_pma"
done

log "GROUP 2 done — plotting"
${PYTHON} plot_training_curves.py \
    --log_dir logs --pattern "gps_mcdo_pma_seed*" \
    --output_dir figures/gps_mcdo \
    --title "GPS + PMA + MCDO"

# =============================================================================
# GROUP 3 — GCN + PMA + Evidential
# =============================================================================
log "GROUP 3: GCN + PMA + Evidential"

for SEED in 999 888 777 666; do
    log "  gcn pma evidential seed=${SEED}"
    ${PYTHON} gnn_regression_evidential.py ${EVI_ARGS} \
        --model_type gcn \
        --seed ${SEED} --data_seed ${SEED} \
        --job_title "gcn_evidential_pma"
done

log "GROUP 3 done — plotting"
${PYTHON} plot_training_curves.py \
    --log_dir logs --pattern "gcn_evidential_pma_seed*" \
    --output_dir figures/gcn_evidential \
    --title "GCN + PMA + Evidential"

# =============================================================================
# GROUP 4 — GPS + PMA + Evidential
# =============================================================================
log "GROUP 4: GPS + PMA + Evidential"

for SEED in 999 888 777 666; do
    log "  gps pma evidential seed=${SEED}"
    ${PYTHON} gnn_regression_gps_evidential.py ${GPS_EVI_ARGS} \
        --seed ${SEED} --data_seed ${SEED} \
        --job_title "gps_evidential_pma"
done

log "GROUP 4 done — plotting"
${PYTHON} plot_training_curves.py \
    --log_dir logs --pattern "gps_evidential_pma_seed*" \
    --output_dir figures/gps_evidential \
    --title "GPS + PMA + Evidential"

# =============================================================================
# Final cross-group comparison plot
# =============================================================================
log "Plotting cross-group comparison"
${PYTHON} plot_training_curves.py \
    --log_dir logs \
    --pattern "*_pma_seed*" \
    --output_dir figures/all_regression \
    --title "All regression experiments"

log "ALL DONE"
echo ""
echo "CSVs:    $(ls logs/*_pma_seed*.csv 2>/dev/null | wc -l) files in logs/"
echo "Plots:   $(ls figures/**/*.png 2>/dev/null | wc -l) files in figures/"
echo "Models:  $(ls save/best_*.pth 2>/dev/null | wc -l) best checkpoints in save/"

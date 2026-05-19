#!/bin/bash
# =============================================================================
# Overnight experiment runner for GNN solubility project
# Replicates 2210.07145, tests convergence, GPS, evidential, and HP tuning.
#
# Launch with:
#   nohup bash run_overnight.sh > logs/master_$(date +%Y%m%d_%H%M%S).log 2>&1 &
#   echo $!   # note the PID so you can kill it if needed
#
# GPU setting: change USE_GPU / GPU_IDX below if you have a GPU.
# =============================================================================

set -eu

# libnvrtc.so.12 lives inside the conda env but is not on the default linker
# path, so DGL 2.1 fails to load its graphbolt extension. Fix that here.
CONDA_ENV_LIB="$HOME/miniconda3/envs/gnn/lib"
export LD_LIBRARY_PATH="${CONDA_ENV_LIB}:${LD_LIBRARY_PATH:-}"

# Use the full path so we don't need `conda activate` (avoids non-interactive-shell issues)
PYTHON="$HOME/miniconda3/envs/gnn/bin/python"

mkdir -p logs save figures

USE_GPU=True
GPU_IDX=0
WORKERS=0    # 0 = safe for DGL graphs; bump to 4 if your system supports it

# Shared base arguments (paper hyperparameters)
COMMON="--use_gpu ${USE_GPU} --gpu_idx ${GPU_IDX} \
        --dataset_name Solubility --split_method scaffold \
        --num_layers 4 --hidden_dim 128 \
        --batch_size 64 --num_workers ${WORKERS} \
        --lr 1e-3 --weight_decay 1e-6 \
        --log_dir logs"

MCDO_COMMON="${COMMON} --readout pma --dropout_prob 0.2 --num_sampling 10"
VANILLA_COMMON="${COMMON} --dropout_prob 0.0"
EVI_COMMON="${COMMON} --readout pma --dropout_prob 0.0 --evidential_coeff 0.01"
GPS_MCDO="${COMMON} --readout pma --dropout_prob 0.2 --num_sampling 10 \
          --num_heads 4 --local_mp_type gin --rwse_k 16"
GPS_EVI="${COMMON} --readout pma --dropout_prob 0.0 --evidential_coeff 0.01 \
         --num_heads 4 --local_mp_type gin --rwse_k 16"
CLS_EVI="${COMMON} --readout pma --dropout_prob 0.0 --evidential_coeff 0.01 \
         --warmup_epochs 10 --out_dim 4"

log() { echo ""; echo "===== [$(date '+%Y-%m-%d %H:%M:%S')] $* ====="; echo ""; }

# Helper: plot one group of CSVs
plot_group() {
    local pattern="$1" outdir="$2" title="$3"
    ${PYTHON} plot_training_curves.py \
        --log_dir logs --pattern "${pattern}" \
        --output_dir "${outdir}" --title "${title}" || true
}

# =============================================================================
# PHASE 0 — Convergence probe: seed 999, GCN+PMA+MCDO, patience=30
# This determines N (convergence epoch) used by ALL subsequent phases.
# =============================================================================
log "PHASE 0: Convergence probe (seed 999, GCN+PMA+MCDO, max 300 epochs)"

CONV_LOG="logs/phase0_conv_probe.log"

${PYTHON} gnn_regression_mcdo.py ${MCDO_COMMON} \
    --model_type gcn --num_epoches 300 --patience 30 \
    --seed 999 --data_seed 999 \
    --job_title conv_mcdo_gcn_pma \
    2>&1 | tee "${CONV_LOG}"

# Extract convergence epoch; fall back to the last CSV row count
CONV_EPOCH=$(grep -oP '(?<=CONVERGENCE_EPOCH=)\d+' "${CONV_LOG}" | tail -1)
if [ -z "${CONV_EPOCH}" ]; then
    CONV_EPOCH=$(tail -n +2 logs/conv_mcdo_gcn_pma_seed999.csv 2>/dev/null | wc -l || echo "150")
fi
echo "${CONV_EPOCH}" > logs/convergence_epoch.txt
log "Convergence epoch determined: N=${CONV_EPOCH}"

# =============================================================================
# PHASE 1 — Paper replication: GCN/GIN/GAT × Mean/PMA × Vanilla/MCDO × 4 seeds
#           All 150 epochs, exactly as in 2210.07145
# =============================================================================
log "PHASE 1: Paper replication (3 models × 2 readouts × 2 methods × 4 seeds = 48 runs)"

for MODEL in gcn gin gat; do
  for READOUT in mean pma; do
    for SEED in 999 888 777 666; do

      # Vanilla regression
      log "  Paper vanilla  model=${MODEL} readout=${READOUT} seed=${SEED}"
      ${PYTHON} gnn_regression.py ${VANILLA_COMMON} \
          --model_type ${MODEL} --readout ${READOUT} \
          --num_epoches 150 \
          --seed ${SEED} --data_seed ${SEED} \
          --job_title "paper_vanilla_${MODEL}_${READOUT}" \
          2>&1 | tee "logs/paper_vanilla_${MODEL}_${READOUT}_seed${SEED}.log"

      # MCDO regression
      log "  Paper MCDO     model=${MODEL} readout=${READOUT} seed=${SEED}"
      ${PYTHON} gnn_regression_mcdo.py ${COMMON} \
          --model_type ${MODEL} --readout ${READOUT} \
          --dropout_prob 0.2 --num_sampling 10 \
          --num_epoches 150 \
          --seed ${SEED} --data_seed ${SEED} \
          --job_title "paper_mcdo_${MODEL}_${READOUT}" \
          2>&1 | tee "logs/paper_mcdo_${MODEL}_${READOUT}_seed${SEED}.log"

    done
  done
done

log "Phase 1 complete — plotting paper replication curves"
plot_group "paper_vanilla_*" "figures/paper_vanilla" "Paper replication: Vanilla"
plot_group "paper_mcdo_*"    "figures/paper_mcdo"    "Paper replication: MCDO"
plot_group "paper_*_gcn_*"   "figures/paper_gcn"     "Paper: GCN (all readouts & methods)"
plot_group "paper_*_pma*"    "figures/paper_pma"     "Paper: PMA readout (all models & methods)"

# =============================================================================
# PHASE 2 — Convergence: remaining seeds for N epochs (seed 999 done in Phase 0)
# =============================================================================
log "PHASE 2: Convergence runs seeds 888/777/666 for N=${CONV_EPOCH} epochs"

for SEED in 888 777 666; do
  ${PYTHON} gnn_regression_mcdo.py ${MCDO_COMMON} \
      --model_type gcn \
      --num_epoches "${CONV_EPOCH}" \
      --seed ${SEED} --data_seed ${SEED} \
      --job_title "conv_mcdo_gcn_pma" \
      2>&1 | tee "logs/conv_mcdo_gcn_pma_seed${SEED}.log"
done

log "Phase 2 complete — plotting convergence curves"
plot_group "conv_mcdo_gcn_pma_seed*" \
           "figures/convergence_mcdo" \
           "GCN+PMA+MCDO convergence (N=${CONV_EPOCH} epochs)"

# =============================================================================
# PHASE 3 — GPS: paper conditions (150 ep) + convergence conditions (N ep)
# =============================================================================
log "PHASE 3a: GPS+MCDO — paper conditions (150 epochs)"
for SEED in 999 888 777 666; do
  ${PYTHON} gnn_regression_mcdo_gps.py ${GPS_MCDO} \
      --num_epoches 150 \
      --seed ${SEED} --data_seed ${SEED} \
      --job_title "gps_mcdo_paper" \
      2>&1 | tee "logs/gps_mcdo_paper_seed${SEED}.log"
done

log "PHASE 3b: GPS+MCDO — convergence conditions (N=${CONV_EPOCH} epochs)"
for SEED in 999 888 777 666; do
  ${PYTHON} gnn_regression_mcdo_gps.py ${GPS_MCDO} \
      --num_epoches "${CONV_EPOCH}" \
      --seed ${SEED} --data_seed ${SEED} \
      --job_title "gps_mcdo_conv" \
      2>&1 | tee "logs/gps_mcdo_conv_seed${SEED}.log"
done

log "Phase 3 complete — plotting GPS curves"
plot_group "gps_mcdo_paper_seed*" "figures/gps_paper" "GPS+MCDO paper conditions (150 ep)"
plot_group "gps_mcdo_conv_seed*"  "figures/gps_conv"  "GPS+MCDO convergence (N=${CONV_EPOCH} ep)"

# =============================================================================
# PHASE 4 — Evidential regression (replaces MCDO): GCN+PMA and GPS+PMA
# =============================================================================
log "PHASE 4a: GCN+PMA+Evidential — paper conditions (150 epochs)"
for SEED in 999 888 777 666; do
  ${PYTHON} gnn_regression_evidential.py ${EVI_COMMON} \
      --model_type gcn --num_epoches 150 \
      --seed ${SEED} --data_seed ${SEED} \
      --job_title "evi_reg_gcn_paper" \
      2>&1 | tee "logs/evi_reg_gcn_paper_seed${SEED}.log"
done

log "PHASE 4b: GCN+PMA+Evidential — convergence conditions (N=${CONV_EPOCH} epochs)"
for SEED in 999 888 777 666; do
  ${PYTHON} gnn_regression_evidential.py ${EVI_COMMON} \
      --model_type gcn --num_epoches "${CONV_EPOCH}" \
      --seed ${SEED} --data_seed ${SEED} \
      --job_title "evi_reg_gcn_conv" \
      2>&1 | tee "logs/evi_reg_gcn_conv_seed${SEED}.log"
done

log "PHASE 4c: GPS+Evidential — paper conditions (150 epochs)"
for SEED in 999 888 777 666; do
  ${PYTHON} gnn_regression_gps_evidential.py ${GPS_EVI} \
      --num_epoches 150 \
      --seed ${SEED} --data_seed ${SEED} \
      --job_title "gps_evi_paper" \
      2>&1 | tee "logs/gps_evi_paper_seed${SEED}.log"
done

log "PHASE 4d: GPS+Evidential — convergence conditions (N=${CONV_EPOCH} epochs)"
for SEED in 999 888 777 666; do
  ${PYTHON} gnn_regression_gps_evidential.py ${GPS_EVI} \
      --num_epoches "${CONV_EPOCH}" \
      --seed ${SEED} --data_seed ${SEED} \
      --job_title "gps_evi_conv" \
      2>&1 | tee "logs/gps_evi_conv_seed${SEED}.log"
done

log "Phase 4 complete — plotting evidential regression curves"
plot_group "evi_reg_gcn_paper_seed*" "figures/evi_gcn_paper" "GCN+PMA+Evidential paper (150 ep)"
plot_group "evi_reg_gcn_conv_seed*"  "figures/evi_gcn_conv"  "GCN+PMA+Evidential convergence (N=${CONV_EPOCH} ep)"
plot_group "gps_evi_paper_seed*"     "figures/gps_evi_paper" "GPS+Evidential paper (150 ep)"
plot_group "gps_evi_conv_seed*"      "figures/gps_evi_conv"  "GPS+Evidential convergence (N=${CONV_EPOCH} ep)"

# =============================================================================
# PHASE 5 — Evidential classification (replaces SWA)
# =============================================================================
log "PHASE 5a: Evidential classification — paper conditions (150 epochs)"
for SEED in 999 888 777 666; do
  ${PYTHON} gnn_classification_evidential.py ${CLS_EVI} \
      --model_type gcn --num_epoches 150 \
      --seed ${SEED} --data_seed ${SEED} \
      --job_title "evi_cls_gcn_paper" \
      2>&1 | tee "logs/evi_cls_gcn_paper_seed${SEED}.log"
done

log "PHASE 5b: Evidential classification — convergence conditions (N=${CONV_EPOCH} epochs)"
for SEED in 999 888 777 666; do
  ${PYTHON} gnn_classification_evidential.py ${CLS_EVI} \
      --model_type gcn --num_epoches "${CONV_EPOCH}" \
      --seed ${SEED} --data_seed ${SEED} \
      --job_title "evi_cls_gcn_conv" \
      2>&1 | tee "logs/evi_cls_gcn_conv_seed${SEED}.log"
done

log "Phase 5 complete — plotting evidential classification curves"
plot_group "evi_cls_gcn_paper_seed*" "figures/evi_cls_paper" "Evidential cls paper (150 ep)"
plot_group "evi_cls_gcn_conv_seed*"  "figures/evi_cls_conv"  "Evidential cls convergence (N=${CONV_EPOCH} ep)"

# =============================================================================
# PHASE 6 — HP tuning: evidential coefficient (lambda/gamma)
#           Optuna (20 trials × 40 epochs) + final 4-seed run
# =============================================================================
log "PHASE 6: HP tuning for evidential regression coefficient (N_final=${CONV_EPOCH} epochs)"

${PYTHON} hp_tune_evidential.py \
    --use_gpu ${USE_GPU} \
    --gpu_idx ${GPU_IDX} \
    --n_trials 20 \
    --tune_epochs 40 \
    --final_epochs "${CONV_EPOCH}" \
    --log_dir logs \
    --output_dir figures/hp_tune \
    2>&1 | tee logs/phase6_hp_tune.log

BEST_COEFF=$(cat logs/best_evidential_coeff.txt 2>/dev/null || echo "0.01")
log "Phase 6 complete — best evidential coeff = ${BEST_COEFF}"
plot_group "evi_tuned_coeff*" "figures/hp_tuned" "Evidential (best coeff=${BEST_COEFF})"

# =============================================================================
# SUMMARY
# =============================================================================
log "ALL PHASES COMPLETE"
echo ""
echo "Results summary:"
echo "  Convergence epoch N = ${CONV_EPOCH}"
echo "  Best evidential coeff = ${BEST_COEFF}"
echo ""
echo "Plots saved in figures/:"
ls figures/ 2>/dev/null || true
echo ""
echo "CSV logs in logs/:"
ls logs/*.csv 2>/dev/null | wc -l || echo "0"
echo " CSV files"

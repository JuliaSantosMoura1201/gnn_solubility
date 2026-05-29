#!/bin/bash
# =============================================================================
# Final evidential regression runs — all with train_nll logging
#
# GCN + PMA + Evidential tuned      (4 seeds)
# GPS + PMA + Evidential tuned      (4 seeds)
# GCN + KerRead + Evidential tuned  (4 seeds)
#
# Launch:
#   nohup bash run_evidential_final.sh > logs/run_evidential_final_$(date +%Y%m%d_%H%M%S).log 2>&1 &
#   echo $!
# =============================================================================

set -eu

GNN_LIB=$HOME/miniconda3/envs/gnn/lib/python3.10/site-packages
GNN_BASE=$HOME/miniconda3/envs/gnn/lib
export LD_LIBRARY_PATH=$GNN_LIB/nvidia/cusparse/lib:$GNN_LIB/nvidia/cublas/lib:$GNN_BASE:${LD_LIBRARY_PATH:-}

PYTHON=$HOME/miniconda3/envs/gnn/bin/python
LAMBDA=$(cat /home/julia/gnn_solubility/logs/best_evidential_coeff.txt)

COMMON="--use_gpu True --gpu_idx 0
        --dataset_name Solubility --split_method scaffold --num_workers 0
        --num_layers 4 --hidden_dim 128
        --dropout_prob 0.0 --evidential_coeff ${LAMBDA}
        --batch_size 64 --lr 1e-3 --weight_decay 1e-6 --num_epoches 150
        --log_dir logs --patience 0"

log() { echo ""; echo "===== [$(date '+%Y-%m-%d %H:%M:%S')] $* ====="; echo ""; }

cd /home/julia/gnn_solubility

# =============================================================================
# GROUP 1 — GCN + PMA + Evidential
# =============================================================================
log "GROUP 1: GCN + PMA + Evidential (lambda=${LAMBDA})"
for SEED in 999 888 777 666; do
    log "  gcn pma evidential seed=${SEED}"
    ${PYTHON} gnn_regression_evidential.py ${COMMON} \
        --model_type gcn --readout pma \
        --seed ${SEED} --data_seed ${SEED} \
        --job_title gcn_evidential_pma_final
done

# =============================================================================
# GROUP 2 — GPS + PMA + Evidential
# =============================================================================
log "GROUP 2: GPS + PMA + Evidential (lambda=${LAMBDA})"
for SEED in 999 888 777 666; do
    log "  gps pma evidential seed=${SEED}"
    ${PYTHON} gnn_regression_gps_evidential.py ${COMMON} \
        --readout pma \
        --num_heads 4 --local_mp_type gin --rwse_k 16 \
        --seed ${SEED} --data_seed ${SEED} \
        --job_title gps_evidential_pma_final
done

# =============================================================================
# GROUP 3 — GCN + KerRead + Evidential
# =============================================================================
log "GROUP 3: GCN + KerRead + Evidential (lambda=${LAMBDA})"
for SEED in 999 888 777 666; do
    log "  gcn kerread evidential seed=${SEED}"
    ${PYTHON} gnn_regression_evidential.py ${COMMON} \
        --model_type gcn --readout kerread \
        --seed ${SEED} --data_seed ${SEED} \
        --job_title gcn_kerread_evidential_final
done

log "ALL DONE"
echo ""
echo "Logs: $(ls logs/*_final_seed*.csv 2>/dev/null | wc -l) CSV files written"

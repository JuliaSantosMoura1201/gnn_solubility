#!/bin/bash
# Runs evidential regression seed 666 (resume), then all 4 classification seeds.
# Launch with: nohup bash run_evidential.sh > logs/run_evidential_master.log 2>&1 &

source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate gnn

BASE_ARGS="--use_gpu False --model_type gcn --num_layers 4 --hidden_dim 128 \
           --readout pma --dropout_prob 0.0 --dataset_name Solubility \
           --split_method scaffold --batch_size 64 --num_epoches 150 \
           --num_workers 0 --lr 0.001 --weight_decay 1e-6 --evidential_coeff 0.01"

echo "===== [$(date)] Starting evidential regression seed 666 ====="
python gnn_regression_evidential.py $BASE_ARGS \
    --job_title Evidential_regression --seed 666 --data_seed 666
echo "===== [$(date)] Done regression seed 666 ====="

for SEED in 999 888 777 666; do
    echo "===== [$(date)] Starting evidential classification seed $SEED ====="
    python gnn_classification_evidential.py $BASE_ARGS \
        --job_title Evidential_classification --seed $SEED --data_seed $SEED
    echo "===== [$(date)] Done classification seed $SEED ====="
done

echo "===== [$(date)] ALL DONE ====="

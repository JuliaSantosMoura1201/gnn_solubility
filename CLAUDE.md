# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment

```bash
conda activate gnn
# GPU runs require this export every session (RTX 4070 Ti, DGL 2.1 / PyTorch 2.2.1):
export LD_LIBRARY_PATH=$HOME/miniconda3/envs/gnn/lib/python3.10/site-packages/nvidia/cusparse/lib:$HOME/miniconda3/envs/gnn/lib/python3.10/site-packages/nvidia/cublas/lib:$LD_LIBRARY_PATH
```

GPU flag pattern for all scripts: `--use_gpu True --gpu_idx 0`

## Running Scripts

**Smoke test (2 epochs, verifies environment):**
```bash
python gnn_regression.py --use_gpu False --num_epoches 2 --model_type gin --readout mean --job_title smoke_test
```

**Full MCDO regression run (standard entry point):**
```bash
python gnn_regression_mcdo.py --job_title paper_mcdo_gcn_pma --model_type gcn --hidden_dim 128 --num_layers 4 --readout pma --out_dim 2 --dropout_prob 0.2 --num_epoches 150 --seed 999 --use_gpu True --gpu_idx 0
```

**GPS + MCDO variant:**
```bash
python gnn_regression_mcdo_gps.py --job_title gps_mcdo_pma --model_type gps --readout pma --hidden_dim 128 --num_layers 4 --out_dim 2 --use_gpu True --gpu_idx 0
```

**Multi-seed batch runs:** see `run_regression.sh` and `run_overnight.sh`.

## Architecture

### Model hierarchy

`libs/models.py` — `MyModel` (GCN/GIN/GAT backbone)  
`libs/gps_model.py` — `GPSModel` (GPS graph transformer backbone)  
Both expose the same interface: `forward(graph, training=False) → (out, alpha)`.

### Readout plug-in point

In both `MyModel` and `GPSModel`, the readout lives at the end of `forward()`:

```python
if self.readout in ['sum', 'mean', 'max']:
    out = dgl.readout_nodes(graph, 'h', op=self.readout)
elif self.readout == 'pma':
    out, alpha = self.pma(graph)
```

`self.pma = PMALayer(...)` is instantiated in `__init__` when `readout == 'pma'`. To add a new readout (e.g. KerRead), follow this same pattern: instantiate it in `__init__` and add a branch in `forward()`.

### Layers (`libs/layers.py`)

| Class | Role |
|-------|------|
| `GraphConvolution` | GCN layer (sum aggregation + residual + LayerNorm) |
| `GraphIsomorphism` | GIN layer (MLP on summed neighbours + residual) |
| `GraphAttention` | GAT layer (multi-head, edge-featured) |
| `PMALayer` | Pooling by Multihead Attention readout (wraps `MultiHeadAttention`) |
| `MLP` | 2-layer MLP utility |

### GPS layers (`libs/gps_layers.py`)
`GPSLayer` = local GIN/GCN + global Transformer + FFN + LayerNorm, with optional RWSE positional encodings computed by `compute_rwse_batched`.

### Data pipeline (`libs/io_utils.py`)
`get_dataset()` downloads via TDC and returns train/valid/test splits. `MyDataset` + `gnn_collate_fn` build batched DGL graphs. Node features = 58-dim atom descriptor; edge features = 6-dim bond descriptor.

### Uncertainty variants
- **MCDO:** `out_dim=2` (predicted mean + log-variance); `heteroscedastic_loss` in `libs/utils.py`; dropout kept active at inference (`training=True`) for `num_sampling` forward passes.
- **Evidential:** `libs/evidential_utils.py`; NIG loss; separate entry-point scripts (`gnn_regression_evidential.py`, `gnn_regression_gps_evidential.py`).

## File Naming Conventions

**Checkpoint:** `save/best_{job_title}_{model_type}_{readout}_{data_seed}_s{seed}.pth`  
**Log:** `logs/{job_title}_seed{seed}.csv`  
Columns: `epoch, train_loss, train_rmse, train_r2, valid_loss, valid_rmse, valid_r2, test_loss, test_rmse, test_r2`

## Key Hyperparameters

| Flag | Note |
|------|------|
| `--readout` | `sum` `mean` `max` `pma`; add new readouts here |
| `--out_dim` | `1` for vanilla regression, `2` for MCDO/heteroscedastic |
| `--data_seed` | Controls scaffold split; keep fixed across seeds for fair comparison |
| `--num_sampling` | MC-Dropout inference passes (default 10) |
| `--patience` | Early-stopping epochs; `0` = disabled |

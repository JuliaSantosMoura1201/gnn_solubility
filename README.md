# gnn_solubility

Graph Neural Networks for molecular solubility and ADME property prediction, with Bayesian uncertainty quantification via Monte Carlo Dropout and heteroscedastic loss.

Molecules are represented as graphs from their SMILES strings (atoms → nodes, bonds → edges) using RDKit and DGL. Four GNN architectures are implemented (GCN, GIN, GIN-E, GAT) together with standard and attention-based graph readouts. Experiments cover regression, four-class solubility classification with Stochastic Weight Averaging (SWA), and MC-Dropout inference with aleatoric/epistemic uncertainty decomposition.

---

## Project Structure

```
gnn_solubility/
├── gnn_regression.py          # Standard regression training (MSE, RMSE, R²)
├── gnn_classification.py      # 4-class solubility classification with SWA
├── gnn_regression_mcdo.py     # MC-Dropout regression + uncertainty estimation
├── inference_solubility_r.py  # Inference & scatter plots from a trained MCDO model
├── attention_visualization.py # PMA attention maps on a custom SMILES CSV
├── summarize_regression.py    # Aggregate results across seeds into bar charts
└── libs/
    ├── models.py              # MyModel, MLP_model
    ├── layers.py              # GCN, GIN, GIN-E, GAT, PMA layer implementations
    ├── io_utils.py            # Dataset classes, molecular graph construction
    ├── io_inference.py        # Data handling for inference scripts
    └── utils.py               # Metrics, loss functions, evaluation helpers
```

---

## Environment Setup

This project uses a pinned stack (Python 3.10, PyTorch 2.2.1, DGL 2.1.0) because the DGL and PyTorch versions must match the graphbolt C++ bindings that DGL ships. The recipe below has been verified on **WSL2 / Ubuntu 18.04, CPU-only**. On newer Ubuntu versions the same commands work; on Windows/macOS, adjust the Miniconda installer URL accordingly.

### Step 1 — Install Miniconda

Miniconda provides an isolated Python installation that doesn't touch system Python. Everything lives under `~/miniconda3` and can be removed with `rm -rf ~/miniconda3` if you ever want to start over.

```bash
# Download installer — using 23.5.2 because the latest installer requires glibc >= 2.28,
# which Ubuntu 18.04 does not ship. On Ubuntu 22.04+, use Miniconda3-latest-Linux-x86_64.sh instead.
wget -O /tmp/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-py310_23.5.2-0-Linux-x86_64.sh

# Run in batch mode (-b auto-accepts the license, -p sets install path)
bash /tmp/miniconda.sh -b -p $HOME/miniconda3

# Register conda in your shell (modifies ~/.bashrc so future terminals see `conda`)
$HOME/miniconda3/bin/conda init bash

# Load conda into the current shell without restarting
source $HOME/miniconda3/etc/profile.d/conda.sh
```

### Step 2 — Create and activate the environment

```bash
# -y auto-accepts the package list
conda create -n gnn python=3.10 -y
conda activate gnn
```

You must run `conda activate gnn` every new terminal before running any script in this repo.

### Step 3 — Install dependencies

```bash
# PyTorch 2.2.1 CPU — must be this version; DGL 2.1.0 ships graphbolt bindings for torch 2.0–2.2.1 only.
# The --index-url flag points pip at PyTorch's CPU-only wheel index.
pip install torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 --index-url https://download.pytorch.org/whl/cpu

# Deep Graph Library
pip install dgl==2.1.0

# torchdata 0.7.1 — newer versions removed `datapipes` that DGL 2.1.0 imports at startup.
pip install "torchdata==0.7.1"

# Chemistry, data, evaluation, plotting (--only-binary skips source builds that require cmake + newer glibc)
pip install --only-binary=:all: pyarrow h5py rdkit scikit-learn matplotlib

# PyTDC (pure Python, no wheel)
pip install PyTDC

# setuptools < 81 — PyTDC's metadata module imports pkg_resources, which was removed in setuptools 81.
pip install "setuptools<81"
```

### Step 4 — Create output directories

```bash
mkdir -p save/ results/ figures/ logs/
```

| Directory | Contents |
|-----------|----------|
| `save/`   | Model checkpoints (`.pth`) |
| `results/`| Prediction CSVs |
| `figures/`| Plots and attention maps |
| `logs/`   | Training logs |

### GPU vs CPU

If your machine has no CUDA GPU (e.g. default WSL2), pass `--use_gpu False` to every training/inference command. The recipe above installs the CPU-only PyTorch build; for GPU, replace the torch index URL with the appropriate CUDA variant from <https://pytorch.org/get-started/previous-versions/> and verify the DGL–torch compatibility table.

---

## Verification

Run the two tests below after setup to confirm everything works end-to-end.

### Test 1 — Imports and basic operations (~2 seconds)

```bash
conda activate gnn
python -c "
import torch, dgl, rdkit, tdc, sklearn, matplotlib, numpy
from rdkit import Chem

print('torch:', torch.__version__)
print('dgl:', dgl.__version__)
print('rdkit:', rdkit.__version__)

g = dgl.graph(([0,1,2],[1,2,0]))
g.ndata['x'] = torch.ones(3, 4)
assert g.ndata['x'].shape == (3, 4)

mol = Chem.MolFromSmiles('CCO')
assert mol.GetNumAtoms() == 3

print('smoke test OK')
"
```

Expected final line: `smoke test OK`.

### Test 2 — End-to-end training smoke test (~a few minutes on CPU)

Runs the full pipeline — TDC dataset download, SMILES → graph featurization, model build, training loop — for 2 epochs. Confirms the project's own code imports and trains before committing to a full 150-epoch run.

```bash
conda activate gnn
python gnn_regression.py \
  --use_gpu False \
  --num_epoches 2 \
  --model_type gin \
  --readout mean \
  --job_title smoke_test
```

On first run TDC downloads `Solubility_AqSolDB` (~a few MB) into `./data/`. A checkpoint should appear under `save/` and a log under `logs/`. If both are produced without errors, the environment is good — re-run with `--num_epoches 150` for real results.

---

## Datasets

Datasets are downloaded automatically on first run via the [TDC](https://tdcommons.ai/) SDK.

**Primary dataset:** `Solubility_AqSolDB` (aqueous solubility, log mol/L)

**Other supported datasets:**

| Task | Datasets |
|------|----------|
| Regression (ADME) | `Lipophilicity_AstraZeneca`, `BBB_martins`, `CYP1A2_Veith`, `CYP2C9_Veith`, `CYP2C19_Veith`, `CYP2D6_Veith`, `CYP3A4_Veith` |
| Classification (Toxicity) | `hERG`, `AMES`, `DILI` |

Pass the dataset name with `--dataset_name` (e.g. `--dataset_name Lipophilicity`).

**Split methods:** `random` | `scaffold` (default: `scaffold`)

---

## Experiments

### 1. Standard regression

Trains a GNN to predict continuous molecular properties. Reports MSE, RMSE, and R².

```bash
python gnn_regression.py \
  --job_title GNN_vanila \
  --model_type gin \
  --hidden_dim 128 \
  --num_layers 4 \
  --readout mean \
  --dataset_name Solubility \
  --split_method scaffold \
  --batch_size 64 \
  --num_epoches 150 \
  --lr 1e-3 \
  --weight_decay 1e-6 \
  --seed 999 \
  --use_gpu True \
  --gpu_idx 0
```

### 2. Classification with SWA

Bins solubility into four classes and trains with Stochastic Weight Averaging. Reports accuracy and Expected Calibration Error (ECE).

| Class | Solubility range |
|-------|-----------------|
| 0 | log(S) > 0 (highly soluble) |
| 1 | −2 < log(S) ≤ 0 |
| 2 | −4 < log(S) ≤ −2 |
| 3 | log(S) ≤ −4 (poorly soluble) |

```bash
python gnn_classification.py \
  --job_title GNN_classification \
  --model_type gcn \
  --hidden_dim 128 \
  --num_layers 4 \
  --readout pma \
  --out_dim 4 \
  --batch_size 64 \
  --num_epoches 150 \
  --swa_start 100 \
  --swa_lr 1e-3 \
  --seed 999 \
  --use_gpu True
```

### 3. MC-Dropout regression (uncertainty quantification)

Outputs a predicted mean and log-variance. At inference, dropout is kept active and the forward pass is repeated `num_sampling` times to estimate epistemic uncertainty. Aleatoric and epistemic uncertainties are reported separately.

```bash
python gnn_regression_mcdo.py \
  --job_title MCDO \
  --model_type gcn \
  --hidden_dim 128 \
  --num_layers 4 \
  --readout pma \
  --out_dim 2 \
  --dropout_prob 0.2 \
  --batch_size 64 \
  --num_epoches 150 \
  --num_sampling 10 \
  --seed 999 \
  --use_gpu True
```

### 4. Inference and visualization

Loads a trained MCDO checkpoint and runs inference on the test set. Generates scatter plots that highlight high- and low-confidence samples.

```bash
python inference_solubility_r.py \
  --job_title MCDO \
  --model_type gcn \
  --hidden_dim 128 \
  --num_layers 4 \
  --readout pma \
  --out_dim 2 \
  --num_sampling 10 \
  --dataset_name Solubility \
  --split_method scaffold \
  --seed 999 \
  --tot_unc_threshold 1.0 \
  --use_gpu True
```

### 5. Attention visualization on custom molecules

Runs inference on a user-supplied CSV of SMILES strings and exports atom-level PMA attention scores as molecular images. Requires a model trained with `--readout pma`.

```bash
python attention_visualization.py \
  --title my_molecules \
  --csv_path path/to/molecules.csv \
  --smi_column SMILES \
  --job_title MCDO \
  --model_type gcn \
  --hidden_dim 128 \
  --num_layers 4 \
  --readout pma \
  --out_dim 2 \
  --num_sampling 30 \
  --seed 999 \
  --use_gpu True
```

### 6. Aggregate results across seeds

Parses training logs from multiple seeds (999, 888, 777, 666) and models (GCN, GIN, GAT) and produces bar charts with error bars.

```bash
python summarize_regression.py
```

---

## Key Hyperparameters

| Flag | Options | Default | Description |
|------|---------|---------|-------------|
| `--model_type` | `gcn` `gin` `gine` `gat` | varies | GNN layer type |
| `--readout` | `sum` `mean` `max` `pma` | varies | Graph readout method |
| `--hidden_dim` | int | `128` | Hidden layer width |
| `--num_layers` | int | `4` | Number of GNN layers |
| `--dropout_prob` | float | `0.0` / `0.2` | Dropout probability |
| `--num_sampling` | int | `10` | MC-Dropout inference passes |
| `--split_method` | `random` `scaffold` | `scaffold` | Dataset split strategy |
| `--seed` | int | `999` | Random seed |
| `--use_gpu` | `True` `False` | `True` | Enable GPU |
| `--gpu_idx` | str | `'1'` | GPU device index |

---

## Expected Outputs

| Script | Output location | Contents |
|--------|----------------|----------|
| `gnn_regression.py` | `logs/`, `save/` | Training log, model checkpoint |
| `gnn_classification.py` | `logs/`, `save/` | Training log, SWA checkpoint |
| `gnn_regression_mcdo.py` | `logs/`, `save/` | Training log, MCDO checkpoint |
| `inference_solubility_r.py` | `figures/` | Scatter plots with uncertainty highlights |
| `attention_visualization.py` | `figures/`, `results/` | Attention map images, predictions CSV |
| `summarize_regression.py` | `figures/` | Bar charts comparing models and seeds |

Checkpoint naming pattern: `save/[job_title]_[model_type]_[hidden_dim]_[readout]_[split_method]_[seed].pth`

---

## Reference

This project is based on the following work:

> *Graph Neural Networks and Bayesian Learning for Molecular Property Prediction*
> (see `Graph Neural Networks and Bayesian Learning paper.pdf` in this repository)

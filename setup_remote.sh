#!/bin/bash
# Remote setup for gnn_solubility — Arch Linux + NVIDIA GPU
# Usage: bash setup_remote.sh 2>&1 | tee setup.log
set -e

CONDA_BASE="$HOME/miniconda3"
ENV_NAME="gnn"
PIP="$CONDA_BASE/envs/$ENV_NAME/bin/pip"
PYTHON="$CONDA_BASE/envs/$ENV_NAME/bin/python"

# ---------------------------------------------------------------------------
echo "====== [1/5] Miniconda ======"
curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/miniconda.sh
bash /tmp/miniconda.sh -b -u -p "$CONDA_BASE"
rm /tmp/miniconda.sh
source "$CONDA_BASE/etc/profile.d/conda.sh"
echo "Done."

# ---------------------------------------------------------------------------
echo "====== [2/5] Terms of Service ======"
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
echo "Done."

# ---------------------------------------------------------------------------
echo "====== [3/5] Detect CUDA ======"
CUDA_STR=$(nvidia-smi 2>/dev/null | grep -oP "CUDA Version: \K[0-9]+\.[0-9]+" || echo "0.0")
CUDA_MAJOR=$(echo "$CUDA_STR" | cut -d. -f1)
echo "System CUDA: $CUDA_STR"
if [ "$CUDA_MAJOR" -ge "12" ]; then
    TORCH_CUDA="cu121"
elif [ "$CUDA_MAJOR" -ge "11" ]; then
    TORCH_CUDA="cu118"
else
    echo "ERROR: CUDA not found or too old."; exit 1
fi
echo "Using PyTorch build: $TORCH_CUDA"

# ---------------------------------------------------------------------------
echo "====== [4/5] Create env '$ENV_NAME' ======"
# Always start fresh
conda env remove -n "$ENV_NAME" -y 2>/dev/null || true
conda create -y -n "$ENV_NAME" python=3.10

# numpy before torch — avoids ABI mismatch
$PIP install --quiet "setuptools<81" "numpy<2.0"

# PyTorch 2.2.1 (pip bundles its own CUDA 12.1 runtime inside torch/lib/)
$PIP install --quiet \
    torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 \
    --index-url "https://download.pytorch.org/whl/${TORCH_CUDA}"

$PIP install --quiet torchdata==0.7.1

# DGL 2.1.0 — graphbolt needs libnvrtc.so.12; install it from the nvidia channel
# so it lands in the env's lib/ directory (no system CUDA toolkit required)
$PIP install --quiet dgl==2.1.0 \
    -f "https://data.dgl.ai/wheels/${TORCH_CUDA}/repo.html"

"$CONDA_BASE/bin/conda" install -y -n "$ENV_NAME" -c nvidia cuda-nvrtc=12.1

$PIP install --quiet pyyaml pydantic rdkit scikit-learn pandas matplotlib

# Make the env's lib/ (which now contains libnvrtc.so.12) visible at runtime
mkdir -p "$CONDA_BASE/envs/$ENV_NAME/etc/conda/activate.d"
cat > "$CONDA_BASE/envs/$ENV_NAME/etc/conda/activate.d/cuda_libs.sh" <<EOF
export LD_LIBRARY_PATH="$CONDA_BASE/envs/$ENV_NAME/lib:\$LD_LIBRARY_PATH"
EOF
echo "LD_LIBRARY_PATH activation script written."

# ---------------------------------------------------------------------------
echo "====== [5/5] Verify ======"
# conda run activates the env (runs activate.d scripts), so LD_LIBRARY_PATH is set
conda run -n "$ENV_NAME" python - <<'PYEOF'
import torch, dgl, rdkit, numpy as np
print(f"torch   {torch.__version__}  |  CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU     {torch.cuda.get_device_name(0)}")
print(f"dgl     {dgl.__version__}")
print(f"rdkit   {rdkit.__version__}")
print(f"numpy   {np.__version__}")
print("\nAll OK — environment is ready.")
PYEOF

echo ""
echo "Setup complete! Now sync the project and run experiments."

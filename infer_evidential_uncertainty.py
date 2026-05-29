"""
Evidential uncertainty inference script.

Loads the best checkpoint for a given job, runs the full test set,
and produces three plots per model:

  1. uncertainty_hist.png     — histogram of aleatoric vs epistemic per molecule
  2. uncertainty_scatter.png  — predicted vs true solubility, coloured by epistemic
  3. uncertainty_retention.png — RMSE as uncertain molecules are discarded

Usage:
    python infer_evidential_uncertainty.py \
        --job_title gcn_evidential_pma_final \
        --model_type gcn \
        --readout pma \
        --seed 999 \
        --data_seed 999 \
        --output_dir figures/uncertainty_analysis/gcn_pma \
        --use_gpu True --gpu_idx 0
"""

import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from libs.io_utils import get_dataset, MyDataset, gnn_collate_fn
from libs.models import MyModel
from libs.gps_model import GPSModel
from libs.evidential_utils import evidential_regression_loss, nig_uncertainty
from libs.utils import str2bool, set_seed, set_device


def load_model(args, device):
    if args.model_type == 'gps':
        model = GPSModel(
            num_layers=4, hidden_dim=128, num_heads=4,
            dropout_prob=0.0, out_dim=4, readout=args.readout,
            local_mp_type='gin', rwse_k=16,
        )
    else:
        model = MyModel(
            model_type=args.model_type, num_layers=4, hidden_dim=128,
            dropout_prob=0.0, out_dim=4, readout=args.readout,
        )

    ckpt_path = os.path.join(
        'save',
        f"best_{args.job_title}_{args.model_type}_{args.readout}_{args.data_seed}_s{args.seed}.pth"
    )
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)
    model.eval()
    print(f"Loaded checkpoint: {ckpt_path}  (epoch {ckpt['epoch']+1})")
    return model


def run_inference(model, test_loader, device, evidential_coeff):
    y_true, y_pred = [], []
    aleatoric_list, epistemic_list = [], []

    with torch.no_grad():
        for batch in test_loader:
            graph = batch[0].to(device)
            y     = batch[1].to(device).float()

            pred_raw, _ = model(graph, training=False)
            _, gamma, nu, alpha, beta = evidential_regression_loss(
                pred_raw, y, coeff=evidential_coeff
            )
            ale, epi = nig_uncertainty(nu, alpha, beta)

            y_true.append(y.cpu())
            y_pred.append(gamma.cpu())
            aleatoric_list.append(ale.cpu())
            epistemic_list.append(epi.cpu())

    return (
        torch.cat(y_true).numpy(),
        torch.cat(y_pred).numpy(),
        torch.cat(aleatoric_list).numpy(),
        torch.cat(epistemic_list).numpy(),
    )


# ---------------------------------------------------------------------------
# Plot 1 — Histogram: aleatoric vs epistemic
# ---------------------------------------------------------------------------
def plot_histogram(aleatoric, epistemic, output_dir, title):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for ax, values, label, color in [
        (axes[0], aleatoric, 'Aleatoric', 'tab:blue'),
        (axes[1], epistemic, 'Epistemic', 'tab:orange'),
    ]:
        # Clip extreme outliers for readability
        p99 = np.percentile(values, 99)
        clipped = values[values <= p99]
        ax.hist(clipped, bins=50, color=color, alpha=0.8, edgecolor='white')
        ax.axvline(np.median(clipped), color='black', linestyle='--',
                   label=f'median={np.median(clipped):.3f}')
        ax.set_xlabel(f'{label} uncertainty', fontsize=11)
        ax.set_ylabel('Count', fontsize=11)
        ax.set_title(label, fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f'{title} — Uncertainty distribution (test set)', fontsize=13, fontweight='bold')
    plt.tight_layout()
    out = os.path.join(output_dir, 'uncertainty_hist.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Plot 2 — Scatter: predicted vs true, coloured by aleatoric and epistemic
# ---------------------------------------------------------------------------
def plot_scatter_full(y_true, y_pred, aleatoric, epistemic, output_dir, title):
    """Two panels: one coloured by aleatoric, one by epistemic."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, unc, unc_label in [
        (axes[0], aleatoric, 'Aleatoric uncertainty'),
        (axes[1], epistemic, 'Epistemic uncertainty'),
    ]:
        p2, p98 = np.percentile(unc, 2), np.percentile(unc, 98)
        sc = ax.scatter(y_true, y_pred, c=unc, cmap='RdYlGn_r', s=8, alpha=0.7,
                        vmin=p2, vmax=p98)
        plt.colorbar(sc, ax=ax, label=unc_label)
        lim = [min(y_true.min(), y_pred.min()) - 0.5,
               max(y_true.max(), y_pred.max()) + 0.5]
        ax.plot(lim, lim, 'k--', linewidth=1)
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_xlabel('True solubility', fontsize=11)
        ax.set_ylabel('Predicted solubility', fontsize=11)
        ax.set_title(f'Coloured by {unc_label}', fontsize=11)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f'{title} — Predicted vs True solubility', fontsize=13, fontweight='bold')
    plt.tight_layout()
    out = os.path.join(output_dir, 'uncertainty_scatter.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Plot 3 — Retention curve: RMSE as uncertain molecules are discarded
# ---------------------------------------------------------------------------
def plot_retention(y_true, y_pred, aleatoric, epistemic, output_dir, title):
    total = aleatoric + epistemic
    fractions = np.linspace(0.1, 1.0, 19)

    def rmse_at_fraction(uncertainty, frac):
        n = int(len(uncertainty) * frac)
        idx = np.argsort(uncertainty)[:n]  # keep lowest-uncertainty molecules
        return np.sqrt(np.mean((y_true[idx] - y_pred[idx]) ** 2))

    fig, ax = plt.subplots(figsize=(8, 5))

    for unc, label, color in [
        (aleatoric, 'Sorted by aleatoric', 'tab:blue'),
        (epistemic, 'Sorted by epistemic', 'tab:orange'),
        (total,     'Sorted by total',     'tab:green'),
    ]:
        rmses = [rmse_at_fraction(unc, f) for f in fractions]
        ax.plot(fractions * 100, rmses, marker='o', markersize=4,
                label=label, color=color, linewidth=2)

    # Random baseline
    rng = np.random.default_rng(42)
    random_rmses = []
    for f in fractions:
        n = int(len(y_true) * f)
        idx = rng.choice(len(y_true), n, replace=False)
        random_rmses.append(np.sqrt(np.mean((y_true[idx] - y_pred[idx]) ** 2)))
    ax.plot(fractions * 100, random_rmses, 'k--', linewidth=1.5,
            label='Random baseline')

    ax.set_xlabel('% of test set retained (lowest uncertainty first)', fontsize=11)
    ax.set_ylabel('RMSE', fontsize=11)
    ax.set_title('Uncertainty retention curve', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=13, fontweight='bold')
    plt.tight_layout()
    out = os.path.join(output_dir, 'uncertainty_retention.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(args):
    set_seed(args.seed)
    device = set_device(args.use_gpu, args.gpu_idx)
    os.makedirs(args.output_dir, exist_ok=True)

    _, _, test_set = get_dataset(
        name='Solubility', method='scaffold', data_seed=args.data_seed
    )
    test_loader = DataLoader(
        MyDataset(test_set), batch_size=64, shuffle=False,
        num_workers=0, collate_fn=gnn_collate_fn
    )

    model = load_model(args, device)

    print("Running inference on test set...")
    y_true, y_pred, aleatoric, epistemic = run_inference(
        model, test_loader, device, args.evidential_coeff
    )

    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    print(f"Test RMSE: {rmse:.4f}")
    print(f"Aleatoric — mean: {aleatoric.mean():.4f}  median: {np.median(aleatoric):.4f}")
    print(f"Epistemic — mean: {epistemic.mean():.4f}  median: {np.median(epistemic):.4f}")
    print(f"Aleatoric / Epistemic ratio: {aleatoric.mean() / epistemic.mean():.2f}")

    title = f"{args.job_title} (seed {args.seed})"
    plot_histogram(aleatoric, epistemic, args.output_dir, title)
    plot_scatter_full(y_true, y_pred, aleatoric, epistemic, args.output_dir, title)
    plot_retention(y_true, y_pred, aleatoric, epistemic, args.output_dir, title)

    print(f"\nAll plots saved to: {args.output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--job_title',        type=str,      required=True)
    parser.add_argument('--model_type',       type=str,      default='gcn')
    parser.add_argument('--readout',          type=str,      default='pma')
    parser.add_argument('--seed',             type=int,      default=999)
    parser.add_argument('--data_seed',        type=int,      default=999)
    parser.add_argument('--evidential_coeff', type=float,    default=0.00023388150738309113)
    parser.add_argument('--use_gpu',          type=str2bool, default=True)
    parser.add_argument('--gpu_idx',          type=str,      default='0')
    parser.add_argument('--output_dir',       type=str,      default='figures/uncertainty_analysis')
    args = parser.parse_args()
    main(args)

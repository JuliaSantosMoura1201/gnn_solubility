"""
Compare aleatoric vs epistemic uncertainty between GCN and GPS (PMA + Evidential).
Aggregates all 4 seeds and produces two comparison plots:

  1. uncertainty_comparison_hist.png  — overlaid histograms per uncertainty type
  2. uncertainty_comparison_box.png   — boxplots of aleatoric and epistemic per model
"""

import os
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from libs.io_utils import get_dataset, MyDataset, gnn_collate_fn
from libs.models import MyModel
from libs.gps_model import GPSModel
from libs.evidential_utils import evidential_regression_loss, nig_uncertainty
from libs.utils import set_seed, set_device

SEEDS        = [999, 888, 777, 666]
COEFF        = 0.00023388150738309113
OUTPUT_DIR   = 'figures/uncertainty_analysis'
MODELS = {
    'GCN': dict(job_title='gcn_evidential_pma_final', model_type='gcn', readout='pma'),
    'GPS': dict(job_title='gps_evidential_pma_final', model_type='gps', readout='pma'),
}


def load_model(cfg, seed, device):
    if cfg['model_type'] == 'gps':
        model = GPSModel(num_layers=4, hidden_dim=128, num_heads=4,
                         dropout_prob=0.0, out_dim=4, readout=cfg['readout'],
                         local_mp_type='gin', rwse_k=16)
    else:
        model = MyModel(model_type=cfg['model_type'], num_layers=4, hidden_dim=128,
                        dropout_prob=0.0, out_dim=4, readout=cfg['readout'])

    ckpt_path = f"save/best_{cfg['job_title']}_{cfg['model_type']}_{cfg['readout']}_{seed}_s{seed}.pth"
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device).eval()
    return model


def collect_uncertainty(model, test_loader, device):
    ale_list, epi_list = [], []
    with torch.no_grad():
        for batch in test_loader:
            graph = batch[0].to(device)
            y     = batch[1].to(device).float()
            pred_raw, _ = model(graph, training=False)
            _, _, nu, alpha, beta = evidential_regression_loss(pred_raw, y, coeff=COEFF)
            ale, epi = nig_uncertainty(nu, alpha, beta)
            ale_list.append(ale.cpu())
            epi_list.append(epi.cpu())
    return torch.cat(ale_list).numpy(), torch.cat(epi_list).numpy()


# Collect uncertainty values across all seeds for each model
device = set_device(True, '0')
results = {}

for name, cfg in MODELS.items():
    all_ale, all_epi = [], []
    for seed in SEEDS:
        set_seed(seed)
        _, _, test_set = get_dataset('Solubility', 'scaffold', seed)
        test_loader = DataLoader(MyDataset(test_set), batch_size=64,
                                 shuffle=False, num_workers=0,
                                 collate_fn=gnn_collate_fn)
        model = load_model(cfg, seed, device)
        ale, epi = collect_uncertainty(model, test_loader, device)
        all_ale.append(ale)
        all_epi.append(epi)
        print(f"{name} seed {seed} — aleatoric: {ale.mean():.4f}  epistemic: {epi.mean():.4f}")

    results[name] = {
        'aleatoric': np.concatenate(all_ale),
        'epistemic': np.concatenate(all_epi),
    }

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Plot 1 — Overlaid histograms: GCN vs GPS, aleatoric and epistemic
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

COLORS = {'GCN': 'tab:blue', 'GPS': 'tab:orange'}

for ax, unc_key, xlabel in [
    (axes[0], 'aleatoric', 'Aleatoric uncertainty'),
    (axes[1], 'epistemic', 'Epistemic uncertainty'),
]:
    for name, color in COLORS.items():
        values = results[name][unc_key]
        p99 = np.percentile(values, 99)
        clipped = values[values <= p99]
        ax.hist(clipped, bins=60, color=color, alpha=0.5,
                label=f"{name} (median={np.median(clipped):.3f})",
                edgecolor='none', density=True)
        ax.axvline(np.median(clipped), color=color, linestyle='--', linewidth=1.5)

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title(xlabel, fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

fig.suptitle('GCN vs GPS — Aleatoric & Epistemic uncertainty (PMA + Evidential, 4 seeds)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
out = os.path.join(OUTPUT_DIR, 'uncertainty_comparison_hist.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Plot 2 — Boxplots: aleatoric and epistemic side by side per model
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

for ax, unc_key, ylabel in [
    (axes[0], 'aleatoric', 'Aleatoric uncertainty'),
    (axes[1], 'epistemic', 'Epistemic uncertainty'),
]:
    data  = [results[name][unc_key] for name in MODELS]
    # Clip outliers for readability
    data  = [d[d <= np.percentile(d, 99)] for d in data]
    bp = ax.boxplot(data, patch_artist=True, widths=0.4,
                    medianprops=dict(color='black', linewidth=2))
    for patch, color in zip(bp['boxes'], COLORS.values()):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(list(MODELS.keys()), fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(ylabel, fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')

fig.suptitle('GCN vs GPS — Uncertainty distribution (PMA + Evidential, 4 seeds)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
out = os.path.join(OUTPUT_DIR, 'uncertainty_comparison_box.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {out}")

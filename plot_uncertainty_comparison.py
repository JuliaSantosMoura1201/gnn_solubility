"""
Two plots for the uncertainty experiments (3rd round / tuned2):

1. NIG-NLL curves — GCN and GPS Evidential (tuned2), valid + test, mean ± std
2. MCDO vs Evidential — RMSE and R² mean ± std for GCN and GPS, both methods
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

LOG_DIR = 'logs'
SEEDS   = [999, 888, 777, 666]


def load_seeds(prefix):
    return [pd.read_csv(f'{LOG_DIR}/{prefix}_seed{s}.csv') for s in SEEDS]


def mean_std(dfs, col):
    min_ep = min(len(df) for df in dfs)
    arr = pd.concat([df[col].iloc[:min_ep].reset_index(drop=True)
                     for df in dfs], axis=1)
    return arr.mean(axis=1), arr.std(axis=1), list(range(1, min_ep + 1))


# =============================================================================
# Plot 1 — NIG-NLL for GCN and GPS (tuned2)
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

nll_configs = [
    ('GCN Evidential (tuned)', 'gcn_evidential_pma_tuned2', 'tab:blue'),
    ('GPS Evidential (tuned)', 'gps_evidential_pma_tuned2', 'tab:orange'),
]

for ax, (split_label, split_col) in zip(axes, [('Validation', 'valid_nll'), ('Test', 'test_nll')]):
    for label, prefix, color in nll_configs:
        dfs = load_seeds(prefix)
        mu, std, epochs = mean_std(dfs, split_col)
        ax.plot(epochs, mu, color=color, label=label, linewidth=2)
        ax.fill_between(epochs, mu - std, mu + std, alpha=0.2, color=color)

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('NIG-NLL', fontsize=12)
    ax.set_title(f'{split_label} NIG-NLL', fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

fig.suptitle('NIG Negative Log-Likelihood — Evidential tuned (mean ± std, 4 seeds)',
             fontsize=14, fontweight='bold')
plt.tight_layout()
out = 'figures/uncertainty/nig_nll_tuned2.png'
os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {out}")


# =============================================================================
# Plot 2 — MCDO vs Evidential (tuned2): RMSE and R² for GCN and GPS
# =============================================================================
configs = [
    ('GCN + MCDO',       'arch_mcdo_gcn_pma',          'tab:blue',   '-'),
    ('GCN + Evidential', 'gcn_evidential_pma_tuned2',   'tab:blue',   '--'),
    ('GPS + MCDO',       'gps_mcdo_pma',                'tab:orange', '-'),
    ('GPS + Evidential', 'gps_evidential_pma_tuned2',   'tab:orange', '--'),
]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, (ylabel, col) in zip(axes, [('Validation RMSE', 'valid_rmse'),
                                     ('Validation R²',   'valid_r2')]):
    for label, prefix, color, ls in configs:
        dfs = load_seeds(prefix)
        mu, std, epochs = mean_std(dfs, col)
        ax.plot(epochs, mu, color=color, linestyle=ls, label=label, linewidth=2)
        ax.fill_between(epochs, mu - std, mu + std, alpha=0.1, color=color)

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(ylabel, fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

fig.suptitle('MCDO vs Evidential (tuned) — GCN and GPS, PMA (mean ± std, 4 seeds)',
             fontsize=14, fontweight='bold')
plt.tight_layout()
out = 'figures/uncertainty/mcdo_vs_evidential_tuned2.png'
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {out}")

"""
Plot GCN vs GIN vs GAT comparison: mean ± std across 4 seeds for MCDO + PMA.
Saves figures/arch_comparison_mcdo_pma.png
"""

import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

LOG_DIR = 'logs'
OUTPUT  = 'figures/arch_comparison_mcdo_pma.png'
SEEDS   = [999, 888, 777, 666]

MODELS = {
    'GCN': 'arch_mcdo_gcn_pma',
    'GIN': 'arch_mcdo_gin_pma',
    'GAT': 'arch_mcdo_gat_pma',
}

COLORS = {
    'GCN': 'tab:blue',
    'GIN': 'tab:orange',
    'GAT': 'tab:green',
}


def load_seeds(prefix):
    dfs = []
    for seed in SEEDS:
        path = os.path.join(LOG_DIR, f"{prefix}_seed{seed}.csv")
        dfs.append(pd.read_csv(path))
    return dfs


def mean_std(dfs, col):
    min_ep = min(len(df) for df in dfs)
    arr = pd.concat([df[col].iloc[:min_ep].reset_index(drop=True)
                     for df in dfs], axis=1)
    return arr.mean(axis=1), arr.std(axis=1), range(1, min_ep + 1)


fig, axes = plt.subplots(1, 2, figsize=(14, 5))

metrics = [
    ('RMSE', 'valid_rmse'),
    ('R²',   'valid_r2'),
]

for ax, (ylabel, col) in zip(axes, metrics):
    for label, prefix in MODELS.items():
        dfs = load_seeds(prefix)
        mu, std, epochs = mean_std(dfs, col)
        color = COLORS[label]
        ax.plot(epochs, mu, color=color, label=label, linewidth=2)
        ax.fill_between(epochs, mu - std, mu + std, alpha=0.2, color=color)

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(f'Validation {ylabel}', fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

fig.suptitle('GCN vs GIN vs GAT — PMA + MCDO (mean ± std, 4 seeds)',
             fontsize=14, fontweight='bold')
plt.tight_layout()
os.makedirs(os.path.dirname(os.path.abspath(OUTPUT)), exist_ok=True)
plt.savefig(OUTPUT, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {OUTPUT}")

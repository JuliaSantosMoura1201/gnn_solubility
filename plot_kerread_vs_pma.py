"""
Plot KerRead vs PMA side-by-side for GCN and GPS backbones.
Each plot: validation RMSE and R² mean ± std across 4 seeds.
Saves:
  figures/kerread_vs_pma/gcn_kerread_vs_pma.png
  figures/kerread_vs_pma/gps_kerread_vs_pma.png
"""

import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

LOG_DIR = 'logs'
SEEDS   = [999, 888, 777, 666]

CONFIGS = {
    'GCN': {
        'PMA':     'arch_mcdo_gcn_pma',
        'KerRead': 'gcn_mcdo_kerread',
        'output':  'figures/kerread_vs_pma/gcn_kerread_vs_pma.png',
        'title':   'GCN — KerRead vs PMA (MCDO, mean ± std, 4 seeds)',
    },
    'GPS': {
        'PMA':     'gps_mcdo_pma',
        'KerRead': 'gps_mcdo_kerread',
        'output':  'figures/kerread_vs_pma/gps_kerread_vs_pma.png',
        'title':   'GPS — KerRead vs PMA (MCDO, mean ± std, 4 seeds)',
    },
}

COLORS = {
    'PMA':     'tab:blue',
    'KerRead': 'tab:orange',
}

METRICS = [
    ('Validation RMSE', 'valid_rmse'),
    ('Validation R²',   'valid_r2'),
]


def load_seeds(prefix):
    return [pd.read_csv(f'{LOG_DIR}/{prefix}_seed{s}.csv') for s in SEEDS]


def mean_std(dfs, col):
    min_ep = min(len(df) for df in dfs)
    arr = pd.concat([df[col].iloc[:min_ep].reset_index(drop=True)
                     for df in dfs], axis=1)
    return arr.mean(axis=1), arr.std(axis=1), range(1, min_ep + 1)


for backbone, cfg in CONFIGS.items():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, (ylabel, col) in zip(axes, METRICS):
        for readout, color in COLORS.items():
            dfs = load_seeds(cfg[readout])
            mu, std, epochs = mean_std(dfs, col)
            ax.plot(epochs, mu, color=color, label=readout, linewidth=2)
            ax.fill_between(epochs, mu - std, mu + std, alpha=0.2, color=color)

        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(ylabel, fontsize=12)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

    fig.suptitle(cfg['title'], fontsize=14, fontweight='bold')
    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(cfg['output'])), exist_ok=True)
    plt.savefig(cfg['output'], dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {cfg['output']}")

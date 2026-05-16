"""
Aggregate MCDO regression results across seeds and plot RMSE / R² bar charts
comparing all four models: GCN, GIN, GAT, GPS.

Expected log files (produced by gnn_regression_mcdo.py / gnn_regression_mcdo_gps.py
with shell redirection: ... | tee logs/MCDO_<model>_pma_scaffold_<seed>.log):

    logs/MCDO_gcn_pma_scaffold_999.log
    logs/MCDO_gcn_pma_scaffold_888.log
    logs/MCDO_gcn_pma_scaffold_777.log
    logs/MCDO_gcn_pma_scaffold_666.log
    logs/MCDO_gin_pma_scaffold_<seed>.log   (same pattern)
    logs/MCDO_gat_pma_scaffold_<seed>.log
    logs/MCDO_gps_pma_scaffold_<seed>.log

Each log must contain lines like:
    End of  150 -th epoch MSE: x  x  x  RMSE: x  x  x  R2: x  x  x
"""

import os
import numpy as np
import matplotlib.pyplot as plt


SEED_LIST  = [999, 888, 777, 666]
MODEL_LIST = ['gcn', 'gin', 'gat', 'gps']
LABELS     = ['GCN', 'GIN', 'GAT', 'GPS']
COLORS     = ['steelblue', 'darkorange', 'forestgreen', 'crimson']

# Column indices after splitting the "End of ..." line on whitespace:
# [train_mse, val_mse, test_mse, train_rmse, val_rmse, test_rmse, train_r2, val_r2, test_r2]
IDX_LIST = [6, 7, 8, 10, 11, 12, 14, 15, 16]

METHOD  = 'MCDO'
READOUT = 'pma'


def load_last_epoch(log_path):
    """Parse log file and return the metric values from the final epoch."""
    with open(log_path) as f:
        lines = f.readlines()
    epoch_lines = [l for l in lines if l.startswith('End of')]
    if not epoch_lines:
        raise ValueError(f"No 'End of' lines found in {log_path}")
    last = epoch_lines[-1].split()
    return [float(last[i]) for i in IDX_LIST]


def collect_model_stats(model):
    """Load results for all seeds; return (mean, std) arrays of shape (9,).

    Returns None if no log files are found for this model.
    """
    seed_results = []
    for seed in SEED_LIST:
        path = f'./logs/{METHOD}_{model}_{READOUT}_scaffold_{seed}.log'
        if not os.path.exists(path):
            print(f"  [warn] missing: {path}")
            continue
        try:
            seed_results.append(load_last_epoch(path))
        except Exception as e:
            print(f"  [warn] could not parse {path}: {e}")

    if not seed_results:
        return None, None

    arr = np.array(seed_results)          # (n_seeds, 9)
    return np.mean(arr, axis=0), np.std(arr, axis=0)


def make_bar_chart(means, stds, metric_indices, ylabel, ylim, yticks, filename, title):
    """Draw a grouped bar chart with one group per split (Train/Valid/Test)."""
    n_models = len(MODEL_LIST)
    width = 0.18
    offsets = np.linspace(-(n_models - 1) * width / 2,
                          (n_models - 1) * width / 2,
                          n_models)
    x = np.arange(3)   # Train, Valid, Test

    fig, ax = plt.subplots()
    ax.set_title(title, fontsize=17)

    for i, (model, label, color) in enumerate(zip(MODEL_LIST, LABELS, COLORS)):
        if means[i] is None:
            continue
        values = means[i][metric_indices]
        errors = stds[i][metric_indices]
        ax.bar(x + offsets[i], values, yerr=errors,
               color=color, width=width, alpha=0.7,
               label=label, capsize=4)

    ax.set_xticks(x)
    ax.set_xticklabels(['Train', 'Valid', 'Test'], fontsize=15)
    ax.set_yticks(yticks)
    ax.set_yticklabels([str(t) for t in yticks], fontsize=14)
    ax.set_ylim(*ylim)
    ax.set_ylabel(ylabel, fontsize=16)
    ax.legend(fontsize=13)
    plt.tight_layout()
    plt.savefig(filename)
    print(f"Saved: {filename}")
    plt.close()


def main():
    all_means = []
    all_stds  = []

    print("Loading logs...")
    for model in MODEL_LIST:
        mean, std = collect_model_stats(model)
        all_means.append(mean)
        all_stds.append(std)
        if mean is not None:
            # indices 3,4,5 = train/valid/test RMSE; 6,7,8 = R2
            print(f"  {model.upper():4s}  RMSE test={mean[5]:.3f}±{std[5]:.3f}   R² test={mean[8]:.3f}±{std[8]:.3f}")
        else:
            print(f"  {model.upper():4s}  no data found")

    title = f'{METHOD}, {READOUT.upper()}'

    # RMSE chart  — indices 3,4,5 in the 9-element vector
    make_bar_chart(
        means=all_means,
        stds=all_stds,
        metric_indices=[3, 4, 5],
        ylabel='RMSE',
        ylim=(0.0, 1.65),
        yticks=[0.0, 0.4, 0.8, 1.2, 1.6],
        filename=f'figures/RMSE_{METHOD}_{READOUT}_all.png',
        title=title,
    )

    # R² chart — indices 6,7,8
    make_bar_chart(
        means=all_means,
        stds=all_stds,
        metric_indices=[6, 7, 8],
        ylabel='R²',
        ylim=(0.0, 1.0),
        yticks=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        filename=f'figures/R2_{METHOD}_{READOUT}_all.png',
        title=title,
    )


if __name__ == '__main__':
    main()

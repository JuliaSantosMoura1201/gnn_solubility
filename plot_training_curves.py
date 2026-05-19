"""
Plot training/validation/test curves from per-epoch CSV logs.

Usage:
    python plot_training_curves.py --log_dir logs --pattern "paper_mcdo_gcn_pma*" \
                                   --output_dir figures/paper --title "GCN+PMA+MCDO paper"
"""

import argparse
import glob
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd


def load_group(csv_files):
    dfs = {}
    for f in sorted(csv_files):
        base = os.path.splitext(os.path.basename(f))[0]
        seed = base.rsplit('_seed', 1)[-1] if '_seed' in base else base
        try:
            dfs[seed] = pd.read_csv(f)
        except Exception as e:
            print(f"  Warning: could not read {f}: {e}")
    return dfs


def is_regression(df):
    return 'train_rmse' in df.columns


def plot_group(dfs, title, output_path):
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red',
              'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray']

    sample_df = next(iter(dfs.values()))
    regression = is_regression(sample_df)

    if regression:
        metrics = [('RMSE', 'train_rmse', 'valid_rmse', 'test_rmse'),
                   ('R²',   'train_r2',   'valid_r2',   'test_r2')]
    else:
        metrics = [('Accuracy', 'train_acc', 'valid_acc', 'test_acc'),
                   ('Loss',     'train_loss','valid_loss','test_loss')]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, (ylabel, train_col, valid_col, test_col) in zip(axes, metrics):
        for i, (seed, df) in enumerate(sorted(dfs.items())):
            c = colors[i % len(colors)]
            epochs = df['epoch']
            ax.plot(epochs, df[train_col], color=c, linestyle='-',
                    alpha=0.85, label=f'seed {seed} train')
            ax.plot(epochs, df[valid_col], color=c, linestyle='--',
                    alpha=0.85, label=f'seed {seed} valid')
            if test_col in df.columns:
                ax.plot(epochs, df[test_col], color=c, linestyle=':',
                        alpha=0.5, label=f'seed {seed} test')

        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=13, fontweight='bold')
    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_mean_std(dfs, title, output_path):
    """Plot mean ± std across seeds for each split."""
    sample_df = next(iter(dfs.values()))
    regression = is_regression(sample_df)

    if regression:
        pairs = [('RMSE', 'train_rmse', 'valid_rmse'),
                 ('R²',   'train_r2',   'valid_r2')]
    else:
        pairs = [('Accuracy', 'train_acc', 'valid_acc'),
                 ('Loss',     'train_loss','valid_loss')]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, (ylabel, train_col, valid_col) in zip(axes, pairs):
        all_dfs = list(dfs.values())
        min_ep = min(len(df) for df in all_dfs)
        train_arr = pd.concat([df[train_col].iloc[:min_ep].reset_index(drop=True)
                                for df in all_dfs], axis=1)
        valid_arr = pd.concat([df[valid_col].iloc[:min_ep].reset_index(drop=True)
                                for df in all_dfs], axis=1)
        epochs = range(1, min_ep + 1)

        for arr, label, color in [(train_arr, 'train', 'tab:blue'),
                                   (valid_arr, 'valid', 'tab:orange')]:
            mu  = arr.mean(axis=1)
            std = arr.std(axis=1)
            ax.plot(epochs, mu, color=color, label=label)
            ax.fill_between(epochs, mu - std, mu + std, alpha=0.2, color=color)

        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f'{title} — mean ± std over {len(dfs)} seeds', fontsize=13, fontweight='bold')
    plt.tight_layout()
    mean_path = output_path.replace('.png', '_mean_std.png')
    os.makedirs(os.path.dirname(os.path.abspath(mean_path)), exist_ok=True)
    plt.savefig(mean_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {mean_path}")


def main(args):
    pattern = os.path.join(args.log_dir, args.pattern)
    csv_files = glob.glob(pattern)

    if not csv_files:
        print(f"No CSV files matched: {pattern}")
        return

    # Auto-group by prefix (everything before '_seed')
    groups = {}
    for f in csv_files:
        base = os.path.splitext(os.path.basename(f))[0]
        prefix = base.rsplit('_seed', 1)[0] if '_seed' in base else base
        groups.setdefault(prefix, []).append(f)

    os.makedirs(args.output_dir, exist_ok=True)

    for prefix, files in sorted(groups.items()):
        print(f"\nGroup: {prefix} ({len(files)} files)")
        dfs = load_group(files)
        if not dfs:
            continue

        title = args.title if args.title else prefix.replace('_', ' ')
        out_path = os.path.join(args.output_dir, f"{prefix}.png")

        plot_group(dfs, title=title, output_path=out_path)

        if len(dfs) >= 2:
            plot_mean_std(dfs, title=title, output_path=out_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir',    type=str, default='logs',
                        help='Directory containing CSV log files')
    parser.add_argument('--pattern',   type=str, default='*.csv',
                        help='Glob pattern relative to log_dir to select CSVs')
    parser.add_argument('--output_dir',type=str, default='figures',
                        help='Directory to save PNG plots')
    parser.add_argument('--title',     type=str, default='',
                        help='Plot title (auto-derived from prefix if blank)')
    args = parser.parse_args()
    main(args)

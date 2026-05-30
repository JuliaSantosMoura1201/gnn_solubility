"""
Compare retention curves between GCN and GPS (PMA + Evidential).
Aggregates all 4 seeds and plots RMSE vs fraction of test set retained
(sorted by aleatoric, epistemic, and total uncertainty) with random baseline.
Saves figures/uncertainty_analysis/retention_comparison.png
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

SEEDS      = [999, 888, 777, 666]
COEFF      = 0.00023388150738309113
OUTPUT_DIR = 'figures/uncertainty_analysis'
FRACTIONS  = np.linspace(0.1, 1.0, 19)

MODELS = {
    'GCN': dict(job_title='gcn_evidential_pma_final', model_type='gcn', readout='pma'),
    'GPS': dict(job_title='gps_evidential_pma_final', model_type='gps', readout='pma'),
}
COLORS = {'GCN': 'tab:blue', 'GPS': 'tab:orange'}


def load_model(cfg, seed, device):
    if cfg['model_type'] == 'gps':
        model = GPSModel(num_layers=4, hidden_dim=128, num_heads=4,
                         dropout_prob=0.0, out_dim=4, readout=cfg['readout'],
                         local_mp_type='gin', rwse_k=16)
    else:
        model = MyModel(model_type=cfg['model_type'], num_layers=4, hidden_dim=128,
                        dropout_prob=0.0, out_dim=4, readout=cfg['readout'])
    ckpt = torch.load(
        f"save/best_{cfg['job_title']}_{cfg['model_type']}_{cfg['readout']}_{seed}_s{seed}.pth",
        map_location=device
    )
    model.load_state_dict(ckpt['model_state_dict'])
    return model.to(device).eval()


def collect(model, test_loader, device):
    y_true, y_pred, ale_list, epi_list = [], [], [], []
    with torch.no_grad():
        for batch in test_loader:
            graph = batch[0].to(device)
            y     = batch[1].to(device).float()
            pred_raw, _ = model(graph, training=False)
            _, gamma, nu, alpha, beta = evidential_regression_loss(pred_raw, y, coeff=COEFF)
            ale, epi = nig_uncertainty(nu, alpha, beta)
            y_true.append(y.cpu()); y_pred.append(gamma.cpu())
            ale_list.append(ale.cpu()); epi_list.append(epi.cpu())
    return (torch.cat(y_true).numpy(), torch.cat(y_pred).numpy(),
            torch.cat(ale_list).numpy(), torch.cat(epi_list).numpy())


def rmse_retained(y_true, y_pred, uncertainty, frac):
    n   = max(1, int(len(uncertainty) * frac))
    idx = np.argsort(uncertainty)[:n]
    return np.sqrt(np.mean((y_true[idx] - y_pred[idx]) ** 2))


# Collect data
device  = set_device(True, '0')
results = {}

for name, cfg in MODELS.items():
    all_y_true, all_y_pred, all_ale, all_epi = [], [], [], []
    for seed in SEEDS:
        set_seed(seed)
        _, _, test_set = get_dataset('Solubility', 'scaffold', seed)
        loader = DataLoader(MyDataset(test_set), batch_size=64,
                            shuffle=False, num_workers=0, collate_fn=gnn_collate_fn)
        model = load_model(cfg, seed, device)
        yt, yp, ale, epi = collect(model, loader, device)
        all_y_true.append(yt); all_y_pred.append(yp)
        all_ale.append(ale);   all_epi.append(epi)
        print(f"{name} seed {seed} done")

    results[name] = dict(
        y_true = np.concatenate(all_y_true),
        y_pred = np.concatenate(all_y_pred),
        ale    = np.concatenate(all_ale),
        epi    = np.concatenate(all_epi),
    )


# Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, name in zip(axes, MODELS):
    r      = results[name]
    color  = COLORS[name]
    total  = r['ale'] + r['epi']

    for unc, label, ls in [
        (r['ale'], 'Aleatoric', '-'),
        (r['epi'], 'Epistemic', '--'),
        (total,    'Total',     ':'),
    ]:
        rmses = [rmse_retained(r['y_true'], r['y_pred'], unc, f) for f in FRACTIONS]
        ax.plot(FRACTIONS * 100, rmses, color=color, linestyle=ls,
                linewidth=2, marker='o', markersize=4, label=label)

    # Random baseline
    rng = np.random.default_rng(42)
    rand_rmses = []
    for f in FRACTIONS:
        n   = max(1, int(len(r['y_true']) * f))
        idx = rng.choice(len(r['y_true']), n, replace=False)
        rand_rmses.append(np.sqrt(np.mean((r['y_true'][idx] - r['y_pred'][idx]) ** 2)))
    ax.plot(FRACTIONS * 100, rand_rmses, 'k--', linewidth=1.5, label='Random baseline')

    ax.set_xlabel('% of test set retained (lowest uncertainty first)', fontsize=11)
    ax.set_ylabel('RMSE', fontsize=11)
    ax.set_title(f'{name} + PMA + Evidential', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

fig.suptitle('Retention curves — GCN vs GPS (PMA + Evidential, 4 seeds aggregated)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
os.makedirs(OUTPUT_DIR, exist_ok=True)
out = os.path.join(OUTPUT_DIR, 'retention_comparison.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {out}")

"""
Scatter plot: predicted vs true solubility coloured by aleatoric / epistemic
uncertainty for GCN and GPS (PMA + Evidential, 4 seeds aggregated).

Produces figures/uncertainty_analysis/scatter_comparison.png — a 2×2 grid:
  row 0 → GCN   (aleatoric | epistemic)
  row 1 → GPS   (aleatoric | epistemic)
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
    ckpt = torch.load(
        f"save/best_{cfg['job_title']}_{cfg['model_type']}_{cfg['readout']}_{seed}_s{seed}.pth",
        map_location=device,
    )
    model.load_state_dict(ckpt['model_state_dict'])
    return model.to(device).eval()


def collect(model, loader, device):
    y_true, y_pred, ale_list, epi_list = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            graph = batch[0].to(device)
            y     = batch[1].to(device).float()
            pred_raw, _ = model(graph, training=False)
            _, gamma, nu, alpha, beta = evidential_regression_loss(pred_raw, y, coeff=COEFF)
            ale, epi = nig_uncertainty(nu, alpha, beta)
            y_true.append(y.cpu()); y_pred.append(gamma.cpu())
            ale_list.append(ale.cpu()); epi_list.append(epi.cpu())
    return (torch.cat(y_true).numpy(), torch.cat(y_pred).numpy(),
            torch.cat(ale_list).numpy(), torch.cat(epi_list).numpy())


# ── collect ──────────────────────────────────────────────────────────────────
device  = set_device(True, '0')
results = {}

for name, cfg in MODELS.items():
    yt_all, yp_all, ale_all, epi_all = [], [], [], []
    for seed in SEEDS:
        set_seed(seed)
        _, _, test_set = get_dataset('Solubility', 'scaffold', seed)
        loader = DataLoader(MyDataset(test_set), batch_size=64,
                            shuffle=False, num_workers=0, collate_fn=gnn_collate_fn)
        model = load_model(cfg, seed, device)
        yt, yp, ale, epi = collect(model, loader, device)
        yt_all.append(yt); yp_all.append(yp)
        ale_all.append(ale); epi_all.append(epi)
        print(f"{name} seed {seed} done")

    results[name] = dict(
        y_true = np.concatenate(yt_all),
        y_pred = np.concatenate(yp_all),
        ale    = np.concatenate(ale_all),
        epi    = np.concatenate(epi_all),
    )


# ── plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

for row, name in enumerate(MODELS):
    r = results[name]

    for col, (unc, unc_label) in enumerate([
        (r['ale'], 'Aleatoric uncertainty'),
        (r['epi'], 'Epistemic uncertainty'),
    ]):
        ax = axes[row, col]

        # Clip colour range to [2nd, 98th] percentile so outliers don't wash out the palette
        vmin, vmax = np.percentile(unc, 2), np.percentile(unc, 98)
        sc = ax.scatter(r['y_true'], r['y_pred'],
                        c=unc, cmap='RdYlGn_r', s=5, alpha=0.5,
                        vmin=vmin, vmax=vmax, rasterized=True)
        plt.colorbar(sc, ax=ax, label=unc_label, fraction=0.046, pad=0.04)

        lim = [min(r['y_true'].min(), r['y_pred'].min()) - 0.5,
               max(r['y_true'].max(), r['y_pred'].max()) + 0.5]
        ax.plot(lim, lim, 'k--', linewidth=1, alpha=0.6)
        ax.set_xlim(lim); ax.set_ylim(lim)

        rmse = np.sqrt(np.mean((r['y_true'] - r['y_pred']) ** 2))
        ax.set_xlabel('True solubility (log mol/L)', fontsize=11)
        ax.set_ylabel('Predicted solubility', fontsize=11)
        ax.set_title(f'{name} — coloured by {unc_label}\n(RMSE = {rmse:.3f}, n = {len(r["y_true"])})',
                     fontsize=11)
        ax.grid(True, alpha=0.25)

fig.suptitle('Predicted vs True solubility coloured by uncertainty\n'
             'GCN vs GPS  |  PMA + Evidential  |  4 seeds aggregated',
             fontsize=13, fontweight='bold')
plt.tight_layout()

os.makedirs(OUTPUT_DIR, exist_ok=True)
out = os.path.join(OUTPUT_DIR, 'scatter_comparison.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {out}")

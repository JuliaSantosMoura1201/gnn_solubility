"""
Hyperparameter search for the evidential regression coefficient (lambda / coeff).

Uses Optuna (TPE sampler + MedianPruner) to minimise validation RMSE of
GCN+PMA+NIG-evidential regression on the Solubility scaffold split.

Improvements over the naive approach:
  - Self-contained inline training loop: no CSV / checkpoint I/O during search.
  - Stdout suppressed during trials so the terminal stays readable.
  - MedianPruner kills unpromising trials early (after epoch n_warmup_steps).
  - Optimization-history and coeff-distribution plots saved to --output_dir.
  - Best coeff is then used for a full multi-seed run via gnn_regression_evidential.

Usage:
    python hp_tune_evidential.py \
        --use_gpu True --gpu_idx 0 \
        --n_trials 30 --tune_epochs 50 --final_epochs 150 \
        --log_dir logs --output_dir figures/hp_tune
"""

import argparse
import contextlib
import io
import os
import sys

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))

from libs.io_utils import get_dataset, MyDataset, gnn_collate_fn
from libs.models import MyModel
from libs.evidential_utils import evidential_regression_loss, nig_nll
from libs.utils import set_seed, set_device

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False
    print("WARNING: optuna not installed.  Run:  pip install optuna")
    print("Falling back to grid search over 5 coeff values.")

# Import training function only for the final multi-seed run
import gnn_regression_evidential as _evi_trainer


# ---------------------------------------------------------------------------
# Inline training function (HP search phase — no file I/O)
# ---------------------------------------------------------------------------

def _train_one_trial(coeff, tune_epochs, use_gpu, gpu_idx, seed, data_seed,
                     trial=None):
    """Train for tune_epochs and return the best validation RMSE seen.

    Stdout is suppressed so Optuna progress stays readable.
    If `trial` is given, reports intermediate validation RMSE each epoch
    and raises TrialPruned when Optuna's pruner decides to stop.
    """
    set_seed(seed)
    device = set_device(use_gpu, gpu_idx)

    train_set, valid_set, _ = get_dataset('Solubility', 'scaffold', data_seed)
    train_loader = DataLoader(MyDataset(train_set), batch_size=64,
                              shuffle=True, num_workers=0,
                              collate_fn=gnn_collate_fn)
    valid_loader = DataLoader(MyDataset(valid_set), batch_size=64,
                              shuffle=False, num_workers=0,
                              collate_fn=gnn_collate_fn)

    model = MyModel(model_type='gcn', num_layers=4, hidden_dim=128,
                    readout='pma', dropout_prob=0.0, out_dim=4,
                    norm_features=False).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3,
                                  weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40,
                                                 gamma=0.1)

    best_valid_nll = float('inf')

    with contextlib.redirect_stdout(io.StringIO()):
        for epoch in range(tune_epochs):
            # --- train ---
            model.train()
            for batch in train_loader:
                graph = batch[0].to(device)
                y     = batch[1].to(device).float()
                optimizer.zero_grad()
                pred_raw, _ = model(graph, training=True)
                loss, *_ = evidential_regression_loss(pred_raw, y, coeff=coeff)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            scheduler.step()

            # --- validate on NIG-NLL (not RMSE) ---
            # NLL jointly rewards accurate means AND calibrated uncertainty,
            # so tuning lambda on NLL finds the coefficient that makes the
            # model know when it doesn't know — not just the best mean fit.
            model.eval()
            nll_batches = []
            with torch.no_grad():
                for batch in valid_loader:
                    graph = batch[0].to(device)
                    y     = batch[1].to(device).float()
                    pred_raw, _ = model(graph, training=False)
                    _, gamma, nu, alpha, beta = evidential_regression_loss(
                        pred_raw, y, coeff=coeff)
                    nll_batches.append(nig_nll(y, gamma, nu, alpha, beta).item())

            valid_nll = sum(nll_batches) / len(nll_batches)
            best_valid_nll = min(best_valid_nll, valid_nll)

            if trial is not None:
                trial.report(valid_nll, epoch)
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()

    return best_valid_nll


# ---------------------------------------------------------------------------
# Optuna search
# ---------------------------------------------------------------------------

def _print_trial_result(study, trial):
    """Optuna callback — prints one summary line per completed/pruned trial."""
    coeff = trial.params.get('evidential_coeff', float('nan'))
    if trial.state == optuna.trial.TrialState.COMPLETE:
        marker = "*" if trial.number == study.best_trial.number else " "
        print(f"  {marker} trial {trial.number:3d}  "
              f"coeff={coeff:.5g}  val_nll={trial.value:.4f}", flush=True)
    elif trial.state == optuna.trial.TrialState.PRUNED:
        print(f"    trial {trial.number:3d}  "
              f"coeff={coeff:.5g}  PRUNED", flush=True)


def optuna_search(args):
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=5,   # never prune the first 5 trials
        n_warmup_steps=args.pruner_warmup,  # don't prune before this epoch
        interval_steps=1,
    )
    study = optuna.create_study(
        direction='minimize',
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=pruner,
        study_name='evidential_coeff_search',
    )

    def objective(trial):
        coeff = trial.suggest_float('evidential_coeff', 1e-4, 1.0, log=True)
        return _train_one_trial(
            coeff=coeff,
            tune_epochs=args.tune_epochs,
            use_gpu=args.use_gpu,
            gpu_idx=args.gpu_idx,
            seed=999,
            data_seed=999,
            trial=trial,
        )

    study.optimize(
        objective,
        n_trials=args.n_trials,
        callbacks=[_print_trial_result],
    )
    return study


# ---------------------------------------------------------------------------
# Grid-search fallback (used when Optuna is not installed)
# ---------------------------------------------------------------------------

def grid_search(args):
    candidates = [1e-4, 1e-3, 0.01, 0.1, 1.0]
    results = {}
    for i, coeff in enumerate(candidates):
        print(f"  Grid trial {i+1}/{len(candidates)}  coeff={coeff:.4g}", flush=True)
        rmse = _train_one_trial(
            coeff=coeff,
            tune_epochs=args.tune_epochs,
            use_gpu=args.use_gpu,
            gpu_idx=args.gpu_idx,
            seed=999,
            data_seed=999,
        )
        results[coeff] = rmse
        print(f"    val_rmse={rmse:.4f}")
    best_coeff = min(results, key=results.get)
    return best_coeff


# ---------------------------------------------------------------------------
# Final multi-seed run (uses the full training script for proper logging)
# ---------------------------------------------------------------------------

def _make_base_args(use_gpu, gpu_idx, log_dir, seed, data_seed,
                    num_epoches, evidential_coeff, job_title, patience=0):
    return argparse.Namespace(
        job_title=job_title,
        use_gpu=use_gpu,
        gpu_idx=gpu_idx,
        seed=seed,
        dataset_name='Solubility',
        split_method='scaffold',
        data_seed=data_seed,
        model_type='gcn',
        num_layers=4,
        hidden_dim=128,
        readout='pma',
        dropout_prob=0.0,
        norm_features=False,
        num_epoches=num_epoches,
        num_workers=0,
        batch_size=64,
        lr=1e-3,
        weight_decay=1e-6,
        evidential_coeff=evidential_coeff,
        log_dir=log_dir,
        patience=patience,
    )


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def _save_study_plots(study, output_dir):
    """Save optimization-history and coeff-distribution plots."""
    os.makedirs(output_dir, exist_ok=True)
    try:
        import optuna.visualization.matplotlib as optuna_mpl
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        ax = optuna_mpl.plot_optimization_history(study)
        ax.get_figure().savefig(
            os.path.join(output_dir, 'opt_history.png'), dpi=150,
            bbox_inches='tight')
        plt.close('all')
        print(f"  Saved opt_history.png")

        ax = optuna_mpl.plot_param_importances(study)
        ax.get_figure().savefig(
            os.path.join(output_dir, 'param_importances.png'), dpi=150,
            bbox_inches='tight')
        plt.close('all')
        print(f"  Saved param_importances.png")

        # Scatter: coeff vs val_rmse for completed trials
        trials = [t for t in study.trials
                  if t.state == optuna.trial.TrialState.COMPLETE]
        if trials:
            coeffs = [t.params['evidential_coeff'] for t in trials]
            rmses  = [t.value for t in trials]
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.scatter(coeffs, rmses, s=40, alpha=0.8)
            ax.axvline(study.best_params['evidential_coeff'], color='red',
                       linestyle='--', label=f"best={study.best_params['evidential_coeff']:.4g}")
            ax.set_xscale('log')
            ax.set_xlabel('evidential_coeff (lambda)')
            ax.set_ylabel('best valid RMSE')
            ax.set_title('Optuna: coeff vs. validation RMSE')
            ax.legend()
            fig.tight_layout()
            fig.savefig(os.path.join(output_dir, 'coeff_vs_rmse.png'),
                        dpi=150, bbox_inches='tight')
            plt.close('all')
            print(f"  Saved coeff_vs_rmse.png")

    except Exception as e:
        print(f"  [warn] Could not save plots: {e}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    print(f"\n{'='*60}")
    print("HP Tuning: evidential regression coefficient (lambda / coeff)")
    print(f"  search range : [1e-4, 1.0]  log-uniform")
    print(f"  n_trials     : {args.n_trials}")
    print(f"  tune_epochs  : {args.tune_epochs}  (per trial)")
    print(f"  pruner       : MedianPruner (startup=5, warmup=10 epochs)")
    if not args.skip_final_run:
        print(f"  final_epochs : {args.final_epochs}  (multi-seed GCN+PMA run)")
    else:
        print(f"  skip_final_run=True  (caller handles downstream runs)")
    print(f"{'='*60}\n")

    if HAS_OPTUNA:
        study = optuna_search(args)
        best_coeff = study.best_params['evidential_coeff']
        n_pruned   = sum(1 for t in study.trials
                         if t.state == optuna.trial.TrialState.PRUNED)
        n_complete = sum(1 for t in study.trials
                         if t.state == optuna.trial.TrialState.COMPLETE)
        print(f"\nOptuna finished: {n_complete} complete, {n_pruned} pruned")
        print(f"Best evidential_coeff = {best_coeff:.6g}  "
              f"(val_nll={study.best_value:.4f})")
        _save_study_plots(study, args.output_dir)
    else:
        best_coeff = grid_search(args)
        print(f"\nGrid search best evidential_coeff = {best_coeff:.6g}")

    # Persist best coeff — always written so the shell script can read it
    os.makedirs(args.log_dir, exist_ok=True)
    result_path = os.path.join(args.log_dir, 'best_evidential_coeff.txt')
    with open(result_path, 'w') as f:
        f.write(f"{best_coeff}\n")
    print(f"Best coeff saved to {result_path}")

    if args.skip_final_run:
        print(f"\n{'='*60}")
        print(f"Tuning complete.  Best evidential_coeff = {best_coeff:.6g}")
        print(f"{'='*60}\n")
        return

    # Full run: 4 seeds with the found coeff
    print(f"\nRunning full experiment  coeff={best_coeff:.6g}  "
          f"epochs={args.final_epochs}  seeds=[999, 888, 777, 666] ...")
    for seed in [999, 888, 777, 666]:
        print(f"\n--- seed {seed} ---")
        run_args = _make_base_args(
            use_gpu=args.use_gpu,
            gpu_idx=args.gpu_idx,
            log_dir=args.log_dir,
            seed=seed,
            data_seed=seed,
            num_epoches=args.final_epochs,
            evidential_coeff=best_coeff,
            job_title=f"evi_tuned_coeff{best_coeff:.4g}",
        )
        _evi_trainer.main(run_args)

    print(f"\n{'='*60}")
    print(f"Done.  Best evidential_coeff = {best_coeff:.6g}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='HP tuning for evidential regression lambda')
    parser.add_argument('--use_gpu',      type=lambda x: x.lower() in ('true','1','yes'),
                        default=False)
    parser.add_argument('--gpu_idx',      type=str,  default='0')
    parser.add_argument('--n_trials',     type=int,  default=30,
                        help='Number of Optuna trials')
    parser.add_argument('--tune_epochs',  type=int,  default=50,
                        help='Epochs per HP-search trial')
    parser.add_argument('--final_epochs', type=int,  default=150,
                        help='Epochs for the final multi-seed run')
    parser.add_argument('--log_dir',      type=str,  default='logs',
                        help='Directory for CSV logs from the final run')
    parser.add_argument('--output_dir',      type=str,            default='figures/hp_tune',
                        help='Directory for Optuna visualization plots')
    parser.add_argument('--skip_final_run',  action='store_true',
                        help='Only run HP search; skip the multi-seed final run')
    parser.add_argument('--pruner_warmup',  type=int,  default=40,
                        help='MedianPruner: epochs before pruning starts (match first LR drop)')
    args = parser.parse_args()
    main(args)

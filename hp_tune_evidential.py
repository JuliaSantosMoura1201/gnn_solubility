"""
Hyperparameter search for the evidential regression coefficient (lambda / coeff).

Uses Optuna to minimise validation RMSE of GCN+PMA+NIG-evidential regression
on the Solubility scaffold split (seed 999 for tuning).

After finding the best coeff, runs a full 4-seed experiment with that coeff
for --final_epochs epochs (default: convergence epoch from Phase 0).

Usage:
    python hp_tune_evidential.py \
        --use_gpu False --n_trials 20 --tune_epochs 40 --final_epochs 150 \
        --log_dir logs --output_dir figures/hp_tune
"""

import argparse
import os
import sys

import pandas as pd

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False
    print("WARNING: optuna not installed. Run: pip install optuna")
    print("Falling back to a simple grid search over 5 coeff values.")

# Import training function from evidential script
sys.path.insert(0, os.path.dirname(__file__))
import gnn_regression_evidential as _evi_trainer


def _make_base_args(use_gpu, gpu_idx, log_dir, seed, data_seed,
                    num_epoches, evidential_coeff, job_title, patience=0):
    """Build an argparse.Namespace that matches gnn_regression_evidential.py."""
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


def run_trial(use_gpu, gpu_idx, coeff, tune_epochs, trial_idx, log_dir):
    job = f"hp_trial{trial_idx:03d}"
    args = _make_base_args(
        use_gpu=use_gpu, gpu_idx=gpu_idx,
        log_dir=os.path.join(log_dir, 'hp_tune'),
        seed=999, data_seed=999,
        num_epoches=tune_epochs,
        evidential_coeff=coeff,
        job_title=job,
    )
    _evi_trainer.main(args)

    csv_path = os.path.join(log_dir, 'hp_tune', f"{job}_seed999.csv")
    if not os.path.exists(csv_path):
        return float('inf')
    df = pd.read_csv(csv_path)
    return float(df['valid_rmse'].min())


def optuna_search(args):
    def objective(trial):
        coeff = trial.suggest_float('evidential_coeff', 1e-4, 1.0, log=True)
        val_rmse = run_trial(
            use_gpu=args.use_gpu, gpu_idx=args.gpu_idx,
            coeff=coeff, tune_epochs=args.tune_epochs,
            trial_idx=trial.number,
            log_dir=args.log_dir,
        )
        return val_rmse

    study = optuna.create_study(
        direction='minimize',
        sampler=optuna.samplers.TPESampler(seed=42),
        study_name='evidential_coeff_search',
        storage=None,
    )
    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=False)
    return study.best_params['evidential_coeff']


def grid_search(args):
    import math
    candidates = [1e-4, 1e-3, 0.01, 0.1, 1.0]
    results = {}
    for i, coeff in enumerate(candidates):
        print(f"Grid search coeff={coeff}")
        rmse = run_trial(
            use_gpu=args.use_gpu, gpu_idx=args.gpu_idx,
            coeff=coeff, tune_epochs=args.tune_epochs,
            trial_idx=i, log_dir=args.log_dir,
        )
        results[coeff] = rmse
        print(f"  coeff={coeff:.4g}  valid_rmse={rmse:.4f}")
    best_coeff = min(results, key=results.get)
    return best_coeff


def main(args):
    print(f"\n{'='*60}")
    print("HP Tuning: evidential regression coefficient (lambda)")
    print(f"  n_trials={args.n_trials}, tune_epochs={args.tune_epochs}")
    print(f"  final_epochs={args.final_epochs}")
    print(f"{'='*60}\n")

    if HAS_OPTUNA:
        best_coeff = optuna_search(args)
        method = "Optuna TPE"
    else:
        best_coeff = grid_search(args)
        method = "grid search"

    print(f"\n[{method}] Best evidential_coeff = {best_coeff:.6g}")

    # Save result
    os.makedirs(args.log_dir, exist_ok=True)
    with open(os.path.join(args.log_dir, 'best_evidential_coeff.txt'), 'w') as f:
        f.write(f"{best_coeff}\n")

    # Full run with best coeff, all 4 seeds
    print(f"\nRunning full experiment with best coeff={best_coeff:.6g} "
          f"for {args.final_epochs} epochs, seeds [999,888,777,666]...")
    for seed in [999, 888, 777, 666]:
        print(f"\n--- Seed {seed} ---")
        full_args = _make_base_args(
            use_gpu=args.use_gpu, gpu_idx=args.gpu_idx,
            log_dir=args.log_dir,
            seed=seed, data_seed=seed,
            num_epoches=args.final_epochs,
            evidential_coeff=best_coeff,
            job_title=f"evi_tuned_coeff{best_coeff:.4g}",
        )
        _evi_trainer.main(full_args)

    print(f"\nHP tuning complete. Best coeff={best_coeff:.6g}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_gpu',       type=lambda x: x.lower() in ('true','1','yes'),
                        default=False)
    parser.add_argument('--gpu_idx',       type=str,   default='0')
    parser.add_argument('--n_trials',      type=int,   default=20,
                        help='Number of Optuna trials (ignored for grid search)')
    parser.add_argument('--tune_epochs',   type=int,   default=40,
                        help='Epochs per trial during HP search')
    parser.add_argument('--final_epochs',  type=int,   default=150,
                        help='Epochs for the final multi-seed run after tuning')
    parser.add_argument('--log_dir',       type=str,   default='logs',
                        help='Directory for CSV logs (trial runs go to log_dir/hp_tune/)')
    parser.add_argument('--output_dir',    type=str,   default='figures/hp_tune',
                        help='Directory for plots (passed to plot_training_curves.py)')
    args = parser.parse_args()
    main(args)

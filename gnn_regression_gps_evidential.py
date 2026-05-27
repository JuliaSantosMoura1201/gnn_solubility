import time
import argparse
import os

import torch
from torch.utils.data import DataLoader

from libs.io_utils import get_dataset
from libs.io_utils import MyDataset
from libs.io_utils import gnn_collate_fn

from libs.gps_model import GPSModel

from libs.evidential_utils import evidential_regression_loss
from libs.evidential_utils import nig_uncertainty, nig_nll

from libs.utils import str2bool
from libs.utils import set_seed
from libs.utils import set_device
from libs.utils import evaluate_regression
from libs.utils import EarlyStopping
from libs.utils import open_csv_logger


def main(args):
    set_seed(seed=args.seed)
    device = set_device(use_gpu=args.use_gpu, gpu_idx=args.gpu_idx)

    train_set, valid_set, test_set = get_dataset(
        name=args.dataset_name,
        method=args.split_method,
        data_seed=args.data_seed,
    )

    train_ds = MyDataset(splitted_set=train_set)
    valid_ds = MyDataset(splitted_set=valid_set)
    test_ds  = MyDataset(splitted_set=test_set)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, collate_fn=gnn_collate_fn)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, collate_fn=gnn_collate_fn)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, collate_fn=gnn_collate_fn)

    model = GPSModel(
        num_layers=args.num_layers,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        readout=args.readout,
        dropout_prob=args.dropout_prob,
        out_dim=4,  # NIG: gamma, nu, alpha, beta
        norm_features=args.norm_features,
        local_mp_type=args.local_mp_type,
        rwse_k=args.rwse_k,
    )
    model = model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer=optimizer, step_size=40, gamma=0.1
    )

    os.makedirs(args.log_dir, exist_ok=True)
    log_path = os.path.join(args.log_dir, f"{args.job_title}_seed{args.seed}.csv")
    csv_fh, csv_writer = open_csv_logger(log_path, [
        'epoch', 'train_loss', 'train_rmse', 'train_r2',
        'valid_loss', 'valid_rmse', 'valid_r2', 'valid_nll',
        'test_loss', 'test_rmse', 'test_r2', 'test_nll',
    ])
    early_stop = EarlyStopping(patience=args.patience, mode='min')
    best_valid_rmse = float('inf')
    best_save_path = os.path.join('./save', f"best_{args.job_title}_gps_{args.readout}_{args.data_seed}_s{args.seed}.pth")

    for epoch in range(args.num_epoches):
        # --- Train ---
        model.train()
        num_batches = len(train_loader)
        train_loss = 0.0
        y_list, pred_list = [], []

        for i, batch in enumerate(train_loader):
            st = time.time()
            optimizer.zero_grad()

            graph, y = batch[0].to(device), batch[1].to(device).float()
            pred_raw, _ = model(graph, training=True)

            loss, gamma, nu, alpha, beta = evidential_regression_loss(
                pred_raw, y, coeff=args.evidential_coeff
            )

            if not torch.isfinite(loss):
                print(f"[warn] non-finite loss at epoch {epoch+1} batch {i+1} — skipping")
                continue

            y_list.append(y)
            pred_list.append(gamma.detach())

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.detach().cpu().numpy()

            et = time.time()
            print("Train!!! Epoch:", epoch + 1,
                  "\t Batch:", i + 1, '/', num_batches,
                  "\t Loss:", loss.detach().cpu().numpy(),
                  "\t Time spent:", round(et - st, 2), "(s)")

        scheduler.step()
        if not y_list:
            print(f"[warn] epoch {epoch+1}: all training batches were non-finite — stopping")
            break
        train_loss /= max(len(y_list), 1)
        train_metrics = evaluate_regression(y_list=y_list, pred_list=pred_list)

        # --- Validation & Test ---
        model.eval()
        with torch.no_grad():
            valid_loss = 0.0
            num_batches = len(valid_loader)
            y_list, pred_list = [], []
            valid_nll_batches = []

            for i, batch in enumerate(valid_loader):
                st = time.time()
                graph, y = batch[0].to(device), batch[1].to(device).float()

                pred_raw, _ = model(graph, training=False)
                loss, gamma, nu, alpha, beta = evidential_regression_loss(
                    pred_raw, y, coeff=args.evidential_coeff
                )
                y_list.append(y)
                pred_list.append(gamma)
                valid_loss += loss.cpu().numpy()
                valid_nll_batches.append(nig_nll(y, gamma, nu, alpha, beta).item())

                et = time.time()
                print("Valid!!! Epoch:", epoch + 1,
                      "\t Batch:", i + 1, '/', num_batches,
                      "\t Loss:", loss.cpu().numpy(),
                      "\t Time spent:", round(et - st, 2), "(s)")

            valid_loss /= num_batches
            valid_metrics = evaluate_regression(y_list=y_list, pred_list=pred_list)
            valid_nll_mean = sum(valid_nll_batches) / len(valid_nll_batches)

            test_loss = 0.0
            num_batches = len(test_loader)
            y_list, pred_list = [], []
            ale_list, epi_list = [], []
            test_nll_batches = []

            for i, batch in enumerate(test_loader):
                st = time.time()
                graph, y = batch[0].to(device), batch[1].to(device).float()

                pred_raw, _ = model(graph, training=False)
                loss, gamma, nu, alpha, beta = evidential_regression_loss(
                    pred_raw, y, coeff=args.evidential_coeff
                )
                aleatoric, epistemic = nig_uncertainty(nu, alpha, beta)

                y_list.append(y)
                pred_list.append(gamma)
                ale_list.append(aleatoric)
                epi_list.append(epistemic)
                test_loss += loss.cpu().numpy()
                test_nll_batches.append(nig_nll(y, gamma, nu, alpha, beta).item())

                et = time.time()
                print("Test!!! Epoch:", epoch + 1,
                      "\t Batch:", i + 1, '/', num_batches,
                      "\t Loss:", loss.cpu().numpy(),
                      "\t Time spent:", round(et - st, 2), "(s)")

            test_loss /= num_batches
            test_metrics = evaluate_regression(y_list=y_list, pred_list=pred_list)
            test_nll_mean = sum(test_nll_batches) / len(test_nll_batches)
            ale_mean = torch.cat(ale_list).mean().item()
            epi_mean = torch.cat(epi_list).mean().item()

        print("End of ", epoch + 1, "-th epoch",
              "MSE:",  round(train_metrics[0], 3), "\t", round(valid_metrics[0], 3), "\t", round(test_metrics[0], 3),
              "RMSE:", round(train_metrics[1], 3), "\t", round(valid_metrics[1], 3), "\t", round(test_metrics[1], 3),
              "R2:",   round(train_metrics[2], 3), "\t", round(valid_metrics[2], 3), "\t", round(test_metrics[2], 3),
              "Aleatoric:", round(ale_mean, 4), "Epistemic:", round(epi_mean, 4))

        csv_writer.writerow({
            'epoch': epoch + 1,
            'train_loss': round(float(train_loss), 6), 'train_rmse': round(train_metrics[1], 6), 'train_r2': round(train_metrics[2], 6),
            'valid_loss': round(float(valid_loss), 6), 'valid_rmse': round(valid_metrics[1], 6), 'valid_r2': round(valid_metrics[2], 6), 'valid_nll': round(valid_nll_mean, 6),
            'test_loss':  round(float(test_loss),  6), 'test_rmse':  round(test_metrics[1],  6), 'test_r2':  round(test_metrics[2],  6), 'test_nll': round(test_nll_mean, 6),
        })
        csv_fh.flush()

        if valid_metrics[1] < best_valid_rmse:
            best_valid_rmse = valid_metrics[1]
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()}, best_save_path)

        if early_stop.step(valid_metrics[1]):
            print(f"CONVERGENCE_EPOCH={epoch+1}")
            break

        save_path = (
            f"./save/{args.job_title}_"
            f"gps_{args.hidden_dim}_{args.readout}_"
            f"{args.split_method}_{args.data_seed}_evidential.pth"
        )
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, save_path)

    csv_fh.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--job_title',     type=str,      default='GPS_evidential')
    parser.add_argument('--use_gpu',       type=str2bool, default=True)
    parser.add_argument('--gpu_idx',       type=str,      default='0')
    parser.add_argument('--seed',          type=int,      default=999)

    parser.add_argument('--dataset_name',  type=str,      default='Solubility')
    parser.add_argument('--split_method',  type=str,      default='scaffold')
    parser.add_argument('--data_seed',     type=int,      default=999)

    parser.add_argument('--num_layers',    type=int,      default=4)
    parser.add_argument('--hidden_dim',    type=int,      default=128)
    parser.add_argument('--num_heads',     type=int,      default=4)
    parser.add_argument('--readout',       type=str,      default='pma')
    parser.add_argument('--dropout_prob',  type=float,    default=0.0)
    parser.add_argument('--norm_features', type=str2bool, default=False)
    parser.add_argument('--local_mp_type', type=str,      default='gin')
    parser.add_argument('--rwse_k',        type=int,      default=16)

    parser.add_argument('--num_epoches',   type=int,      default=150)
    parser.add_argument('--num_workers',   type=int,      default=0)
    parser.add_argument('--batch_size',    type=int,      default=64)
    parser.add_argument('--lr',            type=float,    default=1e-3)
    parser.add_argument('--weight_decay',  type=float,    default=1e-6)

    parser.add_argument('--evidential_coeff', type=float, default=0.01,
                        help='Weight (lambda) for the NIG evidence regularization term')
    parser.add_argument('--log_dir',  type=str, default='logs',
                        help='Directory for per-epoch CSV metric logs')
    parser.add_argument('--patience', type=int, default=0,
                        help='Early-stopping patience in epochs (0 = disabled)')

    args = parser.parse_args()
    print("Arguments")
    for k, v in vars(args).items():
        print(k, ": ", v)
    main(args)

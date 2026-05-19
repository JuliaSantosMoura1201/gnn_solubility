import time
import argparse

import torch
from torch.utils.data import DataLoader

from libs.io_utils import get_dataset
from libs.io_utils import SolubilityClassificationDataset
from libs.io_utils import gnn_collate_fn

from libs.models import MyModel

from libs.evidential_utils import evidential_classification_loss

from libs.utils import str2bool
from libs.utils import set_seed
from libs.utils import set_device
from libs.utils import evaluate_classification_multi
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

    train_ds = SolubilityClassificationDataset(splitted_set=train_set)
    valid_ds = SolubilityClassificationDataset(splitted_set=valid_set)
    test_ds  = SolubilityClassificationDataset(splitted_set=test_set)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, collate_fn=gnn_collate_fn)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, collate_fn=gnn_collate_fn)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, collate_fn=gnn_collate_fn)

    # out_dim = num_classes: raw outputs become Dirichlet evidence after softplus
    model = MyModel(
        model_type=args.model_type,
        num_layers=args.num_layers,
        hidden_dim=args.hidden_dim,
        readout=args.readout,
        dropout_prob=args.dropout_prob,
        out_dim=args.out_dim,
        norm_features=args.norm_features,
    )
    model = model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer=optimizer, step_size=40, gamma=0.1
    )

    import os
    os.makedirs(args.log_dir, exist_ok=True)
    log_path = os.path.join(args.log_dir, f"{args.job_title}_seed{args.seed}.csv")
    csv_fh, csv_writer = open_csv_logger(log_path, [
        'epoch', 'train_loss', 'train_acc', 'train_ece',
        'valid_loss', 'valid_acc', 'valid_ece',
        'test_loss', 'test_acc', 'test_ece',
    ])
    early_stop = EarlyStopping(patience=args.patience, mode='min')
    best_valid_loss = float('inf')
    best_save_path = os.path.join('./save', f"best_{args.job_title}_{args.model_type}_{args.readout}_{args.data_seed}_s{args.seed}.pth")

    for epoch in range(args.num_epoches):
        # --- Train ---
        model.train()
        num_batches = len(train_loader)
        train_loss = 0.0
        y_list, pred_list = [], []

        for i, batch in enumerate(train_loader):
            st = time.time()
            optimizer.zero_grad()

            graph, y = batch[0].to(device), batch[1].to(device).long()

            pred_raw, _ = model(graph, training=False)
            loss, alpha, S, log_probs = evidential_classification_loss(
                pred_raw, y,
                num_classes=args.out_dim,
                coeff=args.evidential_coeff,
                epoch=epoch,
                warmup_epochs=args.warmup_epochs,
            )

            y_list.append(y)
            pred_list.append(log_probs.detach())  # log(alpha/S) for evaluate_classification_multi

            loss.backward()
            optimizer.step()
            train_loss += loss.detach().cpu().numpy()

            et = time.time()
            print("Train!!! Epoch:", epoch + 1,
                  "\t Batch:", i + 1, '/', num_batches,
                  "\t Loss:", loss.detach().cpu().numpy(),
                  "\t Time spent:", round(et - st, 2), "(s)")

        scheduler.step()
        train_loss /= num_batches
        train_metrics = evaluate_classification_multi(y_list=y_list, pred_list=pred_list)

        # --- Validation & Test (single forward pass — no MC sampling) ---
        model.eval()
        with torch.no_grad():
            valid_loss = 0.0
            num_batches = len(valid_loader)
            y_list, pred_list = [], []

            for i, batch in enumerate(valid_loader):
                st = time.time()
                graph, y = batch[0].to(device), batch[1].to(device).long()

                pred_raw, _ = model(graph, training=False)
                loss, alpha, S, log_probs = evidential_classification_loss(
                    pred_raw, y,
                    num_classes=args.out_dim,
                    coeff=args.evidential_coeff,
                    epoch=epoch,
                    warmup_epochs=args.warmup_epochs,
                )

                y_list.append(y)
                pred_list.append(log_probs)
                valid_loss += loss.cpu().numpy()

                et = time.time()
                print("Valid!!! Epoch:", epoch + 1,
                      "\t Batch:", i + 1, '/', num_batches,
                      "\t Loss:", loss.cpu().numpy(),
                      "\t Time spent:", round(et - st, 2), "(s)")

            valid_loss /= num_batches
            valid_metrics = evaluate_classification_multi(y_list=y_list, pred_list=pred_list)

            test_loss = 0.0
            num_batches = len(test_loader)
            y_list, pred_list = [], []
            vacuity_list = []

            for i, batch in enumerate(test_loader):
                st = time.time()
                graph, y = batch[0].to(device), batch[1].to(device).long()

                pred_raw, _ = model(graph, training=False)
                loss, alpha, S, log_probs = evidential_classification_loss(
                    pred_raw, y,
                    num_classes=args.out_dim,
                    coeff=args.evidential_coeff,
                    epoch=epoch,
                    warmup_epochs=args.warmup_epochs,
                )

                # Vacuity: epistemic uncertainty = num_classes / S
                vacuity = args.out_dim / S.squeeze(1)

                y_list.append(y)
                pred_list.append(log_probs)
                vacuity_list.append(vacuity)
                test_loss += loss.cpu().numpy()

                et = time.time()
                print("Test!!! Epoch:", epoch + 1,
                      "\t Batch:", i + 1, '/', num_batches,
                      "\t Loss:", loss.cpu().numpy(),
                      "\t Time spent:", round(et - st, 2), "(s)")

            test_loss /= num_batches
            test_metrics = evaluate_classification_multi(y_list=y_list, pred_list=pred_list)

            vacuity_mean = torch.cat(vacuity_list).mean().item()

        # Log line — same format as gnn_classification.py for easy comparison
        print("End of ", epoch + 1, "-th epoch",
              "Accuracy:", round(train_metrics[0], 3),
              round(valid_metrics[0], 3),
              round(test_metrics[0], 3),
              "ECE:", round(train_metrics[1], 3),
              round(valid_metrics[1], 3),
              round(test_metrics[1], 3),
              "Vacuity:", round(vacuity_mean, 4))

        csv_writer.writerow({
            'epoch': epoch + 1,
            'train_loss': round(float(train_loss), 6), 'train_acc': round(train_metrics[0], 6), 'train_ece': round(float(train_metrics[1]), 6),
            'valid_loss': round(float(valid_loss), 6), 'valid_acc': round(valid_metrics[0], 6), 'valid_ece': round(float(valid_metrics[1]), 6),
            'test_loss':  round(float(test_loss),  6), 'test_acc':  round(test_metrics[0],  6), 'test_ece':  round(float(test_metrics[1]),  6),
        })
        csv_fh.flush()

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()}, best_save_path)

        if early_stop.step(valid_loss):
            print(f"CONVERGENCE_EPOCH={epoch+1}")
            break

        save_path = (
            f"./save/{args.job_title}_"
            f"{args.model_type}_"
            f"{args.hidden_dim}_"
            f"{args.readout}_"
            f"{args.split_method}_"
            f"{args.data_seed}_evidential.pth"
        )
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, save_path)

    csv_fh.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--job_title',      type=str,      default='Evidential_classification')
    parser.add_argument('--use_gpu',        type=str2bool, default=True)
    parser.add_argument('--gpu_idx',        type=str,      default='1')
    parser.add_argument('--seed',           type=int,      default=999)

    parser.add_argument('--dataset_name',   type=str,      default='Solubility')
    parser.add_argument('--split_method',   type=str,      default='scaffold')
    parser.add_argument('--data_seed',      type=int,      default=999)

    parser.add_argument('--model_type',     type=str,      default='gcn')
    parser.add_argument('--num_layers',     type=int,      default=4)
    parser.add_argument('--hidden_dim',     type=int,      default=128)
    parser.add_argument('--out_dim',        type=int,      default=4,
                        help='Number of classes (= Dirichlet output dimension)')
    parser.add_argument('--readout',        type=str,      default='pma')
    parser.add_argument('--dropout_prob',   type=float,    default=0.0)
    parser.add_argument('--norm_features',  type=str2bool, default=False)

    parser.add_argument('--num_epoches',    type=int,      default=150)
    parser.add_argument('--num_workers',    type=int,      default=8)
    parser.add_argument('--batch_size',     type=int,      default=64)
    parser.add_argument('--lr',             type=float,    default=1e-3)
    parser.add_argument('--weight_decay',   type=float,    default=1e-6)

    # Evidential-specific
    parser.add_argument('--evidential_coeff', type=float, default=0.01,
                        help='Maximum weight for the KL regularization term')
    parser.add_argument('--warmup_epochs',    type=int,   default=10,
                        help='Epochs over which to linearly anneal the KL weight from 0 to coeff')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='Directory for per-epoch CSV metric logs')
    parser.add_argument('--patience', type=int, default=0,
                        help='Early-stopping patience in epochs (0 = disabled)')

    args = parser.parse_args()

    print("Arguments")
    for k, v in vars(args).items():
        print(k, ": ", v)
    main(args)

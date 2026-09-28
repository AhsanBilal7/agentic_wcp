"""Standard (non-agentic) training baseline.

Trains SRCNN and/or DnCNN at a single SNR with plain Adam, as in ChannelNet
(https://github.com/Mehran-Soltani/ChannelNet), then evaluates each model across
a list of test SNRs to show how a fixed-SNR estimator degrades under SNR shift.
"""

import argparse
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from channel_utils import load_perfect_channel, make_snr_dataset, to_tensor
from models import MODEL_TYPES, build_model

DEFAULT_TEST_SNRS = [8, 10, 12, 14, 16, 18, 20, 22, 25, 30]


def train_estimator(model_type, train_data, train_label, val_data, val_label, save_path,
                    epochs=300, batch_size=128, lr=1e-3, device='cuda'):
    """Train with MSE + Adam and keep the checkpoint with the lowest validation loss."""
    train_loader = DataLoader(TensorDataset(to_tensor(train_data), to_tensor(train_label)),
                              batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(to_tensor(val_data), to_tensor(val_label)),
                            batch_size=batch_size, shuffle=False)

    model = build_model(model_type).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8)
    best_val_loss = float('inf')

    print(f"Training {model_type} on {device} "
          f"({sum(p.numel() for p in model.parameters()):,} parameters)")

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            loss = criterion(model(data), target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * data.size(0)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                val_loss += criterion(model(data), target).item() * data.size(0)
        val_loss /= len(val_loader.dataset)

        if epoch % 10 == 0:
            print(f"Epoch [{epoch}/{epochs}] Train Loss: {train_loss:.6f}  Val Loss: {val_loss:.6f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)

    print(f"Best validation loss: {best_val_loss:.6f} (saved to {save_path})")


def predict(model_type, checkpoint_path, inputs, batch_size=128, device='cuda'):
    """Run a saved estimator on (N, 72, 14, 1) inputs and return predictions of the same shape."""
    model = build_model(model_type).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    loader = DataLoader(TensorDataset(to_tensor(inputs)), batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        predictions = [model(batch.to(device)).cpu() for (batch,) in loader]
    return torch.cat(predictions).permute(0, 2, 3, 1).numpy()


def evaluate_across_snrs(model_type, checkpoint_path, perfect, test_mask, test_snrs,
                         num_pilots, batch_size=128, device='cuda'):
    """MSE of a fixed-SNR model at every test SNR, on the held-out channels in `test_mask`."""
    results = {}
    for snr in test_snrs:
        inputs, labels = make_snr_dataset(perfect, snr, num_pilots)
        preds = predict(model_type, checkpoint_path, inputs[test_mask], batch_size, device)
        results[snr] = float(np.mean((preds - labels[test_mask]) ** 2))
        print(f"  {model_type} @ {snr:>2} dB: MSE = {results[snr]:.6f}")
    return results


def plot_results(all_results, train_snr, save_path):
    """MSE vs. test SNR for each model, with the training SNR marked."""
    plt.figure(figsize=(9, 5.5))
    for model_type, results in all_results.items():
        snrs = sorted(results)
        plt.semilogy(snrs, [results[s] for s in snrs], 'o-', linewidth=2, markersize=7, label=model_type)
    plt.axvline(train_snr, color='gray', linestyle='--', alpha=0.7, label=f'Training SNR = {train_snr} dB')
    plt.xlabel('Test SNR (dB)', fontsize=12)
    plt.ylabel('MSE', fontsize=12)
    plt.title('Standard training: cross-SNR performance', fontsize=14, fontweight='bold')
    plt.grid(True, which='both', alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to: {save_path}")


def parse_args():
    parser = argparse.ArgumentParser(description='Standard single-SNR training baseline')
    parser.add_argument('--models', nargs='+', default=list(MODEL_TYPES), choices=MODEL_TYPES)
    parser.add_argument('--train_snr', type=int, default=22, help='SNR (dB) used for training')
    parser.add_argument('--test_snrs', type=int, nargs='+', default=DEFAULT_TEST_SNRS)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--num_pilots', type=int, default=48, choices=[8, 16, 24, 36, 48])
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--data_path', default='data/Perfect_H_40000.mat')
    parser.add_argument('--output_dir', default='results/baseline')
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'])
    return parser.parse_args()


def main():
    args = parse_args()
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'
    os.makedirs(args.output_dir, exist_ok=True)

    perfect = load_perfect_channel(args.data_path)
    print(f"Perfect channel shape: {perfect.shape}")
    np.random.seed(args.seed)

    print(f"\nBuilding training data at SNR = {args.train_snr} dB...")
    inputs, labels = make_snr_dataset(perfect, args.train_snr, args.num_pilots)
    train_mask = np.random.rand(len(labels)) < 0.8
    test_mask = ~train_mask
    print(f"Train samples: {train_mask.sum()}, Validation/test samples: {test_mask.sum()}")

    all_results = {}
    for model_type in args.models:
        print(f"\n{'=' * 70}\n{model_type}\n{'=' * 70}")
        ckpt = os.path.join(args.output_dir,
                            f'{model_type}_VehA_{args.num_pilots}pilots_snr{args.train_snr}.pth')
        train_estimator(model_type, inputs[train_mask], labels[train_mask],
                        inputs[test_mask], labels[test_mask], ckpt, epochs=args.epochs,
                        batch_size=args.batch_size, lr=args.lr, device=args.device)
        all_results[model_type] = evaluate_across_snrs(
            model_type, ckpt, perfect, test_mask, args.test_snrs, args.num_pilots,
            args.batch_size, args.device)

    results_path = os.path.join(args.output_dir, f'baseline_train{args.train_snr}_results.json')
    with open(results_path, 'w') as f:
        json.dump({'train_snr': args.train_snr, 'num_pilots': args.num_pilots,
                   'mse': all_results}, f, indent=2)
    print(f"\nResults saved to: {results_path}")
    plot_results(all_results, args.train_snr,
                 os.path.join(args.output_dir, f'baseline_train{args.train_snr}_mse.png'))


if __name__ == '__main__':
    main()

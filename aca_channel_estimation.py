"""Agentic Continual Adaptation (ACA) for deep-learning channel estimation.

A channel estimator (SRCNN or DnCNN) is trained sequentially across SNR regimes.
Before every mini-batch update, an actor-critic RL agent observes the training
state and selects one of seven adaptation strategies. An anchor memory of samples
from past regimes measures forgetting after the update, and the update is rolled
back when forgetting exceeds an adaptive threshold.

Paper: "Agentic Continual Adaptation: Enabling Lifelong Learning in Wireless
Channel Estimation", IEEE Network, 2026. doi:10.1109/MNET.2026.3693331
"""

import argparse
import copy
import json
import os
from collections import deque

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from channel_utils import load_perfect_channel, make_snr_dataset, to_tensor
from models import MODEL_TYPES, build_model

DEFAULT_SNRS = [8, 10, 12, 14, 16, 18, 20, 22, 25, 30]

# The seven adaptation strategies (the agent's action space), ordered from
# plastic to stable. The index doubles as the action's compute cost in the reward.
ACTIONS = [
    'FULL_UPDATE',      # all parameters, base learning rate
    'LAST_LAYER',       # only the last two parameter tensors (output layer)
    'FREEZE_EARLY',     # freeze the first half of the parameter tensors
    'SMALL_LR',         # all parameters, 0.1x learning rate
    'ANCHOR_REPLAY',    # all parameters, 0.5x learning rate, mix in anchor samples
    'ADAPTIVE_MIX',     # freeze the first third, 0.5x learning rate, anchor replay
    'CONSERVATIVE',     # all parameters, 0.05x learning rate
]


class AnchorMemory:
    """Representative samples from past SNR regimes, used to measure forgetting."""

    def __init__(self, max_size=256):
        self.max_size = max_size
        self.buffer = []
        self._stacked = None

    def __len__(self):
        return len(self.buffer)

    def add_samples(self, dataloader, num_samples):
        """Add up to `num_samples` samples from `dataloader` using reservoir sampling."""
        samples_added = 0
        for X_batch, Y_batch in dataloader:
            remaining = num_samples - samples_added
            X_batch, Y_batch = X_batch[:remaining], Y_batch[:remaining]

            for i in range(X_batch.size(0)):
                if len(self.buffer) >= self.max_size:
                    idx = np.random.randint(0, samples_added + i + 1)
                    if idx < self.max_size:
                        self.buffer[idx] = (X_batch[i], Y_batch[i])
                else:
                    self.buffer.append((X_batch[i], Y_batch[i]))

            samples_added += X_batch.size(0)
            if samples_added >= num_samples:
                break
        self._stacked = None

    def _tensors(self):
        if self._stacked is None:
            X_list, Y_list = zip(*self.buffer)
            self._stacked = (torch.stack(X_list), torch.stack(Y_list))
        return self._stacked

    def evaluate(self, model, device, chunk_size=256):
        """Mean per-sample MSE of `model` on the anchor samples (0.0 if empty)."""
        if len(self.buffer) == 0:
            return 0.0

        X_all, Y_all = self._tensors()
        model.eval()
        total_sq_err = 0.0
        with torch.no_grad():
            for start in range(0, len(X_all), chunk_size):
                X = X_all[start:start + chunk_size].to(device)
                Y = Y_all[start:start + chunk_size].to(device)
                total_sq_err += nn.functional.mse_loss(model(X), Y, reduction='sum').item()
        model.train()
        return total_sq_err / Y_all.numel()

    def get_dataloader(self, batch_size=32):
        """Shuffled loader over the anchor samples, or None if the memory is empty."""
        if len(self.buffer) == 0:
            return None
        return DataLoader(TensorDataset(*self._tensors()), batch_size=batch_size, shuffle=True)


class AdaptationAgent(nn.Module):
    """Actor-critic agent that picks an adaptation strategy from the training state.

    State (5-D): [current batch MSE, anchor MSE, SNR progress, loss trend, epoch progress].
    """

    def __init__(self, state_dim=5, hidden_dim=128, num_actions=len(ACTIONS),
                 lr=1e-4, eps=0.2, gamma=0.95):
        super().__init__()
        self.actions = ACTIONS[:num_actions]
        self.eps = eps      # epsilon-greedy exploration rate
        self.gamma = gamma  # TD discount

        self.policy_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_actions),
        )
        self.value_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    @property
    def device(self):
        return next(self.parameters()).device

    @staticmethod
    def get_state(current_mse, anchor_mse, snr_ratio, loss_trend, step_budget):
        return torch.tensor([current_mse, anchor_mse, snr_ratio, loss_trend, step_budget],
                            dtype=torch.float32)

    def select_action(self, state, explore=True):
        """Epsilon-greedy action selection. Returns (action index, its policy probability)."""
        with torch.no_grad():
            action_probs = torch.softmax(self.policy_net(state.unsqueeze(0).to(self.device)), dim=-1)

        if explore and np.random.random() < self.eps:
            action = np.random.randint(0, len(self.actions))
        else:
            action = torch.argmax(action_probs, dim=-1).item()
        return action, action_probs[0, action].item()

    def update(self, states, actions, rewards, next_states):
        """One actor-critic step with TD(0) advantages and an entropy bonus."""
        if len(states) == 0:
            return 0.0

        states = torch.stack(states).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long, device=self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        next_states = torch.stack(next_states).to(self.device)

        # Critic: TD(0) regression
        values = self.value_net(states).squeeze()
        next_values = self.value_net(next_states).squeeze().detach()
        td_target = rewards + self.gamma * next_values
        value_loss = nn.MSELoss()(values, td_target)

        # Actor: policy gradient weighted by the TD advantage
        action_probs = torch.softmax(self.policy_net(states), dim=-1)
        log_probs = torch.log(action_probs.gather(1, actions.unsqueeze(1)).squeeze() + 1e-8)
        advantages = (td_target - values).detach()
        policy_loss = -(log_probs * advantages).mean()

        entropy = -(action_probs * torch.log(action_probs + 1e-8)).sum(dim=-1).mean()
        loss = policy_loss + 0.5 * value_loss - 0.01 * entropy

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        self.optimizer.step()
        return loss.item()


class ACATrainer:
    """Runs agent-guided updates with forgetting detection and rollback."""

    def __init__(self, model, agent, anchor_memory, device,
                 forgetting_threshold=0.15, adaptive_threshold=True,
                 agent_update_every=64):
        self.model = model
        self.agent = agent
        self.anchor_memory = anchor_memory
        self.device = device
        self.criterion = nn.MSELoss()
        self.agent_update_every = agent_update_every

        self.base_forgetting_threshold = forgetting_threshold
        self.adaptive_threshold = adaptive_threshold
        self.min_threshold = 0.08
        self.max_threshold = 0.30

        self.episode_states = []
        self.episode_actions = []
        self.episode_rewards = []

        self.stats = {
            'rollback_count': 0,
            'total_updates': 0,
            'action_counts': {action: 0 for action in agent.actions},
            'forgetting_values': [],
            'threshold_values': [],
            'epoch_losses': [],
            'anchor_mse_history': [],
            'snr_history': [],
        }

    @staticmethod
    def compute_loss_trend(recent_losses):
        """Relative change between the oldest and newest of the recent batch losses."""
        if len(recent_losses) < 2:
            return 0.0
        return (recent_losses[-1] - recent_losses[0]) / max(abs(recent_losses[0]), 1e-6)

    def get_dynamic_threshold(self, anchor_mse):
        """Forgetting tolerance: strict when the anchor set is fit well, looser when it is not."""
        if not self.adaptive_threshold:
            return self.base_forgetting_threshold
        if anchor_mse < 0.001:
            return self.min_threshold
        if anchor_mse < 0.005:
            return self.base_forgetting_threshold
        return min(self.max_threshold, self.base_forgetting_threshold * 1.5)

    def _freeze_leading_params(self, count):
        for i, param in enumerate(self.model.parameters()):
            if i < count:
                param.requires_grad = False

    def _set_lr(self, optimizer, lr):
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def apply_adaptation_action(self, action, optimizer, base_lr):
        """Configure trainable parameters and learning rate. Returns whether to use anchor replay."""
        action_name = self.agent.actions[action]
        self.stats['action_counts'][action_name] += 1

        for param in self.model.parameters():
            param.requires_grad = True
        n_params = sum(1 for _ in self.model.parameters())

        # name: (number of leading parameter tensors to freeze, lr multiplier, anchor replay)
        config = {
            'FULL_UPDATE':   (0,             1.0,  False),
            'LAST_LAYER':    (n_params - 2,  1.0,  False),
            'FREEZE_EARLY':  (n_params // 2, 1.0,  False),
            'SMALL_LR':      (0,             0.1,  False),
            'ANCHOR_REPLAY': (0,             0.5,  True),
            'ADAPTIVE_MIX':  (n_params // 3, 0.5,  True),
            'CONSERVATIVE':  (0,             0.05, False),
        }
        n_frozen, lr_scale, use_anchor = config[action_name]
        self._freeze_leading_params(n_frozen)
        self._set_lr(optimizer, base_lr * lr_scale)
        return use_anchor

    def train_epoch(self, dataloader, optimizer, base_lr, epoch, snr_value,
                    total_epochs, current_snr_idx=0, total_snrs=1):
        """Train one epoch; every mini-batch is one agent decision."""
        self.model.train()
        running_loss = 0.0
        recent_losses = deque(maxlen=10)
        anchor_loader = self.anchor_memory.get_dataloader(batch_size=32)

        loop = tqdm(dataloader, desc=f"SNR={snr_value}dB Epoch {epoch}/{total_epochs}")
        for X_batch, Y_batch in loop:
            X_batch, Y_batch = X_batch.to(self.device), Y_batch.to(self.device)

            # 1. Observe the state
            self.model.eval()
            with torch.no_grad():
                current_mse = self.criterion(self.model(X_batch), Y_batch).item()
            self.model.train()
            anchor_mse = self.anchor_memory.evaluate(self.model, self.device)

            snr_ratio = current_snr_idx / max(total_snrs - 1, 1)
            loss_trend = self.compute_loss_trend(recent_losses)
            step_budget = (epoch - 1) / total_epochs
            state = self.agent.get_state(current_mse, anchor_mse, snr_ratio, loss_trend, step_budget)

            # 2. The agent selects a strategy
            action, _ = self.agent.select_action(state, explore=True)

            # 3. Checkpoint, then apply the strategy
            checkpoint = copy.deepcopy(self.model.state_dict())
            use_anchor = self.apply_adaptation_action(action, optimizer, base_lr)

            # 4. Update on the current batch (optionally mixed with anchor replay)
            optimizer.zero_grad()
            loss = self.criterion(self.model(X_batch), Y_batch)
            if use_anchor and anchor_loader is not None:
                anchor_X, anchor_Y = next(iter(anchor_loader))
                anchor_X, anchor_Y = anchor_X.to(self.device), anchor_Y.to(self.device)
                anchor_loss = self.criterion(self.model(anchor_X), anchor_Y)
                loss = 0.7 * loss + 0.3 * anchor_loss

            if loss.requires_grad:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                self.stats['total_updates'] += 1

            for param in self.model.parameters():
                param.requires_grad = True

            # 5. Measure forgetting on the anchor memory; roll back if it is too large
            new_anchor_mse = self.anchor_memory.evaluate(self.model, self.device)
            forgetting = new_anchor_mse - anchor_mse
            threshold = self.get_dynamic_threshold(anchor_mse)
            self.stats['forgetting_values'].append(forgetting)
            self.stats['threshold_values'].append(threshold)

            if forgetting > threshold:
                self.model.load_state_dict(checkpoint)
                new_anchor_mse = anchor_mse
                forgetting = 0.0
                self.stats['rollback_count'] += 1

            # 6. Reward: accuracy, minus forgetting and compute cost
            improvement = -current_mse
            forgetting_penalty = max(0, forgetting) * 100.0
            compute_cost = 0.01 * action
            reward = 10.0 * (improvement - forgetting_penalty - compute_cost)

            self.episode_states.append(state)
            self.episode_actions.append(action)
            self.episode_rewards.append(reward)

            recent_losses.append(loss.item())
            running_loss += loss.item()
            self.stats['anchor_mse_history'].append(new_anchor_mse)
            self.stats['snr_history'].append(snr_value)

            loop.set_postfix(loss=f"{loss.item():.6f}", action=self.agent.actions[action][:8],
                             anchor=f"{new_anchor_mse:.5f}", rollback=self.stats['rollback_count'])

        avg_loss = running_loss / len(dataloader)
        self.stats['epoch_losses'].append(avg_loss)

        # 7. Update the agent on the collected transitions
        if len(self.episode_states) >= self.agent_update_every:
            next_states = self.episode_states[1:] + [self.episode_states[-1]]
            agent_loss = self.agent.update(self.episode_states, self.episode_actions,
                                           self.episode_rewards, next_states)
            print(f"\nAgent updated - Loss: {agent_loss:.4f} | "
                  f"Rollbacks: {self.stats['rollback_count']}/{self.stats['total_updates']}")
            self.episode_states, self.episode_actions, self.episode_rewards = [], [], []

        return avg_loss

    def get_statistics(self):
        stats = self.stats.copy()
        total = self.stats['total_updates']
        stats['rollback_rate'] = self.stats['rollback_count'] / total if total > 0 else 0.0
        return stats

    def save_statistics(self, filepath, history_len=1000):
        """Save statistics to JSON (per-step histories are truncated to the last `history_len`)."""
        stats = self.get_statistics()
        stats['action_counts'] = dict(stats['action_counts'])
        for key in ('forgetting_values', 'threshold_values', 'anchor_mse_history', 'snr_history'):
            stats[key] = [float(x) for x in stats[key][-history_len:]]
        stats['epoch_losses'] = [float(x) for x in stats['epoch_losses']]

        with open(filepath, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"Statistics saved to {filepath}")


def train_aca_channel_estimator(model_type='DNCNN', snr_list=DEFAULT_SNRS, num_pilots=48,
                                epochs_per_snr=50, save_dir='./aca_checkpoints',
                                forgetting_threshold=0.15, data_path='data/Perfect_H_40000.mat',
                                anchor_memory_size=256, batch_size=128, lr=1e-3, seed=42,
                                device='cuda'):
    """Train one channel estimator sequentially over `snr_list` with ACA.

    Returns the trained model and the training statistics.
    """
    os.makedirs(save_dir, exist_ok=True)

    print("=" * 70)
    print(f"ACA Training: {model_type}")
    print("=" * 70)
    print(f"SNR sequence: {snr_list}")
    print(f"Epochs per SNR: {epochs_per_snr}")
    print(f"Forgetting threshold: {forgetting_threshold}")
    print("=" * 70)

    print(f"\nLoading perfect channel data from {data_path}...")
    perfect = load_perfect_channel(data_path)
    print(f"Perfect channel shape: {perfect.shape}")

    model = build_model(model_type).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    anchor_memory = AnchorMemory(max_size=anchor_memory_size)
    agent = AdaptationAgent(state_dim=5, hidden_dim=128, num_actions=len(ACTIONS)).to(device)
    trainer = ACATrainer(model, agent, anchor_memory, device,
                         forgetting_threshold=forgetting_threshold, adaptive_threshold=True)

    np.random.seed(seed)
    torch.manual_seed(seed)

    for snr_idx, snr in enumerate(snr_list):
        print(f"\n{'=' * 70}")
        print(f"Training on SNR = {snr} dB ({snr_idx + 1}/{len(snr_list)})")
        print(f"{'=' * 70}")

        print("Generating noisy channels and interpolating pilots...")
        inputs, labels = make_snr_dataset(perfect, snr, num_pilots)

        # 80/20 train/validation split; ACA trains on the 80% part
        idx_random = np.random.rand(len(labels)) < 0.8
        train_dataset = TensorDataset(to_tensor(inputs[idx_random]), to_tensor(labels[idx_random]))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        print(f"Train samples: {len(train_dataset)}, Val samples: {int((~idx_random).sum())}")

        optimizer = optim.Adam(model.parameters(), lr=lr)
        for epoch in range(1, epochs_per_snr + 1):
            avg_loss = trainer.train_epoch(train_loader, optimizer, base_lr=lr, epoch=epoch,
                                           snr_value=snr, total_epochs=epochs_per_snr,
                                           current_snr_idx=snr_idx, total_snrs=len(snr_list))
            if epoch % 10 == 0:
                print(f"Epoch {epoch}/{epochs_per_snr} - Avg Loss: {avg_loss:.6f}")

        # Keep a few representative samples of this regime for forgetting detection
        samples_to_add = min(anchor_memory_size // len(snr_list), 50)
        print(f"\nAdding {samples_to_add} samples to anchor memory...")
        anchor_memory.add_samples(train_loader, num_samples=samples_to_add)
        print(f"Anchor memory size: {len(anchor_memory)}")

        checkpoint_path = os.path.join(save_dir, f'aca_{model_type}_snr{snr}.pth')
        torch.save({'model_state_dict': model.state_dict(),
                    'agent_state_dict': agent.state_dict(),
                    'snr': snr, 'snr_idx': snr_idx, 'model_type': model_type}, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")

    stats = trainer.get_statistics()
    torch.save({'model_state_dict': model.state_dict(),
                'agent_state_dict': agent.state_dict(),
                'model_type': model_type, 'snr_list': snr_list,
                'rollback_rate': stats['rollback_rate'],
                'action_counts': dict(stats['action_counts'])},
               os.path.join(save_dir, f'aca_{model_type}_final.pth'))

    print(f"\n{'=' * 70}\nTraining completed!\n{'=' * 70}")
    trainer.save_statistics(os.path.join(save_dir, f'aca_{model_type}_statistics.json'))

    print("\nFinal statistics:")
    print(f"  Total updates: {stats['total_updates']}")
    print(f"  Rollbacks: {stats['rollback_count']}")
    print(f"  Rollback rate: {stats['rollback_rate']:.2%}")
    print("\n  Action distribution:")
    for action, count in stats['action_counts'].items():
        pct = 100 * count / stats['total_updates'] if stats['total_updates'] > 0 else 0
        print(f"    {action:<14} {count:>7} ({pct:.1f}%)")

    return model, stats


def test_aca_model(checkpoint_path, snr_list=DEFAULT_SNRS, num_pilots=48,
                   data_path='data/Perfect_H_40000.mat', batch_size=128, seed=42, device='cuda'):
    """Evaluate a trained checkpoint at every SNR in `snr_list`. Returns {snr: {'mse', 'num_samples'}}."""
    print(f"\nLoading model from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_type = checkpoint['model_type']

    model = build_model(model_type).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    perfect = load_perfect_channel(data_path)
    np.random.seed(seed)
    results = {'model_type': model_type, 'test_results': {}}

    print("\n" + "=" * 70 + "\nTesting on multiple SNRs\n" + "=" * 70)
    for snr in snr_list:
        print(f"\nTesting SNR = {snr} dB...")
        inputs, labels = make_snr_dataset(perfect, snr, num_pilots)

        # Evaluate on the 20% validation split
        idx_random = np.random.rand(len(labels)) < 0.8
        test_dataset = TensorDataset(to_tensor(inputs[~idx_random]), to_tensor(labels[~idx_random]))
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        total_mse, num_batches = 0.0, 0
        with torch.no_grad():
            for X_batch, Y_batch in tqdm(test_loader, desc=f"Testing SNR {snr}dB"):
                predictions = model(X_batch.to(device))
                total_mse += nn.MSELoss()(predictions, Y_batch.to(device)).item()
                num_batches += 1

        avg_mse = total_mse / num_batches
        results['test_results'][snr] = {'mse': avg_mse, 'num_samples': len(test_dataset)}
        print(f"  MSE: {avg_mse:.6f}")

    results_path = checkpoint_path.replace('.pth', '_test_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")
    return results


def plot_aca_performance(results, save_path='aca_performance.png'):
    """Plot test MSE against SNR."""
    snrs = sorted(results['test_results'].keys())
    mses = [results['test_results'][snr]['mse'] for snr in snrs]

    plt.figure(figsize=(10, 6))
    plt.semilogy(snrs, mses, 'o-', linewidth=2, markersize=8)
    plt.xlabel('SNR (dB)', fontsize=12)
    plt.ylabel('MSE', fontsize=12)
    plt.title(f'{results["model_type"]}: ACA test MSE across SNRs', fontsize=14, fontweight='bold')
    plt.grid(True, which='both', alpha=0.3)
    for snr, mse in zip(snrs, mses):
        plt.annotate(f'{mse:.4f}', xy=(snr, mse), xytext=(0, 10),
                     textcoords='offset points', ha='center', fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to: {save_path}")


def parse_args():
    parser = argparse.ArgumentParser(description='Agentic Continual Adaptation (ACA) for channel estimation')
    parser.add_argument('--mode', default='both', choices=['train', 'test', 'both'],
                        help='train, test a checkpoint, or train then test (default: both)')
    parser.add_argument('--model_type', default='DNCNN', choices=MODEL_TYPES,
                        help='channel estimator architecture (default: DNCNN)')
    parser.add_argument('--snr_list', type=int, nargs='+', default=DEFAULT_SNRS,
                        help='SNR regimes in dB, visited in this order during training')
    parser.add_argument('--epochs_per_snr', type=int, default=50, help='training epochs per SNR regime')
    parser.add_argument('--num_pilots', type=int, default=48, choices=[8, 16, 24, 36, 48],
                        help='number of pilots in the 72x14 grid')
    parser.add_argument('--forgetting_threshold', type=float, default=0.15,
                        help='base forgetting threshold for rollback')
    parser.add_argument('--anchor_memory_size', type=int, default=256, help='anchor memory capacity')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3, help='base learning rate of the estimator')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--data_path', default='data/Perfect_H_40000.mat',
                        help='path to the perfect VehA channel .mat file')
    parser.add_argument('--save_dir', default='./aca_checkpoints', help='where checkpoints and logs go')
    parser.add_argument('--checkpoint_path', default=None,
                        help='checkpoint to test (default: <save_dir>/aca_<model_type>_final.pth)')
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'])
    return parser.parse_args()


def main():
    args = parse_args()
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'

    checkpoint_path = args.checkpoint_path or os.path.join(
        args.save_dir, f'aca_{args.model_type}_final.pth')

    if args.mode in ('train', 'both'):
        train_aca_channel_estimator(
            model_type=args.model_type, snr_list=args.snr_list, num_pilots=args.num_pilots,
            epochs_per_snr=args.epochs_per_snr, save_dir=args.save_dir,
            forgetting_threshold=args.forgetting_threshold, data_path=args.data_path,
            anchor_memory_size=args.anchor_memory_size, batch_size=args.batch_size,
            lr=args.lr, seed=args.seed, device=args.device)

    if args.mode in ('test', 'both'):
        results = test_aca_model(checkpoint_path, snr_list=args.snr_list, num_pilots=args.num_pilots,
                                 data_path=args.data_path, batch_size=args.batch_size,
                                 seed=args.seed, device=args.device)
        plot_aca_performance(results, checkpoint_path.replace('.pth', '_performance.png'))


if __name__ == '__main__':
    main()

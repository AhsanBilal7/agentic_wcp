import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from scipy.io import loadmat
import matplotlib.pyplot as plt
import pandas as pd
import json
import os
from tqdm import tqdm
import copy
from collections import deque
import argparse

from models import (interpolation, SRCNN, DNCNN)
from aca_channel_utils import add_awgn_noise


class AnchorMemory:
    """Maintains representative samples from past SNR regimes"""
    def __init__(self, max_size=256):
        self.max_size = max_size
        self.buffer = []
    
    def add_samples(self, dataloader, num_samples):
        """Add representative samples using reservoir sampling"""
        samples_added = 0
        for X_batch, Y_batch in dataloader:
            batch_size = X_batch.size(0)
            if samples_added + batch_size > num_samples:
                remaining = num_samples - samples_added
                X_batch = X_batch[:remaining]
                Y_batch = Y_batch[:remaining]
            
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
    
    def evaluate(self, model, device):
        """Evaluate model performance on anchor samples"""
        if len(self.buffer) == 0:
            return 0.0
        
        model.eval()
        total_mse = 0.0
        
        with torch.no_grad():
            for X, Y in self.buffer:
                X = X.unsqueeze(0).to(device)
                Y = Y.unsqueeze(0).to(device)
                
                output = model(X)
                mse = nn.MSELoss()(output, Y)
                total_mse += mse.item()
        
        model.train()
        return total_mse / len(self.buffer)
    
    def get_dataloader(self, batch_size=32):
        """Get a dataloader from buffer samples"""
        if len(self.buffer) == 0:
            return None
        
        X_list, Y_list = zip(*self.buffer)
        X_tensor = torch.stack(X_list)
        Y_tensor = torch.stack(Y_list)
        
        dataset = TensorDataset(X_tensor, Y_tensor)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)


class AdaptationAgent(nn.Module):
    """Agent that decides adaptation strategy based on SNR conditions"""
    def __init__(self, state_dim=5, hidden_dim=128, num_actions=7):
        super().__init__()
        
        self.policy_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_actions)
        )
        
        self.value_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.actions = [
            'FULL_UPDATE',      # Standard training
            'LAST_LAYER',       # Only update final layers
            'FREEZE_EARLY',     # Freeze early feature extractors
            'SMALL_LR',         # Reduce learning rate
            'ANCHOR_REPLAY',    # Mix current + anchor samples
            'ADAPTIVE_MIX',     # Hybrid approach
            'CONSERVATIVE'      # Very small updates
        ]
        
        self.optimizer = optim.Adam(self.parameters(), lr=1e-4)
        self.eps = 0.2  # Exploration rate
        
    def get_state(self, current_mse, anchor_mse, snr_ratio, loss_trend, step_budget):
        """Construct state representation"""
        return torch.tensor([
            current_mse,
            anchor_mse,
            snr_ratio,
            loss_trend,
            step_budget
        ], dtype=torch.float32)
    
    def select_action(self, state, explore=True):
        """Select action using epsilon-greedy policy"""
        state = state.unsqueeze(0).to(next(self.parameters()).device)
        
        with torch.no_grad():
            action_logits = self.policy_net(state)
            action_probs = torch.softmax(action_logits, dim=-1)
        
        if explore and np.random.random() < self.eps:
            action = np.random.randint(0, len(self.actions))
        else:
            action = torch.argmax(action_probs, dim=-1).item()
        
        return action, action_probs[0, action].item()
    
    def update(self, states, actions, rewards, next_states):
        """Update policy using actor-critic"""
        if len(states) == 0:
            return 0.0
        
        states = torch.stack(states).to(next(self.parameters()).device)
        actions = torch.tensor(actions, dtype=torch.long).to(next(self.parameters()).device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(next(self.parameters()).device)
        next_states = torch.stack(next_states).to(next(self.parameters()).device)
        
        # Value function update
        values = self.value_net(states).squeeze()
        next_values = self.value_net(next_states).squeeze().detach()
        td_target = rewards + 0.95 * next_values
        value_loss = nn.MSELoss()(values, td_target)
        
        # Policy gradient update
        action_logits = self.policy_net(states)
        action_probs = torch.softmax(action_logits, dim=-1)
        log_probs = torch.log(action_probs.gather(1, actions.unsqueeze(1)).squeeze() + 1e-8)
        
        advantages = (td_target - values).detach()
        policy_loss = -(log_probs * advantages).mean()
        
        # Entropy bonus for exploration
        entropy = -(action_probs * torch.log(action_probs + 1e-8)).sum(dim=-1).mean()
        
        loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        return loss.item()


class ACATrainer:
    """Agentic Continual Adaptation trainer for channel estimation"""
    def __init__(self, model, agent, anchor_memory, device, 
                 forgetting_threshold=0.15, adaptive_threshold=True):
        self.model = model
        self.agent = agent
        self.anchor_memory = anchor_memory
        self.device = device
        self.criterion = nn.MSELoss()
        
        # Threshold configuration
        self.base_forgetting_threshold = forgetting_threshold
        self.adaptive_threshold = adaptive_threshold
        self.min_threshold = 0.08
        self.max_threshold = 0.30
        
        # Episode tracking
        self.episode_states = []
        self.episode_actions = []
        self.episode_rewards = []
        
        # Statistics tracking
        self.stats = {
            'rollback_count': 0,
            'total_updates': 0,
            'action_counts': {action: 0 for action in agent.actions},
            'forgetting_values': [],
            'threshold_values': [],
            'epoch_losses': [],
            'anchor_mse_history': [],
            'snr_history': []
        }
        
    def save_checkpoint(self):
        """Save model state for rollback"""
        return copy.deepcopy(self.model.state_dict())
    
    def restore_checkpoint(self, checkpoint):
        """Restore model from checkpoint"""
        self.model.load_state_dict(checkpoint)
    
    def compute_loss_trend(self, recent_losses):
        """Compute trend in recent losses"""
        if len(recent_losses) < 2:
            return 0.0
        losses = list(recent_losses)
        return (losses[-1] - losses[0]) / max(abs(losses[0]), 1e-6)
    
    def get_dynamic_threshold(self, initial_anchor_mse, current_snr=None):
        """Compute adaptive forgetting threshold"""
        if not self.adaptive_threshold:
            return self.base_forgetting_threshold
        
        # Scale threshold based on anchor performance
        if initial_anchor_mse < 0.001:
            dynamic_threshold = self.min_threshold
        elif initial_anchor_mse < 0.005:
            dynamic_threshold = self.base_forgetting_threshold
        else:
            dynamic_threshold = min(self.max_threshold, 
                                   self.base_forgetting_threshold * 1.5)
        
        return dynamic_threshold
    
    def apply_adaptation_action(self, action, optimizer, base_lr, anchor_loader=None):
        """Apply selected adaptation strategy"""
        action_name = self.agent.actions[action]
        self.stats['action_counts'][action_name] += 1
        
        # Reset all parameters to trainable first
        for param in self.model.parameters():
            param.requires_grad = True
        
        if action_name == 'FULL_UPDATE':
            for param_group in optimizer.param_groups:
                param_group['lr'] = base_lr
            use_anchor = False
            
        elif action_name == 'LAST_LAYER':
            # Freeze all but last layers
            layer_count = 0
            total_layers = sum(1 for _ in self.model.parameters())
            for param in self.model.parameters():
                if layer_count < total_layers - 2:
                    param.requires_grad = False
                layer_count += 1
            for param_group in optimizer.param_groups:
                param_group['lr'] = base_lr
            use_anchor = False
            
        elif action_name == 'FREEZE_EARLY':
            # Freeze first half of network
            layer_count = 0
            total_layers = sum(1 for _ in self.model.parameters())
            for param in self.model.parameters():
                if layer_count < total_layers // 2:
                    param.requires_grad = False
                layer_count += 1
            for param_group in optimizer.param_groups:
                param_group['lr'] = base_lr
            use_anchor = False
            
        elif action_name == 'SMALL_LR':
            for param_group in optimizer.param_groups:
                param_group['lr'] = base_lr * 0.1
            use_anchor = False
            
        elif action_name == 'ANCHOR_REPLAY':
            for param_group in optimizer.param_groups:
                param_group['lr'] = base_lr * 0.5
            use_anchor = True
            
        elif action_name == 'ADAPTIVE_MIX':
            # Freeze first third
            layer_count = 0
            total_layers = sum(1 for _ in self.model.parameters())
            for param in self.model.parameters():
                if layer_count < total_layers // 3:
                    param.requires_grad = False
                layer_count += 1
            for param_group in optimizer.param_groups:
                param_group['lr'] = base_lr * 0.5
            use_anchor = True
            
        else:  # CONSERVATIVE
            for param_group in optimizer.param_groups:
                param_group['lr'] = base_lr * 0.05
            use_anchor = False
        
        return use_anchor
    
    def train_epoch(self, dataloader, optimizer, base_lr, epoch, snr_value, 
                   total_epochs, current_snr_idx=0, total_snrs=1):
        """Train one epoch with agentic adaptation"""
        self.model.train()
        running_loss = 0.0
        recent_losses = deque(maxlen=10)
        total_batches = len(dataloader)
        
        # Get anchor replay loader if available
        anchor_loader = self.anchor_memory.get_dataloader(batch_size=32)
        
        loop = tqdm(
            enumerate(dataloader, 1),
            total=total_batches,
            desc=f"SNR={snr_value}dB Epoch {epoch}/{total_epochs}"
        )
        
        for batch_idx, (X_batch, Y_batch) in loop:
            X_batch = X_batch.to(self.device)
            Y_batch = Y_batch.to(self.device)
            
            # Evaluate current performance
            self.model.eval()
            with torch.no_grad():
                output_eval = self.model(X_batch)
                current_mse = self.criterion(output_eval, Y_batch).item()
            self.model.train()
            
            # Get anchor performance
            anchor_mse = self.anchor_memory.evaluate(self.model, self.device)
            
            # Compute state features
            snr_ratio = current_snr_idx / max(total_snrs - 1, 1)
            loss_trend = self.compute_loss_trend(recent_losses)
            step_budget = (epoch - 1) / total_epochs
            
            # Agent selects action
            state = self.agent.get_state(current_mse, anchor_mse, snr_ratio, 
                                        loss_trend, step_budget)
            action, action_prob = self.agent.select_action(state, explore=True)
            
            # Save checkpoint
            checkpoint = self.save_checkpoint()
            initial_anchor_mse = anchor_mse
            
            # Apply adaptation strategy
            use_anchor = self.apply_adaptation_action(action, optimizer, base_lr, anchor_loader)
            
            # Perform update on current batch
            optimizer.zero_grad()
            output = self.model(X_batch)
            loss = self.criterion(output, Y_batch)
            
            # Add anchor replay if selected
            if use_anchor and anchor_loader is not None:
                try:
                    anchor_X, anchor_Y = next(iter(anchor_loader))
                    anchor_X = anchor_X.to(self.device)
                    anchor_Y = anchor_Y.to(self.device)
                    
                    anchor_output = self.model(anchor_X)
                    anchor_loss = self.criterion(anchor_output, anchor_Y)
                    
                    # Combine losses
                    loss = 0.7 * loss + 0.3 * anchor_loss
                except:
                    pass
            
            if loss.requires_grad:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                self.stats['total_updates'] += 1
            
            # Re-enable all gradients
            for param in self.model.parameters():
                param.requires_grad = True
            
            # Check for forgetting
            new_anchor_mse = self.anchor_memory.evaluate(self.model, self.device)
            forgetting = new_anchor_mse - initial_anchor_mse
            
            # Get dynamic threshold
            dynamic_threshold = self.get_dynamic_threshold(initial_anchor_mse, snr_value)
            self.stats['forgetting_values'].append(forgetting)
            self.stats['threshold_values'].append(dynamic_threshold)
            
            # Rollback if excessive forgetting
            if forgetting > dynamic_threshold:
                self.restore_checkpoint(checkpoint)
                new_anchor_mse = initial_anchor_mse
                forgetting = 0.0
                self.stats['rollback_count'] += 1
            
            # Compute reward
            improvement = -current_mse
            forgetting_penalty = max(0, forgetting) * 100.0
            compute_cost = 0.01 * action
            reward = 10.0 * (improvement - forgetting_penalty - compute_cost)
            
            # Store episode data
            next_state = self.agent.get_state(loss.item(), new_anchor_mse, 
                                             snr_ratio, loss_trend, step_budget)
            self.episode_states.append(state)
            self.episode_actions.append(action)
            self.episode_rewards.append(reward)
            
            # Track statistics
            recent_losses.append(loss.item())
            running_loss += loss.item()
            self.stats['anchor_mse_history'].append(new_anchor_mse)
            self.stats['snr_history'].append(snr_value)
            
            # Update progress bar
            loop.set_postfix(
                loss=f"{loss.item():.6f}",
                action=self.agent.actions[action][:8],
                anchor=f"{new_anchor_mse:.5f}",
                rollback=self.stats['rollback_count']
            )
        
        avg_loss = running_loss / total_batches
        self.stats['epoch_losses'].append(avg_loss)
        
        # Update agent periodically
        if len(self.episode_states) >= 64:
            next_states = self.episode_states[1:] + [self.episode_states[-1]]
            agent_loss = self.agent.update(
                self.episode_states,
                self.episode_actions,
                self.episode_rewards,
                next_states
            )
            print(f"\nAgent updated - Loss: {agent_loss:.4f} | "
                  f"Rollbacks: {self.stats['rollback_count']}/{self.stats['total_updates']}")
            
            self.episode_states = []
            self.episode_actions = []
            self.episode_rewards = []
        
        return avg_loss
    
    def get_statistics(self):
        """Return training statistics"""
        stats = self.stats.copy()
        if self.stats['total_updates'] > 0:
            stats['rollback_rate'] = self.stats['rollback_count'] / self.stats['total_updates']
        else:
            stats['rollback_rate'] = 0.0
        return stats
    
    def save_statistics(self, filepath):
        """Save statistics to JSON"""
        stats = self.get_statistics()
        stats['action_counts'] = dict(stats['action_counts'])
        
        # Convert lists to serializable format
        stats['forgetting_values'] = [float(x) for x in stats['forgetting_values'][-1000:]]
        stats['threshold_values'] = [float(x) for x in stats['threshold_values'][-1000:]]
        stats['epoch_losses'] = [float(x) for x in stats['epoch_losses']]
        stats['anchor_mse_history'] = [float(x) for x in stats['anchor_mse_history'][-1000:]]
        stats['snr_history'] = [float(x) for x in stats['snr_history'][-1000:]]
        
        with open(filepath, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"Statistics saved to {filepath}")


def train_aca_channel_estimator(model_type='SRCNN', snr_list=[8, 10, 12, 14, 16, 18, 20, 22],
                                 num_pilots=48, epochs_per_snr=50, save_dir='./aca_checkpoints',
                                 forgetting_threshold=0.15, device='cuda'):
    """
    Train channel estimator with ACA across multiple SNR regimes
    
    Args:
        model_type: 'SRCNN' or 'DNCNN'
        snr_list: List of SNR values to train on sequentially
        num_pilots: Number of pilot signals
        epochs_per_snr: Training epochs per SNR value
        save_dir: Directory to save checkpoints
        forgetting_threshold: Base forgetting threshold
        device: Device to use
    
    Returns:
        Trained model and statistics
    """
    os.makedirs(save_dir, exist_ok=True)
    
    print("="*70)
    print(f"ACA Training: {model_type}")
    print("="*70)
    print(f"SNR Sequence: {snr_list}")
    print(f"Epochs per SNR: {epochs_per_snr}")
    print(f"Forgetting Threshold: {forgetting_threshold}")
    print("="*70)
    
    # Load perfect channel data
    print("\nLoading perfect channel data...")
    perfect = loadmat("Perfect_H_40000.mat")['My_perfect_H']
    print(f"Perfect channel shape: {perfect.shape}")
    
    # Initialize model
    if model_type == 'SRCNN':
        model = SRCNN().to(device)
    else:
        model = DNCNN(depth=20, n_channels=64).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Initialize ACA components
    anchor_memory = AnchorMemory(max_size=256)
    agent = AdaptationAgent(state_dim=5, hidden_dim=128, num_actions=7).to(device)
    aca_trainer = ACATrainer(model, agent, anchor_memory, device,
                            forgetting_threshold=forgetting_threshold,
                            adaptive_threshold=True)
    
    # Set random seed
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Train on each SNR sequentially
    for snr_idx, snr in enumerate(snr_list):
        print(f"\n{'='*70}")
        print(f"Training on SNR = {snr} dB ({snr_idx+1}/{len(snr_list)})")
        print(f"{'='*70}")
        
        # Generate noisy data for this SNR
        print(f"Generating noisy channel data (SNR={snr}dB)...")
        noisy_channel = add_awgn_noise(perfect, snr)
        
        # Interpolate
        print("Interpolating channel estimates...")
        interp_noisy = interpolation(noisy_channel, snr, num_pilots, 'rbf')
        
        # Prepare perfect labels
        perfect_image = np.zeros((len(perfect), 72, 14, 2))
        perfect_image[:, :, :, 0] = np.real(perfect)
        perfect_image[:, :, :, 1] = np.imag(perfect)
        perfect_image = np.concatenate(
            (perfect_image[:, :, :, 0], perfect_image[:, :, :, 1]),
            axis=0
        ).reshape(2*len(perfect), 72, 14, 1)
        
        # Train/val split
        idx_random = np.random.rand(len(perfect_image)) < 0.8
        train_data = interp_noisy[idx_random]
        train_label = perfect_image[idx_random]
        val_data = interp_noisy[~idx_random]
        val_label = perfect_image[~idx_random]
        
        print(f"Train samples: {len(train_data)}, Val samples: {len(val_data)}")
        
        # Convert to tensors
        train_data_tensor = torch.FloatTensor(train_data).permute(0, 3, 1, 2)
        train_label_tensor = torch.FloatTensor(train_label).permute(0, 3, 1, 2)
        
        # Create dataloader
        train_dataset = TensorDataset(train_data_tensor, train_label_tensor)
        train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
        
        # Initialize optimizer for this SNR
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        
        # Train with ACA
        for epoch in range(1, epochs_per_snr + 1):
            avg_loss = aca_trainer.train_epoch(
                train_loader, optimizer, base_lr=1e-3,
                epoch=epoch, snr_value=snr, total_epochs=epochs_per_snr,
                current_snr_idx=snr_idx, total_snrs=len(snr_list)
            )
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}/{epochs_per_snr} - Avg Loss: {avg_loss:.6f}")
        
        # Add samples to anchor memory
        samples_to_add = min(256 // len(snr_list), 50)
        print(f"\nAdding {samples_to_add} samples to anchor memory...")
        anchor_memory.add_samples(train_loader, num_samples=samples_to_add)
        print(f"Anchor memory size: {len(anchor_memory.buffer)}")
        
        # Save checkpoint after each SNR
        checkpoint_path = os.path.join(save_dir, f'aca_{model_type}_snr{snr}.pth')
        torch.save({
            'model_state_dict': model.state_dict(),
            'agent_state_dict': agent.state_dict(),
            'snr': snr,
            'snr_idx': snr_idx,
            'model_type': model_type
        }, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")
    
    # Save final model and statistics
    final_checkpoint = os.path.join(save_dir, f'aca_{model_type}_final.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'agent_state_dict': agent.state_dict(),
        'model_type': model_type,
        'snr_list': snr_list,
        'statistics': aca_trainer.get_statistics()
    }, final_checkpoint)
    
    print(f"\n{'='*70}")
    print("Training completed!")
    print(f"{'='*70}")
    
    # Save and print statistics
    stats_path = os.path.join(save_dir, f'aca_{model_type}_statistics.json')
    aca_trainer.save_statistics(stats_path)
    
    stats = aca_trainer.get_statistics()
    print(f"\nFinal Statistics:")
    print(f"  Total Updates: {stats['total_updates']}")
    print(f"  Rollbacks: {stats['rollback_count']}")
    print(f"  Rollback Rate: {stats['rollback_rate']:.2%}")
    print(f"\n  Action Distribution:")
    for action, count in stats['action_counts'].items():
        percentage = (count / stats['total_updates'] * 100) if stats['total_updates'] > 0 else 0
        print(f"    {action}: {count} ({percentage:.1f}%)")
    
    return model, stats


def test_aca_model(checkpoint_path, snr_list=[8, 10, 12, 14, 16, 18, 20, 22],
                   num_pilots=48, batch_size=128, device='cuda'):
    """
    Test trained ACA model across multiple SNRs (with batch processing to avoid OOM)
    
    Args:
        checkpoint_path: Path to trained model checkpoint
        snr_list: List of SNR values to test on
        num_pilots: Number of pilots
        batch_size: Batch size for testing (default: 128)
        device: Device to use
    
    Returns:
        Dictionary of MSE results for each SNR
    """
    print(f"\nLoading model from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_type = checkpoint['model_type']
    
    # Initialize model
    if model_type == 'SRCNN':
        model = SRCNN().to(device)
    else:
        model = DNCNN(depth=20, n_channels=64).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load perfect channel
    perfect = loadmat("Perfect_H_40000.mat")['My_perfect_H']
    
    # Set random seed
    np.random.seed(42)
    
    results = {'model_type': model_type, 'test_results': {}}
    
    print("\n" + "="*70)
    print("Testing on Multiple SNRs")
    print("="*70)
    
    for snr in snr_list:
        print(f"\nTesting SNR = {snr} dB...")
        
        # Generate test data
        noisy_channel = add_awgn_noise(perfect, snr)
        interp_noisy = interpolation(noisy_channel, snr, num_pilots, 'rbf')
        
        perfect_image = np.zeros((len(perfect), 72, 14, 2))
        perfect_image[:, :, :, 0] = np.real(perfect)
        perfect_image[:, :, :, 1] = np.imag(perfect)
        perfect_image = np.concatenate(
            (perfect_image[:, :, :, 0], perfect_image[:, :, :, 1]),
            axis=0
        ).reshape(2*len(perfect), 72, 14, 1)
        
        # Use validation split
        idx_random = np.random.rand(len(perfect_image)) < 0.8
        test_data = interp_noisy[~idx_random]
        test_label = perfect_image[~idx_random]
        
        # Convert to tensors
        test_data_tensor = torch.FloatTensor(test_data).permute(0, 3, 1, 2)
        test_label_tensor = torch.FloatTensor(test_label).permute(0, 3, 1, 2)
        
        # Create test dataloader
        test_dataset = TensorDataset(test_data_tensor, test_label_tensor)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        # Predict in batches
        total_mse = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for X_batch, Y_batch in tqdm(test_loader, desc=f"Testing SNR {snr}dB"):
                X_batch = X_batch.to(device)
                Y_batch = Y_batch.to(device)
                
                predictions = model(X_batch)
                batch_mse = nn.MSELoss()(predictions, Y_batch).item()
                total_mse += batch_mse
                num_batches += 1
                
                # Clear cache periodically
                if num_batches % 10 == 0:
                    torch.cuda.empty_cache()
        
        avg_mse = total_mse / num_batches
        results['test_results'][snr] = {'mse': avg_mse, 'num_samples': len(test_data)}
        print(f"  MSE: {avg_mse:.6f}")
        
        # Clear cache after each SNR
        torch.cuda.empty_cache()
    
    # Save results
    results_path = checkpoint_path.replace('.pth', '_test_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")
    
    return results


def plot_aca_performance(results, save_path='aca_performance.png'):
    """Plot performance across SNRs"""
    snrs = sorted(results['test_results'].keys())
    mses = [results['test_results'][snr]['mse'] for snr in snrs]
    
    plt.figure(figsize=(10, 6))
    plt.plot(snrs, mses, 'o-', linewidth=2, markersize=10)
    plt.xlabel('SNR (dB)', fontsize=12)
    plt.ylabel('MSE', fontsize=12)
    plt.title(f'{results["model_type"]} - ACA Cross-SNR Performance', 
              fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    for snr, mse in zip(snrs, mses):
        plt.annotate(f'{mse:.4f}', xy=(snr, mse), xytext=(0, 10),
                    textcoords='offset points', ha='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {save_path}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ACA Channel Estimation Training')
    parser.add_argument('--mode', type=str, default='both', 
                       choices=['train', 'test', 'both'],
                       help='Mode: train, test, or both')
    parser.add_argument('--model_type', type=str, default='DNCNN',
                       choices=['SRCNN', 'DNCNN'],
                       help='Model architecture')
    parser.add_argument('--snr_list', type=int, nargs='+', 
                       default=[8, 10, 12, 14, 16, 18, 20, 22, 25, 30],
                       help='List of SNR values to train/test on')
    parser.add_argument('--epochs_per_snr', type=int, default=50,
                       help='Training epochs per SNR')
    parser.add_argument('--num_pilots', type=int, default=48,
                       help='Number of pilot signals')
    parser.add_argument('--forgetting_threshold', type=float, default=0.15,
                       help='Forgetting threshold')
    parser.add_argument('--save_dir', type=str, default='./aca_checkpoints',
                       help='Directory to save checkpoints')
    parser.add_argument('--checkpoint_path', type=str, default="./aca_checkpoints/aca_DNCNN_snr22.pth",
                       help='Path to checkpoint for testing')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to use')
    
    args = parser.parse_args()
    
    # Check device availability
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'
    
    if args.mode in ['train', 'both']:
        model, stats = train_aca_channel_estimator(
            model_type=args.model_type,
            snr_list=args.snr_list,
            num_pilots=args.num_pilots,
            epochs_per_snr=args.epochs_per_snr,
            save_dir=args.save_dir,
            forgetting_threshold=args.forgetting_threshold,
            device=args.device
        )
        
        final_checkpoint = os.path.join(args.save_dir, f'aca_{args.model_type}_final.pth')
        
        if args.mode == 'both':
            results = test_aca_model(final_checkpoint, args.snr_list, 
                                    args.num_pilots, args.device)
            plot_aca_performance(results, 
                               f'{args.save_dir}/aca_{args.model_type}_performance.png')
    
    elif args.mode == 'test':
        if args.checkpoint_path is None:
            print("Error: --checkpoint_path required for test mode")
        else:
            results = test_aca_model(args.checkpoint_path, args.snr_list,
                                    args.num_pilots,batch_size=128, device=args.device)
            plot_aca_performance(results,
                               f'aca_{args.model_type}_test_performance.png')
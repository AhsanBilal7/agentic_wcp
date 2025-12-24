import argparse
import random
import csv
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, Dataset
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR
from model import *
from dataloader import get_all_datasets
from utils import compute_device, evaluate_nmse_vs_snr_masked
from torch.nn.utils import clip_grad_norm_
from loss import *
import math
import copy
from collections import deque

device = compute_device()
BATCH_SIZE = 2048
SEQ_LEN    = 32
NUM_EPOCHS = 100
ALPHA      = 0.2
LR         = 1e-4
SNR_LIST   = [0,5,10,12,14,16,18,20,22,24,26,28,30]

# ACA-specific hyperparameters
ANCHOR_SIZE = 128
FORGETTING_THRESHOLD = 0.15
AGENT_LR = 1e-3
REWARD_SCALE = 10.0

print("Loading datasets...")
train_S1, test_S1, train_loader_S1, test_loader_S1, \
train_S2, test_S2, train_loader_S2, test_loader_S2, \
train_S3, test_S3, train_loader_S3, test_loader_S3 = \
    get_all_datasets(
        data_dir      = "../dataset/outputs/",
        batch_size    = BATCH_SIZE,
        dataset_id    = "all",
        normalization = "log_min_max",
        per_user      = True,
        seq_len       = SEQ_LEN
    )
print("Loaded datasets successfully.")

parser = argparse.ArgumentParser()
parser.add_argument('--model_type', type=str, default='LSTM',
                    choices=['LSTM', 'GRU', 'TRANS'],
                    help='Model type for continual learning')
args = parser.parse_args()


class AnchorMemory:
    """Maintains a small buffer of representative samples from past tasks."""
    def __init__(self, max_size=ANCHOR_SIZE):
        self.max_size = max_size
        self.buffer = []
    
    def add_samples(self, dataloader, num_samples):
        """Add representative samples from a dataloader using reservoir sampling."""
        samples_added = 0
        for X_batch, Y_batch in dataloader:
            batch_size = X_batch.size(0)
            if samples_added + batch_size > num_samples:
                remaining = num_samples - samples_added
                X_batch = X_batch[:remaining]
                Y_batch = Y_batch[:remaining]
            
            for i in range(X_batch.size(0)):
                if len(self.buffer) >= self.max_size:
                    idx = random.randint(0, samples_added + i)
                    if idx < self.max_size:
                        self.buffer[idx] = (X_batch[i], Y_batch[i])
                else:
                    self.buffer.append((X_batch[i], Y_batch[i]))
            
            samples_added += X_batch.size(0)
            if samples_added >= num_samples:
                break
    
    def evaluate(self, model, device):
        """Evaluate model performance on anchor samples."""
        if len(self.buffer) == 0:
            return 0.0
        
        model.eval()
        total_nmse = 0.0
        
        with torch.no_grad():
            for X, Y in self.buffer:
                X = X.unsqueeze(0).to(device)
                Y = Y.unsqueeze(0).to(device)
                mag_t, mask_t = Y[:, 0], Y[:, 1]
                
                mag_p, mask_logits = model(X)
                nmse = masked_nmse(mag_p, mag_t, mask_t)
                total_nmse += nmse.item()
        
        model.train()
        return total_nmse / len(self.buffer)


class AdaptationAgent(nn.Module):
    """Agent that decides how to adapt the model based on observed state."""
    def __init__(self, state_dim=4, hidden_dim=64, num_actions=6):
        super().__init__()
        self.policy_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
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
            'FULL_UPDATE',
            'LAST_LAYER',
            'FREEZE_EARLY',
            'SMALL_LR',
            'FEW_STEPS',
            'ADAPTIVE_MIX'
        ]
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=AGENT_LR)
        self.eps = 0.2
        
    def get_state(self, current_nmse, anchor_nmse, drift_signal, step_budget):
        """Construct agent state from observations."""
        return torch.tensor([
            current_nmse,
            anchor_nmse,
            drift_signal,
            step_budget
        ], dtype=torch.float32)
    
    def select_action(self, state, explore=True):
        """Select adaptation action based on policy."""
        state = state.unsqueeze(0).to(next(self.parameters()).device)
        
        with torch.no_grad():
            action_logits = self.policy_net(state)
            action_probs = torch.softmax(action_logits, dim=-1)
        
        if explore and random.random() < self.eps:
            action = random.randint(0, len(self.actions) - 1)
        else:
            action = torch.argmax(action_probs, dim=-1).item()
        
        return action, action_probs[0, action].item()
    
    def update(self, states, actions, rewards, next_states):
        """Update agent policy using policy gradient and value function."""
        states = torch.stack(states).to(next(self.parameters()).device)
        actions = torch.tensor(actions, dtype=torch.long).to(next(self.parameters()).device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(next(self.parameters()).device)
        next_states = torch.stack(next_states).to(next(self.parameters()).device)
        
        values = self.value_net(states).squeeze()
        next_values = self.value_net(next_states).squeeze().detach()
        
        td_target = rewards + 0.99 * next_values
        value_loss = nn.MSELoss()(values, td_target)
        
        action_logits = self.policy_net(states)
        action_probs = torch.softmax(action_logits, dim=-1)
        log_probs = torch.log(action_probs.gather(1, actions.unsqueeze(1)).squeeze() + 1e-8)
        
        advantages = (td_target - values).detach()
        policy_loss = -(log_probs * advantages).mean()
        
        loss = policy_loss + 0.5 * value_loss
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()


class ACATrainer:
    """Agentic Continual Adaptation trainer."""
    def __init__(self, model, agent, anchor_memory, device):
        self.model = model
        self.agent = agent
        self.anchor_memory = anchor_memory
        self.device = device
        self.bce_loss = nn.BCEWithLogitsLoss()
        
        self.episode_states = []
        self.episode_actions = []
        self.episode_rewards = []
        
    def save_checkpoint(self):
        """Save model state for potential rollback."""
        return copy.deepcopy(self.model.state_dict())
    
    def restore_checkpoint(self, checkpoint):
        """Restore model from checkpoint."""
        self.model.load_state_dict(checkpoint)
    
    def compute_drift_signal(self, recent_nmse_buffer):
        """Compute channel drift signal from recent NMSE statistics."""
        if len(recent_nmse_buffer) < 2:
            return 0.0
        nmse_list = list(recent_nmse_buffer)
        return abs(nmse_list[-1] - nmse_list[-2])
    
    def apply_adaptation_action(self, action, optimizer, base_lr):
        """Apply the selected adaptation action to the model."""
        action_name = self.agent.actions[action]
        
        if action_name == 'FULL_UPDATE':
            for param in self.model.parameters():
                param.requires_grad = True
            for param_group in optimizer.param_groups:
                param_group['lr'] = base_lr
            num_steps = 1
            
        elif action_name == 'LAST_LAYER':
            for name, param in self.model.named_parameters():
                if 'fc_out' in name or 'output' in name or 'decoder' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
            num_steps = 1
            
        elif action_name == 'FREEZE_EARLY':
            total_layers = len(list(self.model.parameters()))
            for idx, param in enumerate(self.model.parameters()):
                if idx < total_layers // 2:
                    param.requires_grad = False
                else:
                    param.requires_grad = True
            num_steps = 1
            
        elif action_name == 'SMALL_LR':
            for param in self.model.parameters():
                param.requires_grad = True
            for param_group in optimizer.param_groups:
                param_group['lr'] = base_lr * 0.1
            num_steps = 1
            
        elif action_name == 'FEW_STEPS':
            for param in self.model.parameters():
                param.requires_grad = True
            for param_group in optimizer.param_groups:
                param_group['lr'] = base_lr
            num_steps = 1
            
        else:
            total_layers = len(list(self.model.parameters()))
            for idx, param in enumerate(self.model.parameters()):
                if idx < total_layers // 3:
                    param.requires_grad = False
                else:
                    param.requires_grad = True
            for param_group in optimizer.param_groups:
                param_group['lr'] = base_lr * 0.5
            num_steps = 1
        
        return num_steps
    
    def train_epoch(self, dataloader, optimizer, base_lr, epoch, task_name, step_budget=1.0):
        """Train one epoch with agentic adaptation."""
        self.model.train()
        running_loss = 0.0
        recent_nmse_buffer = deque(maxlen=10)
        total_batches = len(dataloader)
        
        loop = tqdm(
            enumerate(dataloader, 1),
            total=total_batches,
            desc=f"{task_name} Epoch {epoch}/{NUM_EPOCHS}"
        )
        
        for batch_idx, (X_batch, Y_batch) in loop:
            X_batch = X_batch.to(self.device)
            Y_batch = Y_batch.to(self.device)
            mag_t, mask_t = Y_batch[:, 0], Y_batch[:, 1]
            
            self.model.eval()
            with torch.no_grad():
                mag_p_eval, mask_logits_eval = self.model(X_batch)
                current_nmse = masked_nmse(mag_p_eval, mag_t, mask_t).item()
            self.model.train()
            
            anchor_nmse = self.anchor_memory.evaluate(self.model, self.device)
            drift_signal = self.compute_drift_signal(recent_nmse_buffer)
            recent_nmse_buffer.append(current_nmse)
            
            state = self.agent.get_state(current_nmse, anchor_nmse, drift_signal, step_budget)
            action, action_prob = self.agent.select_action(state, explore=True)
            
            checkpoint = self.save_checkpoint()
            initial_anchor_nmse = anchor_nmse
            
            num_steps = self.apply_adaptation_action(action, optimizer, base_lr)
            
            optimizer.zero_grad()
            mag_p, mask_logits = self.model(X_batch)
            loss_mag = masked_nmse(mag_p, mag_t, mask_t)
            loss_mask = self.bce_loss(mask_logits, mask_t)
            loss = loss_mag + ALPHA * loss_mask
            
            loss.backward()
            clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()
            
            new_anchor_nmse = self.anchor_memory.evaluate(self.model, self.device)
            forgetting = new_anchor_nmse - initial_anchor_nmse
            
            if forgetting > FORGETTING_THRESHOLD:
                self.restore_checkpoint(checkpoint)
                new_anchor_nmse = initial_anchor_nmse
                forgetting = 0.0
            
            improvement = -loss_mag.item()
            forgetting_penalty = max(0, forgetting) * 10.0
            compute_cost = 0.01 * action
            reward = REWARD_SCALE * (improvement - forgetting_penalty - compute_cost)
            
            next_state = self.agent.get_state(loss_mag.item(), new_anchor_nmse, drift_signal, step_budget)
            self.episode_states.append(state)
            self.episode_actions.append(action)
            self.episode_rewards.append(reward)
            
            for param in self.model.parameters():
                param.requires_grad = True
            
            running_loss += loss.item()
            loop.set_postfix(
                nmse=loss_mag.item(),
                bce=loss_mask.item(),
                action=self.agent.actions[action][:8],
                anchor=f"{new_anchor_nmse:.4f}"
            )
        
        avg_loss = running_loss / total_batches
        
        if len(self.episode_states) > 32:
            next_states = self.episode_states[1:] + [self.episode_states[-1]]
            agent_loss = self.agent.update(
                self.episode_states,
                self.episode_actions,
                self.episode_rewards,
                next_states
            )
            print(f"Agent updated - Loss: {agent_loss:.4f}")
            
            self.episode_states = []
            self.episode_actions = []
            self.episode_rewards = []
        
        return avg_loss


if args.model_type == 'GRU':
    model = GRUModel(input_dim=1, hidden_dim=32, output_dim=1, n_layers=3, H=16, W=9).to(device)
elif args.model_type == 'LSTM':
    model = LSTMChannelPredictor().to(device)
else:
    model = TransformerModel(dim_val=128, n_heads=4, n_encoder_layers=1,
                             n_decoder_layers=1, out_channels=2, H=16, W=9).to(device)

anchor_memory = AnchorMemory(max_size=ANCHOR_SIZE)
agent = AdaptationAgent(state_dim=4, hidden_dim=64, num_actions=6).to(device)
aca_trainer = ACATrainer(model, agent, anchor_memory, device)

print("=== Task 1: S1 ===")
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
sched = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)

for epoch in range(1, NUM_EPOCHS + 1):
    avg_loss = aca_trainer.train_epoch(
        train_loader_S1, optimizer, base_lr=1e-3, 
        epoch=epoch, task_name="S1", step_budget=1.0
    )
    sched.step()
    print(f"Epoch {epoch} S1 — Avg Loss: {avg_loss:.4f}")

print("Adding S1 samples to anchor memory...")
anchor_memory.add_samples(train_loader_S1, num_samples=ANCHOR_SIZE // 3)

print("=== Task 2: S2 ===")
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
sched = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)

for epoch in range(1, NUM_EPOCHS + 1):
    avg_loss = aca_trainer.train_epoch(
        train_loader_S2, optimizer, base_lr=1e-3,
        epoch=epoch, task_name="S2", step_budget=1.0
    )
    sched.step()
    print(f"Epoch {epoch} S2 — Avg Loss: {avg_loss:.4f}")

print("Adding S2 samples to anchor memory...")
anchor_memory.add_samples(train_loader_S2, num_samples=ANCHOR_SIZE // 3)

print("=== Task 3: S3 ===")
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
sched = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)

for epoch in range(1, NUM_EPOCHS + 1):
    avg_loss = aca_trainer.train_epoch(
        train_loader_S3, optimizer, base_lr=1e-3,
        epoch=epoch, task_name="S3", step_budget=1.0
    )
    sched.step()
    print(f"Epoch {epoch} S3 — Avg Loss: {avg_loss:.4f}")

print("Adding S3 samples to anchor memory...")
anchor_memory.add_samples(train_loader_S3, num_samples=ANCHOR_SIZE // 3)

print("\n=== NMSE Evaluation ===")
nmse_results = {
    'S1_Compact': evaluate_nmse_vs_snr_masked(model, test_loader_S1, device, SNR_LIST),
    'S2_Dense': evaluate_nmse_vs_snr_masked(model, test_loader_S2, device, SNR_LIST),
    'S3_Standard': evaluate_nmse_vs_snr_masked(model, test_loader_S3, device, SNR_LIST),
}

csv_rows = [['Task', 'SNR', 'NMSE Masked', 'NMSE (dB)']]
for task, res in nmse_results.items():
    for snr, nmse in res.items():
        nmse_db = -10 * math.log10(nmse + 1e-12)
        print(f"Task {task} | SNR {snr:2d} → NMSE {nmse:.6f} | {nmse_db:.2f} dB")
        csv_rows.append([task, snr, f"{nmse:.6f}", f"{nmse_db:.2f}"])

csv_path = f"aca_{args.model_type}_nmse_results.csv"
with open(csv_path, 'w', newline='') as f:
    csv.writer(f).writerows(csv_rows)

print(f"\nResults saved to {csv_path}")
# Agentic Continual Adaptation for Channel Estimation

## Overview

This implementation applies **Agentic Continual Adaptation (ACA)** to the channel estimation problem, enabling models (SRCNN/DNCNN) to learn across multiple SNR regimes without catastrophic forgetting.

## Key Problem Being Solved

**Traditional approach**: Train a separate model for each SNR value → inefficient, no knowledge transfer

**Our approach**: Train a single model sequentially across SNR values with an RL agent dynamically selecting the best adaptation strategy → efficient learning with knowledge retention

## How the Agentic System Works

### 1. **Architecture Components**

#### AdaptationAgent (RL Agent)
- **State Space** (5D):
  - `current_mse`: Model performance on current SNR batch
  - `anchor_mse`: Performance on historical samples (detects forgetting)
  - `snr_ratio`: Progress through SNR sequence (0 to 1)
  - `loss_trend`: Recent loss trajectory (improving/degrading)
  - `step_budget`: Training progress within current SNR (0 to 1)

- **Action Space** (7 discrete strategies):
  1. `FULL_UPDATE`: Standard gradient descent (all parameters)
  2. `LAST_LAYER`: Only update output layers (preserve features)
  3. `FREEZE_EARLY`: Freeze first 50% of network (protect learned representations)
  4. `SMALL_LR`: Reduce learning rate 10x (conservative updates)
  5. `ANCHOR_REPLAY`: Mix current + historical samples (50% LR reduction)
  6. `ADAPTIVE_MIX`: Freeze first 33%, reduce LR 50%, use anchor replay
  7. `CONSERVATIVE`: Very small updates (5% LR)

- **Policy**: Actor-Critic with epsilon-greedy exploration (ε=0.2)
- **Learning**: Policy gradient + value function with TD learning

#### AnchorMemory (Experience Replay)
- Maintains 256 representative samples from past SNR regimes
- Uses reservoir sampling for diversity
- Evaluates forgetting by testing model on historical data
- Can generate mini-batches for anchor replay

#### ACATrainer (Training Coordinator)
- Manages training loop with adaptive strategies
- Implements checkpoint/rollback mechanism
- Computes dynamic forgetting thresholds
- Tracks extensive statistics

### 2. **Training Flow Per Batch**

```
1. OBSERVE STATE
   ├─ Evaluate current batch performance (MSE)
   ├─ Test on anchor memory (historical performance)
   ├─ Compute SNR progression, loss trend
   └─ Construct 5D state vector

2. AGENT DECIDES
   ├─ Agent policy network processes state
   ├─ Selects action (strategy) via ε-greedy
   └─ Returns action ID and probability

3. APPLY STRATEGY
   ├─ Freeze/unfreeze layers based on action
   ├─ Adjust learning rate
   ├─ Optionally mix in anchor samples
   └─ Save model checkpoint

4. EXECUTE UPDATE
   ├─ Forward pass on current batch
   ├─ (Optional) Forward pass on anchor batch
   ├─ Compute weighted loss
   ├─ Backward pass with gradient clipping
   └─ Optimizer step

5. CHECK FORGETTING
   ├─ Re-evaluate anchor memory performance
   ├─ Compute forgetting = new_anchor_mse - old_anchor_mse
   ├─ Compare to dynamic threshold
   └─ If forgetting > threshold → ROLLBACK to checkpoint

6. COMPUTE REWARD
   reward = 10 * (improvement - 100*forgetting - 0.01*action_cost)
   ├─ Improvement: Negative MSE (lower is better)
   ├─ Forgetting penalty: 100x amplification (strongly discourage)
   └─ Compute cost: Slight bias toward simpler actions

7. LEARN FROM EXPERIENCE
   ├─ Store (state, action, reward, next_state)
   ├─ Every 64 steps: Update agent via actor-critic
   └─ Policy gradient with advantage estimation
```

### 3. **Dynamic Forgetting Thresholds**

The system adapts forgetting tolerance based on current performance:

```python
if anchor_mse < 0.001:      # Excellent performance
    threshold = 0.08        # Be very strict
elif anchor_mse < 0.005:    # Good performance  
    threshold = 0.15        # Moderate strictness (base)
else:                       # Poor performance
    threshold = 0.30        # Allow more forgetting
```

**Rationale**: When the model is performing well on historical data, we protect that knowledge aggressively. When performance is already poor, we allow more plasticity to adapt to new conditions.

### 4. **Multi-SNR Training Strategy**

```
For each SNR in [8, 10, 12, 14, 16, 18, 20, 22] dB:
    1. Generate noisy channel data at current SNR
    2. Train for N epochs with ACA
    3. Add representative samples to anchor memory
    4. Save checkpoint
    → Model learns progressively harder/different noise conditions
```

**Key insight**: The agent learns to recognize different noise regimes and select appropriate adaptation strategies automatically.

## Advantages Over Standard Training

### Standard Training
- ❌ One model per SNR (8 models needed)
- ❌ No knowledge transfer between SNRs
- ❌ Catastrophic forgetting when training sequentially
- ❌ Fixed update strategy (no adaptation)

### ACA Training  
- ✅ Single model for all SNRs
- ✅ Knowledge transfer via anchor memory
- ✅ Forgetting prevention via rollback
- ✅ Adaptive strategies via RL agent
- ✅ Efficient compute (selective updates)

## Usage Examples

### Train from scratch
```bash
python aca_channel_estimation.py \
    --mode train \
    --model_type SRCNN \
    --snr_list 8 10 12 14 16 18 20 22 \
    --epochs_per_snr 50 \
    --forgetting_threshold 0.15 \
    --save_dir ./aca_checkpoints
```

### Test trained model
```bash
python aca_channel_estimation.py \
    --mode test \
    --model_type SRCNN \
    --checkpoint_path ./aca_checkpoints/aca_SRCNN_final.pth \
    --snr_list 8 10 12 14 16 18 20 22
```

### Train and test
```bash
python aca_channel_estimation.py \
    --mode both \
    --model_type DNCNN \
    --epochs_per_snr 30 \
    --forgetting_threshold 0.12
```

## Key Hyperparameters

- `epochs_per_snr`: Training epochs per SNR value (default: 50)
  - Higher → Better adaptation, longer training
  - Lower → Faster training, potential underfitting

- `forgetting_threshold`: Base forgetting tolerance (default: 0.15)
  - Higher → More plasticity, less protection
  - Lower → More protection, less adaptation
  - Adaptive mode adjusts this automatically

- `anchor_memory_size`: Number of historical samples (default: 256)
  - Larger → Better forgetting detection, more memory
  - Smaller → Less accurate detection, faster

## Expected Outputs

### Training
```
./aca_checkpoints/
├── aca_SRCNN_snr8.pth          # Checkpoint after SNR=8
├── aca_SRCNN_snr10.pth         # Checkpoint after SNR=10
├── ...
├── aca_SRCNN_final.pth         # Final trained model
├── aca_SRCNN_statistics.json   # Training statistics
└── aca_SRCNN_performance.png   # Performance plot
```

### Statistics File
```json
{
  "total_updates": 12800,
  "rollback_count": 423,
  "rollback_rate": 0.033,
  "action_counts": {
    "FULL_UPDATE": 6234,
    "LAST_LAYER": 2145,
    "FREEZE_EARLY": 1832,
    "SMALL_LR": 987,
    "ANCHOR_REPLAY": 654,
    "ADAPTIVE_MIX": 732,
    "CONSERVATIVE": 216
  },
  "epoch_losses": [...],
  "anchor_mse_history": [...],
  "forgetting_values": [...],
  "threshold_values": [...]
}
```

## Understanding the Statistics

### Rollback Rate
- **0-5%**: Excellent - Agent learned good strategies, minimal forgetting
- **5-10%**: Good - Some forgetting, but controlled
- **10-20%**: Fair - Frequent forgetting, may need tuning
- **>20%**: Poor - Excessive forgetting, increase threshold or reduce LR

### Action Distribution
- High `FULL_UPDATE`: Agent confident, stable learning
- High `ANCHOR_REPLAY`: Significant domain shift between SNRs
- High `CONSERVATIVE`: Agent cautious, protecting knowledge
- Balanced distribution: Agent adapting strategies appropriately

## Integration with Your Existing Code

The ACA system is designed to work with your existing models. You need to:

1. **Import your model classes** in `aca_channel_estimation.py`:
```python
from models import SRCNN, DNCNN, interpolation
from channel_utils import add_awgn_noise
```

2. **Ensure your models inherit from `nn.Module`** (already done)

3. **Use the same data preprocessing** (interpolation, normalization)

## Comparison to Original Multi-SNR Script

| Aspect | Original Script | ACA Script |
|--------|----------------|------------|
| Training | Fixed strategy | Adaptive strategy |
| Model count | 1 per SNR | 1 for all SNRs |
| Knowledge transfer | None | Via anchor memory |
| Forgetting | Catastrophic | Prevented |
| Optimization | Manual | Learned by agent |
| Flexibility | None | High (7 strategies) |

## Advanced: Customizing the Agent

### Add new actions
```python
self.actions = [
    'FULL_UPDATE',
    'YOUR_NEW_STRATEGY',  # Add here
    ...
]

# Then implement in apply_adaptation_action():
elif action_name == 'YOUR_NEW_STRATEGY':
    # Your custom parameter freezing / LR adjustment
    for param_group in optimizer.param_groups:
        param_group['lr'] = base_lr * YOUR_FACTOR
    use_anchor = YOUR_CHOICE
```

### Modify state representation
```python
def get_state(self, current_mse, anchor_mse, snr_ratio, loss_trend, 
              step_budget, YOUR_NEW_FEATURE):
    return torch.tensor([
        current_mse,
        anchor_mse,
        snr_ratio,
        loss_trend,
        step_budget,
        YOUR_NEW_FEATURE  # Add dimension
    ], dtype=torch.float32)
```

### Tune reward function
```python
# Current:
reward = 10.0 * (improvement - 100*forgetting - 0.01*action_cost)

# More conservative (penalize forgetting more):
reward = 10.0 * (improvement - 200*forgetting - 0.01*action_cost)

# Encourage efficiency (penalize complex actions):
reward = 10.0 * (improvement - 100*forgetting - 0.1*action_cost)
```

## Troubleshooting

### High rollback rate (>20%)
- **Cause**: Forgetting threshold too strict or agent learning poorly
- **Fix**: Increase `forgetting_threshold` or train agent longer

### Low rollback rate but poor performance
- **Cause**: Threshold too loose, not protecting knowledge
- **Fix**: Decrease `forgetting_threshold`, enable adaptive mode

### Agent always selects same action
- **Cause**: Insufficient exploration or reward signal too weak
- **Fix**: Increase epsilon, adjust reward scale, check state normalization

### Memory errors
- **Cause**: Anchor memory too large or batch size too big
- **Fix**: Reduce `max_size` in AnchorMemory, reduce batch size

## Research Contributions

This implementation demonstrates:

1. **Meta-learning for continual learning**: Agent learns adaptation policies
2. **Dynamic plasticity-stability tradeoff**: Automated via forgetting detection
3. **Efficient multi-domain learning**: Single model, multiple noise regimes
4. **Interpretable adaptation**: Action statistics reveal learned strategies

## Citation

If you use this code for research, please cite the original ACA work and acknowledge the channel estimation application.

## Next Steps

1. **Experiment with different SNR sequences** (random order, curriculum learning)
2. **Try different model architectures** (deeper networks, attention mechanisms)
3. **Analyze learned agent policies** (which states trigger which actions?)
4. **Compare to other continual learning methods** (EWC, Progressive Neural Networks)
5. **Extend to other wireless scenarios** (different channel models, fading types)

## Questions?

Check the statistics JSON file for training insights, and adjust hyperparameters based on the rollback rate and action distribution.
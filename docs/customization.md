# Extending and Tuning ACA

This guide covers how to read the training statistics, which knobs to turn, and how to
change the agent's state, actions, or reward. All code references are in
[`aca_channel_estimation.py`](../aca_channel_estimation.py).

## Reading the training statistics

Training writes `aca_<MODEL>_statistics.json`:

| Key | Meaning |
|---|---|
| `total_updates` | Parameter updates attempted (one per mini-batch) |
| `rollback_count`, `rollback_rate` | Updates rejected by the forgetting check, and their fraction |
| `action_counts` | How often the agent picked each of the seven strategies |
| `epoch_losses` | Mean training loss per epoch, across all SNR regimes |
| `forgetting_values`, `threshold_values` | Per-step forgetting Δ and threshold τ (last 1000 steps) |
| `anchor_mse_history`, `snr_history` | Per-step anchor MSE and SNR regime (last 1000 steps) |

How to read them:

- **Rollback rate.** The paper reports rollback on 7.5% of updates. A much higher rate
  (above ~20%) means the threshold is too strict or the learning rate is too high. A rate
  near 0% with poor accuracy on earlier regimes means the threshold is too loose.
- **Action distribution.** A high share of `FULL_UPDATE` means stable learning. A rising
  share of `ANCHOR_REPLAY` / `ADAPTIVE_MIX` points to a large shift between regimes. A high
  share of `CONSERVATIVE` means the agent is protecting old knowledge.

## Hyperparameters

| Flag | Default | Effect |
|---|---|---|
| `--epochs_per_snr` | 50 | Longer training per regime: better fit, more time |
| `--forgetting_threshold` | 0.15 | Base τ. Lower protects old regimes more; higher allows more plasticity |
| `--anchor_memory_size` | 256 | Anchor capacity. Each regime adds `min(size // n_regimes, 50)` samples |
| `--lr` | 1e-3 | Base learning rate, scaled per action |
| `--snr_list` | 8 … 30 | The regimes and the order they are visited in |

Fixed in code (`AdaptationAgent.__init__` and `ACATrainer.__init__`): exploration
ε = 0.2, discount γ = 0.95, agent learning rate 1e-4, agent update every 64 decisions,
threshold bounds τ ∈ [0.08, 0.30].

### Adaptive threshold

`ACATrainer.get_dynamic_threshold` sets τ from the anchor MSE measured before the update:

| Anchor MSE | τ | Reasoning |
|---|---|---|
| < 0.001 | 0.08 | Old regimes are fit well: protect them strictly |
| 0.001 – 0.005 | `--forgetting_threshold` | Default tolerance |
| ≥ 0.005 | min(0.30, 1.5 × base) | Old regimes are fit poorly: allow more plasticity |

Pass `adaptive_threshold=False` to `ACATrainer` to use a fixed τ.

## Adding an action

1. Append the name to `ACTIONS`. Its list index is also its compute cost in the reward
   (`0.01 × index`), so place cheap actions first.
2. Add its configuration to the `config` table in `ACATrainer.apply_adaptation_action`:

```python
config = {
    ...
    'MY_STRATEGY': (n_params // 4, 0.2, True),  # (tensors frozen from the input side, LR scale, anchor replay)
}
```

The agent's output layer is sized from `len(ACTIONS)`, so nothing else needs to change.
Old checkpoints will not load into an agent with a different number of actions.

## Changing the state

`AdaptationAgent.get_state` builds the 5-D state
`[current batch MSE, anchor MSE, SNR progress, loss trend, epoch progress]`.
To add a feature, extend `get_state`, pass the new value where it is called in
`ACATrainer.train_epoch` (both `state` and `next_state`), and set `state_dim` when
constructing the agent in `train_aca_channel_estimator`.

## Changing the reward

The reward is computed in `ACATrainer.train_epoch`:

```python
reward = 10.0 * (-current_mse - 100.0 * max(0, forgetting) - 0.01 * action)
```

Examples:

```python
# Penalize forgetting more strongly
reward = 10.0 * (-current_mse - 200.0 * max(0, forgetting) - 0.01 * action)

# Prefer cheaper actions
reward = 10.0 * (-current_mse - 100.0 * max(0, forgetting) - 0.1 * action)
```

## Using another estimator

Any `nn.Module` that maps a `(B, 1, 72, 14)` tensor to the same shape works. Add it to
`build_model` in [`models.py`](../models.py) and to `MODEL_TYPES`. The partial-freezing
actions work on the order of `model.parameters()`, so define layers from input to output.

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| Rollback rate above ~20% | Threshold too strict or LR too high | Raise `--forgetting_threshold` or lower `--lr` |
| Low rollback rate but earlier SNRs degrade | Threshold too loose | Lower `--forgetting_threshold` |
| Agent always picks the same action | Weak reward signal or too little exploration | Raise `eps` in `AdaptationAgent`, rescale the reward |
| Out of memory | Batch or anchor memory too large | Lower `--batch_size` or `--anchor_memory_size` |
| Data preparation is slow | RBF interpolation of 40,000 grids per SNR runs on the CPU | Expected. Use fewer SNRs for quick tests |

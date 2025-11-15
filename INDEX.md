# BalatroNN - Complete File Index

## 📖 Documentation (Start Here!)

| File | Purpose | When to Read |
|------|---------|--------------|
| **README.md** | Project overview, features, examples | First |
| **QUICKSTART.md** | Get running in 5 minutes | First time setup |
| **GETTING_STARTED.md** | Detailed beginner's guide | Before training |
| **ARCHITECTURE.md** | Technical deep dive | Understanding internals |
| **PROJECT_SUMMARY.md** | Complete project overview | Understanding scope |
| **CONTRIBUTING.md** | How to contribute | Before modifying code |
| **PROJECT_STRUCTURE.txt** | Visual file structure | Reference |
| **LICENSE** | MIT License | Legal |
| **INDEX.md** | This file | Navigation |

## 🐍 Python Scripts

### Main Scripts
| File | Purpose | Usage |
|------|---------|-------|
| **train.py** | Main training script | `python train.py --config configs/default.yaml` |
| **evaluate.py** | Model evaluation | `python evaluate.py --checkpoint path/to/model.pt` |
| **visualize_training.py** | Generate plots | `python visualize_training.py --log-dir logs/` |
| **test_setup.py** | Verify installation | `python test_setup.py` |

### Source Code (`src/`)

#### Environment (`src/environment/`)
- **balatro_env.py** (600 lines)
  - `BalatroEnv` - Main environment class
  - `GameState` - Game state representation
  - `Card`, `Joker`, `Blind` - Game entities
  - Poker hand evaluation
  - Reward calculation

#### Models (`src/models/`)
- **balatro_network.py** (550 lines)
  - `CardEncoder` - Transformer for card sequences
  - `JokerEncoder` - MLP for joker encoding
  - `StateEncoder` - MLP for game state
  - `PolicyValueNetwork` - Main network
  - ~15M parameters (default config)

#### Training (`src/training/`)
- **ppo.py** (400 lines)
  - `PPOConfig` - Configuration dataclass
  - `PPOBuffer` - Experience replay with GAE
  - `PPOTrainer` - Main training loop
  - Clipped surrogate objective
  - Value function optimization

#### Utilities (`src/utils/`)
- **config.py** - YAML configuration management
- **logging.py** - TensorBoard & WandB integration

## ⚙️ Configuration Files

| File | Model Size | Training Time | Use Case |
|------|-----------|---------------|----------|
| **configs/quick_test.yaml** | Small (128d) | 5-10 min | Testing, debugging |
| **configs/default.yaml** | Medium (256d) | 6 hours | Standard training |
| **configs/h100_large.yaml** | XL (512d) | 12-15 hours | Production, H100 |

### Config Structure
```yaml
environment:      # Game settings
model:           # Architecture (d_model, layers, etc.)
training:        # PPO hyperparameters
logging:         # TensorBoard, WandB
checkpoints:     # Saving strategy
evaluation:      # Eval frequency
curriculum:      # Progressive difficulty
```

## 🔧 Setup & Installation

| File | Purpose |
|------|---------|
| **setup.sh** | Automated installation script |
| **requirements.txt** | Python dependencies |
| **.gitignore** | Git ignore patterns |

## 📁 Generated Directories

These are created during training:

- **checkpoints/** - Saved model checkpoints
  - `checkpoint_XXXXX.pt` - Periodic saves
  - `final_model.pt` - Final trained model
  - `interrupted.pt` - Save on Ctrl+C

- **logs/** - Training logs
  - `tensorboard/` - TensorBoard event files
  - `metrics.jsonl` - JSON-formatted metrics
  - `config.yaml` - Copy of training config

## 🎯 Quick Reference

### First Time Setup
```bash
./setup.sh
python test_setup.py
```

### Training
```bash
# Quick test (10 minutes)
python train.py --config configs/quick_test.yaml

# Standard training (6 hours)
python train.py --config configs/default.yaml

# H100 optimized (12-15 hours)
python train.py --config configs/h100_large.yaml --wandb
```

### Monitoring
```bash
# TensorBoard
tensorboard --logdir logs/

# Watch logs
tail -f logs/default/metrics.jsonl
```

### Evaluation
```bash
# Run 100 episodes
python evaluate.py --checkpoint checkpoints/final_model.pt --episodes 100

# Visualize (first 10 episodes)
python evaluate.py --checkpoint checkpoints/final_model.pt --episodes 10 --render

# Deterministic policy
python evaluate.py --checkpoint checkpoints/final_model.pt --deterministic
```

### Visualization
```bash
# Generate plots
python visualize_training.py --log-dir logs/default/ --save plots/
```

## 🔍 Finding Things

### "I want to..."

**...understand how the game works**
→ Read `src/environment/balatro_env.py`, especially `GameState` and `BalatroEnv`

**...modify the neural network**
→ Edit `src/models/balatro_network.py`, see `PolicyValueNetwork`

**...change training hyperparameters**
→ Edit config files in `configs/` or create your own

**...add new game features**
→ Extend `src/environment/balatro_env.py`, add new `Card` effects or `Joker` types

**...implement a different RL algorithm**
→ Create new file in `src/training/` following structure of `ppo.py`

**...improve logging**
→ Modify `src/utils/logging.py` and `PPOTrainer.train()` in `ppo.py`

**...run experiments**
→ Create new config in `configs/`, use different seeds

**...deploy the model**
→ Load checkpoint with `torch.load()`, use `model.get_action_and_value()`

## 📊 Code Map

### Key Classes

```
BalatroEnv (src/environment/balatro_env.py)
├── reset() - Initialize new episode
├── step() - Execute action
├── _play_hand() - Play cards
├── _evaluate_hand() - Poker evaluation
└── render() - Display game state

PolicyValueNetwork (src/models/balatro_network.py)
├── forward() - Encode observations
├── get_action_and_value() - Sample actions
└── CardEncoder, JokerEncoder, StateEncoder

PPOTrainer (src/training/ppo.py)
├── collect_rollouts() - Gather experience
├── train_step() - Optimize policy
├── train() - Main training loop
└── save_checkpoint() / load_checkpoint()
```

### Key Functions

- `evaluate_hand()` - Determine poker hand type
- `calculate_score()` - Compute chips and mult
- `compute_returns_and_advantages()` - GAE computation
- `get_action_and_value()` - Sample actions from policy

## 💾 File Sizes

```
Total project: 228 KB
Python code: 2,636 lines across 12 files

Breakdown:
├── balatro_env.py:     600 lines (23%)
├── balatro_network.py: 550 lines (21%)
├── ppo.py:             400 lines (15%)
├── train.py:           300 lines (11%)
├── evaluate.py:        250 lines (10%)
├── visualize_training.py: 200 lines (8%)
├── test_setup.py:      200 lines (8%)
└── Other:              136 lines (5%)
```

## 🎓 Learning Path

1. **Beginner**
   - Read: QUICKSTART.md, GETTING_STARTED.md
   - Run: test_setup.py, quick_test.yaml
   - Explore: configs/, train.py

2. **Intermediate**
   - Read: ARCHITECTURE.md, balatro_env.py
   - Modify: Create custom config, tune hyperparameters
   - Experiment: Different model sizes, learning rates

3. **Advanced**
   - Read: balatro_network.py, ppo.py
   - Implement: New features, different architectures
   - Contribute: See CONTRIBUTING.md

## 🔗 Related Files

Files that work together:

- `train.py` ↔ `configs/*.yaml` ↔ `src/training/ppo.py`
- `evaluate.py` ↔ checkpoints ↔ `src/models/`
- `visualize_training.py` ↔ `logs/*.jsonl`
- `setup.sh` ↔ `requirements.txt`

## 📈 Metrics Reference

### Training Metrics (in logs/)
- `rollout/mean_reward` - Average episode reward
- `rollout/mean_length` - Average episode length
- `train/policy_loss` - Policy optimization loss
- `train/value_loss` - Value function loss
- `train/kl_divergence` - Policy change magnitude
- `train/entropy_loss` - Exploration measure
- `train/clip_fraction` - Fraction of clipped updates

### Evaluation Metrics
- Mean/std reward
- Mean/std episode length
- Mean/max ante reached
- Win rate (ante 8+)

## 🆘 Troubleshooting

| Issue | Check File | Look For |
|-------|-----------|----------|
| Import errors | test_setup.py, requirements.txt | Dependencies |
| CUDA errors | configs/*.yaml | batch_size, d_model |
| Training slow | ppo.py, train.py | GPU utilization |
| No learning | configs/*.yaml | learning_rate, ent_coef |
| Bad performance | balatro_env.py | Reward function |

## 🎉 You Made It!

This index should help you navigate the entire project. Start with:

1. **QUICKSTART.md** if you want to train immediately
2. **GETTING_STARTED.md** if you want detailed guidance
3. **ARCHITECTURE.md** if you want to understand the system

Happy coding! 🃏🤖


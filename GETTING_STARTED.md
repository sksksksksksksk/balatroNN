# Getting Started with BalatroNN

## 🎯 Quick Overview

**What**: Train a neural network to play Balatro using reinforcement learning  
**How**: PPO algorithm with transformer-based architecture  
**Hardware**: Single H100 (or RTX 3090/4090 for smaller models)  
**Time**: 1-2 hours for meaningful results on H100

## 📋 Prerequisites

```bash
# Check Python version (need 3.8+)
python3 --version

# Check if you have a GPU (optional but recommended)
nvidia-smi
```

## 🚀 Installation (5 minutes)

### Automatic Setup

```bash
# Clone or navigate to the project directory
cd balatroNN

# Run setup script
./setup.sh

# Activate virtual environment
source venv/bin/activate
```

### Manual Setup

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install PyTorch (CUDA 11.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install dependencies
pip install -r requirements.txt

# Verify installation
python test_setup.py
```

## ✅ Verify Everything Works

```bash
python test_setup.py
```

You should see:
```
✓ PyTorch 2.1.0
  - CUDA available: NVIDIA H100
  - CUDA version: 11.8
✓ NumPy 1.24.3
✓ Gymnasium 0.29.1
✓ All tests passed!
```

## 🎮 Your First Training Run (10 minutes)

### Step 1: Quick Test

```bash
python train.py --config configs/quick_test.yaml
```

This will:
- Train a small model (128 dimensions)
- Run for 100K timesteps (~5-10 minutes)
- Save checkpoints to `checkpoints/quick_test/`
- Log to `logs/quick_test/`

You should see output like:
```
===== Update 1 | Timesteps 512 =====
rollout/mean_reward: 15.32
rollout/mean_length: 45.20
train/policy_loss: 0.1234
train/value_loss: 12.5678
```

### Step 2: Monitor Training

**Option A: TensorBoard**
```bash
# In a new terminal
tensorboard --logdir logs/quick_test/
```
Open http://localhost:6006

**Option B: Watch Logs**
```bash
tail -f logs/quick_test/metrics.jsonl
```

### Step 3: Evaluate the Model

```bash
python evaluate.py --checkpoint checkpoints/quick_test/final_model.pt --episodes 10
```

Output:
```
Episode 1/10: Reward = 45.23, Length = 67, Max Ante = 2
Episode 2/10: Reward = 38.12, Length = 54, Max Ante = 3
...
Mean Reward: 42.15 ± 8.34
Mean Ante Reached: 2.50
```

## 🚀 Production Training

### For RTX 3090/4090

```bash
python train.py --config configs/default.yaml
```

Settings:
- Model size: 256 dimensions
- Training time: ~6 hours for 10M steps
- Memory usage: ~8GB VRAM
- Expected performance: Reach ante 3-4 consistently

### For H100 (Recommended)

```bash
python train.py --config configs/h100_large.yaml --wandb
```

Settings:
- Model size: 512 dimensions  
- Training time: ~12-15 hours for 100M steps
- Memory usage: ~20GB VRAM
- Expected performance: Reach ante 6-7, ~30% win rate

### Enable Weights & Biases Logging

```bash
# First time only
wandb login

# Then train with wandb
python train.py --config configs/h100_large.yaml --wandb
```

## 📊 Understanding the Output

### Training Metrics

| Metric | What it Means | Good Value |
|--------|--------------|------------|
| `rollout/mean_reward` | Average reward per episode | Increasing trend |
| `rollout/mean_length` | Average steps per episode | 50-200 |
| `train/policy_loss` | How much policy is changing | Decreasing |
| `train/value_loss` | Value prediction error | Decreasing |
| `train/kl_divergence` | Policy update magnitude | < 0.02 |
| `train/entropy` | Exploration level | > 1.0 |

### What to Look For

**Good Signs:**
- ✅ Mean reward increasing over time
- ✅ Policy loss decreasing
- ✅ Episode length stabilizing
- ✅ KL divergence staying low (<0.02)

**Warning Signs:**
- ⚠️ Reward plateauing early
- ⚠️ Very high KL divergence (>0.1)
- ⚠️ Entropy dropping to zero
- ⚠️ Value loss exploding

## 🎯 Next Steps

### Experiment with Hyperparameters

Create a custom config:

```yaml
# configs/my_experiment.yaml
model:
  d_model: 384  # Bigger model
  num_layers: 8  # Deeper

training:
  learning_rate: 0.0001  # Lower learning rate
  batch_size: 512  # Larger batches
  ent_coef: 0.02  # More exploration
```

Run:
```bash
python train.py --config configs/my_experiment.yaml
```

### Compare Different Runs

```bash
# Train multiple configurations
python train.py --config configs/default.yaml --seed 1
python train.py --config configs/default.yaml --seed 2
python train.py --config configs/h100_large.yaml --seed 1

# Compare in TensorBoard
tensorboard --logdir logs/
```

### Visualize Training Progress

```bash
python visualize_training.py --log-dir logs/h100_large/ --save plots/
```

Generates:
- Training reward curves
- Loss curves
- Episode length progression
- KL divergence tracking

## 🔧 Troubleshooting

### "CUDA out of memory"

**Solution 1**: Reduce batch size
```yaml
training:
  batch_size: 128  # Instead of 256
```

**Solution 2**: Reduce model size
```yaml
model:
  d_model: 128  # Instead of 256
  num_layers: 4  # Instead of 6
```

### "Training is very slow"

**Check GPU utilization:**
```bash
watch -n 1 nvidia-smi
```

Should see:
- GPU utilization: 80-100%
- Memory usage: High but not maxed out

**If GPU is underutilized:**
- Increase batch size
- Increase number of rollout steps

### "No learning progress"

**Try these:**

1. **Increase exploration**
```yaml
training:
  ent_coef: 0.02  # More entropy
```

2. **Adjust learning rate**
```yaml
training:
  learning_rate: 0.001  # Higher
```

3. **Check reward scale**
Look at `rollout/mean_reward`. Should be in range -100 to +100.

### "Import errors"

```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall

# Verify
python test_setup.py
```

## 📈 Training Strategies

### Strategy 1: Quick Iteration

```bash
# Use small model for fast experiments
python train.py --config configs/quick_test.yaml --seed 1
python train.py --config configs/quick_test.yaml --seed 2
python train.py --config configs/quick_test.yaml --seed 3

# Find best hyperparameters, then scale up
```

### Strategy 2: Curriculum Learning

Manually increase difficulty:
1. Train on ante 1-2 for 1M steps
2. Train on ante 1-4 for 5M steps  
3. Train on all antes for 10M steps

### Strategy 3: Ensemble

Train multiple models:
```bash
python train.py --config configs/default.yaml --seed 1
python train.py --config configs/default.yaml --seed 2
python train.py --config configs/default.yaml --seed 3

# Evaluate all and pick best
python evaluate.py --checkpoint checkpoints/checkpoint_seed1_final.pt
python evaluate.py --checkpoint checkpoints/checkpoint_seed2_final.pt
python evaluate.py --checkpoint checkpoints/checkpoint_seed3_final.pt
```

## 💡 Pro Tips

1. **Use multiple seeds**: Results can vary significantly
2. **Monitor early**: Check TensorBoard after 1M steps
3. **Save often**: Checkpoints save automatically every 100 updates
4. **GPU memory**: Start with default batch size, then increase
5. **Learning rate**: Most important hyperparameter to tune
6. **Patience**: Good results take 10M+ steps

## 📚 Learn More

- **ARCHITECTURE.md** - Deep dive into the neural network
- **README.md** - Full feature list and documentation
- **CONTRIBUTING.md** - How to extend the project
- **PROJECT_SUMMARY.md** - Complete project overview

## 🆘 Getting Help

1. Check existing logs: `logs/*/metrics.jsonl`
2. Run tests: `python test_setup.py`
3. Check GPU: `nvidia-smi`
4. Review config: `configs/default.yaml`
5. Open an issue with logs and config

## 🎉 Success Checklist

- [ ] Installation verified with `test_setup.py`
- [ ] Quick test completed successfully
- [ ] TensorBoard showing training curves
- [ ] First checkpoint saved
- [ ] Evaluation script works
- [ ] Ready for production training

**You're all set! Happy training! 🃏🤖**

---

**Next**: Run `python train.py --config configs/h100_large.yaml` for serious training


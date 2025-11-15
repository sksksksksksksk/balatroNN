# Continuing Training Guide

After your initial training run completes, you can continue training from where you left off!

## Quick Answer

### Option 1: Continue from Final Model (Easiest)

```bash
python train.py --config configs/colab.yaml --resume checkpoints/colab/final_model.pt
```

### Option 2: Continue from Latest Checkpoint

```bash
# Find the latest checkpoint
ls -lt checkpoints/colab/*.pt | head -5

# Resume from it
python train.py --config configs/colab.yaml --resume checkpoints/colab/checkpoint_2000000.pt
```

### Option 3: Increase Total Timesteps in Config

Edit your config file to train for longer from the start:

```yaml
training:
  total_timesteps: 4_000_000  # From 2M to 4M
```

Then resume:

```bash
python train.py --config configs/colab.yaml --resume checkpoints/colab/final_model.pt
```

## Understanding Checkpoints

### What Gets Saved

During training, several checkpoints are saved:

1. **Periodic Checkpoints** (e.g., `checkpoint_100000.pt`)
   - Saved every N updates (default: 50-100)
   - Named with the total timestep count
   - Location: `checkpoints/colab/checkpoint_XXXXX.pt`

2. **Final Model** (`final_model.pt`)
   - Saved when training completes normally
   - Contains the final trained state
   - Location: `checkpoints/colab/final_model.pt`

3. **Interrupted Model** (`interrupted.pt`)
   - Saved if you press Ctrl+C
   - Allows resuming after manual stop
   - Location: `checkpoints/colab/interrupted.pt`

### What's in a Checkpoint

Each checkpoint contains:
- Model weights (neural network parameters)
- Optimizer state (for smooth resuming)
- Number of timesteps completed
- Number of updates completed
- Training progress

## Step-by-Step Examples

### Example 1: Train Another 2M Steps

You finished 2M steps, want 2M more (4M total):

```bash
# 1. Edit config to increase total_timesteps
nano configs/colab.yaml
# Change: total_timesteps: 4_000_000

# 2. Resume from final model
python train.py --config configs/colab.yaml --resume checkpoints/colab/final_model.pt
```

The trainer will:
- Start from 2,000,000 timesteps
- Continue until 4,000,000 timesteps
- Train for another ~3-4 hours on T4

### Example 2: Continue After Interruption

You pressed Ctrl+C at 1.5M steps:

```bash
# Resume from the interrupted checkpoint
python train.py --config configs/colab.yaml --resume checkpoints/colab/interrupted.pt
```

Or from the latest saved checkpoint:

```bash
# Find latest checkpoint
ls -lt checkpoints/colab/checkpoint_*.pt | head -1

# Resume (example shows 1,500,000 steps)
python train.py --config configs/colab.yaml --resume checkpoints/colab/checkpoint_1500000.pt
```

### Example 3: Train to 10M Steps in Stages

For very long training, break it into chunks:

**Stage 1: 0 → 2M**
```bash
# configs/colab.yaml: total_timesteps: 2_000_000
python train.py --config configs/colab.yaml
```

**Stage 2: 2M → 4M**
```bash
# Edit config: total_timesteps: 4_000_000
python train.py --config configs/colab.yaml --resume checkpoints/colab/final_model.pt
```

**Stage 3: 4M → 6M**
```bash
# Edit config: total_timesteps: 6_000_000
python train.py --config configs/colab.yaml --resume checkpoints/colab/final_model.pt
```

And so on...

### Example 4: Continue with Different Config

You can change hyperparameters when resuming:

```bash
# Create a new config for continued training
cp configs/colab.yaml configs/colab_continue.yaml

# Edit configs/colab_continue.yaml:
# - total_timesteps: 4_000_000
# - learning_rate: 0.0001  (reduce for fine-tuning)
# - target_kl: 0.02  (relax if needed)

# Resume with new config
python train.py --config configs/colab_continue.yaml --resume checkpoints/colab/final_model.pt
```

**Note:** Only training hyperparameters change. Model architecture must stay the same!

## Google Colab Specific

### Continuing in a New Colab Session

If your Colab session ended:

**Step 1: Download checkpoints** (from previous session)
```python
# Before session ends:
from google.colab import files
!tar -czf checkpoints.tar.gz checkpoints/
files.download('checkpoints.tar.gz')
```

**Step 2: Upload and extract** (in new session)
```python
# Upload the downloaded file
from google.colab import files
uploaded = files.upload()  # Select checkpoints.tar.gz

# Extract
!tar -xzf checkpoints.tar.gz
```

**Step 3: Resume training**
```bash
python train.py --config configs/colab.yaml --resume checkpoints/colab/final_model.pt
```

### Auto-Continue Script for Colab

Add this to your Colab notebook:

```python
# Cell: Auto-resume if checkpoint exists
import os

config_file = "configs/colab.yaml"
checkpoint_file = "checkpoints/colab/final_model.pt"

if os.path.exists(checkpoint_file):
    print(f"✓ Found existing checkpoint: {checkpoint_file}")
    print("Continuing training from checkpoint...")
    !python train.py --config {config_file} --resume {checkpoint_file}
else:
    print("No checkpoint found, starting fresh training...")
    !python train.py --config {config_file}
```

## Finding the Right Checkpoint

### List All Checkpoints

```bash
# See all checkpoints with timestamps
ls -lht checkpoints/colab/*.pt

# See just periodic checkpoints
ls -1 checkpoints/colab/checkpoint_*.pt | sort -V
```

### Choose the Best Checkpoint

**Use `final_model.pt` if:**
- Training completed successfully
- You want to continue from the end
- It exists

**Use latest `checkpoint_XXXXX.pt` if:**
- Training was interrupted
- `final_model.pt` doesn't exist
- You want to resume from a specific point

**Use `interrupted.pt` if:**
- You manually stopped training (Ctrl+C)
- You want to resume immediately

### Check Checkpoint Contents

```python
import torch

# Load checkpoint to inspect
checkpoint = torch.load('checkpoints/colab/final_model.pt', map_location='cpu')

print(f"Timesteps completed: {checkpoint['num_timesteps']:,}")
print(f"Updates completed: {checkpoint['num_updates']:,}")
print(f"Keys in checkpoint: {list(checkpoint.keys())}")
```

## Important Notes

### ✅ What WILL Continue

- Total timestep count
- Training progress
- Model weights
- Optimizer state (momentum, etc.)
- Learning is smooth and continuous

### ⚠️ What MIGHT Change

If you edit the config:
- Learning rate
- Target KL
- Batch size
- Entropy coefficient
- Other hyperparameters

These will use NEW values from the config file.

### ❌ What CANNOT Change

- Model architecture (d_model, num_layers, etc.)
- Environment settings (if they affect state/action space)
- Device type (but can move CPU ↔ GPU)

If you change these, you'll get an error or undefined behavior!

## Advanced: Checkpoint Management

### Keep Only Recent Checkpoints

To save disk space:

```bash
# Keep only the 5 most recent checkpoints
cd checkpoints/colab
ls -t checkpoint_*.pt | tail -n +6 | xargs rm -f
```

Or configure in `colab.yaml`:

```yaml
checkpoints:
  dir: "checkpoints/colab"
  save_best: true
  keep_last_n: 5  # Only keep 5 most recent
```

### Backup Important Checkpoints

```bash
# Backup a good checkpoint
cp checkpoints/colab/final_model.pt checkpoints/colab/backup_2M_steps.pt

# Backup with timestamp
cp checkpoints/colab/final_model.pt checkpoints/colab/model_$(date +%Y%m%d_%H%M%S).pt
```

### Resume from Older Checkpoint

If your latest training went wrong:

```bash
# List all checkpoints with their timesteps
ls -1 checkpoints/colab/checkpoint_*.pt | sort -V

# Resume from an earlier one
python train.py --config configs/colab.yaml --resume checkpoints/colab/checkpoint_1000000.pt
```

## Troubleshooting

### Error: "Checkpoint file not found"

**Problem:** File path is wrong

**Solution:**
```bash
# Check if file exists
ls -l checkpoints/colab/final_model.pt

# Use absolute path if needed
python train.py --config configs/colab.yaml --resume /absolute/path/to/checkpoint.pt
```

### Error: "Size mismatch" or "Unexpected key"

**Problem:** Model architecture changed between training runs

**Solution:** Use the SAME config that created the checkpoint:
```bash
# Find the original config (saved during training)
cat logs/colab/config.yaml

# Use that config
python train.py --config logs/colab/config.yaml --resume checkpoints/colab/final_model.pt
```

### Training Restarts from 0

**Problem:** Not using `--resume` flag

**Solution:** Always include `--resume`:
```bash
# Wrong (starts from scratch)
python train.py --config configs/colab.yaml

# Correct (continues training)
python train.py --config configs/colab.yaml --resume checkpoints/colab/final_model.pt
```

### Want to Change Model Architecture

**Problem:** Can't resume with different architecture

**Solution:** 
1. Option A: Start fresh training with new architecture
2. Option B: Use transfer learning (advanced, not covered here)

## Examples for Different Scenarios

### Scenario 1: Colab Free Tier (12-hour limit)

Train in 3 sessions:

**Session 1:**
```bash
# Train for 2M steps
python train.py --config configs/colab.yaml
# Download checkpoints before session ends
```

**Session 2:**
```bash
# Upload checkpoints, then:
# Edit config: total_timesteps: 4_000_000
python train.py --config configs/colab.yaml --resume checkpoints/colab/final_model.pt
# Download again
```

**Session 3:**
```bash
# Upload checkpoints, then:
# Edit config: total_timesteps: 6_000_000
python train.py --config configs/colab.yaml --resume checkpoints/colab/final_model.pt
```

### Scenario 2: Local Training, Want to Continue

```bash
# After 2M steps complete
# Edit configs/default.yaml: total_timesteps: 10_000_000
python train.py --config configs/default.yaml --resume checkpoints/default/final_model.pt

# Will train from 2M → 10M
# Takes another ~6-8 hours on H100
```

### Scenario 3: Fine-tuning After Main Training

```bash
# Main training done (2M steps), now fine-tune with lower learning rate
cp configs/colab.yaml configs/colab_finetune.yaml

# Edit configs/colab_finetune.yaml:
# - total_timesteps: 2_500_000  (just 500k more)
# - learning_rate: 0.00005  (much lower)
# - target_kl: 0.005  (more conservative)

python train.py --config configs/colab_finetune.yaml --resume checkpoints/colab/final_model.pt
```

## Quick Reference Commands

```bash
# Continue from final model (most common)
python train.py --config configs/colab.yaml --resume checkpoints/colab/final_model.pt

# Continue from interrupted training
python train.py --config configs/colab.yaml --resume checkpoints/colab/interrupted.pt

# Continue from specific checkpoint
python train.py --config configs/colab.yaml --resume checkpoints/colab/checkpoint_1500000.pt

# List all checkpoints
ls -lht checkpoints/colab/*.pt

# Check checkpoint progress
python -c "import torch; c=torch.load('checkpoints/colab/final_model.pt', map_location='cpu'); print(f'Steps: {c[\"num_timesteps\"]:,}')"

# Continue with different hyperparameters
python train.py --config configs/new_config.yaml --resume checkpoints/colab/final_model.pt
```

## Summary

✅ **Always use `--resume` to continue training**

✅ **Increase `total_timesteps` in config for longer training**

✅ **Checkpoints are automatically saved every N updates**

✅ **Can change hyperparameters, but not model architecture**

✅ **Perfect for Colab's time limits - train in stages!**

---

**Pro Tip:** Always download checkpoints after each Colab session! They're your progress insurance. 💾


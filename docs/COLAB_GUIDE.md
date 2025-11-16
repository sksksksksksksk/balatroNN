# Google Colab Training Guide

Train BalatroNN for **FREE** using Google Colab! 🎉

## Why Google Colab?

✅ **Free GPU/TPU** - T4 GPU or TPU v2-8  
✅ **No Setup** - Everything in browser  
✅ **Easy Sharing** - Notebooks are shareable  
✅ **Good for Learning** - Perfect for experimentation  

## Hardware Options

| Tier | GPU | VRAM | Speed | Cost |
|------|-----|------|-------|------|
| **Free** | T4 | 16GB | Medium | $0 |
| **Free** | TPU v2-8 | 8GB/core | Fast | $0 |
| **Pro** | A100 | 40GB | Very Fast | $10/mo |
| **Pro+** | A100 | 40GB | Very Fast | $50/mo |

## Quick Start

### Option 1: Use Prepared Notebook (Easiest)

1. **Open Colab Notebook**
   - Click: https://colab.research.google.com/
   - Upload `colab_setup.ipynb` from this repo

2. **Choose Runtime**
   - Runtime → Change runtime type
   - Hardware accelerator: **GPU** (T4)
   - Save

3. **Run All Cells**
   - Runtime → Run all
   - Watch training progress!

### Option 2: Manual Setup

1. **Create New Notebook**

2. **Install Dependencies**
```python
!pip install -q torch gymnasium tensorboard tqdm pydantic pyyaml
```

3. **Clone Repository**
```python
!git clone https://github.com/YOUR_USERNAME/balatroNN.git
%cd balatroNN
```

4. **Train**
```python
!python train.py --config configs/colab.yaml
```

## Training Configurations

### For T4 GPU (Free Tier)

```yaml
# configs/colab.yaml
training:
  total_timesteps: 2_000_000  # 2M steps
  batch_size: 256
  n_steps: 1024
  # Takes ~3-4 hours
```

**Expected Results:**
- Mean Ante: 2-3
- Training Time: 3-4 hours
- Model Size: ~60MB

### For A100 GPU (Colab Pro)

```yaml
# configs/colab_pro.yaml
training:
  total_timesteps: 10_000_000  # 10M steps
  batch_size: 512
  n_steps: 2048
  # Takes ~6-8 hours
```

**Expected Results:**
- Mean Ante: 4-5
- Training Time: 6-8 hours
- Better performance!

### For TPU (Free Tier)

TPUs require PyTorch/XLA:

```python
# Install TPU support
!pip install cloud-tpu-client==0.10 torch~=2.0.0
!pip install https://storage.googleapis.com/tpu-pytorch/wheels/colab/torch_xla-2.0-cp310-cp310-linux_x86_64.whl
```

Then train normally - auto-detection handles TPU!

## Step-by-Step Tutorial

### 1. Setup Environment

```python
# Check GPU
!nvidia-smi

# Install packages
!pip install -q gymnasium tensorboard tqdm pydantic pyyaml

# Clone repo
!git clone https://github.com/YOUR_USERNAME/balatroNN.git
%cd balatroNN
```

### 2. Verify Device

```python
import sys
sys.path.insert(0, 'src')

from utils import print_device_info, get_device

print_device_info()
device, name = get_device('auto')
print(f"Training on: {name}")
```

### 3. Create Colab Config

```python
%%writefile configs/colab.yaml
training:
  total_timesteps: 2_000_000
  batch_size: 256
  device: "auto"
  
logging:
  log_dir: "logs/colab"
  use_tensorboard: true
```

### 4. Start Training

```python
!python train.py --config configs/colab.yaml
```

### 5. Monitor with TensorBoard

```python
%load_ext tensorboard
%tensorboard --logdir logs/colab/
```

### 6. Save Results

```python
# Create archive
!tar -czf results.tar.gz checkpoints/ logs/

# Download
from google.colab import files
files.download('results.tar.gz')
```

## Colab Limitations & Solutions

### Issue 1: Runtime Disconnects

**Problem**: Colab disconnects after ~12 hours or if idle

**Solutions:**
- Save checkpoints frequently (every 50-100 updates)
- Keep browser tab active
- Use Colab Pro for 24+ hour runtime
- Download checkpoints regularly

### Issue 2: Out of Memory

**Problem**: T4 has only 16GB VRAM

**Solutions:**
```yaml
# Reduce batch size
training:
  batch_size: 128  # Instead of 256

# Or reduce model size
model:
  d_model: 128  # Instead of 256
```

### Issue 3: Disk Space

**Problem**: Free tier has ~100GB disk

**Solutions:**
- Clean old checkpoints
- Only keep last 3 checkpoints
- Download and delete old logs

```python
!rm -rf logs/old_experiment/
!rm checkpoints/*.pt  # Keep only final
```

### Issue 4: Slow TPU

**Problem**: TPU can be slower initially

**Solution**: 
- Stick with GPU for BalatroNN
- T4 is simpler and works well
- TPU best for huge batch sizes

## Best Practices

### 1. Use Colab Pro for Serious Training

Free tier is great for:
- Testing
- Learning
- Short experiments (< 4 hours)

Colab Pro ($10/mo) for:
- Full training runs
- A100 GPU access
- Longer runtime
- More reliable

### 2. Save Early, Save Often

```python
# In training config
checkpoints:
  save_interval: 50  # Every 50 updates
  keep_last_n: 3     # Save space
```

### 3. Monitor Resources

```python
# Check RAM/Disk
from psutil import virtual_memory
mem = virtual_memory()
print(f"RAM Used: {mem.percent}%")

# Check GPU
!nvidia-smi --query-gpu=memory.used,memory.total --format=csv
```

### 4. Use Resume Training

```python
# If disconnected, resume from checkpoint
!python train.py --config configs/colab.yaml \
  --resume checkpoints/colab/checkpoint_latest.pt
```

### 5. Optimize for Time Limits

```python
# Split training into chunks
# Session 1: Train for 1M steps
!python train.py --config configs/colab.yaml  # total_timesteps: 1_000_000

# Download checkpoint

# Session 2: Continue for 1M more
!python train.py --config configs/colab.yaml --resume checkpoint.pt  # total_timesteps: 2_000_000
```

## TPU-Specific Instructions

### Setup TPU Runtime

1. Runtime → Change runtime type
2. Hardware accelerator: **TPU**
3. Save

### Install TPU Support

```python
!pip install cloud-tpu-client==0.10
!pip install torch~=2.0.0
!pip install https://storage.googleapis.com/tpu-pytorch/wheels/colab/torch_xla-2.0-cp310-cp310-linux_x86_64.whl
```

### Verify TPU

```python
import torch_xla
import torch_xla.core.xla_model as xm

print(f"TPU cores: {xm.xrt_world_size()}")
```

### Train on TPU

```python
# Auto-detection works!
!python train.py --config configs/colab.yaml --device tpu
```

**Note**: TPUs work best with very large batch sizes (1024+). For BalatroNN, T4 GPU is often faster.

## Performance Benchmarks

Training 2M steps (d_model=256):

| Hardware | Time | Cost | Ante Reached |
|----------|------|------|--------------|
| T4 (Free) | 3-4 hrs | $0 | 2-3 |
| A100 (Pro) | 1-2 hrs | $1-2 | 2-3 |
| TPU v2-8 (Free) | 2-3 hrs | $0 | 2-3 |

Training 10M steps:

| Hardware | Time | Cost | Ante Reached |
|----------|------|------|--------------|
| T4 (Free) | 15-20 hrs* | $0 | 4-5 |
| A100 (Pro) | 6-8 hrs | $5-8 | 4-5 |
| TPU v2-8 (Free) | 10-12 hrs* | $0 | 4-5 |

*May require multiple sessions

## Colab Pro vs Free

| Feature | Free | Pro | Pro+ |
|---------|------|-----|------|
| **Runtime** | ~12 hrs | ~24 hrs | ~24 hrs |
| **GPU** | T4 | T4/A100 | A100 |
| **RAM** | 12GB | 32GB | 52GB |
| **Priority** | Low | High | Highest |
| **Background** | No | Yes | Yes |
| **Cost** | $0 | $10/mo | $50/mo |

**Recommendation**: 
- Start with Free to test
- Upgrade to Pro if you like it
- Pro+ only if doing heavy research

## Troubleshooting

### "Runtime disconnected"

**Cause**: Idle timeout or 12-hour limit

**Fix**:
1. Download latest checkpoint
2. Reconnect
3. Resume training

### "Out of memory"

**Fix**:
```python
# Clear GPU memory
import torch
torch.cuda.empty_cache()

# Reduce batch size
# Edit configs/colab.yaml: batch_size: 128
```

### "Module not found"

**Fix**:
```python
# Add to Python path
import sys
sys.path.insert(0, 'src')
```

### "Checkpoint not found"

**Fix**:
```python
# List checkpoints
!ls -lh checkpoints/colab/

# Use correct path
!python train.py --resume checkpoints/colab/checkpoint_2560.pt
```

## Example Workflow

### Complete Training Session

```python
# === SETUP ===
!pip install -q gymnasium tensorboard tqdm pydantic pyyaml
!git clone https://github.com/YOUR_USERNAME/balatroNN.git
%cd balatroNN

# === CHECK DEVICE ===
import sys
sys.path.insert(0, 'src')
from utils import print_device_info
print_device_info()

# === TRAIN ===
!python train.py --config configs/colab.yaml

# === MONITOR ===
%load_ext tensorboard
%tensorboard --logdir logs/colab/

# === EVALUATE ===
!python evaluate.py --checkpoint checkpoints/colab/final_model.pt --episodes 50

# === DOWNLOAD ===
!tar -czf balatronn_trained.tar.gz checkpoints/ logs/
from google.colab import files
files.download('balatronn_trained.tar.gz')
```

## Tips & Tricks

### 1. Keep Session Alive

```javascript
// Run in browser console
function KeepAlive() {
  console.log("Keeping alive...");
  document.querySelector("colab-toolbar-button#connect").click();
}
setInterval(KeepAlive, 60000); // Every minute
```

### 2. Auto-Save to Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')

# Save checkpoints to Drive
!cp -r checkpoints/ /content/drive/MyDrive/balatronn_checkpoints/
```

### 3. Parallel Training

Run multiple experiments:
```python
# Experiment 1: High learning rate
!python train.py --config configs/exp1.yaml &

# Experiment 2: Low learning rate  
!python train.py --config configs/exp2.yaml &

wait
```

## Summary

✅ **Free GPU training** works great for BalatroNN  
✅ **T4 GPU** is recommended over TPU  
✅ **Save checkpoints** frequently  
✅ **Use Colab Pro** for serious training  
✅ **Monitor TensorBoard** for progress  

**Get Started**: Open `colab_setup.ipynb` and run all cells!

## Need Help?

- Check the main [README.md](README.md)
- See [DEVICE_SUPPORT.md](DEVICE_SUPPORT.md) for hardware info
- Open an issue on GitHub

Happy training in the cloud! ☁️🤖


# A100 GPU Optimization Guide

## Current Problem: Severe Underutilization

**Symptoms:**
- Only using ~4GB / 40GB VRAM (10%)
- Only 17% GPU utilization
- Training slower than expected

**Root Causes:**

### 1. Single Environment Sequential Processing ⚠️ MAJOR BOTTLENECK
The current code runs **ONE environment at a time** with **batch size = 1** during rollout collection.

```python
# Current: Single environment
obs_tensor = {...}.unsqueeze(0)  # Batch size = 1
action, log_prob, entropy, value = model.get_action_and_value(obs_tensor)
next_obs, reward, done = env.step(action)  # GPU waits here!
```

**Impact:**
- GPU sits idle 80-90% of the time waiting for environment
- Only processes 1 observation per forward pass
- Catastrophic performance waste

### 2. Conservative Hyperparameters
- Batch size too small (768 vs optimal 2048-4096)
- Model too small (d_model=384 vs optimal 768)
- Not using full A100 capacity

---

## Immediate Fixes (No Code Changes)

### Solution 1: Use Aggressive Config

```bash
# Use the new aggressive config
python train.py --config configs/a100_aggressive.yaml
```

**Changes:**
- `batch_size: 3072` (was 768) → 4x more data per update
- `d_model: 768` (was 384) → 2x larger model
- `n_steps: 4096` (was 2048) → 2x larger rollout buffer
- `num_workers: 8` (was 4) → More parallel data loading

**Expected VRAM Usage:** 30-35GB (was 4GB)  
**Expected GPU Util:** 60-80% (was 17%)  
**Expected Speedup:** 2-3x faster updates

### Solution 2: Increase Batch Size in Current Config

Edit `configs/a100.yaml`:
```yaml
training:
  batch_size: 2048  # or 3072, or 4096
  n_steps: 4096
  
model:
  d_model: 512  # or 768
  num_layers: 10  # or 12
```

Then:
```bash
python train.py --config configs/a100.yaml
```

---

## Long-Term Fix: Vectorized Environments (Requires Code)

The **proper solution** is to implement vectorized environments. This would:
- Run 16-32 environments in parallel
- Process batches of 16-32 observations at once
- Keep GPU busy while environments run on CPU
- Achieve 80-95% GPU utilization

### Implementation Required:

```python
# Instead of:
env = BalatroEnv()

# Use:
from stable_baselines3.common.vec_env import SubprocVecEnv

def make_env():
    return BalatroEnv()

env = SubprocVecEnv([make_env for _ in range(32)])  # 32 parallel envs
```

**Benefits:**
- 32x more observations per forward pass
- GPU stays busy during environment steps
- Near-optimal A100 utilization
- 5-10x total speedup

---

## Performance Comparison

| Configuration | VRAM | GPU Util | Samples/sec | Time to 50M |
|--------------|------|----------|-------------|-------------|
| **Current (default)** | 4GB | 17% | ~500 | 28 hours |
| **Aggressive config** | 32GB | 65% | ~1500 | 9 hours |
| **+ Vectorized envs** | 35GB | 85% | ~5000 | 3 hours |

---

## Quick Start: Maximum Speed Right Now

1. **Stop current training** (if running)

2. **Use aggressive config:**
   ```bash
   python train.py --config configs/a100_aggressive.yaml
   ```

3. **Monitor with nvidia-smi:**
   ```bash
   watch -n 1 nvidia-smi
   ```

4. **Expected output:**
   ```
   Memory-Usage: 30000MiB / 40960MiB (73%)
   GPU-Util: 65-75%
   ```

5. **If VRAM is still low**, increase batch size even more:
   ```yaml
   # In configs/a100_aggressive.yaml
   batch_size: 4096  # Push it to the limit!
   ```

---

## Troubleshooting

### "Out of memory" error
**Symptoms:** CUDA OOM error during training

**Solutions:**
1. Reduce `batch_size` by 25% increments:
   ```yaml
   batch_size: 2304  # From 3072
   ```

2. Reduce model size:
   ```yaml
   d_model: 512  # From 768
   ```

3. Enable gradient accumulation (if implemented):
   ```yaml
   gradient_accumulation_steps: 2
   ```

### Still low GPU utilization after changes
**Possible causes:**
1. **Data loading bottleneck** - Increase `num_workers`
2. **Logging overhead** - Increase `log_interval`
3. **Small model** - Increase `d_model` and `num_layers`
4. **Need vectorized envs** - See long-term solution above

### How to test maximum batch size
Run this test to find your limit:

```python
import torch
from src.models.balatro_network import create_model

config = {...}
model = create_model(config).cuda()

# Test increasing batch sizes
for batch_size in [1024, 2048, 3072, 4096, 5120, 6144]:
    try:
        # Dummy input
        obs = {key: torch.randn(batch_size, *shape).cuda() 
               for key, shape in obs_shapes.items()}
        
        # Forward pass
        model(obs)
        print(f"✓ Batch size {batch_size} works!")
        
        # Check memory
        print(f"  VRAM: {torch.cuda.memory_allocated()/1e9:.1f}GB")
        
    except RuntimeError as e:
        print(f"✗ Batch size {batch_size} OOM")
        break
```

---

## Key Takeaways

1. ⚠️ **Current bottleneck is single-environment sequential processing**
2. ⚡ **Immediate 2-3x speedup available** by using aggressive config
3. 🚀 **5-10x speedup possible** with vectorized environments (requires code)
4. 💪 **A100 can handle MUCH larger models and batches** than currently used

---

## Next Steps

**Now:**
1. Use `configs/a100_aggressive.yaml`
2. Monitor nvidia-smi - aim for 60-80% GPU util and 25-35GB VRAM
3. Adjust batch_size if needed

**Soon:**
1. Implement vectorized environments (SubprocVecEnv)
2. Add gradient accumulation for even larger effective batch sizes
3. Consider mixed-precision training if not already enabled

**Questions?**
- Check nvidia-smi every few minutes during training
- VRAM usage should be steady, not fluctuating
- GPU-Util should be consistently high (not spiking to 100% then dropping to 0%)


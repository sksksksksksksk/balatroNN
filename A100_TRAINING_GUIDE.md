# A100 Training Guide

This guide provides comprehensive instructions for training BalatroNN on NVIDIA A100 GPUs.

## Overview

The A100 configuration is optimized for serious training runs with:
- **Model Size**: 384-dimensional embeddings (44.17M parameters)
- **Batch Size**: 768 (leveraging A100's high memory bandwidth)
- **Training Duration**: 50M timesteps (~12-16 hours)
- **Expected Performance**: Significantly better than Colab T4 runs

## Hardware Requirements

### Minimum
- NVIDIA A100 40GB
- 32GB RAM
- 100GB disk space (for checkpoints and logs)

### Recommended
- NVIDIA A100 80GB
- 64GB RAM
- 500GB disk space (for long-term experiments)

## Quick Start

### 1. Setup Environment

```bash
# Clone repository
git clone https://github.com/yourusername/balatroNN.git
cd balatroNN

# Install dependencies
pip install -r requirements.txt
```

### 2. Verify GPU

```bash
# Check GPU availability
nvidia-smi

# Expected output should show A100 GPU
```

### 3. Run Training

#### Option A: Using Python Script (Recommended)

```bash
# Start training with A100 config
python train.py --config configs/a100.yaml
```

#### Option B: Using Jupyter Notebook

```bash
# Launch Jupyter
jupyter notebook a100_training.ipynb
```

Then run cells sequentially.

## Configuration Details

### Model Architecture

```yaml
model:
  d_model: 384          # Embedding dimension
  num_heads: 12         # Attention heads
  num_layers: 8         # Transformer layers
  dropout: 0.1
```

**Model Size**: ~44.17 MB (float32), ~176 MB with optimizer state

### Training Hyperparameters

```yaml
training:
  total_timesteps: 50_000_000
  learning_rate: 0.00015
  batch_size: 768
  n_steps: 2048
  n_epochs: 10
  
  # PPO parameters
  gamma: 0.995
  gae_lambda: 0.95
  clip_range: 0.2
  ent_coef: 0.012
```

### Performance Optimizations

#### Mixed Precision Training
- **Enabled by default** for A100
- Uses FP16/BF16 for faster training
- A100 has dedicated Tensor Cores for mixed precision
- Expected speedup: 2-3x

#### Model Compilation (PyTorch 2.0+)
- **Enabled** with `torch.compile()`
- Optimizes model graph for inference
- Expected speedup: 10-20%

#### Memory Optimizations
- Pin memory for faster data transfer
- Multiple workers for data loading
- Efficient gradient accumulation

## Monitoring Training

### TensorBoard

```bash
# Start TensorBoard (in separate terminal)
tensorboard --logdir logs/a100_training
```

Access at: http://localhost:6006

### Weights & Biases (Optional)

Set `use_wandb: true` in config and run:

```bash
# Login to W&B
wandb login

# Training will automatically log to W&B
python train.py --config configs/a100.yaml
```

### GPU Monitoring

```bash
# Real-time GPU monitoring
watch -n 1 nvidia-smi

# Or use gpustat
gpustat -i 1
```

## Checkpointing and Resuming

### Automatic Checkpointing

Training automatically saves checkpoints every 100 updates to:
```
checkpoints/a100/experiment_name/
├── checkpoint_0100.pt
├── checkpoint_0200.pt
├── ...
└── final_model.pt
```

### Resume Training

```bash
# Training automatically detects and offers to resume from latest checkpoint
python train.py --config configs/a100.yaml --resume
```

### Manual Resume

```python
import torch
from src.models.balatro_network import create_model

# Load model
model = create_model(config['model'])
checkpoint = torch.load('checkpoints/a100/experiment_name/checkpoint_0500.pt')
model.load_state_dict(checkpoint['model_state_dict'])
```

## Performance Expectations

### Training Speed
- **FPS**: ~8,000-10,000 frames/second
- **Update Time**: ~2-3 seconds per update
- **Full Training**: 12-16 hours for 50M timesteps

### Memory Usage
- **Model**: ~0.5-1 GB
- **Batch Processing**: ~15-20 GB
- **Peak Usage**: ~25-30 GB (A100 40GB has room to spare)

### Expected Results
After 50M timesteps, expect:
- Consistent wins on Ante 1-3
- ~70% win rate on Ante 4-5
- Ante 8 completion occasionally

## Advanced Features

### Curriculum Learning

Enable progressive difficulty increase:

```yaml
curriculum:
  enabled: true
  start_ante: 1
  max_ante: 8
  ante_increase_threshold: 0.75
```

The agent starts at Ante 1 and progresses to harder antes as performance improves.

### Distributed Training (Multi-GPU)

For multiple A100s:

```bash
# Using PyTorch DDP
torchrun --nproc_per_node=4 train.py --config configs/a100.yaml --distributed
```

### Custom Hyperparameter Sweeps

Use W&B Sweeps for hyperparameter optimization:

```bash
# Create sweep
wandb sweep sweep_config.yaml

# Run sweep agent
wandb agent sweep_id
```

## Troubleshooting

### Out of Memory (OOM)

If you encounter OOM errors:

1. **Reduce batch size**:
   ```yaml
   batch_size: 512  # Instead of 768
   ```

2. **Enable gradient checkpointing**:
   ```yaml
   optimization:
     gradient_checkpointing: true
   ```

3. **Reduce model size**:
   ```yaml
   model:
     d_model: 256  # Instead of 384
     num_layers: 6  # Instead of 8
   ```

### Slow Training

If training is slower than expected:

1. **Verify mixed precision is enabled**:
   ```yaml
   training:
     use_mixed_precision: true
   ```

2. **Check GPU utilization**:
   ```bash
   nvidia-smi dmon -i 0
   ```
   Should show ~90-100% GPU utilization.

3. **Increase number of workers**:
   ```yaml
   optimization:
     num_workers: 8  # Adjust based on CPU cores
   ```

### Training Instability

If loss diverges or becomes NaN:

1. **Reduce learning rate**:
   ```yaml
   learning_rate: 0.0001  # Instead of 0.00015
   ```

2. **Enable gradient clipping** (already enabled):
   ```yaml
   max_grad_norm: 0.8
   ```

3. **Reduce entropy coefficient**:
   ```yaml
   ent_coef: 0.008  # Instead of 0.012
   ```

## Cost Estimation

### Cloud Providers

| Provider | Instance | GPU | Cost/Hour | 50M Steps Cost |
|----------|----------|-----|-----------|----------------|
| AWS | p4d.24xlarge | 8x A100 | $32.77 | $5-7 per GPU |
| GCP | a2-highgpu-1g | 1x A100 | $3.67 | ~$50 |
| Azure | Standard_ND96asr_v4 | 8x A100 | $27.20 | $4-6 per GPU |
| Lambda Labs | gpu_1x_a100 | 1x A100 | $1.10 | ~$18 |

**Recommendation**: Lambda Labs or GCP Spot instances for cost-effective training.

## Best Practices

1. **Start with smaller timesteps** (1M) to verify setup
2. **Enable W&B** for experiment tracking
3. **Monitor GPU utilization** to ensure efficient training
4. **Use curriculum learning** for better convergence
5. **Save checkpoints frequently** (every 100 updates)
6. **Run evaluation** periodically to track progress
7. **Use mixed precision** for 2-3x speedup

## Example Training Run

Here's a complete example workflow:

```bash
# 1. Verify setup
python -c "import torch; print(torch.cuda.is_available())"

# 2. Quick test (1M timesteps, ~3 minutes)
python train.py --config configs/quick_test.yaml

# 3. Full A100 training
python train.py --config configs/a100.yaml

# 4. Monitor in separate terminal
tensorboard --logdir logs/a100_training

# 5. After training, evaluate
python evaluate.py --checkpoint checkpoints/a100/*/final_model.pt
```

## Comparison with Other Configs

| Config | GPU | Timesteps | Duration | Model Size | Expected Performance |
|--------|-----|-----------|----------|------------|---------------------|
| Colab | T4 | 2M | 3-4h | Small (256d) | Ante 1-2 |
| A100 | A100 | 50M | 12-16h | Medium (384d) | Ante 1-5 |
| H100 | H100 | 100M | 18-24h | Large (512d) | Ante 1-8 |

## Support

For issues or questions:
1. Check [GitHub Issues](https://github.com/yourusername/balatroNN/issues)
2. Review [Training Logs](logs/)
3. Join our [Discord](https://discord.gg/example)

## References

- [A100 Architecture](https://www.nvidia.com/en-us/data-center/a100/)
- [PyTorch Mixed Precision](https://pytorch.org/docs/stable/amp.html)
- [PPO Algorithm](https://arxiv.org/abs/1707.06347)
- [Balatro Game](https://www.playbalatro.com/)


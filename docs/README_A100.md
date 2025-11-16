# A100 Training Setup - Quick Reference

## Files Created

1. **`configs/a100.yaml`** - A100-optimized configuration
2. **`a100_training.ipynb`** - Interactive Jupyter notebook
3. **`A100_TRAINING_GUIDE.md`** - Comprehensive guide

## Quick Start

### Command Line (Simplest)
```bash
python train.py --config configs/a100.yaml
```

### Jupyter Notebook (Interactive)
```bash
jupyter notebook a100_training.ipynb
```

## Key Features

### Configuration Highlights
- **Model**: 384d embeddings, 12 attention heads, 8 layers
- **Training**: 50M timesteps, batch size 768
- **Optimizations**: Mixed precision, model compilation
- **Duration**: 12-16 hours on A100

### Performance Optimizations
✅ Mixed precision (FP16/BF16) - 2-3x speedup  
✅ PyTorch 2.0 compilation - 10-20% speedup  
✅ Efficient data loading - Pin memory + prefetching  
✅ Curriculum learning - Progressive difficulty  

### Monitoring
- **TensorBoard**: Real-time metrics
- **Weights & Biases**: Cloud-based tracking (optional)
- **GPU monitoring**: nvidia-smi integration

## Expected Results

| Metric | Value |
|--------|-------|
| Training Speed | 8,000-10,000 FPS |
| GPU Memory Usage | 25-30 GB (peak) |
| Model Parameters | 44.17M |
| Win Rate (Ante 4-5) | ~70% |

## Comparison

| Setup | GPU | Duration | Timesteps | Performance |
|-------|-----|----------|-----------|-------------|
| **Colab** | T4 (16GB) | 3-4h | 2M | Ante 1-2 |
| **A100** | A100 (40/80GB) | 12-16h | 50M | Ante 1-5 ⭐ |
| **H100** | H100 (80GB) | 18-24h | 100M | Ante 1-8 |

## Configuration Comparison

```yaml
# Colab (T4)          # A100 (This)        # H100
d_model: 256          d_model: 384         d_model: 512
batch_size: 256       batch_size: 768      batch_size: 1024
timesteps: 2M         timesteps: 50M       timesteps: 100M
```

## Next Steps

1. **Verify GPU**: `nvidia-smi` should show A100
2. **Start Training**: Run the command or notebook above
3. **Monitor**: Open TensorBoard or W&B
4. **Evaluate**: Test model after training completes

## Troubleshooting

**Out of Memory?**
- Reduce batch_size to 512 or 384
- Enable gradient_checkpointing in config

**Slow Training?**
- Verify mixed_precision is enabled
- Check GPU utilization with `nvidia-smi dmon`

**Loss Diverging?**
- Reduce learning_rate to 0.0001
- Decrease ent_coef to 0.008

---

**Need Help?** See `A100_TRAINING_GUIDE.md` for detailed documentation.


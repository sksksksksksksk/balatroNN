# Quick Start Guide

Get up and running with BalatroNN in minutes!

## Prerequisites

- Python 3.8 or higher
- NVIDIA GPU with CUDA support (recommended for training)
- 16GB+ RAM (32GB+ recommended)
- 50GB+ free disk space

## Installation

### Option 1: Automated Setup (Recommended)

```bash
./setup.sh
```

This script will:
- Check system requirements
- Create a virtual environment
- Install all dependencies
- Set up directory structure

### Option 2: Manual Setup

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install PyTorch with CUDA support (for NVIDIA GPUs)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Or for CPU only
pip install torch torchvision torchaudio

# Install other dependencies
pip install -r requirements.txt
```

## Verify Installation

```bash
python test_setup.py
```

This will verify that all components are working correctly.

## Training

### Quick Test (5-10 minutes)

Test the training pipeline with a small model:

```bash
python train.py --config configs/quick_test.yaml
```

### Default Training (Hours)

Train with default settings:

```bash
python train.py --config configs/default.yaml
```

### H100 Optimized Training (Days)

For serious training on an H100 GPU:

```bash
python train.py --config configs/h100_large.yaml
```

### Resume Training

```bash
python train.py --config configs/default.yaml --resume checkpoints/checkpoint_XXXXX.pt
```

## Monitoring

### TensorBoard

```bash
tensorboard --logdir logs/
```

Then open http://localhost:6006 in your browser.

### Weights & Biases

```bash
# Login to wandb
wandb login

# Train with wandb logging
python train.py --config configs/default.yaml --wandb
```

### Visualize Progress

```bash
# Generate plots from logs
python visualize_training.py --log-dir logs/ --save plots/
```

## Evaluation

Evaluate a trained model:

```bash
# Run 100 evaluation episodes
python evaluate.py --checkpoint checkpoints/final_model.pt --episodes 100

# Run with visualization
python evaluate.py --checkpoint checkpoints/final_model.pt --render --episodes 10

# Use deterministic policy
python evaluate.py --checkpoint checkpoints/final_model.pt --deterministic --episodes 100
```

## Configuration

Configuration files are in `configs/`:

- **default.yaml** - Standard training configuration
- **quick_test.yaml** - Fast testing (small model, few timesteps)
- **h100_large.yaml** - Large model optimized for H100 GPU

Key parameters:

```yaml
training:
  total_timesteps: 10_000_000  # Total training steps
  learning_rate: 0.0003        # Learning rate
  batch_size: 256              # Batch size
  n_steps: 2048               # Rollout length
  
model:
  d_model: 256                # Model dimension
  num_heads: 8                # Attention heads
  num_layers: 6               # Transformer layers
```

## Hardware Recommendations

### Training

| Hardware | Model Size | Training Time | Expected Performance |
|----------|-----------|---------------|---------------------|
| RTX 3090 | Small (128) | ~2-3 days | Good for ante 1-3 |
| RTX 4090 | Medium (256) | ~1-2 days | Good for ante 1-5 |
| A100 | Large (384) | ~1 day | Good for ante 1-6 |
| H100 | XL (512) | ~12 hours | Good for ante 1-8 |

### Inference

- CPU: ~10-50ms per decision
- GPU: ~1-5ms per decision

## Tips

1. **Start Small**: Use `quick_test.yaml` to verify everything works
2. **Monitor Progress**: Use TensorBoard to watch training in real-time
3. **Save Often**: The default saves checkpoints every 100 updates
4. **Tune Hyperparameters**: Adjust learning rate, batch size based on your hardware
5. **Use Mixed Precision**: Enable automatic mixed precision for faster training on modern GPUs

## Troubleshooting

### CUDA Out of Memory

Reduce batch size or model size:

```yaml
training:
  batch_size: 128  # Reduce from 256

model:
  d_model: 128  # Reduce from 256
```

### Training is Slow

- Check GPU utilization: `nvidia-smi -l 1`
- Reduce `n_steps` for more frequent updates
- Increase batch size if GPU memory allows

### No Learning Progress

- Increase learning rate (try 1e-3)
- Increase entropy coefficient for more exploration
- Check reward scaling in environment

## Next Steps

1. **Customize the Environment**: Modify `src/environment/balatro_env.py` to add more game mechanics
2. **Experiment with Architecture**: Try different model sizes and architectures
3. **Implement Curriculum Learning**: Gradually increase difficulty as agent improves
4. **Add More Jokers**: Extend the game simulation with more joker effects

## Getting Help

- Check the main README.md for detailed documentation
- Review the code comments for implementation details
- Open an issue on GitHub for bugs or questions

Happy training! 🃏🎰


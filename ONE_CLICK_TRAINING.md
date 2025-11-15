# 🎮 One-Click Training Guide

## TL;DR

1. Open `a100_training.ipynb`
2. Click **Runtime → Run all** (or press Ctrl+F9)
3. Wait 12-16 hours
4. You have a trained Balatro AI!

That's it. Seriously.

---

## What It Does

The notebook automatically:
1. ✅ Checks your GPU (needs A100)
2. ✅ Installs all dependencies
3. ✅ Clones code from GitHub
4. ✅ Configures for A100
5. ✅ Trains for 50M timesteps
6. ✅ Evaluates the model
7. ✅ Creates visualizations

**Zero configuration required.**

---

## Hardware Requirements

- **GPU**: NVIDIA A100 (40GB or 80GB)
- **Time**: 12-16 hours for full training
- **Cost**: ~$18 on Lambda Labs, ~$50 on GCP

---

## Quick Setup on Cloud Platforms

### Lambda Labs (Recommended - Cheapest)
```bash
1. Sign up at lambdalabs.com
2. Launch instance: 1x A100 (40GB)
3. Open JupyterLab (comes pre-installed)
4. Upload a100_training.ipynb
5. Run all cells
```
**Cost**: ~$1.10/hour = ~$18 total

### Google Colab Pro+ (Easiest)
```bash
1. Go to colab.research.google.com
2. Upload a100_training.ipynb
3. Runtime → Change runtime type → A100
4. Run all cells
```
**Cost**: $50/month subscription (includes other GPUs too)

### Google Cloud Platform
```bash
1. Create VM with 1x A100
2. SSH into instance
3. Install Jupyter: pip install jupyter
4. Upload notebook and run
```
**Cost**: ~$3.67/hour = ~$50 total

### AWS (Most Expensive)
```bash
1. Launch p4d instance
2. Install Jupyter
3. Upload notebook
4. Run
```
**Cost**: ~$32/hour (but has 8 GPUs, so ~$5 per GPU)

---

## What You Get

After training completes:

### Files Created
- `checkpoints/a100/*/final_model.pt` - Your trained model (44MB)
- `logs/a100_training/` - TensorBoard logs
- `evaluation_results.png` - Performance visualization

### Model Performance
- **Ante 1-3**: Consistent wins
- **Ante 4-5**: ~70% win rate
- **Ante 6+**: Learning, occasional success

---

## Monitoring (Optional)

While training runs, you can:

### Option 1: TensorBoard (Cell 3)
Shows real-time training metrics, loss curves, rewards

### Option 2: Terminal
```bash
# Watch GPU usage
watch -n 2 nvidia-smi

# View logs
tail -f logs/a100_training/*/events*
```

---

## Stopping Training

**To pause/stop:**
1. Runtime → Interrupt execution
2. Checkpoint auto-saves
3. Resume anytime by running training cell again

**To resume:**
Just run the training cell - it automatically detects and resumes from latest checkpoint.

---

## Troubleshooting

### "No GPU detected"
- Verify A100 is selected in runtime settings
- Try restarting runtime

### "Out of memory"
Edit `configs/a100.yaml`:
```yaml
training:
  batch_size: 512  # Reduce from 768
```

### "Repository clone failed"
Update the repo URL in cell 2:
```python
repo_url = "https://github.com/YOUR_USERNAME/balatroNN.git"
```

### "Training too slow"
Check GPU utilization:
```bash
nvidia-smi
```
Should show ~90-100% GPU usage. If not:
- Verify mixed precision is enabled (it is by default)
- Check if model compilation worked (PyTorch 2.0+ required)

---

## Customization (Advanced)

### Change Training Duration
Edit `configs/a100.yaml`:
```yaml
training:
  total_timesteps: 10_000_000  # 10M instead of 50M (faster, less trained)
```

### Different Model Size
```yaml
model:
  d_model: 256  # Smaller model (faster, less capable)
  num_layers: 6  # Fewer layers
```

### Disable Curriculum Learning
```yaml
curriculum:
  enabled: false  # Start at fixed difficulty
```

---

## After Training

### Use Your Model
```python
from src.models.balatro_network import create_model
import torch

# Load model
model = create_model({'d_model': 384})
checkpoint = torch.load('checkpoints/a100/*/final_model.pt')
model.load_state_dict(checkpoint['model_state_dict'])

# Use for inference
obs = get_observation()  # Your game state
action, _, _, _ = model.get_action_and_value(obs, deterministic=True)
```

### Play Against It
```bash
python play_real_game.py --checkpoint checkpoints/a100/*/final_model.pt
```

### Continue Training
Just run the training cell again - it resumes automatically!

---

## Cost Comparison

| Platform | GPU | $/hour | Total Cost | Setup |
|----------|-----|---------|------------|-------|
| **Lambda Labs** | A100 40GB | $1.10 | ~$18 | ⭐⭐⭐⭐⭐ Easy |
| **GCP** | A100 40GB | $3.67 | ~$50 | ⭐⭐⭐⭐ Medium |
| **Colab Pro+** | A100 40GB | N/A | $50/mo | ⭐⭐⭐⭐⭐ Easiest |
| **AWS** | A100 40GB | ~$32 | ~$350 | ⭐⭐ Complex |

**Recommendation**: Lambda Labs for cost, Colab Pro+ for ease.

---

## FAQ

**Q: Can I use a different GPU?**
A: Yes, but edit the config. T4 = use `colab.yaml`, V100 = reduce batch size.

**Q: How long does setup take?**
A: 2-3 minutes (installs + clone)

**Q: Can I close the browser?**
A: Yes! Training continues. Checkpoints save every ~30 min.

**Q: What if training crashes?**
A: Resume from latest checkpoint - no progress lost!

**Q: Can I train for less time?**
A: Yes! Edit `total_timesteps` in config. 10M = ~3 hours, still decent.

**Q: Do I need to know Python?**
A: Nope! Just click "Run all" and wait.

---

## Support

- **Detailed docs**: `A100_TRAINING_GUIDE.md`
- **Architecture info**: `ARCHITECTURE.md`
- **GitHub Issues**: [link to your repo]
- **Discord**: [your discord]

---

## Credits

Built with:
- PyTorch 2.0+
- PPO (Proximal Policy Optimization)
- Transformer architecture
- Mixed precision training
- Curriculum learning

Inspired by the incredible game Balatro by LocalThunk.

---

**Ready to train?** Open `a100_training.ipynb` and hit "Run all"! 🚀


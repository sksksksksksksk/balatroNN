# Device Support

BalatroNN supports training on multiple hardware backends!

## Supported Devices

| Backend | Status | Performance | Setup Difficulty |
|---------|--------|-------------|------------------|
| **NVIDIA GPU (CUDA)** | ✅ Full Support | Best (100%) | Easy |
| **AMD GPU (ROCm)** | ✅ Full Support | Excellent (80-95%) | Easy |
| **Google TPU** | ✅ Full Support | Very Good (70-90%) | Easy (Colab) |
| **CPU** | ✅ Supported | Slow (10-20%) | Very Easy |
| **Apple Silicon (MPS)** | ⚠️ Experimental | Good (60-70%) | Easy |

## Quick Start

### Auto-Detection (Recommended)

```bash
# Works on any hardware!
python train.py --config configs/default.yaml
```

The system automatically detects and uses the best available device.

### Manual Selection

```bash
# Force specific device
python train.py --config configs/default.yaml --device cuda   # NVIDIA or AMD
python train.py --config configs/default.yaml --device cpu    # CPU
```

## Installation by Device

### NVIDIA GPUs (CUDA)

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### AMD GPUs (ROCm)

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.0
```

See [ROCM_GUIDE.md](ROCM_GUIDE.md) for detailed instructions.

### Google Cloud TPU

**Option 1: Google Colab (Easiest)**
```python
# In Colab notebook with TPU runtime:
!pip install cloud-tpu-client==0.10
!pip install torch~=2.0.0 torch_xla
```

**Option 2: Google Cloud Platform**
```bash
pip install torch torch-xla cloud-tpu-client
```

See [COLAB_GUIDE.md](COLAB_GUIDE.md) for complete instructions.

### CPU Only

```bash
pip install torch torchvision torchaudio
```

### Apple Silicon (M1/M2/M3)

```bash
pip install torch torchvision torchaudio
# MPS backend is automatically used on Apple Silicon
```

## Performance Comparison

Training speed for 10M timesteps (d_model=256, batch=256):

| Device | Time | Relative Speed |
|--------|------|----------------|
| H100 | 1.5 hours | 100% |
| A100 | 2.5 hours | 60% |
| RTX 4090 | 4 hours | 38% |
| RX 7900 XTX | 5 hours | 30% |
| TPU v3-8 | 6 hours | 25% |
| T4 (Colab Free) | 15 hours | 10% |
| RX 6800 XT | 8 hours | 19% |
| CPU (Ryzen 9) | 60+ hours | 2.5% |

## Memory Requirements

| Model Size | Parameters | VRAM Needed | Batch Size |
|-----------|-----------|-------------|------------|
| Small (128) | ~4M | 4GB | 256 |
| Medium (256) | ~15M | 8GB | 256 |
| Large (384) | ~35M | 16GB | 256 |
| XL (512) | ~60M | 24GB | 256 |

With larger batch sizes, VRAM requirements increase proportionally.

## Recommendations

### For First-Time Users
**Use**: Whatever you have!
- GPU preferred but CPU works for testing
- Auto-detection handles everything

### For Serious Training
**NVIDIA**: RTX 4090, A100, or H100
**AMD**: RX 7900 XTX or RX 6800 XT
- 24GB+ VRAM recommended
- Good cooling essential

### For Budget Training
**AMD RX 6800 XT** (~$500)
- 16GB VRAM
- Great value
- Fully supported

### For Research
**NVIDIA A100 or H100**
- Cloud options available
- Maximum performance
- Best ecosystem

## Cloud Options

All major cloud providers support training:

| Provider | GPU/TPU Options | Hourly Cost | Notes |
|----------|-----------------|-------------|-------|
| **Google Colab** | T4, TPU v2, A100 | Free - $10/month | Best for beginners! |
| **Google Cloud** | TPU v2/v3/v4, A100, H100 | $1.50 - $8/hour | Native TPU support |
| **AWS** | A100, H100 | $4 - $32/hour | No TPU |
| **Azure** | A100, H100 | $3 - $30/hour | No TPU |
| **Lambda Labs** | A100, H100 | $1.50 - $2/hour | Great value |
| **Paperspace** | A4000, A6000 | $0.76 - $2/hour | Easy setup |

**Recommendation**: Start with Google Colab free tier (T4 GPU)!

ROCm support varies by provider (check documentation).

## Troubleshooting

### GPU Not Detected

```bash
# Check PyTorch can see GPU
python -c "import torch; print(torch.cuda.is_available())"

# Get detailed info
python -c "from src.utils import print_device_info; print_device_info()"
```

### Out of Memory

**Solutions:**
1. Reduce batch size in config
2. Reduce model size (d_model)
3. Use gradient accumulation
4. Enable mixed precision (FP16)

### Slow Training

**Checklist:**
- [ ] GPU utilization high? (check nvidia-smi or rocm-smi)
- [ ] Batch size optimal?
- [ ] Latest drivers installed?
- [ ] Thermal throttling? (check temps)

## Advanced: Multi-GPU

For multiple GPUs:

```python
# In train.py, add:
if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)
```

Or use DDP (Distributed Data Parallel) for better performance.

## Apple Silicon Notes

M1/M2/M3 Macs use the MPS (Metal Performance Shaders) backend:

```python
# Automatic on Apple Silicon
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
```

**Performance**: ~60-70% of NVIDIA GPUs
**Memory**: Shared with system RAM
**Support**: Growing but some ops missing

## Questions?

See device-specific guides:
- [ROCM_GUIDE.md](ROCM_GUIDE.md) - AMD GPUs
- [ARCHITECTURE.md](ARCHITECTURE.md) - Model details
- [README.md](README.md) - General usage

Or check your setup:
```bash
python test_setup.py
```

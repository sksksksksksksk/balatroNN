# BalatroNN - Neural Network Training for Balatro

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sksksksksksksk/balatroNN/blob/main/colab_setup.ipynb)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

A reinforcement learning system to train neural networks to play Balatro, the poker-based roguelike deck-building game.

**🚀 Train for FREE in Google Colab!** Click the badge above to get started in 5 minutes.

## Overview

This project implements a PPO (Proximal Policy Optimization) agent capable of learning to play Balatro through self-play and game simulation. The architecture uses transformer-based models to handle the complex state space of cards, jokers, and game mechanics.

## Hardware Requirements

### NVIDIA GPUs (CUDA)
- **Recommended**: H100 (80GB VRAM)
- **Minimum**: RTX 3090, RTX 4090, A100 (24GB+ VRAM)

### AMD GPUs (ROCm) ✨ NEW!
- **Recommended**: RX 7900 XTX (24GB VRAM)
- **Also Supported**: RX 7900 XT, RX 6800 XT, Radeon VII
- **See**: [ROCM_GUIDE.md](ROCM_GUIDE.md) for setup

### Google Cloud TPU ☁️ NEW!
- **Free**: TPU v2-8 on Google Colab (8 cores)
- **Paid**: TPU v3/v4 on Google Cloud
- **See**: [COLAB_GUIDE.md](COLAB_GUIDE.md) for instructions

### Google Colab (FREE Training!) 🎉
- **T4 GPU**: Free tier - perfect for learning/testing
- **A100 GPU**: Pro tier ($10/mo) - production training
- **See**: [COLAB_GUIDE.md](COLAB_GUIDE.md) for complete setup

### System
- **RAM**: 32GB+ system memory (16GB+ for smaller models)
- **Storage**: 50GB+ for checkpoints and logs
- **OS**: Linux (recommended), Windows, macOS

## Project Structure

```
balatroNN/
├── src/
│   ├── environment/      # Balatro game environment
│   ├── models/          # Neural network architectures
│   ├── training/        # PPO training logic
│   └── utils/           # Helper functions
├── configs/             # Training configurations
├── checkpoints/         # Model checkpoints
├── logs/               # Training logs
└── train.py            # Main training script
```

## Installation

### Local Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Google Colab Setup (Free!)

No installation needed! Open [COLAB_GUIDE.md](COLAB_GUIDE.md) for instructions.

```python
# In Colab notebook:
!pip install gymnasium tensorboard tqdm pydantic pyyaml
!python train.py --config configs/colab.yaml
```

## Usage

### Basic Training

```bash
python train.py --config configs/default.yaml
```

### Resume Training

```bash
python train.py --config configs/default.yaml --resume checkpoints/latest.pt
```

### Monitor Training

```bash
# TensorBoard
tensorboard --logdir logs/
```

## Architecture

### State Representation
- Hand cards (suit, rank, enhancements)
- Deck composition
- Jokers and their effects
- Consumables (tarot cards, planet cards)
- Game state (blind level, chips required, money, etc.)
- Shop state and available purchases

### Action Space
- Select cards to play
- Discard cards
- Use consumables
- Buy/sell items in shop
- Reroll shop
- Navigate menus

### Neural Network
- Transformer encoder for card sequences
- Attention mechanisms for card interactions
- Policy head for action selection
- Value head for state evaluation

## Training Strategy

1. **Curriculum Learning**: Start with easier blinds and gradually increase difficulty
2. **Self-Play**: Agent learns from playing against itself
3. **Reward Shaping**: Rewards for reaching higher blinds, efficient chip scoring
4. **Exploration**: Entropy bonus to encourage diverse strategies

## Performance Metrics

- Average blind reached per run
- Win rate on different stakes
- Chip efficiency (chips scored per hand)
- Decision time per action

## License

MIT License


### 🕹️ Playing the Real Game

**NEW:** Connect your trained model to the actual Balatro game!

```bash
# 1. Calibrate screen regions (interactive)
python calibrate_game.py

# 2. Test calibration
python calibrate_game.py --test

# 3. Play with your trained model
python play_real_game.py --checkpoint checkpoints/final_model.pt

# See REAL_GAME_GUIDE.md for complete setup instructions
```

**Features:**
- Computer vision for game state detection
- OCR for reading numbers/text
- Automated mouse/keyboard control
- Drop-in replacement for simulated environment
- Works with any trained model

**Requirements:**
- Tesseract OCR installed
- Balatro in windowed mode (1920x1080 recommended)
- One-time calibration setup


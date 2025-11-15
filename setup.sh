#!/bin/bash
# Setup script for BalatroNN

set -e

echo "=========================================="
echo "BalatroNN Setup"
echo "=========================================="
echo ""

# Check Python version
echo "Checking Python version..."
PYTHON_VERSION=$(python3 --version 2>&1 | grep -oP '(?<=Python )\d+\.\d+')
REQUIRED_VERSION="3.8"

if (( $(echo "$PYTHON_VERSION < $REQUIRED_VERSION" | bc -l) )); then
    echo "Error: Python $REQUIRED_VERSION or higher is required (found $PYTHON_VERSION)"
    exit 1
fi

echo "Python version: $PYTHON_VERSION ✓"
echo ""

# Check CUDA availability
echo "Checking CUDA availability..."
if command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    echo ""
else
    echo "No NVIDIA GPU detected. Training will use CPU (slower)."
    echo ""
fi

# Create virtual environment
echo "Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "Virtual environment created ✓"
else
    echo "Virtual environment already exists ✓"
fi
echo ""

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate
echo ""

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip > /dev/null 2>&1
echo "pip upgraded ✓"
echo ""

# Install PyTorch
echo "Installing PyTorch..."
if command -v nvidia-smi &> /dev/null; then
    echo "Installing PyTorch with CUDA support..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
else
    echo "Installing PyTorch (CPU only)..."
    pip install torch torchvision torchaudio
fi
echo "PyTorch installed ✓"
echo ""

# Install other requirements
echo "Installing other dependencies..."
pip install -r requirements.txt
echo "Dependencies installed ✓"
echo ""

# Create directories
echo "Creating directory structure..."
mkdir -p checkpoints logs configs/archive data plots
echo "Directories created ✓"
echo ""

# Test installation
echo "Testing installation..."
python3 -c "
import torch
import numpy
import gymnasium
print('✓ All imports successful')

if torch.cuda.is_available():
    print(f'✓ CUDA available: {torch.cuda.get_device_name(0)}')
    print(f'✓ CUDA version: {torch.version.cuda}')
else:
    print('! CUDA not available, will use CPU')
"
echo ""

# Print summary
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "To get started:"
echo "  1. Activate the virtual environment:"
echo "     source venv/bin/activate"
echo ""
echo "  2. Run a quick test:"
echo "     python train.py --config configs/quick_test.yaml"
echo ""
echo "  3. Start full training:"
echo "     python train.py --config configs/default.yaml"
echo ""
echo "  4. For H100 optimized training:"
echo "     python train.py --config configs/h100_large.yaml"
echo ""
echo "  5. Monitor training:"
echo "     tensorboard --logdir logs/"
echo ""
echo "Happy training! 🃏🎰"


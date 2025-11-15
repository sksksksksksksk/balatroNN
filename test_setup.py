#!/usr/bin/env python3
"""
Quick test to verify the setup is working correctly

Usage:
    python test_setup.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))


def test_imports():
    """Test that all required packages can be imported"""
    print("Testing imports...")
    
    try:
        import torch
        print(f"  ✓ PyTorch {torch.__version__}")
        
        if torch.cuda.is_available():
            print(f"    - CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"    - CUDA version: {torch.version.cuda}")
        else:
            print("    - CUDA not available (CPU only)")
    except ImportError as e:
        print(f"  ✗ PyTorch import failed: {e}")
        return False
    
    try:
        import numpy as np
        print(f"  ✓ NumPy {np.__version__}")
    except ImportError as e:
        print(f"  ✗ NumPy import failed: {e}")
        return False
    
    try:
        import gymnasium as gym
        print(f"  ✓ Gymnasium {gym.__version__}")
    except ImportError as e:
        print(f"  ✗ Gymnasium import failed: {e}")
        return False
    
    try:
        import yaml
        print("  ✓ PyYAML")
    except ImportError as e:
        print(f"  ✗ PyYAML import failed: {e}")
        return False
    
    try:
        from torch.utils.tensorboard import SummaryWriter
        print("  ✓ TensorBoard")
    except ImportError as e:
        print(f"  ✗ TensorBoard import failed: {e}")
        return False
    
    return True


def test_modules():
    """Test that our custom modules can be imported"""
    print("\nTesting custom modules...")
    
    try:
        from environment import BalatroEnv, GameState, Card
        print("  ✓ Environment module")
    except Exception as e:
        print(f"  ✗ Environment module failed: {e}")
        return False
    
    try:
        from models import PolicyValueNetwork
        print("  ✓ Models module")
    except Exception as e:
        print(f"  ✗ Models module failed: {e}")
        return False
    
    try:
        from training import PPOTrainer, PPOConfig
        print("  ✓ Training module")
    except Exception as e:
        print(f"  ✗ Training module failed: {e}")
        return False
    
    try:
        from utils import load_config, Logger
        print("  ✓ Utils module")
    except Exception as e:
        print(f"  ✗ Utils module failed: {e}")
        return False
    
    return True


def test_environment():
    """Test that the environment works"""
    print("\nTesting environment...")
    
    try:
        from environment import BalatroEnv
        
        env = BalatroEnv()
        obs, info = env.reset()
        
        print(f"  ✓ Environment created")
        print(f"    - Observation space: {env.observation_space}")
        print(f"    - Action space: {env.action_space}")
        
        # Take a random action
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        print(f"  ✓ Environment step successful")
        print(f"    - Reward: {reward}")
        print(f"    - Info: {info}")
        
    except Exception as e:
        print(f"  ✗ Environment test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def test_model():
    """Test that the model can be created and run"""
    print("\nTesting model...")
    
    try:
        import torch
        from models import PolicyValueNetwork
        from environment import BalatroEnv
        
        env = BalatroEnv()
        model = PolicyValueNetwork(d_model=128, dropout=0.1)
        
        print(f"  ✓ Model created")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        print(f"    - Parameters: {total_params:,}")
        
        # Test forward pass
        obs, _ = env.reset()
        obs_tensor = {
            key: torch.as_tensor(val, dtype=torch.float32).unsqueeze(0)
            for key, val in obs.items()
        }
        
        with torch.no_grad():
            action, log_prob, entropy, value = model.get_action_and_value(obs_tensor)
        
        print(f"  ✓ Forward pass successful")
        print(f"    - Action type: {action['action_type'].item()}")
        print(f"    - Value: {value.item():.3f}")
        
    except Exception as e:
        print(f"  ✗ Model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def test_config():
    """Test that config files can be loaded"""
    print("\nTesting configuration...")
    
    try:
        from utils import load_config
        
        config = load_config("configs/default.yaml")
        print("  ✓ Default config loaded")
        print(f"    - Total timesteps: {config['training']['total_timesteps']:,}")
        print(f"    - Model d_model: {config['model']['d_model']}")
        
        config = load_config("configs/quick_test.yaml")
        print("  ✓ Quick test config loaded")
        
        config = load_config("configs/h100_large.yaml")
        print("  ✓ H100 config loaded")
        
    except Exception as e:
        print(f"  ✗ Config test failed: {e}")
        return False
    
    return True


def main():
    print("="*60)
    print("BalatroNN Setup Test")
    print("="*60)
    print()
    
    all_passed = True
    
    # Run tests
    all_passed &= test_imports()
    all_passed &= test_modules()
    all_passed &= test_environment()
    all_passed &= test_model()
    all_passed &= test_config()
    
    print()
    print("="*60)
    if all_passed:
        print("✓ All tests passed! Setup is working correctly.")
        print()
        print("You're ready to start training:")
        print("  python train.py --config configs/quick_test.yaml")
    else:
        print("✗ Some tests failed. Please check the error messages above.")
        sys.exit(1)
    print("="*60)


if __name__ == "__main__":
    main()


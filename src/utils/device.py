"""
Device detection and management for CUDA, ROCm, TPU, and CPU

Supports:
- NVIDIA GPUs (CUDA)
- AMD GPUs (ROCm)
- Google TPUs (PyTorch/XLA)
- CPU fallback
"""

import torch
from typing import Tuple
import sys


def get_device(prefer_device: str = "auto") -> Tuple[torch.device, str]:
    """
    Get best available device
    
    Args:
        prefer_device: "auto", "cuda", "rocm", "tpu", "cpu"
    
    Returns:
        (device, device_name) tuple
    """
    if prefer_device == "cpu":
        return torch.device("cpu"), "CPU"
    
    # Check for TPU (Google Cloud TPU)
    if prefer_device in ["auto", "tpu"]:
        try:
            import torch_xla.core.xla_model as xm
            device = xm.xla_device()
            device_name = f"TPU (cores: {xm.xrt_world_size()})"
            return device, device_name
        except (ImportError, RuntimeError):
            if prefer_device == "tpu":
                print("Warning: TPU requested but torch_xla not available")
    
    # Check for CUDA (NVIDIA)
    if torch.cuda.is_available():
        if prefer_device in ["auto", "cuda"]:
            device = torch.device("cuda")
            device_name = torch.cuda.get_device_name(0)
            return device, f"CUDA - {device_name}"
    
    # Check for ROCm (AMD)
    # PyTorch with ROCm also uses torch.cuda API but with different backend
    if hasattr(torch.version, 'hip') and torch.version.hip is not None:
        if prefer_device in ["auto", "rocm", "cuda"]:  # "cuda" works for ROCm too
            device = torch.device("cuda")  # ROCm uses same API as CUDA
            # Try to get AMD GPU name
            try:
                device_name = torch.cuda.get_device_name(0)
            except:
                device_name = "AMD GPU (ROCm)"
            return device, f"ROCm - {device_name}"
    
    # Fallback to CPU
    return torch.device("cpu"), "CPU"


def is_cuda_available() -> bool:
    """Check if CUDA (NVIDIA) is available"""
    return torch.cuda.is_available() and not is_rocm()


def is_rocm() -> bool:
    """Check if ROCm (AMD) is available"""
    return hasattr(torch.version, 'hip') and torch.version.hip is not None


def is_tpu() -> bool:
    """Check if TPU is available"""
    try:
        import torch_xla
        return True
    except ImportError:
        return False


def get_device_info() -> dict:
    """
    Get detailed device information
    
    Returns:
        Dictionary with device details
    """
    info = {
        "cuda_available": torch.cuda.is_available(),
        "rocm_available": is_rocm(),
        "tpu_available": is_tpu(),
        "device_count": 0,
        "pytorch_version": torch.__version__,
    }
    
    # Check TPU first
    if is_tpu():
        try:
            import torch_xla.core.xla_model as xm
            info["backend"] = "TPU"
            info["device_count"] = xm.xrt_world_size()
            info["tpu_cores"] = xm.xrt_world_size()
            return info
        except Exception as e:
            print(f"TPU info error: {e}")
    
    # Check GPU (CUDA/ROCm)
    if torch.cuda.is_available():
        info["device_count"] = torch.cuda.device_count()
        info["current_device"] = torch.cuda.current_device()
        info["device_name"] = torch.cuda.get_device_name(0)
        
        if is_rocm():
            info["backend"] = "ROCm"
            info["rocm_version"] = torch.version.hip if hasattr(torch.version, 'hip') else "unknown"
        else:
            info["backend"] = "CUDA"
            info["cuda_version"] = torch.version.cuda
        
        # Memory info
        if hasattr(torch.cuda, 'get_device_properties'):
            props = torch.cuda.get_device_properties(0)
            info["total_memory_gb"] = props.total_memory / (1024**3)
    else:
        info["backend"] = "CPU"
    
    return info


def print_device_info():
    """Print device information in a nice format"""
    info = get_device_info()
    
    print("\n" + "="*60)
    print("DEVICE INFORMATION")
    print("="*60)
    
    print(f"PyTorch Version: {info['pytorch_version']}")
    print(f"Backend: {info['backend']}")
    
    if info['backend'] == 'TPU':
        print(f"TPU Available: Yes")
        print(f"TPU Cores: {info.get('tpu_cores', 'unknown')}")
    elif info['cuda_available']:
        print(f"GPU Available: Yes")
        print(f"Device Name: {info['device_name']}")
        print(f"Device Count: {info['device_count']}")
        
        if info['backend'] == 'CUDA':
            print(f"CUDA Version: {info['cuda_version']}")
        elif info['backend'] == 'ROCm':
            print(f"ROCm Version: {info['rocm_version']}")
        
        if 'total_memory_gb' in info:
            print(f"GPU Memory: {info['total_memory_gb']:.2f} GB")
    else:
        print(f"GPU/TPU Available: No (using CPU)")
    
    print("="*60 + "\n")


def optimize_for_device(model: torch.nn.Module, device: torch.device) -> torch.nn.Module:
    """
    Apply device-specific optimizations
    
    Args:
        model: PyTorch model
        device: Target device
    
    Returns:
        Optimized model
    """
    model = model.to(device)
    
    # Enable TF32 for better performance on Ampere+ (CUDA) and RDNA3+ (ROCm)
    if device.type == "cuda":
        if is_rocm():
            # ROCm-specific optimizations
            # Most optimizations are automatic, but we can enable some features
            pass
        else:
            # CUDA-specific optimizations
            # Enable TF32 for matrix multiplications (Ampere and newer)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    
    return model


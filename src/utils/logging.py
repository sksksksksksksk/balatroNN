"""
Logging utilities for training
"""

import os
from typing import Dict, Any, Optional
import json
from pathlib import Path
import torch
from torch.utils.tensorboard import SummaryWriter


class Logger:
    """
    Training logger that supports multiple backends
    """
    
    def __init__(self, log_dir: str, use_tensorboard: bool = True):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.use_tensorboard = use_tensorboard
        
        # Initialize tensorboard
        self.tb_writer = None
        if use_tensorboard:
            self.tb_writer = SummaryWriter(log_dir=str(self.log_dir / "tensorboard"))
        
        # JSON log file
        self.json_log_path = self.log_dir / "metrics.jsonl"
        
    def log_scalar(self, tag: str, value: float, step: int):
        """Log a scalar value"""
        if self.tb_writer:
            self.tb_writer.add_scalar(tag, value, step)
    
    def log_scalars(self, metrics: Dict[str, float], step: int):
        """Log multiple scalar values"""
        for tag, value in metrics.items():
            self.log_scalar(tag, value, step)
        
        # Also log to JSON file
        with open(self.json_log_path, 'a') as f:
            log_entry = {"step": step, **metrics}
            f.write(json.dumps(log_entry) + "\n")
    
    def log_histogram(self, tag: str, values: torch.Tensor, step: int):
        """Log histogram of values"""
        if self.tb_writer:
            self.tb_writer.add_histogram(tag, values, step)
    
    def log_text(self, tag: str, text: str, step: int):
        """Log text"""
        if self.tb_writer:
            self.tb_writer.add_text(tag, text, step)
    
    def close(self):
        """Close all loggers"""
        if self.tb_writer:
            self.tb_writer.close()


def setup_tensorboard(log_dir: str) -> SummaryWriter:
    """
    Setup TensorBoard writer
    
    Args:
        log_dir: Directory to save logs
    
    Returns:
        TensorBoard SummaryWriter
    """
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    return SummaryWriter(log_dir=log_dir)


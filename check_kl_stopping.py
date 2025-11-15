#!/usr/bin/env python3
"""
Quick script to check how often KL early stopping occurs
"""

import sys
import re
from pathlib import Path

def analyze_training_log(log_file):
    """Analyze training log for KL early stopping frequency"""
    
    if not Path(log_file).exists():
        print(f"❌ Log file not found: {log_file}")
        print("\nTo create a log, run:")
        print("  python train.py --config configs/default.yaml | tee training.log")
        return
    
    with open(log_file, 'r') as f:
        lines = f.readlines()
    
    total_updates = 0
    early_stops = 0
    
    for line in lines:
        # Count updates
        if 'Update' in line or 'Timesteps' in line:
            total_updates += 1
        
        # Count early stops
        if 'Early stopping' in line and 'KL divergence' in line:
            early_stops += 1
    
    if total_updates == 0:
        print("⚠️  No training updates found in log")
        return
    
    percentage = (early_stops / total_updates) * 100
    
    print("="*60)
    print("KL DIVERGENCE EARLY STOPPING ANALYSIS")
    print("="*60)
    print(f"Total Updates: {total_updates}")
    print(f"Early Stops: {early_stops}")
    print(f"Percentage: {percentage:.1f}%")
    print("="*60)
    
    if percentage < 20:
        print("✅ Status: GOOD")
        print("Your model is learning smoothly with occasional safety stops.")
        print("No action needed!")
    elif percentage < 50:
        print("⚠️  Status: MODERATE")
        print("Early stopping is common. This is often normal in early training.")
        print("Monitor if it continues - may want to adjust target_kl.")
    elif percentage < 80:
        print("🔶 Status: FREQUENT")
        print("Early stopping is happening a lot!")
        print("\nSuggestions:")
        print("1. Increase target_kl: 0.01 → 0.02")
        print("2. Reduce learning_rate by 25-50%")
        print("3. Wait - it may stabilize as training progresses")
    else:
        print("🔴 Status: CRITICAL")
        print("Early stopping almost always! Your learning is being severely constrained.")
        print("\nImmediate actions:")
        print("1. Increase target_kl: 0.01 → 0.03")
        print("2. Or reduce learning_rate by 50%")
        print("3. Check if your reward signal is too noisy")
    
    print("="*60)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python check_kl_stopping.py <training_log_file>")
        print("\nExample:")
        print("  # First, run training with logging:")
        print("  python train.py --config configs/default.yaml 2>&1 | tee training.log")
        print()
        print("  # Then analyze:")
        print("  python check_kl_stopping.py training.log")
        sys.exit(1)
    
    analyze_training_log(sys.argv[1])


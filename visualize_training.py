#!/usr/bin/env python3
"""
Visualize training progress from logs

Usage:
    python visualize_training.py --log-dir logs/
    python visualize_training.py --log-dir logs/ --save plots/
"""

import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize training progress")
    
    parser.add_argument(
        "--log-dir",
        type=str,
        required=True,
        help="Path to log directory"
    )
    
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Directory to save plots"
    )
    
    parser.add_argument(
        "--smooth",
        type=int,
        default=10,
        help="Smoothing window for plots"
    )
    
    return parser.parse_args()


def smooth(data, window=10):
    """Apply moving average smoothing"""
    if len(data) < window:
        return data
    return np.convolve(data, np.ones(window)/window, mode='valid')


def load_metrics(log_dir: Path):
    """Load metrics from JSON log file"""
    metrics_file = log_dir / "metrics.jsonl"
    
    if not metrics_file.exists():
        print(f"No metrics file found at {metrics_file}")
        return None
    
    metrics = []
    with open(metrics_file, 'r') as f:
        for line in f:
            metrics.append(json.loads(line))
    
    return metrics


def plot_metrics(metrics, save_dir=None, smooth_window=10):
    """Create plots from metrics"""
    
    # Extract data
    steps = [m['step'] for m in metrics]
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('BalatroNN Training Progress', fontsize=16)
    
    # Plot 1: Mean Reward
    if 'rollout/mean_reward' in metrics[0]:
        rewards = [m['rollout/mean_reward'] for m in metrics]
        ax = axes[0, 0]
        ax.plot(steps, rewards, alpha=0.3, label='Raw')
        if len(rewards) > smooth_window:
            smooth_rewards = smooth(rewards, smooth_window)
            ax.plot(steps[:len(smooth_rewards)], smooth_rewards, label='Smoothed')
        ax.set_xlabel('Timesteps')
        ax.set_ylabel('Mean Reward')
        ax.set_title('Episode Reward')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Plot 2: Episode Length
    if 'rollout/mean_length' in metrics[0]:
        lengths = [m['rollout/mean_length'] for m in metrics]
        ax = axes[0, 1]
        ax.plot(steps, lengths, alpha=0.3, label='Raw')
        if len(lengths) > smooth_window:
            smooth_lengths = smooth(lengths, smooth_window)
            ax.plot(steps[:len(smooth_lengths)], smooth_lengths, label='Smoothed')
        ax.set_xlabel('Timesteps')
        ax.set_ylabel('Mean Length')
        ax.set_title('Episode Length')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Plot 3: Policy Loss
    if 'train/policy_loss' in metrics[0]:
        policy_loss = [m['train/policy_loss'] for m in metrics]
        ax = axes[0, 2]
        ax.plot(steps, policy_loss, alpha=0.3, label='Raw')
        if len(policy_loss) > smooth_window:
            smooth_policy_loss = smooth(policy_loss, smooth_window)
            ax.plot(steps[:len(smooth_policy_loss)], smooth_policy_loss, label='Smoothed')
        ax.set_xlabel('Timesteps')
        ax.set_ylabel('Policy Loss')
        ax.set_title('Policy Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Plot 4: Value Loss
    if 'train/value_loss' in metrics[0]:
        value_loss = [m['train/value_loss'] for m in metrics]
        ax = axes[1, 0]
        ax.plot(steps, value_loss, alpha=0.3, label='Raw')
        if len(value_loss) > smooth_window:
            smooth_value_loss = smooth(value_loss, smooth_window)
            ax.plot(steps[:len(smooth_value_loss)], smooth_value_loss, label='Smoothed')
        ax.set_xlabel('Timesteps')
        ax.set_ylabel('Value Loss')
        ax.set_title('Value Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Plot 5: KL Divergence
    if 'train/kl_divergence' in metrics[0]:
        kl_div = [m['train/kl_divergence'] for m in metrics]
        ax = axes[1, 1]
        ax.plot(steps, kl_div, alpha=0.5)
        if len(kl_div) > smooth_window:
            smooth_kl = smooth(kl_div, smooth_window)
            ax.plot(steps[:len(smooth_kl)], smooth_kl, linewidth=2)
        ax.set_xlabel('Timesteps')
        ax.set_ylabel('KL Divergence')
        ax.set_title('KL Divergence')
        ax.grid(True, alpha=0.3)
    
    # Plot 6: Entropy
    if 'train/entropy_loss' in metrics[0]:
        entropy = [-m['train/entropy_loss'] for m in metrics]  # Negate to show actual entropy
        ax = axes[1, 2]
        ax.plot(steps, entropy, alpha=0.5)
        if len(entropy) > smooth_window:
            smooth_entropy = smooth(entropy, smooth_window)
            ax.plot(steps[:len(smooth_entropy)], smooth_entropy, linewidth=2)
        ax.set_xlabel('Timesteps')
        ax.set_ylabel('Entropy')
        ax.set_title('Policy Entropy')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_dir:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        plot_file = save_path / 'training_progress.png'
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {plot_file}")
    else:
        plt.show()


def print_summary(metrics):
    """Print summary statistics"""
    if not metrics:
        return
    
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    
    # Get latest metrics
    latest = metrics[-1]
    
    print(f"Total timesteps: {latest['step']:,}")
    
    if 'rollout/mean_reward' in latest:
        # Get last 10% of training for final stats
        final_10_percent = metrics[int(len(metrics)*0.9):]
        final_rewards = [m['rollout/mean_reward'] for m in final_10_percent]
        
        print(f"\nFinal Mean Reward (last 10%): {np.mean(final_rewards):.2f} ± {np.std(final_rewards):.2f}")
        print(f"Best Mean Reward: {max(m['rollout/mean_reward'] for m in metrics):.2f}")
    
    if 'train/policy_loss' in latest:
        print(f"\nFinal Policy Loss: {latest['train/policy_loss']:.4f}")
    
    if 'train/value_loss' in latest:
        print(f"Final Value Loss: {latest['train/value_loss']:.4f}")
    
    if 'train/kl_divergence' in latest:
        print(f"Final KL Divergence: {latest['train/kl_divergence']:.4f}")
    
    print("="*60)


def main():
    args = parse_args()
    
    log_dir = Path(args.log_dir)
    
    if not log_dir.exists():
        print(f"Error: Log directory {log_dir} does not exist")
        return
    
    print(f"Loading metrics from {log_dir}")
    metrics = load_metrics(log_dir)
    
    if not metrics:
        print("No metrics found")
        return
    
    print(f"Loaded {len(metrics)} metric entries")
    
    # Print summary
    print_summary(metrics)
    
    # Create plots
    print("\nGenerating plots...")
    plot_metrics(metrics, args.save, args.smooth)


if __name__ == "__main__":
    main()


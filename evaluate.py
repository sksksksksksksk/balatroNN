#!/usr/bin/env python3
"""
Evaluation script for trained BalatroNN models

Usage:
    python evaluate.py --checkpoint checkpoints/final_model.pt --episodes 100
    python evaluate.py --checkpoint checkpoints/final_model.pt --render
"""

import argparse
import sys
from pathlib import Path
import torch
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent / "src"))

from environment import BalatroEnv
from models import PolicyValueNetwork
from utils import get_device


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate BalatroNN agent")
    
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint"
    )
    
    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="Number of evaluation episodes"
    )
    
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use deterministic policy (no sampling)"
    )
    
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render environment during evaluation"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use (auto/cuda/rocm/cpu)"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed"
    )
    
    return parser.parse_args()


def evaluate(model, env, num_episodes: int, deterministic: bool = True, 
             render: bool = False, device: str = "cuda"):
    """
    Evaluate the model
    
    Args:
        model: Trained model
        env: Environment
        num_episodes: Number of episodes to evaluate
        deterministic: Use deterministic policy
        render: Render environment
        device: Device to use
    
    Returns:
        Dictionary of evaluation metrics
    """
    model.eval()
    
    episode_rewards = []
    episode_lengths = []
    episode_antes = []
    
    for episode in tqdm(range(num_episodes), desc="Evaluating"):
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        episode_length = 0
        max_ante = 1
        
        while not done:
            # Convert observation to tensor
            obs_tensor = {
                key: torch.as_tensor(val, dtype=torch.float32, device=device).unsqueeze(0)
                for key, val in obs.items()
            }
            
            # Get action
            with torch.no_grad():
                action, _, _, value = model.get_action_and_value(obs_tensor, deterministic=deterministic)
            
            # Convert to numpy
            action_np = {
                "action_type": action["action_type"].cpu().numpy()[0],
                "card_selection": action["card_selection"].cpu().numpy()[0],
                "shop_selection": action["shop_selection"].cpu().numpy()[0]
            }
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action_np)
            done = terminated or truncated
            
            episode_reward += reward
            episode_length += 1
            
            # Track max ante reached
            if "ante" in info:
                max_ante = max(max_ante, info["ante"])
            
            # Render if requested
            if render and episode == 0:  # Only render first episode
                env.render()
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        episode_antes.append(max_ante)
        
        if render or episode < 3:  # Print first 3 episodes
            print(f"\nEpisode {episode + 1}: Reward = {episode_reward:.2f}, Length = {episode_length}, Max Ante = {max_ante}")
    
    # Compute statistics
    metrics = {
        "mean_reward": np.mean(episode_rewards),
        "std_reward": np.std(episode_rewards),
        "min_reward": np.min(episode_rewards),
        "max_reward": np.max(episode_rewards),
        "mean_length": np.mean(episode_lengths),
        "std_length": np.std(episode_lengths),
        "mean_ante": np.mean(episode_antes),
        "max_ante": np.max(episode_antes),
        "win_rate": np.mean([ante >= 8 for ante in episode_antes])
    }
    
    return metrics, episode_rewards, episode_lengths, episode_antes


def main():
    args = parse_args()
    
    # Set seed
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
    
    # Get device
    device, device_name = get_device(args.device)
    print(f"Using device: {device_name}")
    
    print(f"Loading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # Create environment
    env = BalatroEnv()
    
    # Create model
    model = PolicyValueNetwork()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    
    print(f"Model loaded. Timesteps trained: {checkpoint.get('num_timesteps', 'unknown')}")
    print(f"Device: {args.device}")
    print(f"Evaluating for {args.episodes} episodes...")
    print("")
    
    # Evaluate
    metrics, rewards, lengths, antes = evaluate(
        model, env, args.episodes, 
        deterministic=args.deterministic,
        render=args.render,
        device=args.device
    )
    
    # Print results
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Episodes: {args.episodes}")
    print(f"Policy: {'Deterministic' if args.deterministic else 'Stochastic'}")
    print("")
    print(f"Mean Reward: {metrics['mean_reward']:.2f} ± {metrics['std_reward']:.2f}")
    print(f"Reward Range: [{metrics['min_reward']:.2f}, {metrics['max_reward']:.2f}]")
    print("")
    print(f"Mean Episode Length: {metrics['mean_length']:.1f} ± {metrics['std_length']:.1f}")
    print("")
    print(f"Mean Ante Reached: {metrics['mean_ante']:.2f}")
    print(f"Max Ante Reached: {metrics['max_ante']}")
    print(f"Win Rate (Ante 8+): {metrics['win_rate']*100:.1f}%")
    print("="*60)
    
    # Save results
    results_path = Path(args.checkpoint).parent / "evaluation_results.txt"
    with open(results_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("EVALUATION RESULTS\n")
        f.write("="*60 + "\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Episodes: {args.episodes}\n")
        f.write(f"Policy: {'Deterministic' if args.deterministic else 'Stochastic'}\n\n")
        for key, value in metrics.items():
            f.write(f"{key}: {value}\n")
        f.write("\n")
        f.write("Per-episode results:\n")
        for i, (r, l, a) in enumerate(zip(rewards, lengths, antes)):
            f.write(f"Episode {i+1}: Reward={r:.2f}, Length={l}, Ante={a}\n")
    
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()


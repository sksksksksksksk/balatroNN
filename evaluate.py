#!/usr/bin/env python3
"""
Evaluation script for trained BalatroNN models

Usage:
    # Evaluate with automatic config detection (infers from checkpoint weights)
    python evaluate.py --checkpoint checkpoints/final_model.pt --episodes 100
    
    # Evaluate with explicit config (recommended for clarity)
    python evaluate.py --checkpoint checkpoints/a100_aggressive/checkpoint_*.pt --config configs/a100_aggressive.yaml --episodes 100
    
    # Verbose mode - see what actions the model takes
    python evaluate.py --checkpoint checkpoints/final_model.pt --episodes 1 --verbose
    
    # Verbose mode with more steps shown
    python evaluate.py --checkpoint checkpoints/final_model.pt --episodes 1 --verbose --max-steps-to-show 100
    
    # Evaluate with rendering
    python evaluate.py --checkpoint checkpoints/final_model.pt --config configs/a100_aggressive.yaml --render
    
Note:
    The config file must match the configuration used during training, especially the d_model parameter.
    If the checkpoint was trained with a100_aggressive.yaml (d_model=768), use that same config for evaluation.
"""

import argparse
import sys
from pathlib import Path
import torch
import numpy as np
import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent / "src"))

from environment import BalatroEnv
from models import PolicyValueNetwork, create_model
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
        "--config",
        type=str,
        default=None,
        help="Path to config file (required if checkpoint doesn't contain config). "
             "Should match the config used for training."
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
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed action logs during evaluation"
    )
    
    parser.add_argument(
        "--max-steps-to-show",
        type=int,
        default=50,
        help="Maximum steps to show in verbose mode per episode (default: 50)"
    )
    
    return parser.parse_args()


def format_action(action: dict, info: dict) -> str:
    """Format action for human-readable output"""
    action_type_names = [
        "Play Hand",
        "Discard",
        "Buy Joker",
        "Buy Pack",
        "Buy Card",
        "Buy Voucher",
        "Sell Joker",
        "Use Tarot",
        "Use Planet",
        "Use Spectral",
        "Reroll Shop",
        "Skip/Continue"
    ]
    
    action_type = int(action["action_type"])
    action_name = action_type_names[action_type] if action_type < len(action_type_names) else f"Unknown({action_type})"
    
    details = []
    
    # Add action-specific details
    if action_type == 0:  # Play hand
        selected = np.where(action["card_selection"] > 0.5)[0]
        if len(selected) > 0:
            details.append(f"cards={list(selected)}")
    elif action_type == 1:  # Discard
        selected = np.where(action["card_selection"] > 0.5)[0]
        if len(selected) > 0:
            details.append(f"cards={list(selected)}")
    elif action_type in [2, 3, 4, 5]:  # Shop purchases
        details.append(f"item={action.get('shop_item_index', '?')}")
    elif action_type == 6:  # Sell joker
        details.append(f"slot={action.get('joker_slot', '?')}")
    elif action_type in [7, 9]:  # Tarot/Spectral
        details.append(f"slot={action.get('consumable_slot', '?')}")
        details.append(f"target={action.get('target_card_index', '?')}")
    elif action_type == 8:  # Planet
        details.append(f"slot={action.get('consumable_slot', '?')}")
    
    detail_str = ", ".join(details) if details else ""
    return f"{action_name}" + (f" ({detail_str})" if detail_str else "")


def format_game_state(info: dict) -> str:
    """Format game state for display"""
    parts = []
    parts.append(f"Ante {info.get('ante', '?')}")
    
    chips = info.get('chips', 0)
    parts.append(f"Chips: {chips:,}")
    
    money = info.get('money', 0)
    parts.append(f"$: {money}")
    
    hands = info.get('hands_left', 0)
    parts.append(f"Hands: {hands}")
    
    return " | ".join(parts)


def evaluate(model, env, num_episodes: int, deterministic: bool = True, 
             render: bool = False, device: str = "cuda", verbose: bool = False,
             max_steps_to_show: int = 50):
    """
    Evaluate the model
    
    Args:
        model: Trained model
        env: Environment
        num_episodes: Number of episodes to evaluate
        deterministic: Use deterministic policy
        render: Render environment
        device: Device to use
        verbose: Show detailed action logs
        max_steps_to_show: Maximum steps to log in verbose mode
    
    Returns:
        Dictionary of evaluation metrics
    """
    model.eval()
    
    episode_rewards = []
    episode_lengths = []
    episode_antes = []
    
    for episode in tqdm(range(num_episodes), desc="Evaluating", disable=verbose):
        obs, info = env.reset()
        done = False
        episode_reward = 0
        episode_length = 0
        max_ante = 1
        step_rewards = []
        
        if verbose:
            print(f"\n{'='*80}")
            print(f"📺 EPISODE {episode + 1}/{num_episodes}")
            print(f"{'='*80}")
            print(f"Starting state: {format_game_state(info)}")
            print()
        
        while not done:
            # Convert observation to tensor
            obs_tensor = {
                key: torch.as_tensor(val, dtype=torch.float32, device=device).unsqueeze(0)
                for key, val in obs.items()
            }
            
            # Get action
            with torch.no_grad():
                action, _, _, value = model.get_action_and_value(obs_tensor, deterministic=deterministic)
            
            # Convert to numpy - handle all action components
            action_np = {}
            for key, val in action.items():
                if torch.is_tensor(val):
                    action_np[key] = val.cpu().numpy()[0]
                else:
                    action_np[key] = val
            
            # Verbose logging
            if verbose and episode_length < max_steps_to_show:
                action_str = format_action(action_np, info)
                value_est = value.cpu().item() if torch.is_tensor(value) else value
                print(f"Step {episode_length + 1:3d} | {action_str:<30} | V={value_est:7.2f}", end="")
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action_np)
            done = terminated or truncated
            
            # More verbose logging
            if verbose and episode_length < max_steps_to_show:
                print(f" | R={reward:7.2f} | {format_game_state(info)}")
            
            episode_reward += reward
            episode_length += 1
            step_rewards.append(reward)
            
            # Track max ante reached
            if "ante" in info:
                max_ante = max(max_ante, info["ante"])
            
            # Render if requested
            if render and episode == 0:  # Only render first episode
                env.render()
        
        # Episode summary
        if verbose:
            if episode_length >= max_steps_to_show:
                print(f"\n... (showing first {max_steps_to_show} steps, episode ran {episode_length} total)")
            print(f"\n{'─'*80}")
            print(f"Episode ended: {'TIMEOUT' if truncated else 'TERMINATED'}")
            print(f"Final state: {format_game_state(info)}")
            print(f"Total Reward: {episode_reward:.2f}")
            print(f"Total Steps: {episode_length}")
            print(f"Max Ante: {max_ante}")
            if len(step_rewards) > 0:
                positive_rewards = sum(1 for r in step_rewards if r > 0)
                negative_rewards = sum(1 for r in step_rewards if r < 0)
                print(f"Reward breakdown: {positive_rewards} positive, {negative_rewards} negative, {len(step_rewards) - positive_rewards - negative_rewards} neutral")
            print(f"{'='*80}\n")
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        episode_antes.append(max_ante)
        
        if not verbose and (render or episode < 3):  # Print first 3 episodes if not verbose
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
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    
    # Load configuration
    model_config = None
    
    # First, try to get config from checkpoint
    if "config" in checkpoint:
        config_obj = checkpoint["config"]
        # Handle both dict and config object
        if hasattr(config_obj, 'model'):
            # It's a config object (e.g., PPOConfig)
            model_config = vars(config_obj.model) if hasattr(config_obj.model, '__dict__') else config_obj.model
        elif isinstance(config_obj, dict):
            # It's a dictionary
            model_config = config_obj.get("model", {})
        else:
            model_config = None
        
        if model_config:
            d_model_val = model_config.get('d_model') if isinstance(model_config, dict) else getattr(model_config, 'd_model', 'unknown')
            print(f"✓ Loaded config from checkpoint: d_model={d_model_val}")
    # Otherwise, load from config file if provided
    if not model_config and args.config:
        with open(args.config, 'r') as f:
            full_config = yaml.safe_load(f)
            model_config = full_config.get("model", {})
        print(f"✓ Loaded config from {args.config}: d_model={model_config.get('d_model', 'unknown')}")
    
    # If still no model config, try to infer from checkpoint state dict
    if not model_config:
        try:
            # Check the size of a known parameter to infer d_model
            param_shape = checkpoint["model_state_dict"]["backbone.card_encoder.card_embedding.0.weight"].shape
            inferred_d_model = param_shape[0]  # Output dimension
            model_config = {"d_model": inferred_d_model, "dropout": 0.1}
            print(f"✓ Inferred model config from checkpoint: d_model={inferred_d_model}")
        except Exception as e:
            print(f"⚠ Warning: Could not infer model config from checkpoint")
            print(f"   Error: {e}")
            print(f"   Please provide --config argument with the correct config file")
            raise ValueError(
                "Could not determine model architecture. Please provide --config argument.\n"
                f"For example: --config configs/a100_aggressive.yaml"
            )
    
    # Create environment
    env = BalatroEnv()
    
    # Convert model_config to dict if it's an object
    if model_config and not isinstance(model_config, dict):
        model_config = vars(model_config) if hasattr(model_config, '__dict__') else {}
    
    # Create model with correct config
    model = create_model(model_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    
    print(f"Model loaded. Timesteps trained: {checkpoint.get('num_timesteps', 'unknown')}")
    print(f"Model architecture: d_model={model_config.get('d_model')}, "
          f"num_layers={model_config.get('num_layers', 'default')}")
    print(f"Device: {device_name}")
    print(f"Evaluating for {args.episodes} episodes...")
    print("")
    
    # Evaluate
    metrics, rewards, lengths, antes = evaluate(
        model, env, args.episodes, 
        deterministic=args.deterministic,
        render=args.render,
        device=device,
        verbose=args.verbose,
        max_steps_to_show=args.max_steps_to_show
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


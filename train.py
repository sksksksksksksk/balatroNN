#!/usr/bin/env python3
"""
Main training script for BalatroNN

Usage:
    python train.py --config configs/default.yaml
    python train.py --config configs/default.yaml --resume checkpoints/latest.pt
"""

import argparse
import os
import sys
from pathlib import Path
import torch
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from environment import BalatroEnv
from models import PolicyValueNetwork
from training import PPOTrainer, PPOConfig
from training.curriculum import JokerCurriculum
from utils import load_config, save_config, Logger, get_device, print_device_info


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train BalatroNN agent")
    
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/cpu). Overrides config."
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Only run evaluation, no training"
    )
    
    return parser.parse_args()


def set_seed(seed: int):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    

def main():
    """Main training function"""
    args = parse_args()
    
    # Load configuration
    print(f"Loading configuration from {args.config}")
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.device:
        config["training"]["device"] = args.device
    
    # Set random seed
    set_seed(args.seed)
    print(f"Random seed: {args.seed}")
    
    # Check device availability with ROCm support
    device_str = config["training"]["device"]
    device, device_name = get_device(device_str)
    
    # Update config with actual device
    config["training"]["device"] = str(device)
    
    print(f"Using device: {device_name}")
    
    # Print detailed device info
    print_device_info()
    
    # Create environment
    print("\nInitializing environment...")
    env_config = config.get("environment", {})
    env = BalatroEnv(config=env_config)
    print(f"Environment created: {env}")
    
    # Create model
    print("\nInitializing model...")
    model_config = config.get("model", {})
    model = PolicyValueNetwork(
        d_model=model_config.get("d_model", 256),
        dropout=model_config.get("dropout", 0.1)
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size: {total_params * 4 / 1024**2:.2f} MB (float32)")
    
    # Create PPO config
    ppo_config = PPOConfig(
        learning_rate=config["training"]["learning_rate"],
        batch_size=config["training"]["batch_size"],
        n_epochs=config["training"]["n_epochs"],
        n_steps=config["training"]["n_steps"],
        gamma=config["training"]["gamma"],
        gae_lambda=config["training"]["gae_lambda"],
        clip_range=config["training"]["clip_range"],
        clip_range_vf=config["training"].get("clip_range_vf"),
        ent_coef=config["training"]["ent_coef"],
        vf_coef=config["training"]["vf_coef"],
        max_grad_norm=config["training"]["max_grad_norm"],
        normalize_advantage=config["training"]["normalize_advantage"],
        target_kl=config["training"].get("target_kl"),
        device=device
    )
    
    # Create curriculum if enabled
    curriculum = None
    curriculum_config = config.get("curriculum", {})
    if curriculum_config.get("enabled", False):
        print("\nInitializing curriculum learning...")
        curriculum = JokerCurriculum(total_timesteps=config["training"]["total_timesteps"])
        curriculum.print_curriculum_summary()
        # Store curriculum config for trainer
        curriculum.adapt_learning_rate = curriculum_config.get("adapt_learning_rate", False)
        curriculum.lr_decay_factor = curriculum_config.get("lr_decay_factor", 0.95)
    
    # Create trainer
    print("\nInitializing PPO trainer...")
    trainer = PPOTrainer(model, env, ppo_config, curriculum=curriculum)
    
    # Resume from checkpoint if specified
    if args.resume:
        print(f"\nResuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    # Setup logging
    log_dir = config["logging"]["log_dir"]
    os.makedirs(log_dir, exist_ok=True)
    
    logger = Logger(
        log_dir=log_dir,
        use_tensorboard=config["logging"]["use_tensorboard"]
    )
    
    # Save configuration
    config_save_path = os.path.join(log_dir, "config.yaml")
    save_config(config, config_save_path)
    print(f"\nSaved configuration to {config_save_path}")
    
    # Evaluation only mode
    if args.eval_only:
        print("\n=== Running Evaluation Only ===")
        eval_episodes = config["evaluation"]["n_eval_episodes"]
        deterministic = config["evaluation"]["deterministic"]
        
        episode_rewards = []
        episode_lengths = []
        
        for i in range(eval_episodes):
            obs, _ = env.reset()
            done = False
            episode_reward = 0
            episode_length = 0
            
            while not done:
                obs_tensor = {
                    key: torch.as_tensor(val, dtype=torch.float32, device=device).unsqueeze(0)
                    for key, val in obs.items()
                }
                
                with torch.no_grad():
                    action, _, _, _ = model.get_action_and_value(obs_tensor, deterministic=deterministic)
                
                action_np = {
                    "action_type": action["action_type"].cpu().numpy()[0],
                    "card_selection": action["card_selection"].cpu().numpy()[0],
                    "shop_selection": action["shop_selection"].cpu().numpy()[0]
                }
                
                obs, reward, terminated, truncated, info = env.step(action_np)
                done = terminated or truncated
                episode_reward += reward
                episode_length += 1
                
                # Optional: render environment
                if i == 0:  # Render first episode
                    env.render()
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            print(f"Episode {i+1}/{eval_episodes}: Reward = {episode_reward:.2f}, Length = {episode_length}")
        
        print(f"\n=== Evaluation Results ===")
        print(f"Mean Reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
        print(f"Mean Length: {np.mean(episode_lengths):.2f} ± {np.std(episode_lengths):.2f}")
        
        return
    
    # Training mode
    print("\n=== Starting Training ===")
    print(f"Total timesteps: {config['training']['total_timesteps']:,}")
    print(f"Rollout steps: {config['training']['n_steps']}")
    print(f"Batch size: {config['training']['batch_size']}")
    print(f"Learning rate: {config['training']['learning_rate']}")
    print(f"Device: {device}")
    print("\n")
    
    # Train
    checkpoint_dir = config["checkpoints"]["dir"]
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    try:
        history = trainer.train(
            total_timesteps=config["training"]["total_timesteps"],
            log_interval=config["logging"]["log_interval"],
            save_interval=config["logging"]["save_interval"],
            checkpoint_dir=checkpoint_dir
        )
        
        print("\n=== Training Complete ===")
        print(f"Final mean reward: {history['mean_reward'][-1]:.2f}")
        print(f"Total updates: {trainer.num_updates}")
        print(f"Total timesteps: {trainer.num_timesteps:,}")
        
        # Save final model
        final_checkpoint_path = os.path.join(checkpoint_dir, "final_model.pt")
        trainer.save_checkpoint(final_checkpoint_path)
        print(f"Saved final model to {final_checkpoint_path}")
        
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        # Save checkpoint on interrupt
        interrupt_checkpoint_path = os.path.join(checkpoint_dir, "interrupted.pt")
        trainer.save_checkpoint(interrupt_checkpoint_path)
        print(f"Saved checkpoint to {interrupt_checkpoint_path}")
    
    finally:
        logger.close()
        env.close()
    
    print("\nDone!")


if __name__ == "__main__":
    main()


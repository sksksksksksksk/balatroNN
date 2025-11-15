"""
Proximal Policy Optimization (PPO) Training Algorithm

Implements PPO with:
- Clipped surrogate objective
- Generalized Advantage Estimation (GAE)
- Value function clipping
- Entropy bonus for exploration
- Gradient clipping
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import time


@dataclass
class PPOConfig:
    """Configuration for PPO training"""
    
    # Training hyperparameters
    learning_rate: float = 3e-4
    batch_size: int = 256
    n_epochs: int = 10
    n_steps: int = 2048  # Steps per rollout
    
    # PPO hyperparameters
    gamma: float = 0.99  # Discount factor
    gae_lambda: float = 0.95  # GAE parameter
    clip_range: float = 0.2  # PPO clipping parameter
    clip_range_vf: Optional[float] = None  # Value function clipping
    ent_coef: float = 0.01  # Entropy coefficient
    vf_coef: float = 0.5  # Value function coefficient
    max_grad_norm: float = 0.5  # Gradient clipping
    
    # Optimization
    normalize_advantage: bool = True
    target_kl: Optional[float] = 0.01  # Early stopping KL divergence
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class PPOBuffer:
    """
    Rollout buffer for PPO
    
    Stores trajectories and computes advantages using GAE
    """
    
    def __init__(self, buffer_size: int, obs_space: Dict, config: PPOConfig):
        self.buffer_size = buffer_size
        self.config = config
        self.reset()
        
    def reset(self):
        """Clear the buffer"""
        self.observations = {
            "hand": [],
            "jokers": [],
            "blind": [],
            "scalar": []
        }
        self.actions = {
            "action_type": [],
            "card_selection": [],
            "shop_selection": []
        }
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        self.advantages = []
        self.returns = []
        self.pos = 0
        self.full = False
        
    def add(self, obs: Dict, action: Dict, reward: float, value: float, 
            log_prob: float, done: bool):
        """Add a transition to the buffer"""
        for key in self.observations:
            self.observations[key].append(obs[key])
        
        for key in self.actions:
            self.actions[key].append(action[key])
        
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
        
        self.pos += 1
        if self.pos >= self.buffer_size:
            self.full = True
    
    def compute_returns_and_advantages(self, last_value: float):
        """
        Compute returns and advantages using GAE
        
        Args:
            last_value: Value estimate for the last state (bootstrap value)
        """
        advantages = np.zeros(len(self.rewards))
        returns = np.zeros(len(self.rewards))
        
        last_gae_lam = 0
        
        # Compute advantages using GAE
        for t in reversed(range(len(self.rewards))):
            if t == len(self.rewards) - 1:
                next_non_terminal = 1.0 - float(self.dones[t])
                next_value = last_value
            else:
                next_non_terminal = 1.0 - float(self.dones[t])
                next_value = self.values[t + 1]
            
            delta = self.rewards[t] + self.config.gamma * next_value * next_non_terminal - self.values[t]
            last_gae_lam = delta + self.config.gamma * self.config.gae_lambda * next_non_terminal * last_gae_lam
            advantages[t] = last_gae_lam
        
        # Returns are advantages + values
        returns = advantages + np.array(self.values)
        
        self.advantages = advantages
        self.returns = returns
    
    def get(self, batch_size: Optional[int] = None) -> Dict:
        """
        Get batches of data from the buffer
        
        Args:
            batch_size: Size of mini-batches. If None, return all data.
        
        Yields:
            Dictionary containing batched data
        """
        indices = np.arange(len(self.rewards))
        
        if batch_size is None:
            batch_size = len(self.rewards)
        
        # Normalize advantages
        advantages = self.advantages
        if self.config.normalize_advantage:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Convert lists to tensors
        obs_batch = {
            key: torch.as_tensor(np.array(val), dtype=torch.float32, device=self.config.device)
            for key, val in self.observations.items()
        }
        
        actions_batch = {
            "action_type": torch.as_tensor(np.array(self.actions["action_type"]), dtype=torch.long, device=self.config.device),
            "card_selection": torch.as_tensor(np.array(self.actions["card_selection"]), dtype=torch.float32, device=self.config.device),
            "shop_selection": torch.as_tensor(np.array(self.actions["shop_selection"]), dtype=torch.long, device=self.config.device)
        }
        
        log_probs_batch = torch.as_tensor(self.log_probs, dtype=torch.float32, device=self.config.device)
        advantages_batch = torch.as_tensor(advantages, dtype=torch.float32, device=self.config.device)
        returns_batch = torch.as_tensor(self.returns, dtype=torch.float32, device=self.config.device)
        values_batch = torch.as_tensor(self.values, dtype=torch.float32, device=self.config.device)
        
        # Yield mini-batches
        np.random.shuffle(indices)
        
        for start_idx in range(0, len(indices), batch_size):
            end_idx = start_idx + batch_size
            batch_indices = indices[start_idx:end_idx]
            
            yield {
                "observations": {key: val[batch_indices] for key, val in obs_batch.items()},
                "actions": {key: val[batch_indices] for key, val in actions_batch.items()},
                "old_log_probs": log_probs_batch[batch_indices],
                "advantages": advantages_batch[batch_indices],
                "returns": returns_batch[batch_indices],
                "old_values": values_batch[batch_indices]
            }


class PPOTrainer:
    """
    PPO Trainer
    
    Handles the training loop, optimization, and logging
    """
    
    def __init__(self, model: nn.Module, env: Any, config: PPOConfig):
        self.model = model.to(config.device)
        self.env = env
        self.config = config
        
        # Optimizer
        self.optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
        
        # Rollout buffer
        self.buffer = PPOBuffer(config.n_steps, env.observation_space, config)
        
        # Statistics
        self.num_timesteps = 0
        self.num_updates = 0
        
    def collect_rollouts(self) -> Dict[str, float]:
        """
        Collect rollouts from the environment
        
        Returns:
            Dictionary of statistics
        """
        self.model.eval()
        self.buffer.reset()
        
        episode_rewards = []
        episode_lengths = []
        current_episode_reward = 0
        current_episode_length = 0
        
        obs, _ = self.env.reset()
        
        for step in range(self.config.n_steps):
            # Convert observation to tensor
            obs_tensor = {
                key: torch.as_tensor(val, dtype=torch.float32, device=self.config.device).unsqueeze(0)
                for key, val in obs.items()
            }
            
            # Get action from policy
            with torch.no_grad():
                action, log_prob, entropy, value = self.model.get_action_and_value(obs_tensor)
            
            # Convert action to numpy for environment
            action_np = {
                "action_type": action["action_type"].cpu().numpy()[0],
                "card_selection": action["card_selection"].cpu().numpy()[0],
                "shop_selection": action["shop_selection"].cpu().numpy()[0]
            }
            
            # Step environment
            next_obs, reward, terminated, truncated, info = self.env.step(action_np)
            done = terminated or truncated
            
            # Store transition
            self.buffer.add(
                obs=obs,
                action={key: val.cpu().numpy()[0] for key, val in action.items()},
                reward=reward,
                value=value.cpu().numpy()[0],
                log_prob=log_prob.cpu().numpy()[0],
                done=done
            )
            
            obs = next_obs
            current_episode_reward += reward
            current_episode_length += 1
            self.num_timesteps += 1
            
            if done:
                episode_rewards.append(current_episode_reward)
                episode_lengths.append(current_episode_length)
                current_episode_reward = 0
                current_episode_length = 0
                obs, _ = self.env.reset()
        
        # Compute value for last state (bootstrap value)
        with torch.no_grad():
            obs_tensor = {
                key: torch.as_tensor(val, dtype=torch.float32, device=self.config.device).unsqueeze(0)
                for key, val in obs.items()
            }
            _, _, _, last_value = self.model.get_action_and_value(obs_tensor)
            last_value = last_value.cpu().numpy()[0]
        
        # Compute returns and advantages
        self.buffer.compute_returns_and_advantages(last_value)
        
        # Statistics
        stats = {
            "rollout/mean_reward": np.mean(episode_rewards) if episode_rewards else 0.0,
            "rollout/mean_length": np.mean(episode_lengths) if episode_lengths else 0.0,
            "rollout/num_episodes": len(episode_rewards)
        }
        
        return stats
    
    def train_step(self) -> Dict[str, float]:
        """
        Perform one PPO training step
        
        Returns:
            Dictionary of training statistics
        """
        self.model.train()
        
        policy_losses = []
        value_losses = []
        entropy_losses = []
        kl_divergences = []
        clip_fractions = []
        
        # Multiple epochs of optimization
        for epoch in range(self.config.n_epochs):
            # Mini-batch training
            for batch in self.buffer.get(self.config.batch_size):
                # Get current policy predictions
                action, log_prob, entropy, value = self.model.get_action_and_value(
                    batch["observations"],
                    batch["actions"]
                )
                
                # Policy loss (clipped surrogate objective)
                ratio = torch.exp(log_prob - batch["old_log_probs"])
                
                policy_loss_1 = batch["advantages"] * ratio
                policy_loss_2 = batch["advantages"] * torch.clamp(
                    ratio, 1 - self.config.clip_range, 1 + self.config.clip_range
                )
                policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()
                
                # Value loss
                if self.config.clip_range_vf is not None:
                    # Clipped value loss
                    value_pred_clipped = batch["old_values"] + torch.clamp(
                        value - batch["old_values"],
                        -self.config.clip_range_vf,
                        self.config.clip_range_vf
                    )
                    value_loss_1 = (value - batch["returns"]).pow(2)
                    value_loss_2 = (value_pred_clipped - batch["returns"]).pow(2)
                    value_loss = torch.max(value_loss_1, value_loss_2).mean()
                else:
                    value_loss = (value - batch["returns"]).pow(2).mean()
                
                # Entropy loss
                entropy_loss = -entropy.mean()
                
                # Total loss
                loss = (policy_loss + 
                       self.config.vf_coef * value_loss + 
                       self.config.ent_coef * entropy_loss)
                
                # Optimization step
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.optimizer.step()
                
                # Logging
                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropy_losses.append(entropy_loss.item())
                
                # KL divergence (approximate)
                with torch.no_grad():
                    kl_div = (batch["old_log_probs"] - log_prob).mean()
                    kl_divergences.append(kl_div.item())
                    
                    # Clip fraction
                    clip_fraction = ((ratio - 1.0).abs() > self.config.clip_range).float().mean()
                    clip_fractions.append(clip_fraction.item())
            
            # Early stopping based on KL divergence
            mean_kl = np.mean(kl_divergences)
            if self.config.target_kl is not None and mean_kl > 1.5 * self.config.target_kl:
                print(f"Early stopping at epoch {epoch} due to reaching max KL divergence")
                break
        
        self.num_updates += 1
        
        # Training statistics
        stats = {
            "train/policy_loss": np.mean(policy_losses),
            "train/value_loss": np.mean(value_losses),
            "train/entropy_loss": np.mean(entropy_losses),
            "train/kl_divergence": np.mean(kl_divergences),
            "train/clip_fraction": np.mean(clip_fractions),
            "train/num_updates": self.num_updates
        }
        
        return stats
    
    def train(self, total_timesteps: int, log_interval: int = 10, 
              save_interval: int = 100, checkpoint_dir: str = "checkpoints") -> Dict[str, List]:
        """
        Main training loop
        
        Args:
            total_timesteps: Total number of environment steps
            log_interval: Log every N updates
            save_interval: Save checkpoint every N updates
            checkpoint_dir: Directory to save checkpoints
        
        Returns:
            Dictionary of training history
        """
        import os
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        history = {
            "timesteps": [],
            "mean_reward": [],
            "mean_length": [],
            "policy_loss": [],
            "value_loss": []
        }
        
        start_time = time.time()
        
        while self.num_timesteps < total_timesteps:
            # Collect rollouts
            rollout_stats = self.collect_rollouts()
            
            # Train on rollouts
            train_stats = self.train_step()
            
            # Combine statistics
            stats = {**rollout_stats, **train_stats}
            stats["time/fps"] = self.num_timesteps / (time.time() - start_time)
            stats["time/timesteps"] = self.num_timesteps
            
            # Logging
            if self.num_updates % log_interval == 0:
                print(f"\n===== Update {self.num_updates} | Timesteps {self.num_timesteps} =====")
                for key, value in stats.items():
                    print(f"{key}: {value:.4f}")
            
            # Save checkpoint
            if self.num_updates % save_interval == 0:
                checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{self.num_timesteps}.pt")
                self.save_checkpoint(checkpoint_path)
                print(f"Saved checkpoint to {checkpoint_path}")
            
            # Update history
            history["timesteps"].append(self.num_timesteps)
            history["mean_reward"].append(rollout_stats.get("rollout/mean_reward", 0))
            history["mean_length"].append(rollout_stats.get("rollout/mean_length", 0))
            history["policy_loss"].append(train_stats.get("train/policy_loss", 0))
            history["value_loss"].append(train_stats.get("train/value_loss", 0))
        
        return history
    
    def save_checkpoint(self, path: str):
        """Save training checkpoint"""
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "num_timesteps": self.num_timesteps,
            "num_updates": self.num_updates,
            "config": self.config
        }
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: str):
        """Load training checkpoint"""
        checkpoint = torch.load(path, map_location=self.config.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.num_timesteps = checkpoint["num_timesteps"]
        self.num_updates = checkpoint["num_updates"]
        print(f"Loaded checkpoint from {path}")


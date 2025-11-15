"""
Alternative reward shaping strategies for Balatro RL

This module provides different reward calculation approaches
that can be experimented with to improve learning.
"""

import numpy as np


class RewardShaper:
    """Base class for reward shaping strategies"""
    
    def __init__(self, config: dict = None):
        self.config = config or {}
    
    def shape_hand_reward(self, chips_scored: int, chips_required: int, 
                         hands_remaining: int) -> float:
        """Calculate reward for playing a hand"""
        raise NotImplementedError
    
    def shape_blind_reward(self, ante: int, blind_type: str) -> float:
        """Calculate reward for beating a blind"""
        raise NotImplementedError
    
    def shape_failure_penalty(self, ante: int, chips_scored: int, 
                             chips_required: int) -> float:
        """Calculate penalty for failing a blind"""
        raise NotImplementedError


class DefaultRewardShaper(RewardShaper):
    """Current default reward shaping (baseline)"""
    
    def shape_hand_reward(self, chips_scored: int, chips_required: int, 
                         hands_remaining: int) -> float:
        return np.log1p(chips_scored) / 10.0
    
    def shape_blind_reward(self, ante: int, blind_type: str) -> float:
        return 100.0 * ante
    
    def shape_failure_penalty(self, ante: int, chips_scored: int, 
                             chips_required: int) -> float:
        return -50.0


class ProgressiveRewardShaper(RewardShaper):
    """
    Reward based on progress toward goal
    
    Encourages efficient chip accumulation and penalizes
    wasting hands.
    """
    
    def shape_hand_reward(self, chips_scored: int, chips_required: int, 
                         hands_remaining: int) -> float:
        # Base reward for chips
        base_reward = np.log1p(chips_scored) / 10.0
        
        # Bonus for reaching milestones
        progress = chips_scored / max(chips_required, 1)
        milestone_bonus = 0.0
        
        if progress >= 0.25:
            milestone_bonus += 10.0
        if progress >= 0.5:
            milestone_bonus += 20.0
        if progress >= 0.75:
            milestone_bonus += 30.0
        if progress >= 1.0:
            milestone_bonus += 50.0
        
        # Efficiency bonus for not wasting hands
        efficiency_bonus = hands_remaining * 5.0 if progress >= 1.0 else 0.0
        
        return base_reward + milestone_bonus + efficiency_bonus
    
    def shape_blind_reward(self, ante: int, blind_type: str) -> float:
        # Scale more aggressively with ante
        base = 100.0 * ante
        
        # Boss blinds worth more
        if blind_type == "boss":
            base *= 1.5
        
        return base
    
    def shape_failure_penalty(self, ante: int, chips_scored: int, 
                             chips_required: int) -> float:
        # Harsher penalty for early failure
        base_penalty = -50.0 * ante
        
        # Partial credit for getting close
        progress = chips_scored / max(chips_required, 1)
        partial_credit = 20.0 * progress
        
        return base_penalty + partial_credit


class DenseRewardShaper(RewardShaper):
    """
    Dense rewards for every action
    
    Provides more frequent feedback to help learning,
    especially in early training.
    """
    
    def shape_hand_reward(self, chips_scored: int, chips_required: int, 
                         hands_remaining: int) -> float:
        # More granular reward
        base_reward = chips_scored / 100.0  # Direct proportion
        
        # Small constant reward for making progress
        progress_reward = 1.0
        
        # Penalty for inefficiency (scoring very little)
        if chips_scored < 50:
            progress_reward = -0.5
        
        return base_reward + progress_reward
    
    def shape_blind_reward(self, ante: int, blind_type: str) -> float:
        # Same as default but with completion bonus
        base = 100.0 * ante
        completion_bonus = 50.0 * ante  # Extra for actually winning
        return base + completion_bonus
    
    def shape_failure_penalty(self, ante: int, chips_scored: int, 
                             chips_required: int) -> float:
        # Less harsh to encourage exploration
        return -25.0


class SparseRewardShaper(RewardShaper):
    """
    Sparse rewards - only reward blind completion
    
    More challenging but potentially more robust to
    reward hacking once learning starts working.
    """
    
    def shape_hand_reward(self, chips_scored: int, chips_required: int, 
                         hands_remaining: int) -> float:
        # No reward for individual hands
        return 0.0
    
    def shape_blind_reward(self, ante: int, blind_type: str) -> float:
        # All reward comes from beating blinds
        base = 200.0 * ante
        
        if blind_type == "boss":
            base *= 2.0
        
        return base
    
    def shape_failure_penalty(self, ante: int, chips_scored: int, 
                             chips_required: int) -> float:
        # Harsh penalty to make success more valuable
        return -100.0 * ante


class CurriculumRewardShaper(RewardShaper):
    """
    Adaptive reward based on training progress
    
    Start with dense rewards, gradually move to sparse
    as the agent improves.
    """
    
    def __init__(self, config: dict = None):
        super().__init__(config)
        self.training_step = 0
        self.max_steps = config.get("total_timesteps", 10_000_000) if config else 10_000_000
    
    def update_progress(self, step: int):
        """Update training progress"""
        self.training_step = step
    
    def _get_sparsity(self) -> float:
        """Get current sparsity level (0=dense, 1=sparse)"""
        return min(1.0, self.training_step / (self.max_steps * 0.5))
    
    def shape_hand_reward(self, chips_scored: int, chips_required: int, 
                         hands_remaining: int) -> float:
        sparsity = self._get_sparsity()
        
        # Dense component
        dense_reward = np.log1p(chips_scored) / 10.0
        
        # Sparse component (only progress toward goal)
        progress = chips_scored / max(chips_required, 1)
        sparse_reward = 50.0 * progress if progress >= 1.0 else 0.0
        
        # Interpolate
        return (1 - sparsity) * dense_reward + sparsity * sparse_reward
    
    def shape_blind_reward(self, ante: int, blind_type: str) -> float:
        # Always reward blind completion
        return 100.0 * ante * (1 + self._get_sparsity())
    
    def shape_failure_penalty(self, ante: int, chips_scored: int, 
                             chips_required: int) -> float:
        # Gradually increase penalty
        sparsity = self._get_sparsity()
        return -50.0 * (1 + sparsity * ante)


# Factory function
def create_reward_shaper(strategy: str = "default", config: dict = None) -> RewardShaper:
    """
    Create a reward shaper
    
    Args:
        strategy: One of "default", "progressive", "dense", "sparse", "curriculum"
        config: Optional configuration dict
    
    Returns:
        RewardShaper instance
    """
    shapers = {
        "default": DefaultRewardShaper,
        "progressive": ProgressiveRewardShaper,
        "dense": DenseRewardShaper,
        "sparse": SparseRewardShaper,
        "curriculum": CurriculumRewardShaper
    }
    
    shaper_class = shapers.get(strategy, DefaultRewardShaper)
    return shaper_class(config)


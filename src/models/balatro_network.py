"""
Neural Network Architecture for Balatro Agent

Uses transformer-based architecture to handle card sequences and game state,
with separate policy and value heads for PPO training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple
import math


class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism"""
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, 
                mask: torch.Tensor = None) -> torch.Tensor:
        batch_size = query.size(0)
        
        # Linear projections and reshape for multi-head
        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            # Reshape mask for multi-head attention: (batch, 1, seq_len) -> (batch, 1, 1, seq_len)
            mask = mask.unsqueeze(1) if mask.dim() == 3 else mask
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)
        
        # Apply attention to values
        context = torch.matmul(attention, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        return self.W_o(context)


class TransformerBlock(nn.Module):
    """Transformer encoder block"""
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # Self-attention with residual connection
        attended = self.attention(x, x, x, mask)
        x = self.norm1(x + attended)
        
        # Feed-forward with residual connection
        fed_forward = self.ff(x)
        x = self.norm2(x + fed_forward)
        
        return x


class CardEncoder(nn.Module):
    """Encodes card sequences using transformers"""
    
    def __init__(self, card_dim: int = 32, d_model: int = 256, num_heads: int = 8, 
                 num_layers: int = 4, dropout: float = 0.1):
        super().__init__()
        
        self.d_model = d_model
        
        # Project card features to model dimension
        self.card_embedding = nn.Sequential(
            nn.Linear(card_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Positional encoding for card positions in hand
        self.pos_encoding = nn.Parameter(torch.randn(1, 8, d_model))
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_model * 4, dropout)
            for _ in range(num_layers)
        ])
        
    def forward(self, cards: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            cards: (batch_size, num_cards, card_dim)
            mask: (batch_size, num_cards) - 1 for valid cards, 0 for padding
        
        Returns:
            Encoded cards: (batch_size, num_cards, d_model)
        """
        batch_size, num_cards, _ = cards.shape
        
        # Embed cards
        x = self.card_embedding(cards)
        
        # Add positional encoding
        x = x + self.pos_encoding[:, :num_cards, :]
        
        # Apply transformer layers
        for layer in self.layers:
            x = layer(x, mask)
        
        return x


class JokerEncoder(nn.Module):
    """Encodes joker cards"""
    
    def __init__(self, joker_dim: int = 64, d_model: int = 256, dropout: float = 0.1):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(joker_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU()
        )
        
    def forward(self, jokers: torch.Tensor) -> torch.Tensor:
        """
        Args:
            jokers: (batch_size, num_jokers, joker_dim)
        
        Returns:
            Encoded jokers: (batch_size, num_jokers, d_model)
        """
        return self.encoder(jokers)


class StateEncoder(nn.Module):
    """Encodes scalar game state and blind information"""
    
    def __init__(self, scalar_dim: int = 32, blind_dim: int = 16, d_model: int = 256, 
                 dropout: float = 0.1):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(scalar_dim + blind_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU()
        )
        
    def forward(self, scalar: torch.Tensor, blind: torch.Tensor) -> torch.Tensor:
        """
        Args:
            scalar: (batch_size, scalar_dim)
            blind: (batch_size, blind_dim)
        
        Returns:
            Encoded state: (batch_size, d_model)
        """
        combined = torch.cat([scalar, blind], dim=-1)
        return self.encoder(combined)


class BalatroNetwork(nn.Module):
    """
    Main network architecture for Balatro agent
    
    Combines card, joker, and state encoders with cross-attention
    to produce action distributions and value estimates.
    """
    
    def __init__(self, d_model: int = 256, num_heads: int = 8, num_layers: int = 6, 
                 dropout: float = 0.1):
        super().__init__()
        
        self.d_model = d_model
        
        # Encoders (EXPANDED for full game)
        self.card_encoder = CardEncoder(card_dim=32, d_model=d_model, num_heads=num_heads, 
                                       num_layers=num_layers, dropout=dropout)
        self.joker_encoder = JokerEncoder(joker_dim=128, d_model=d_model, dropout=dropout)  # Expanded to 128
        self.consumable_encoder = JokerEncoder(joker_dim=64, d_model=d_model, dropout=dropout)  # Reuse joker encoder
        self.shop_encoder = JokerEncoder(joker_dim=128, d_model=d_model, dropout=dropout)  # For shop items
        self.voucher_encoder = JokerEncoder(joker_dim=32, d_model=d_model, dropout=dropout)  # For vouchers
        self.state_encoder = StateEncoder(scalar_dim=64, blind_dim=16, d_model=d_model,  # Expanded to 64
                                         dropout=dropout)
        
        # Synergy encoder (NEW)
        self.synergy_encoder = nn.Sequential(
            nn.Linear(32, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model)
        )
        
        # Cross-attention to combine all information
        self.cross_attention = MultiHeadAttention(d_model, num_heads, dropout)
        
        # Additional processing layers
        self.fusion_layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_model * 4, dropout)
            for _ in range(2)
        ])
        
        # Global pooling
        self.pool = nn.AdaptiveAvgPool1d(1)
        
    def forward(self, obs: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network (EXPANDED for full game)
        
        Args:
            obs: Dictionary containing:
                - hand: (batch_size, 8, 32)
                - jokers: (batch_size, 5, 128)
                - consumables: (batch_size, 6, 64)
                - shop_items: (batch_size, 10, 128)
                - vouchers: (batch_size, 5, 32)
                - blind: (batch_size, 16)
                - scalar: (batch_size, 64)
                - synergies: (batch_size, 32)
        
        Returns:
            card_features: (batch_size, 8, d_model) - features for each card
            global_features: (batch_size, d_model) - global state features
        """
        batch_size = obs["hand"].size(0)
        
        # Encode cards
        card_mask = (obs["hand"].sum(dim=-1) != 0).float()
        card_mask = card_mask.unsqueeze(1)
        card_features = self.card_encoder(obs["hand"], card_mask)
        
        # Encode jokers (expanded to 128 dims)
        joker_features = self.joker_encoder(obs["jokers"])
        
        # Encode consumables (new)
        consumable_features = self.consumable_encoder(obs["consumables"])
        
        # Encode shop items (new)
        shop_features = self.shop_encoder(obs["shop_items"])
        
        # Encode vouchers (new)
        voucher_features = self.voucher_encoder(obs["vouchers"])
        
        # Encode game state (expanded to 64 scalar dims)
        state_features = self.state_encoder(obs["scalar"], obs["blind"])
        state_features = state_features.unsqueeze(1)  # (batch_size, 1, d_model)
        
        # Encode synergy features (NEW)
        synergy_features = self.synergy_encoder(obs["synergies"])
        synergy_features = synergy_features.unsqueeze(1)  # (batch_size, 1, d_model)
        
        # Combine all features (cards + jokers + consumables + shop + vouchers + state + synergies)
        combined_features = torch.cat([
            card_features,      # 8 cards
            joker_features,     # 5 jokers
            consumable_features,  # 6 consumables
            shop_features,      # 10 shop items
            voucher_features,   # 5 vouchers
            state_features,     # 1 state
            synergy_features    # 1 synergy
        ], dim=1)  # Total: 36 tokens
        
        # Apply fusion layers
        for layer in self.fusion_layers:
            combined_features = layer(combined_features)
        
        # Split back into card features and global features
        card_features = combined_features[:, :8, :]
        
        # Global pooling for value estimation (average over all tokens)
        global_features = combined_features.mean(dim=1)
        
        return card_features, global_features


class PolicyValueNetwork(nn.Module):
    """
    Policy and Value network for PPO
    
    Takes encoded features from BalatroNetwork and outputs:
    - Action distributions (policy)
    - Value estimate (critic)
    """
    
    def __init__(self, d_model: int = 256, dropout: float = 0.1):
        super().__init__()
        
        self.backbone = BalatroNetwork(d_model=d_model, dropout=dropout)
        
        # Policy heads (EXPANDED for full game)
        self.action_type_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 12)  # 12 action types (expanded)
        )
        
        self.card_selection_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)  # Per-card selection probability
        )
        
        self.shop_item_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 7)  # Shop item index (up to 7 items total)
        )
        
        self.joker_slot_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 5)  # 5 joker slots
        )
        
        self.consumable_slot_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 2)  # 2 consumable slots per type
        )
        
        self.target_card_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 8)  # Target card index (for tarot/spectral)
        )
        
        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )
        
    def forward(self, obs: Dict[str, torch.Tensor]) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Forward pass
        
        Args:
            obs: Observation dictionary
        
        Returns:
            action_logits: Dictionary of action logits
            value: State value estimate
        """
        card_features, global_features = self.backbone(obs)
        
        # Policy outputs (EXPANDED)
        action_type_logits = self.action_type_head(global_features)
        
        # Per-card selection logits
        card_selection_logits = self.card_selection_head(card_features).squeeze(-1)
        
        # Shop item selection logits
        shop_item_logits = self.shop_item_head(global_features)
        
        # Joker slot selection logits
        joker_slot_logits = self.joker_slot_head(global_features)
        
        # Consumable slot selection logits
        consumable_slot_logits = self.consumable_slot_head(global_features)
        
        # Target card selection logits (for tarot/spectral effects)
        target_card_logits = self.target_card_head(global_features)
        
        # Value estimate
        value = self.value_head(global_features)
        
        action_logits = {
            "action_type": action_type_logits,
            "card_selection": card_selection_logits,
            "shop_item_index": shop_item_logits,
            "joker_slot": joker_slot_logits,
            "consumable_slot": consumable_slot_logits,
            "target_card_index": target_card_logits,
        }
        
        return action_logits, value
    
    def get_action_and_value(self, obs: Dict[str, torch.Tensor], 
                            action: Dict[str, torch.Tensor] = None,
                            deterministic: bool = False) -> Tuple[Dict[str, torch.Tensor], 
                                                                  torch.Tensor, 
                                                                  torch.Tensor, 
                                                                  torch.Tensor]:
        """
        Get action, log probability, entropy, and value
        
        Args:
            obs: Observation dictionary
            action: If provided, compute log prob for this action
            deterministic: If True, select most likely action
        
        Returns:
            action: Sampled or provided action
            log_prob: Log probability of action
            entropy: Entropy of action distribution
            value: State value estimate
        """
        action_logits, value = self.forward(obs)
        
        # Action type distribution
        action_type_dist = torch.distributions.Categorical(logits=action_logits["action_type"])
        
        # Card selection distribution (independent Bernoulli for each card)
        card_selection_dist = torch.distributions.Bernoulli(logits=action_logits["card_selection"])
        
        # Shop item selection distribution
        shop_item_dist = torch.distributions.Categorical(logits=action_logits["shop_item_index"])
        
        # Joker slot selection distribution
        joker_slot_dist = torch.distributions.Categorical(logits=action_logits["joker_slot"])
        
        # Consumable slot selection distribution
        consumable_slot_dist = torch.distributions.Categorical(logits=action_logits["consumable_slot"])
        
        # Target card selection distribution
        target_card_dist = torch.distributions.Categorical(logits=action_logits["target_card_index"])
        
        if action is None:
            # Sample new action
            if deterministic:
                action_type = action_logits["action_type"].argmax(dim=-1)
                card_selection = (action_logits["card_selection"] > 0).float()
                shop_item_index = action_logits["shop_item_index"].argmax(dim=-1)
                joker_slot = action_logits["joker_slot"].argmax(dim=-1)
                consumable_slot = action_logits["consumable_slot"].argmax(dim=-1)
                target_card_index = action_logits["target_card_index"].argmax(dim=-1)
            else:
                action_type = action_type_dist.sample()
                card_selection = card_selection_dist.sample()
                shop_item_index = shop_item_dist.sample()
                joker_slot = joker_slot_dist.sample()
                consumable_slot = consumable_slot_dist.sample()
                target_card_index = target_card_dist.sample()
            
            action = {
                "action_type": action_type,
                "card_selection": card_selection,
                "shop_item_index": shop_item_index,
                "joker_slot": joker_slot,
                "consumable_slot": consumable_slot,
                "target_card_index": target_card_index
            }
        
        # Compute log probabilities
        action_type_log_prob = action_type_dist.log_prob(action["action_type"])
        card_selection_log_prob = card_selection_dist.log_prob(action["card_selection"]).sum(dim=-1)
        shop_item_index_log_prob = shop_item_dist.log_prob(action["shop_item_index"])
        joker_slot_log_prob = joker_slot_dist.log_prob(action["joker_slot"])
        consumable_slot_log_prob = consumable_slot_dist.log_prob(action["consumable_slot"])
        target_card_index_log_prob = target_card_dist.log_prob(action["target_card_index"])
        
        # Total log probability
        log_prob = (action_type_log_prob + card_selection_log_prob + shop_item_index_log_prob + 
                   joker_slot_log_prob + consumable_slot_log_prob + target_card_index_log_prob)
        
        # Compute entropy
        entropy = (action_type_dist.entropy() + card_selection_dist.entropy().sum(dim=-1) + 
                  shop_item_dist.entropy() + joker_slot_dist.entropy() + 
                  consumable_slot_dist.entropy() + target_card_dist.entropy())
        
        return action, log_prob, entropy, value.squeeze(-1)


def create_model(config: Dict = None) -> PolicyValueNetwork:
    """
    Factory function to create model with configuration
    
    Args:
        config: Model configuration dictionary
    
    Returns:
        Initialized model
    """
    config = config or {}
    
    d_model = config.get("d_model", 256)
    dropout = config.get("dropout", 0.1)
    
    model = PolicyValueNetwork(d_model=d_model, dropout=dropout)
    
    # Initialize weights
    def init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.orthogonal_(m.weight, gain=0.01)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
    
    model.apply(init_weights)
    
    return model


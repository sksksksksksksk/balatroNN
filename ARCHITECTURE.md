# BalatroNN Architecture

Technical documentation of the neural network architecture and training system.

## System Overview

```
┌─────────────────┐
│  Balatro Game   │
│   Environment   │
└────────┬────────┘
         │ State
         ▼
┌─────────────────┐
│  Card Encoder   │ ─┐
│  (Transformer)  │  │
└─────────────────┘  │
                     │
┌─────────────────┐  │
│  Joker Encoder  │ ─┼──► Cross Attention ──► Policy Head ──► Action
│     (MLP)       │  │                     └──► Value Head ──► Value Estimate
└─────────────────┘  │
                     │
┌─────────────────┐  │
│  State Encoder  │ ─┘
│     (MLP)       │
└─────────────────┘
```

## Component Details

### 1. Environment (`src/environment/balatro_env.py`)

**Purpose**: Simulates Balatro game mechanics and provides Gym-compatible interface.

**State Space**:
- **Hand**: 8 cards × 32 features each
  - Rank (normalized)
  - Suit (one-hot, 4 dims)
  - Enhancement (one-hot, 9 dims)
  - Edition (one-hot, 4 dims)
  - Seal (one-hot, 5 dims)
  - Debuff flag
  - Base chips

- **Jokers**: 5 jokers × 64 features each
  - Rarity encoding
  - Effect parameters (chips, mult, xmult)
  - Trigger conditions

- **Blind Info**: 16 features
  - Ante level
  - Blind type (small/big/boss)
  - Chip requirement
  - Reward

- **Scalar State**: 32 features
  - Current chips scored
  - Hands remaining
  - Discards remaining
  - Money
  - Deck size
  - etc.

**Action Space**:
- **Action Type**: Discrete(5)
  - 0: Play hand
  - 1: Discard
  - 2: Shop buy
  - 3: Skip
  - 4: Reroll shop

- **Card Selection**: MultiBinary(8)
  - Binary mask for which cards to select

- **Shop Selection**: Discrete(7)
  - Which shop item to buy

### 2. Neural Network Architecture (`src/models/balatro_network.py`)

#### Card Encoder

```
Input: (batch, 8 cards, 32 features)
  ↓
Linear Projection: 32 → d_model (256)
  ↓
+ Positional Encoding (learned)
  ↓
Transformer Blocks × 4:
  - Multi-head Self-Attention (8 heads)
  - Feed-Forward Network (d_model × 4)
  - Layer Normalization
  - Residual Connections
  ↓
Output: (batch, 8, 256) card features
```

**Parameters**: ~2.5M for d_model=256

#### Joker Encoder

```
Input: (batch, 5 jokers, 64 features)
  ↓
MLP:
  Linear: 64 → 256
  LayerNorm + ReLU
  Linear: 256 → 256
  LayerNorm + ReLU
  ↓
Output: (batch, 5, 256) joker features
```

**Parameters**: ~130K

#### State Encoder

```
Input: (batch, 48) [32 scalar + 16 blind]
  ↓
MLP:
  Linear: 48 → 256
  LayerNorm + ReLU
  Linear: 256 → 256
  LayerNorm + ReLU
  ↓
Output: (batch, 256) state features
```

**Parameters**: ~150K

#### Feature Fusion

```
Card Features (8, 256)
Joker Features (5, 256)   ──► Concatenate ──► Transformer × 2 ──► Split
State Features (1, 256)
```

#### Policy Heads

**Action Type Head**:
```
Input: (batch, 256) global features
  ↓
Linear: 256 → 256 + ReLU
Linear: 256 → 5
  ↓
Output: (batch, 5) action type logits
```

**Card Selection Head**:
```
Input: (batch, 8, 256) card features
  ↓
Linear: 256 → 128 + ReLU
Linear: 128 → 1
  ↓
Output: (batch, 8) per-card selection logits
```

**Shop Selection Head**:
```
Input: (batch, 256) global features
  ↓
Linear: 256 → 256 + ReLU
Linear: 256 → 7
  ↓
Output: (batch, 7) shop selection logits
```

#### Value Head

```
Input: (batch, 256) global features
  ↓
Linear: 256 → 256 + ReLU
Linear: 256 → 128 + ReLU
Linear: 128 → 1
  ↓
Output: (batch, 1) value estimate
```

**Total Parameters** (d_model=256): ~15M

### 3. PPO Training Algorithm (`src/training/ppo.py`)

#### Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| Learning Rate | 3e-4 | Adam optimizer learning rate |
| Batch Size | 256 | Mini-batch size for updates |
| N Steps | 2048 | Steps per rollout |
| N Epochs | 10 | Optimization epochs per rollout |
| Gamma | 0.99 | Discount factor |
| GAE Lambda | 0.95 | GAE parameter |
| Clip Range | 0.2 | PPO clip parameter |
| Entropy Coef | 0.01 | Entropy bonus weight |
| Value Coef | 0.5 | Value loss weight |
| Max Grad Norm | 0.5 | Gradient clipping threshold |

#### Training Loop

```python
for iteration in range(num_iterations):
    # 1. Collect rollouts (n_steps)
    for step in range(n_steps):
        action, log_prob, value = policy(observation)
        next_obs, reward, done = env.step(action)
        buffer.add(obs, action, reward, value, log_prob, done)
    
    # 2. Compute advantages using GAE
    advantages = compute_gae(rewards, values, dones)
    returns = advantages + values
    
    # 3. Optimize policy and value function
    for epoch in range(n_epochs):
        for batch in buffer.get_batches(batch_size):
            # Forward pass
            _, new_log_prob, entropy, new_value = policy(
                batch.obs, batch.action
            )
            
            # Policy loss (clipped surrogate)
            ratio = exp(new_log_prob - batch.old_log_prob)
            policy_loss = -min(
                ratio * batch.advantages,
                clip(ratio, 1-ε, 1+ε) * batch.advantages
            )
            
            # Value loss (MSE)
            value_loss = (new_value - batch.returns)²
            
            # Entropy bonus
            entropy_loss = -entropy
            
            # Total loss
            loss = policy_loss + vf_coef * value_loss + ent_coef * entropy_loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm(model.parameters(), max_grad_norm)
            optimizer.step()
```

#### Generalized Advantage Estimation (GAE)

```
A_t = Σ_(l=0)^∞ (γλ)^l δ_(t+l)

where δ_t = r_t + γV(s_(t+1)) - V(s_t)
```

Benefits:
- Reduces variance of policy gradient
- Biases estimates toward short-term rewards (controlled by λ)
- Smooth trade-off between bias and variance

## Memory Requirements

### Model Memory (FP32)

| Configuration | Parameters | Memory | Batch Size 1 |
|---------------|-----------|---------|--------------|
| Small (128) | ~4M | ~16 MB | ~50 MB |
| Medium (256) | ~15M | ~60 MB | ~150 MB |
| Large (384) | ~35M | ~140 MB | ~300 MB |
| XL (512) | ~60M | ~240 MB | ~500 MB |

### Training Memory (per batch)

- Gradients: ~2× model size
- Optimizer states (Adam): ~2× model size
- Activations: ~batch_size × rollout_length × feature_size

**Example** (d_model=256, batch=256):
- Model: 60 MB
- Gradients: 120 MB
- Adam states: 120 MB
- Activations: ~2 GB
- **Total**: ~2.3 GB

### Rollout Buffer

- Storage: n_steps × observation_size × 4 bytes
- Example (2048 steps): ~50 MB

### H100 Memory Budget (80 GB)

With 80 GB VRAM on H100:
- Model (512): 240 MB
- Training overhead: 2 GB
- Batch size 1024: ~4 GB per forward pass
- **Can handle**: XL model with batch size ~1024

## Performance Characteristics

### Throughput

| Hardware | Model Size | FPS (Training) | FPS (Inference) |
|----------|-----------|----------------|-----------------|
| RTX 3090 | Medium | ~500 | ~2000 |
| RTX 4090 | Large | ~800 | ~3000 |
| A100 | Large | ~1200 | ~4000 |
| H100 | XL | ~2000 | ~6000 |

### Training Time Estimates

To reach 10M timesteps:

| Hardware | Time | Wall Clock |
|----------|------|------------|
| RTX 3090 | 5-6 hours | ~6 hours |
| RTX 4090 | 3-4 hours | ~4 hours |
| A100 | 2-3 hours | ~2.5 hours |
| H100 | 1-2 hours | ~1.5 hours |

## Key Design Decisions

### 1. Transformer for Card Sequences

**Why**: Cards interact with each other (e.g., pairs, flushes), so self-attention captures these relationships naturally.

**Alternative**: CNN or RNN would require more engineering to capture card interactions.

### 2. Separate Encoders for Different Modalities

**Why**: Cards, jokers, and game state have different structural properties. Specialized encoders learn better representations.

**Alternative**: Single unified encoder would be simpler but less expressive.

### 3. Multi-Discrete Action Space

**Why**: Actions have different types (what to do) and targets (which cards). Factoring the action space makes learning easier.

**Alternative**: Flat action space would be exponentially large.

### 4. PPO over Other RL Algorithms

**Why**: 
- Sample efficient
- Stable training
- Well-understood hyperparameters
- Good for complex action spaces

**Alternatives**:
- **A3C**: Harder to tune, needs multiple environments
- **SAC**: Requires continuous actions (we have discrete)
- **DQN**: Difficult with large action spaces

### 5. Curriculum Learning (Optional)

Start with easier blinds and gradually increase difficulty. Helps with:
- Faster initial learning
- More stable training
- Better final performance

## Future Improvements

1. **Hierarchical Policy**: Separate high-level (shop vs play) and low-level (which cards) policies

2. **Memory/Recurrence**: Add LSTM/GRU to remember past rounds

3. **Self-Play**: Train against past versions for robustness

4. **Model-Based RL**: Learn world model of game mechanics

5. **Multi-Task Learning**: Train on multiple stakes simultaneously

6. **Attention Visualization**: Visualize what cards the model focuses on

7. **Meta-Learning**: Quick adaptation to new joker combinations


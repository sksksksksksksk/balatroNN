# 🏗️ Model Architecture Analysis for Balatro RL

## Current Architecture (512d/10layers)

### Network Components

```python
BalatroNetwork (backbone):
├── Card Encoder: 10 transformer layers, 512d
├── Joker Encoder: 128→512d MLP
├── Consumable Encoder: 64→512d MLP  
├── Shop Encoder: 128→512d MLP
├── Voucher Encoder: 32→512d MLP
├── State Encoder: 64→512d MLP
├── Synergy Encoder: 32→512d MLP
└── Fusion: 2 transformer layers, 512d

PolicyValueNetwork (heads):
├── Action Type Head: 512→512→12
├── Card Selection Head: 512→256→1 (per card)
├── Shop Item Head: 512→512→7
├── Joker Slot Head: 512→256→5
├── Consumable Slot Head: 512→256→2
├── Target Card Head: 512→256→8
└── Value Head: 512→512→256→1
```

### Token Processing

**36 tokens total**:
- 8 cards (hand)
- 5 jokers
- 6 consumables  
- 10 shop items
- 5 vouchers
- 1 state
- 1 synergy

All processed through 10-layer transformer (card encoder) + 2 fusion layers.

## Parameter Count Estimation

### Transformer Layers (main cost):
```
Per transformer layer:
  - Attention: 4 × d_model² ≈ 4 × 512² = 1.05M params
  - FFN: 2 × d_model × (4×d_model) ≈ 2 × 512 × 2048 = 2.1M params
  - Total per layer: ~3.2M params

10 card encoder layers: 10 × 3.2M = 32M params
2 fusion layers: 2 × 3.2M = 6.4M params
```

### Encoders & Heads:
```
Encoders: ~5M params (various sizes)
Policy heads: ~3M params
Value head: ~1M params
```

### **Total: ~47M parameters**

Compare to original config:
- 768d/12layers: ~100M parameters (2x larger!)
- 512d/10layers: ~47M parameters (current)

## Task Complexity Analysis

### Balatro Game Complexity:

**State Space**:
- 52 cards × 150 jokers × combos × consumables = HUGE
- But much of it is structured/sparse

**Action Space**:
- 12 action types
- Multi-headed (cards, shop, slots)
- ~100-200 meaningful actions per state

**Planning Horizon**:
- Ante 1→8 progression
- ~50-200 steps per episode
- Medium-term planning needed

### Comparison to Other Games:

| Game | Complexity | Successful Model Size |
|------|-----------|----------------------|
| **Atari** | Low | 1-5M params |
| **Poker (Pluribus)** | Medium | 5-10M params |
| **Dota 2 (OpenAI Five)** | High | 4.5M params per agent |
| **Go (AlphaGo)** | Very High | 100M+ params |
| **StarCraft (AlphaStar)** | Extreme | 300M+ params |
| **Balatro** | Medium-High | ??? |

Balatro sits between Poker and Dota in complexity:
- More complex than poker (150 jokers, synergies, multi-step)
- Less complex than real-time strategy
- Structured state space (cards, not pixels)

## Architecture Assessment

### Current (512d/10layers - 47M params): ✅ **PROBABLY GOOD**

**Pros**:
- ✅ Large enough to capture joker interactions
- ✅ Transformer can learn synergies
- ✅ Not too large (faster training)
- ✅ Fits well in memory with batch_size=7680

**Cons**:
- ⚠️ Might still be overkill
- ⚠️ 36 tokens × 10 layers = expensive
- ⚠️ Could be slower to converge

### Alternative Architectures

#### Option 1: Smaller Model (384d/8layers - ~25M params)
```yaml
model:
  d_model: 384  # 25% smaller
  num_layers: 8  # 2 fewer layers
  num_heads: 12  # Keep attention heads
```

**Benefits**:
- ✅ Faster training (~30% speedup)
- ✅ Can fit larger batch (9000-10000)
- ✅ Simpler = easier to debug
- ✅ Less prone to overfitting

**Risks**:
- ❌ May struggle with complex joker interactions
- ❌ Might need more training steps

#### Option 2: Current (512d/10layers - ~47M params) 
```yaml
model:
  d_model: 512  # Current
  num_layers: 10
  num_heads: 16
```

**Benefits**:
- ✅ Good capacity for complexity
- ✅ Can learn intricate synergies
- ✅ Proven size for similar tasks

**Risks**:
- ⚠️ Slower training
- ⚠️ Could overfit with only 20M timesteps

#### Option 3: Larger Model (640d/12layers - ~80M params)
```yaml
model:
  d_model: 640  # 25% larger
  num_layers: 12  # Original depth
  num_heads: 16
```

**Benefits**:
- ✅ Maximum capacity
- ✅ Can learn very subtle patterns

**Risks**:
- ❌ Much slower (40% longer training)
- ❌ Requires more data (50M+ steps)
- ❌ Risk of overfitting
- ❌ Smaller batch size

## Recommendations

### For Fast Iteration (Recommended): **384d/8layers**

If you want to quickly test the new V3 rewards and exponential penalties:

```yaml
model:
  d_model: 384
  num_heads: 12
  num_layers: 8
  dropout: 0.1

training:
  batch_size: 9216  # Even larger!
  total_timesteps: 20_000_000
```

**Why**:
- Get results in ~6-8 hours instead of 12-16
- Iterate faster on reward design
- Probably still enough capacity for Balatro
- Can always scale up later if needed

### For Production Quality: **512d/10layers** (Current)

If you want maximum quality and have time:

```yaml
model:
  d_model: 512
  num_heads: 16
  num_layers: 10
  dropout: 0.1

training:
  batch_size: 7680  # Current
  total_timesteps: 50_000_000  # More data
```

**Why**:
- Good balance of capacity vs speed
- Proven architecture size
- Enough capacity for all 150 jokers
- Can learn complex synergies

### For Research/Max Performance: **640d/12layers**

Only if you need absolute best performance and have lots of compute:

```yaml
model:
  d_model: 640
  num_heads: 16
  num_layers: 12
  dropout: 0.1

training:
  batch_size: 5120  # Smaller due to memory
  total_timesteps: 100_000_000
```

## Specific Concerns

### 1. Skip-Spam Learning

**Not an architecture problem**. Even a 10M param model would learn skip-spam with bad rewards. The V3 exponential penalties will fix this regardless of model size.

### 2. Joker Interaction Learning

**May benefit from larger model**. 150 jokers with synergies = lots of combinations. 512d should be fine, but 384d might struggle with rare joker combos.

### 3. Training Stability

**Larger models = more unstable**. Your batch_size=7680 helps, but 512d/10layers might oscillate more than 384d/8layers.

### 4. Overfitting Risk

With only 20M timesteps:
- 384d/8layers: Low risk ✅
- 512d/10layers: Medium risk ⚠️
- 640d/12layers: High risk ❌

## My Recommendation

**Start with 384d/8layers for initial testing**, then scale up if needed:

1. **Phase 1 (Current)**: Test V3 rewards with 384d/8layers
   - Fast iteration (6-8 hours)
   - Verify exponential penalties work
   - See if it learns basic Balatro gameplay
   
2. **Phase 2**: If good results, train 512d/10layers
   - Full 50M timestep run
   - Should reach Ante 3-5 consistently
   
3. **Phase 3**: If needed, try 640d/12layers
   - Only if 512d hits a ceiling
   - 100M timesteps for convergence

## Quick Change

To try the smaller architecture:

```bash
# Edit configs/a100_aggressive.yaml
vim configs/a100_aggressive.yaml

# Change:
d_model: 384  # from 512
num_layers: 8  # from 10
batch_size: 9216  # from 7680 (can go bigger!)

# Train
python train.py --config configs/a100_aggressive.yaml
```

## Bottom Line

**Your current 512d/10layers is good** - not too big, not too small. But if you want faster iteration to test the V3 rewards, **384d/8layers would be better** for initial experiments. You can always scale up once you confirm the rewards work!

The architecture isn't your bottleneck - the reward structure was. With V3 exponential penalties, even a 256d/6layer model would probably learn to play! 🎮


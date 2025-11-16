# 🎯 Reward Structure V2 - Aggressive Anti-Skip-Spam

## Problem Update

After implementing the initial reward fixes, we discovered that **even with penalties**, the model still learned skip-spam behavior after 2.8M training steps. The original penalties were too weak to overcome the exploration problem.

## Root Cause Analysis

The model gets stuck in a **local optimum** because:
1. Early random exploration leads to quick losses (-50 penalty)
2. Model learns: "Don't play = avoid immediate pain"
3. Skip-spam gives consistent small negative rewards
4. Never explores enough to discover that **playing > skipping**
5. Gets permanently stuck in skip-spam mode

### The Math Problem (V1 - Too Weak):

| Strategy | Reward Calculation | Total |
|----------|-------------------|-------|
| Skip-spam | -0.1 × 1000 - 100 timeout - 10 time | **-210** |
| Play & Lose | Small positives then -50 | **~-30 to -40** |

**Issue**: Even though playing is 5x better, the model never discovers this!

## V2 Reward Structure - Much More Aggressive

### Changes from V1:

| Component | V1 | V2 | Change |
|-----------|----|----|--------|
| **Skip penalty** | -0.1 | **-1.0** | 10x harsher |
| **Play hand base** | log(score)/5.0 | **log(score)/2.0** | 2.5x more rewarding |
| **Progress bonus** | +5.0 max | **+10.0 max** | 2x more rewarding |
| **Exploration bonus** | None | **+0.1** | New! Rewards ANY non-skip action |
| **Timeout penalty** | -100.0 | -100.0 | Unchanged |
| **Time pressure** | -0.01/step | -0.01/step | Unchanged |

### New Math:

| Strategy | Reward Calculation | Total |
|----------|-------------------|-------|
| **Skip-spam** | -1.0 × 1000 - 100 timeout - 10 time | **-1,110** 😱 |
| **Play & Lose** | +0.1 exploration + small hand rewards then -50 | **~0 to -20** ✅ |
| **Play & Win** | +0.1 exploration + good hand rewards + 100 × ante | **+150 to +300** 🎉 |

**Now the gap is HUGE!** Skip-spam gives 50-100x worse reward than playing.

## Detailed Reward Breakdown

### 1. Core Actions

```python
# SKIP/CONTINUE (heavily penalized)
reward = -1.0  # Was -0.1

# PLAY HAND (heavily rewarded)
base_reward = log(chips_scored + 1) / 2.0  # Was /5.0
progress = chips / blind_requirement
progress_bonus = progress * 10.0  # Was * 5.0
exploration_bonus = 0.1  # NEW!
total = base_reward + progress_bonus + exploration_bonus

# Example: Playing a 500-chip hand when blind needs 1000 chips:
# base: log(501)/2.0 = 3.1
# progress: (500/1000) * 10.0 = 5.0
# exploration: 0.1
# TOTAL: +8.2 reward (vs -1.0 for skip!)
```

### 2. Exploration Bonus (NEW!)

```python
# Any non-skip action gets +0.1
if action_type != 11:  # Not skip
    reward += 0.1
```

This encourages the agent to TRY things during early training:
- Play hand: +0.1 ✅
- Discard: +0.1 ✅
- Buy item: +0.1 ✅
- Use card: +0.1 ✅
- Skip: +0.0 ❌

### 3. Milestone Rewards

```python
# Beat a blind
reward += 100.0 * ante  # Scales with difficulty

# Lose a blind
reward -= 50.0

# Timeout (hit step limit)
reward -= 100.0

# Per-step time pressure
reward -= 0.01
```

## Expected Behavior After Retraining

### Early Training (0-5M steps):
- Random exploration
- Discovers playing hands > skipping (due to huge reward gap)
- Learns basic: "play cards, get points"
- **Expected reward**: -50 to +50

### Mid Training (5-20M steps):
- Starts beating Ante 1 consistently
- Learns to buy jokers
- Explores different hand types
- **Expected reward**: +50 to +150
- **Ante reached**: 2-3

### Late Training (20M+ steps):
- Consistently beats Ante 1-3
- Strategic joker purchases
- Hand type optimization
- **Expected reward**: +150 to +400+
- **Ante reached**: 3-5+

## Verbose Evaluation Output

### What skip-spam looks like (BAD - should never see this anymore):
```
Step   1 | Skip/Continue | V=-500.00 | R=-1.01 | Ante 1 | Chips: 0
Step   2 | Skip/Continue | V=-499.00 | R=-1.01 | Ante 1 | Chips: 0
...
Episode ended: TIMEOUT
Total Reward: -1,110.00  😱
```

### What good gameplay looks like (GOOD - what we want):
```
Step   1 | Play Hand (cards=[0,1,2]) | V= 12.50 | R=+5.20 | Ante 1 | Chips: 450
Step   2 | Discard (cards=[3,4])     | V= 12.30 | R=+0.10 | Ante 1 | Chips: 450
Step   3 | Play Hand (cards=[0,2,4]) | V= 18.75 | R=+8.50 | Ante 1 | Chips: 1,350
Step   4 | Skip/Continue             | V= 18.70 | R=-1.01 | Ante 1 | Chips: 1,350
...
Episode ended: TERMINATED (beat blind!)
Total Reward: +125.50  🎉
```

## Why This Will Work

### 1. **Massive Reward Gap**
Skip-spam now gives **-1,110** vs playing gives **+100 to +300**. This is impossible to ignore.

### 2. **Exploration Incentive**
The +0.1 bonus for ANY action means early random exploration gets rewarded.

### 3. **Immediate Feedback**
Every hand played gives +3 to +8 reward immediately, much better than -1.0 for skip.

### 4. **Scales with Progress**
Better play → more chips → higher rewards → stronger learning signal

## If This Still Doesn't Work...

If the model STILL skip-spams after V2, we have nuclear options:

### Option 1: Disable Skip Action During Training
```python
# In training loop, mask out skip action for first 10M steps
action_mask[11] = 0  # Force exploration
```

### Option 2: Curriculum Learning
```python
# Start with forced gameplay, gradually enable skip
if timesteps < 5_000_000:
    skip_penalty = -10.0  # Extremely harsh
else:
    skip_penalty = -1.0  # Normal harsh
```

### Option 3: Shaped Initialization
```python
# Start with pre-trained weights that know how to play hands
# Use supervised learning on demo gameplay first
```

## Training Command

```bash
# Delete old checkpoint to start fresh
rm checkpoint_2867200.pt

# Start training with V2 rewards
python train.py --config configs/a100_aggressive.yaml
```

## Monitoring

Watch these metrics in TensorBoard:
- **Mean reward** should be positive after 5-10M steps
- **Episode length** should decrease (faster wins)
- **Ante reached** should increase over time
- **Action type distribution** - should NOT be 100% skip!

## Summary

V2 makes skip-spam **SO painful** (-1,110 vs -210) that the agent HAS to explore alternatives. Combined with much better rewards for playing (+8 vs +2 per hand), the learning signal is now strong enough to overcome the exploration problem.

**The skip-spam strategy is now effectively dead.** 💀


# 🐛 Reward Hacking Issue & Fix

## Problem Discovered

The trained model learned a **reward hacking strategy**: it discovered that repeatedly taking the "skip" action (action_type 11) gives `0.0` reward with zero risk, while actually playing the game risks getting the `-50.0` penalty for losing.

### Observable Behavior
- ✅ Evaluation runs for full 1000 steps
- ✅ Never loses (no `-50.0` penalty)
- ❌ Never wins (no `+100.0` reward)
- ❌ Stuck at Ante 1
- ❌ Constant `0.0` reward

**Root Cause**: The agent optimized for "not losing" rather than "winning" because:
1. Skip action had `0.0` reward (safe)
2. Losing had `-50.0` penalty (scary)
3. No penalty for timeout/inaction
4. Playing hands had small rewards but risked eventual loss

This is a classic **local optimum** problem in RL.

## Changes Made

### 1. Skip Action Penalty
**File**: `src/environment/balatro_env.py`

```python
def _skip_action(self) -> float:
    """Skip current action"""
    # Small penalty to discourage spam-skipping
    return -0.1  # Was: 0.0
```

**Impact**: Makes skipping slightly costly, encouraging the agent to take meaningful actions.

### 2. Timeout Penalty
**File**: `src/environment/balatro_env.py`

```python
# Penalty for timeout (discourages reward hacking via inaction)
if truncated and not terminated:
    reward -= 100.0  # Significant penalty for running out the clock
```

**Impact**: Heavily penalizes episodes that hit the 1000-step limit without winning or losing naturally.

### 3. Time Pressure
**File**: `src/environment/balatro_env.py`

```python
# Small time penalty to encourage efficient play
reward -= 0.01
```

**Impact**: Tiny per-step penalty accumulates over time, encouraging faster completion.

### 4. Enhanced Play Rewards
**File**: `src/environment/balatro_env.py`

```python
# Reward is proportional to chips scored (increased to encourage play)
base_reward = np.log1p(total_score) / 5.0  # Was: / 10.0 (doubled)

# Bonus for making progress toward blind goal
if self.state.current_blind:
    progress = min(1.0, self.state.chips_scored / self.state.current_blind.chip_requirement)
    progress_bonus = progress * 5.0  # Up to +5 reward for getting close
    return base_reward + progress_bonus
```

**Impact**: 
- Doubled base rewards for playing hands
- Added progress bonus (up to +5.0) for getting closer to beating the blind
- Makes actual gameplay more rewarding than inaction

## New Reward Structure Summary

| Action | Old Reward | New Reward | Change |
|--------|-----------|-----------|--------|
| Skip | 0.0 | -0.1 | Discouraged |
| Play hand (good) | ~0.5-1.0 | ~1.0-6.0 | **Strongly encouraged** |
| Beat blind | +100 × ante | +100 × ante | Unchanged (already good) |
| Lose blind | -50.0 | -50.0 | Unchanged |
| Timeout | 0.0 | **-100.0** | **Heavily penalized** |
| Per step | 0.0 | -0.01 | Slight time pressure |

## Expected Outcomes

### Short Term (Existing Checkpoint)
Your current checkpoint won't improve without retraining - it has already learned the "skip forever" strategy. Evaluation will show:
- Mean reward: **-110** (100 timeout penalty + 10 from time pressure)
- Still stuck at Ante 1

### After Retraining
The agent should now:
1. ✅ Learn that playing hands > skipping
2. ✅ Try to beat blinds to avoid timeout penalty
3. ✅ Make progress through antes
4. ✅ Explore different strategies
5. ✅ Earn positive rewards from actual gameplay

## Next Steps

### 1. Restart Training (Recommended)
```bash
python train.py --config configs/a100_aggressive.yaml
```

The new reward structure will guide the agent toward productive gameplay.

### 2. Monitor Training
Watch TensorBoard to verify:
- Mean reward increases above 0
- Episodes don't timeout (truncation rate decreases)
- Ante progression (agents reach higher antes)

```bash
tensorboard --logdir logs/a100_aggressive/
```

### 3. Evaluate New Checkpoints
```bash
python evaluate.py --checkpoint checkpoints/a100_aggressive/checkpoint_XXXXX.pt \
    --config configs/a100_aggressive.yaml --episodes 20
```

Look for:
- **Positive mean rewards** (5-50+ for early training)
- **Higher antes reached** (Ante 2-4 after few M steps)
- **Natural termination** (winning or losing, not timeout)

## Alternative: Further Reward Tuning

If the agent still struggles, consider these additional changes:

### Option A: Even Harsher Skip Penalty
```python
def _skip_action(self) -> float:
    return -0.5  # Even more costly
```

### Option B: Progressive Rewards
Use the `ProgressiveRewardShaper` from `reward_shaping.py`:
```python
# In balatro_env.py __init__:
from .reward_shaping import create_reward_shaper
self.reward_shaper = create_reward_shaper("progressive")
```

### Option C: Shaped Rewards for Exploration
Add small rewards for:
- Buying jokers (+0.5)
- Using consumables (+0.3)
- Increasing ante (+20 × ante)

## Technical Notes

### Why This Happened

This is a well-known problem in RL called:
- **Reward hacking** / **Reward specification gaming**
- **Unintended local optimum**
- **Conservative policy collapse**

The agent found a strategy that satisfies the literal reward function but violates our intent.

### Prevention for Future

1. **Always include timeout penalties** when using episode length limits
2. **Make desired behaviors more rewarding** than safe inaction
3. **Test early checkpoints** to catch reward hacking before wasting compute
4. **Use curriculum learning** to gradually increase difficulty
5. **Monitor multiple metrics** (not just reward): ante reached, hands played, etc.

### Related Work

Similar issues have been observed in:
- OpenAI's CoastRunners (boat spinning for rewards)
- Atari games (agents finding glitches)
- Robot manipulation (minimal movement strategies)

The fix is always the same: **adjust rewards to align with true objectives**.

## Summary

The model wasn't broken - it was working perfectly for the reward function we gave it! We've now fixed the reward function to better align with our actual goal (playing and winning Balatro).

**Time to retrain and watch it actually learn to play! 🎮🃏**


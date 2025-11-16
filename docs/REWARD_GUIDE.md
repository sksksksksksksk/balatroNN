# Reward System Guide

## Current Reward Structure

Your model is **already rewarded for both** high scores and winning blinds!

### Breakdown

| Reward Type | Amount | When |
|------------|--------|------|
| **Hand Score** | `log(1 + chips) / 10` | Each hand played |
| **Beat Blind** | `100 × ante` | Complete blind requirement |
| **Win Run** | `+1000` | Beat all 8 antes |
| **Fail Blind** | `-50` | Run out of hands |
| **Invalid Action** | `-1` | Try impossible action |

### Example Calculation

**Scenario**: Playing in Ante 3, score 1000 chips with a hand

1. Hand reward: `log(1 + 1000) / 10 = 0.69`
2. If this completes blind (total ≥ requirement):
   - Blind bonus: `100 × 3 = 300`
   - **Total reward**: `0.69 + 300 = 300.69`
3. If this is 8th ante:
   - Win bonus: `+1000`
   - **Total reward**: `1300.69`

## Why This Design?

### ✓ Combines Short and Long Term

- **Short-term**: Immediate feedback for scoring chips
- **Long-term**: Large bonus for strategic success (beating blinds)

### ✓ Scales with Difficulty

- Ante 1 blind: +100 reward
- Ante 8 blind: +800 reward
- Encourages reaching higher antes

### ✓ Prevents Reward Hacking

- `log(chips)` prevents exploiting huge scores
- Failure penalty discourages risky play
- Blind completion ensures strategic progress

## Alternative Reward Strategies

I've created 5 different reward shapers you can experiment with:

### 1. **Default** (Current)
```python
# Balanced approach
Hand: log(1 + chips) / 10
Blind: 100 × ante
```
**Best for**: General learning

### 2. **Progressive**
```python
# Milestone-based rewards
Hand: base + milestone_bonus + efficiency_bonus
  - 25% progress: +10
  - 50% progress: +20  
  - 75% progress: +30
  - 100% progress: +50 + (5 × hands_remaining)
Blind: 100 × ante × 1.5 (if boss)
```
**Best for**: Encouraging efficient play

### 3. **Dense**
```python
# Frequent feedback
Hand: chips / 100 + progress_bonus
Blind: 150 × ante
```
**Best for**: Early training, exploration

### 4. **Sparse**
```python
# Only reward completion
Hand: 0
Blind: 200 × ante × 2 (if boss)
```
**Best for**: Robust learning (once it works)

### 5. **Curriculum**
```python
# Adaptive: starts dense → becomes sparse
Hand: (1 - progress) × dense + progress × sparse
  progress = training_step / (max_steps / 2)
```
**Best for**: Long training runs

## How to Experiment

### Option 1: Modify Environment Directly

Edit `src/environment/balatro_env.py`:

```python
def _play_hand(self, card_selection):
    # ... existing code ...
    total_score = chips * mult
    self.state.chips_scored += total_score
    
    # MODIFY THIS LINE:
    # return np.log1p(total_score) / 10.0  # Default
    
    # Try progressive:
    progress = self.state.chips_scored / self.state.current_blind.chip_requirement
    milestone_bonus = 0
    if progress >= 0.25: milestone_bonus += 10
    if progress >= 0.5: milestone_bonus += 20
    if progress >= 0.75: milestone_bonus += 30
    return np.log1p(total_score) / 10.0 + milestone_bonus
```

### Option 2: Use Reward Shaper Module

```python
from environment.reward_shaping import create_reward_shaper

# In BalatroEnv.__init__:
self.reward_shaper = create_reward_shaper("progressive", config)

# In step():
reward = self.reward_shaper.shape_hand_reward(
    chips_scored=total_score,
    chips_required=self.state.current_blind.chip_requirement,
    hands_remaining=self.state.hands_remaining
)
```

### Option 3: Config-Based

Add to your YAML config:

```yaml
environment:
  reward_strategy: "progressive"  # or "dense", "sparse", "curriculum"
  reward_config:
    milestone_rewards: [10, 20, 30, 50]
    efficiency_bonus: 5.0
```

## Recommendations

### For Your First Training Run
**Use Default** ✓
- Proven to work
- Balanced feedback
- Good baseline

### If Learning is Slow
**Try Dense** 
- More frequent rewards
- Helps exploration
- Switch to Default after 1M steps

### For Maximum Performance
**Try Progressive or Curriculum**
- Encourages efficiency
- Better long-term strategy
- May take longer to learn initially

### For Research
**Compare All**
- Train 5 models with different shapers
- Same seeds, configs
- See which learns fastest

## Common Issues & Solutions

### Issue: Model just discards repeatedly
**Solution**: 
- Increase penalty for useless actions
- Add reward for using hands efficiently

```python
# Penalty for wasting discards
if discards_remaining < starting_discards / 2:
    reward -= 0.5  # Penalize excessive discarding
```

### Issue: Model scores but doesn't win blinds
**Solution**:
- Increase blind completion bonus
- Add progress milestones

```python
# Current: 100 × ante
# Try: 200 × ante (or even 300)
reward += 200.0 * self.state.ante
```

### Issue: Model wins early antes but fails later
**Solution**:
- Scale rewards more aggressively
- Use exponential scaling

```python
# Instead of linear (100, 200, 300...)
# Use exponential (100, 200, 400, 800...)
reward += 100.0 * (2 ** (self.state.ante - 1))
```

### Issue: Model plays same strategy every time
**Solution**:
- Add exploration bonus
- Increase entropy coefficient in PPO

```yaml
training:
  ent_coef: 0.02  # Increase from 0.01
```

## Metrics to Monitor

Track these in TensorBoard:

1. **Mean Reward per Episode**
   - Should increase over time
   - Expect: -100 → +500 → +1000+

2. **Mean Ante Reached**
   - Should increase over time
   - Good: 3-5, Excellent: 6-8

3. **Win Rate** (Ante 8+)
   - % of episodes reaching ante 8
   - Good: 10%, Excellent: 30%+

4. **Efficiency** (Chips per Hand)
   - Average chips scored per hand played
   - Higher = better strategy

5. **Exploration** (Entropy)
   - Should start high, gradually decrease
   - Too low too fast = premature convergence

## Advanced: Curriculum Learning

Gradually increase difficulty:

```python
class CurriculumEnv(BalatroEnv):
    def __init__(self, config):
        super().__init__(config)
        self.start_ante = 1
        self.win_rate_threshold = 0.7
        self.recent_wins = []
    
    def reset(self):
        obs, info = super().reset()
        # Start at appropriate ante
        for _ in range(self.start_ante - 1):
            self._advance_to_next_blind()
        return obs, info
    
    def update_curriculum(self, won: bool):
        self.recent_wins.append(won)
        if len(self.recent_wins) > 100:
            self.recent_wins.pop(0)
        
        win_rate = sum(self.recent_wins) / len(self.recent_wins)
        
        # Increase difficulty if doing well
        if win_rate > self.win_rate_threshold and self.start_ante < 8:
            self.start_ante += 1
            print(f"Curriculum: Now starting at Ante {self.start_ante}")
```

## Summary

✅ **You're already rewarding both high scores and blind completion!**

The current system is well-designed with:
- Immediate feedback for good plays
- Strategic incentives for blind completion  
- Difficulty scaling with ante progression
- Penalties for failures

**To improve learning:**
1. Start with **default** rewards (current)
2. Monitor training metrics
3. If stuck, try **dense** or **progressive**
4. For best performance, experiment with **curriculum**

**Remember**: Reward shaping is an art! What works best depends on:
- Your model architecture
- Training time available
- Desired final behavior
- Hardware constraints

Experiment and iterate! 🎮🤖


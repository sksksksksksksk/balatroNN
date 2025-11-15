# KL Divergence Early Stopping Guide

## What Is It?

The "Early stopping at epoch X due to reaching max KL divergence" message is a **safety feature** in PPO (Proximal Policy Optimization) that prevents your policy from changing too drastically in a single update.

### The Math

KL divergence measures how different two probability distributions are. In PPO:
- We want to improve the policy
- But not change it TOO much at once (could destabilize training)
- So we stop an update early if KL divergence > `1.5 * target_kl`

## Is This Normal? 🤔

| Frequency | Status | Meaning | Action |
|-----------|--------|---------|--------|
| **10-30%** | ✅ GOOD | Healthy learning | None needed |
| **30-50%** | ⚠️ MODERATE | Common early in training | Monitor |
| **50-80%** | 🔶 FREQUENT | Learning constrained | Consider tuning |
| **80%+** | 🔴 CRITICAL | Severely limited | Adjust immediately |

## Why Does It Happen?

### 1. **Early Training (Most Common)**
When your model is learning rapidly, the policy changes a lot between updates. This is GOOD - it means learning is happening! The early stopping just keeps it safe.

**Solution:** Wait it out. Usually stabilizes after 1000-5000 updates.

### 2. **Learning Rate Too High**
If your learning rate is too aggressive, each gradient step tries to change the policy too much.

**Solution:** Reduce `learning_rate`:
```yaml
training:
  learning_rate: 0.0003  # If you had 0.001
```

### 3. **Target KL Too Low**
Your `target_kl` might be set conservatively low, causing frequent stops even when changes are reasonable.

**Solution:** Increase `target_kl`:
```yaml
training:
  target_kl: 0.02  # From default 0.01
```

### 4. **Noisy Rewards**
If your reward signal is very noisy or sparse, the policy might oscillate, triggering frequent stops.

**Solution:** Check reward shaping (see `REWARD_GUIDE.md`)

## How to Check Your Training

### Option 1: Live Monitoring

Watch for the pattern during training:
```bash
python train.py --config configs/default.yaml 2>&1 | tee training.log
```

Look for:
- Frequent "Early stopping" messages
- High `train/kl_divergence` values in TensorBoard

### Option 2: Analyze Logs

```bash
# After training, analyze your log
python check_kl_stopping.py training.log
```

This will show you:
- Total updates vs early stops
- Percentage of constrained updates
- Recommendations

### Option 3: TensorBoard

```bash
tensorboard --logdir logs/
```

Look at the `train/kl_divergence` graph:
- Should decrease over time
- Spikes near `target_kl` threshold are fine
- Consistently at threshold = too constrained

## Configuration Guide

### Conservative (Slower, More Stable)
```yaml
training:
  learning_rate: 0.0001
  target_kl: 0.005
  n_epochs: 15
```
- Use when: Training is unstable, getting NaN losses
- Pros: Very stable, reliable convergence
- Cons: Slower learning

### Balanced (Recommended Default)
```yaml
training:
  learning_rate: 0.0003
  target_kl: 0.01
  n_epochs: 10
```
- Use when: Standard training, production runs
- Pros: Good balance of speed and stability
- Cons: May see 20-40% early stops early on

### Aggressive (Faster, Less Stable)
```yaml
training:
  learning_rate: 0.001
  target_kl: 0.03
  n_epochs: 8
```
- Use when: Experimenting, trying to learn quickly
- Pros: Faster convergence if it works
- Cons: Risk of instability, may need to restart

### Very Aggressive (Research/Experimentation)
```yaml
training:
  learning_rate: 0.003
  target_kl: null  # Disable KL constraint!
  n_epochs: 6
```
- Use when: You know what you're doing
- Pros: Maximum learning speed
- Cons: High risk of policy collapse

## Tuning Process

### Step 1: Diagnose

Run for 10,000 timesteps and check early stopping frequency:

```bash
python train.py --config configs/default.yaml 2>&1 | tee training.log
# After a few minutes, Ctrl+C
python check_kl_stopping.py training.log
```

### Step 2: Adjust

Based on results:

**If 50%+ early stops:**
```yaml
# Option A: Increase target_kl
training:
  target_kl: 0.02  # Double it

# Option B: Reduce learning rate
training:
  learning_rate: 0.00015  # Half it

# Option C: Both (recommended)
training:
  learning_rate: 0.0002
  target_kl: 0.015
```

**If <10% early stops:**
You can afford to be more aggressive:
```yaml
training:
  learning_rate: 0.0005  # Increase
  target_kl: 0.01  # Keep same
```

### Step 3: Monitor

Continue training and watch:
- TensorBoard: `train/kl_divergence`
- Console: Early stopping frequency
- Performance: Mean episode reward

### Step 4: Fine-tune

Once early stopping stabilizes (usually after 5,000-10,000 steps), you can:
- Slightly increase learning rate for faster convergence
- Slightly decrease target_kl for more stable policy

## Advanced: Disabling KL Constraint

For research or experimentation, you can disable it:

```yaml
training:
  target_kl: null  # Disable
```

**⚠️ Warning:** This removes a safety net! Use only if:
- You're experimenting
- You'll monitor training closely
- You're okay with potential instability

Without KL constraint, you rely entirely on the PPO clipping mechanism (`clip_range`) for stability.

## Real-World Example

### Scenario: 80% Early Stops

```
Training log shows:
- 1000 updates
- 800 early stops
- KL divergence: 0.015-0.020 (threshold: 0.015)
```

**Analysis:** Policy is trying to change faster than allowed.

**Fix:**
```yaml
# Before
training:
  learning_rate: 0.0003
  target_kl: 0.01
  n_epochs: 10

# After
training:
  learning_rate: 0.0003
  target_kl: 0.02  # ← Doubled
  n_epochs: 10
```

**Result:**
- Early stops drop to 30%
- Learning proceeds 2x faster
- Policy still stable

## Common Mistakes

### ❌ Mistake 1: Ignoring It
**Problem:** "I see early stops but I'll just let it run"

**Why bad:** Your model is learning much slower than it could

**Fix:** Analyze and tune

### ❌ Mistake 2: Disabling Too Early
**Problem:** Set `target_kl: null` immediately

**Why bad:** Training might become unstable

**Fix:** First try increasing `target_kl` to 0.02-0.03

### ❌ Mistake 3: Over-Tuning
**Problem:** Changing settings every 100 steps

**Why bad:** Need time to see effects (1000+ steps)

**Fix:** Make one change, wait 5,000+ steps, assess

### ❌ Mistake 4: Only Looking at KL
**Problem:** "KL is perfect, why isn't it learning?"

**Why bad:** KL is one metric; check rewards, loss, entropy too

**Fix:** Monitor all metrics in TensorBoard

## Quick Reference

### Common Configurations

```yaml
# For Balatro (default)
training:
  learning_rate: 0.0003
  target_kl: 0.01
  
# If seeing 50%+ early stops
training:
  learning_rate: 0.0003
  target_kl: 0.02
  
# For stable but slow training
training:
  learning_rate: 0.0001
  target_kl: 0.005
  
# For fast experimentation (risky)
training:
  learning_rate: 0.001
  target_kl: 0.05
```

### Target KL Values

| Value | Effect | Use Case |
|-------|--------|----------|
| 0.005 | Very conservative | Unstable training |
| 0.01 | Standard | Production (default) |
| 0.02 | Relaxed | Faster learning |
| 0.03 | Aggressive | Experimentation |
| 0.05+ | Very aggressive | Research only |
| null | Disabled | Expert use only |

## Summary

✅ **Early KL stopping is NORMAL and GOOD**
- It's a safety feature, not an error
- Protects against training instability
- 10-40% early stops is healthy

⚠️ **Take action if:**
- Early stops > 60% for extended period (5000+ steps)
- Learning is very slow
- You're comfortable tuning hyperparameters

🎯 **Best practice:**
1. Start with defaults
2. Monitor for 5,000-10,000 steps
3. Adjust if needed
4. Recheck after 5,000 more steps
5. Fine-tune when stable

📊 **Always monitor:**
- TensorBoard metrics
- Episode rewards
- Policy/value losses
- Not just KL divergence alone

---

**Still confused?** 
- Check your TensorBoard: `tensorboard --logdir logs/`
- Run analysis: `python check_kl_stopping.py training.log`
- See configs: `configs/` folder for examples


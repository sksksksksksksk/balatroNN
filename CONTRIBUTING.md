# Contributing to BalatroNN

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## Ways to Contribute

### 1. Bug Reports
- Use clear, descriptive titles
- Include steps to reproduce
- Provide system information (OS, GPU, Python version)
- Include relevant logs or error messages

### 2. Feature Requests
- Describe the feature and its benefits
- Explain why it would be useful
- Provide examples or mockups if applicable

### 3. Code Contributions

#### Getting Started
```bash
# Fork and clone the repository
git clone https://github.com/yourusername/balatroNN.git
cd balatroNN

# Create a branch for your feature
git checkout -b feature/your-feature-name

# Set up development environment
./setup.sh
```

#### Development Guidelines

**Code Style**
- Follow PEP 8 style guide
- Use type hints where possible
- Add docstrings for functions and classes
- Keep functions focused and small

**Testing**
```bash
# Run tests before submitting
python test_setup.py

# Test your changes
python train.py --config configs/quick_test.yaml
```

**Commits**
- Use clear, descriptive commit messages
- Reference issues in commits (e.g., "Fix #123")
- Keep commits focused on single changes

#### Pull Request Process

1. **Update Documentation**
   - Update README.md if needed
   - Add docstrings to new code
   - Update ARCHITECTURE.md for major changes

2. **Test Your Changes**
   - Ensure all tests pass
   - Add new tests for new features
   - Test on different hardware if possible

3. **Submit PR**
   - Provide clear description of changes
   - Link related issues
   - Include before/after comparisons if applicable
   - Request review from maintainers

## Areas for Contribution

### High Priority

1. **Environment Improvements**
   - More accurate Balatro mechanics
   - Additional joker effects
   - Consumable cards (Tarot, Planet)
   - Shop mechanics
   - Stake variations

2. **Performance Optimization**
   - Faster training loops
   - Better GPU utilization
   - Memory optimization
   - Parallelization

3. **Documentation**
   - Tutorial videos
   - More examples
   - Advanced guides
   - API documentation

### Medium Priority

1. **Additional Features**
   - Curriculum learning implementation
   - Model ensembling
   - Self-play training
   - Attention visualization

2. **Experiments**
   - Different architectures (CNN, RNN)
   - Alternative RL algorithms (SAC, DQN)
   - Reward shaping strategies
   - Hyperparameter studies

3. **Tools**
   - Better visualization tools
   - Model comparison utilities
   - Checkpoint management
   - Distributed training

### Low Priority

1. **Nice to Have**
   - Web interface for evaluation
   - Real-time game interface
   - Model zoo with pretrained models
   - Automated hyperparameter tuning

## Code Examples

### Adding a New Joker

```python
# In src/environment/balatro_env.py

def create_blueprint_joker():
    return Joker(
        name="Blueprint",
        rarity="rare",
        effect="Copies ability of joker to the right",
        chips_bonus=0,
        mult_bonus=0,
        xmult_bonus=1.0,
        triggers=["hand_played"]
    )
```

### Adding a New Metric

```python
# In src/training/ppo.py

def train_step(self):
    # ... existing code ...
    
    # Add new metric
    stats["train/new_metric"] = compute_new_metric()
    
    return stats
```

### Adding a New Configuration

```yaml
# In configs/my_config.yaml

model:
  d_model: 384
  num_heads: 12
  num_layers: 8

training:
  learning_rate: 0.0001
  batch_size: 512
```

## Testing Guidelines

### Unit Tests
```python
# test_environment.py
import pytest
from src.environment import BalatroEnv

def test_environment_reset():
    env = BalatroEnv()
    obs, info = env.reset()
    assert "hand" in obs
    assert len(obs["hand"]) == (8, 32)
```

### Integration Tests
```python
# test_training.py
def test_training_loop():
    env = BalatroEnv()
    model = PolicyValueNetwork()
    trainer = PPOTrainer(model, env, config)
    
    # Train for a few steps
    history = trainer.train(total_timesteps=1000)
    
    assert len(history["mean_reward"]) > 0
```

## Documentation Standards

### Docstring Format
```python
def evaluate_hand(cards: List[Card]) -> Tuple[HandType, int, int]:
    """
    Evaluate a poker hand and calculate score.
    
    Args:
        cards: List of Card objects to evaluate
        
    Returns:
        Tuple of (hand_type, chips, mult)
        
    Example:
        >>> cards = [Card(Rank.ACE, Suit.HEARTS), ...]
        >>> hand_type, chips, mult = evaluate_hand(cards)
        >>> print(f"Score: {chips * mult}")
    """
    pass
```

### README Updates
- Keep it concise and scannable
- Use screenshots/GIFs for visual features
- Update badges and metrics
- Maintain table of contents

## Review Process

1. **Initial Review** (1-3 days)
   - Maintainers review code and design
   - Provide feedback on approach
   - Request changes if needed

2. **Revision** (as needed)
   - Address feedback
   - Update based on comments
   - Re-request review

3. **Approval**
   - Two maintainer approvals required
   - All tests must pass
   - Documentation must be complete

4. **Merge**
   - Squash and merge by maintainer
   - Update changelog
   - Close related issues

## Community Guidelines

- Be respectful and inclusive
- Help others learn and grow
- Share knowledge and insights
- Give constructive feedback
- Celebrate contributions

## Questions?

- Open an issue for questions
- Join our Discord (if applicable)
- Email maintainers
- Check existing issues and discussions

Thank you for contributing to BalatroNN! 🃏🤖


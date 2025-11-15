# Environment Scope - What's Implemented

## TL;DR

**The current environment simulates:**
- ✅ Core card playing mechanics
- ✅ Hand detection and scoring
- ✅ Blinds and antes
- ✅ Card enhancements/editions/seals

**It does NOT fully simulate:**
- ❌ Shops and shop economics
- ❌ Booster packs
- ❌ Planet/Tarot/Spectral cards
- ❌ Complex joker effects (150+ unique jokers)
- ❌ Vouchers
- ❌ Tags
- ❌ Deck variations

The model is learning to play **optimal poker hands** given a deck, but not the full meta-game of building decks, buying jokers, etc.

## Current Implementation Status

### ✅ Fully Implemented

#### Core Gameplay
- **Card Playing**: Play hands, select cards, calculate scores
- **Hand Detection**: All poker hands (pair, flush, straight, etc.)
- **Discarding**: Discard and draw new cards
- **Blinds**: Small/Big/Boss blinds with scaling difficulty
- **Antes**: Progress through antes 1-8
- **Money System**: Earn money from playing cards
- **Win/Loss Conditions**: Beat chip requirements or fail

#### Card Mechanics
- **Base Cards**: All 52 standard playing cards
- **Enhancements** (8 types):
  - Bonus (+30 chips)
  - Mult (+4 mult)
  - Wild (any suit)
  - Glass (x2 mult, fragile)
  - Steel (x1.5 mult in hand)
  - Stone (+50 chips, no rank/suit)
  - Gold ($3 per round)
  - Lucky (random bonuses)

- **Editions** (4 types):
  - Base (none)
  - Foil (+50 chips)
  - Holographic (+10 mult)
  - Polychrome (x1.5 mult)

- **Seals** (5 types):
  - None
  - Gold (earn $3 when played)
  - Red (retrigger card)
  - Blue (create Planet card)
  - Purple (create Tarot when discarded)

### ⚠️ Partially Implemented (Stubs)

#### Jokers
**What exists:**
- `Joker` class with name, chips_bonus, mult_bonus
- Space in observation for up to 5 jokers
- Generic joker effect application

**What's missing:**
- Specific joker implementations (150+ unique jokers)
- Conditional effects (e.g., "Baron" only activates with Kings)
- Triggered effects (e.g., "Egg" creates Joker when selling)
- Scoring jokers vs. effect jokers
- Joker combinations and synergies

**Example of what's missing:**
```python
# Current (simplified):
def _apply_joker_effects(self, chips, mult, hand_type, cards):
    for joker in self.state.jokers:
        chips += joker.chips_bonus
        mult += joker.mult_bonus
    return chips, mult

# What's needed for real jokers:
def _apply_joker_effects(self, chips, mult, hand_type, cards):
    for joker in self.state.jokers:
        if joker.name == "Baron":
            # +1.5x mult if played hand has a King
            if any(c.rank == Rank.KING for c in cards):
                mult *= 1.5
        elif joker.name == "Fibonacci":
            # Each Ace, 2, 3, 5, or 8 gives +8 mult
            fib_ranks = [Rank.ACE, Rank.TWO, Rank.THREE, Rank.FIVE, Rank.EIGHT]
            fib_count = sum(1 for c in cards if c.rank in fib_ranks)
            mult += 8 * fib_count
        elif joker.name == "Ride the Bus":
            # +1 mult per consecutive hand with no face cards
            # Reset if face card played
            # (requires persistent state!)
        # ... and 147 more unique jokers!
```

#### Shops
**What exists:**
- `in_shop` flag
- `shop_items` list
- Shop action types (buy, reroll, skip)
- Shop reroll tracking

**What's missing:**
- Shop generation (what items appear)
- Pricing system
- Item purchasing logic
- Pack opening mechanics
- Shop refreshing between rounds

**Current implementation:**
```python
def _shop_buy(self, item_index: int) -> float:
    if not self.state.in_shop:
        return -1.0
    # Simplified shop logic
    # In real implementation, would handle actual shop items
    return 0.0  # Does nothing!
```

### ❌ Not Implemented

#### Booster Packs
- **Arcana Pack**: 2 Tarot cards
- **Celestial Pack**: 2 Planet cards
- **Spectral Pack**: 2 Spectral cards
- **Standard Pack**: 4 playing cards
- **Buffoon Pack**: 2 Jokers
- Pack opening UI and selection
- Skip tag rewards

#### Consumable Cards

**Planet Cards (11 total)**
- Level up specific hand types
- Permanent upgrades to chips/mult
- Examples: Jupiter (4 of a Kind), Mars (Pair), etc.

**Tarot Cards (22 total)**
- Single-use effects on cards
- Examples:
  - "The Fool" - Creates last played hand type Tarot
  - "The Magician" - Enhances 2 cards to Lucky
  - "The High Priestess" - Creates 2 Planet cards
  - etc.

**Spectral Cards (15 total)**
- Powerful single-use effects
- Often have drawbacks
- Examples:
  - "Familiar" - Destroy 1 card, add 3 random Enhanced face cards
  - "Cryptid" - Create 2 copies of selected card
  - etc.

#### Vouchers (32 total)
- Permanent upgrades
- Examples:
  - "Overstock" - +1 card slot in shop
  - "Crystal Ball" - +1 consumable slot
  - "Paint Brush" - +2 hand size
  - etc.
- Voucher tiers and upgrades

#### Tags
- Skip rewards (money, vouchers, rare jokers)
- Tag types (Uncommon, Rare, Negative, etc.)
- Tag selection and consequences

#### Deck Types (15 total)
- Red Deck (standard)
- Blue Deck (+1 hand per round)
- Yellow Deck (start with extra money)
- Green Deck (earn more interest)
- Painted Deck (+2 hand size, -1 hand per round)
- etc.

#### Other Features
- **Stakes**: Difficulty modifiers (White through Gold)
- **Boss Blind Effects**: Special mechanics per boss
- **Challenge Runs**: Pre-set restrictions and goals
- **Seeded Runs**: Reproducible games
- **Collection**: Unlocks and progress tracking

## Why This Matters for Training

### What the Current Model Learns

With the current implementation, the model learns:
1. ✅ **Hand Recognition**: Identifying optimal poker hands
2. ✅ **Card Selection**: Which cards to play vs. discard
3. ✅ **Resource Management**: When to use hands vs. discards
4. ✅ **Strategic Planning**: Building towards specific hand types
5. ✅ **Blind Scaling**: Adapting to increasing difficulty

### What the Model Doesn't Learn

Without full shop/joker implementation:
1. ❌ **Economy Management**: When to buy vs. save money
2. ❌ **Build Planning**: Selecting jokers that synergize
3. ❌ **Resource Allocation**: Buying packs vs. jokers vs. consumables
4. ❌ **Deck Building**: Adding/removing cards strategically
5. ❌ **Long-term Planning**: Building towards specific strategies (e.g., "flush build")

### Training Implications

**Current State:**
- Model learns to **play cards optimally** with what it has
- Good for: Learning poker hand values and basic strategy
- Limited by: Can't adapt deck or build synergies

**With Full Implementation:**
- Model would learn **complete Balatro strategy**
- Much larger action space (100+ joker choices, 30+ card effects, etc.)
- Longer training time (10x-100x more complexity)
- Harder reward signal (must plan many rounds ahead)

## Should You Implement the Full Game?

### Pros of Current Simplified Version
✅ Faster training (weeks vs. months)
✅ Easier to debug
✅ Focuses on core gameplay
✅ Good proof-of-concept
✅ Works with modest hardware

### Pros of Full Implementation
✅ More interesting gameplay
✅ Better resembles actual Balatro
✅ Can learn advanced strategies
✅ More impressive final result

### Recommendation

**Phase 1 (Current):** Train on simplified environment
- Prove the approach works
- Get baseline performance
- Understand what works

**Phase 2 (Future):** Gradually add complexity
- Start with basic shop (buy jokers only)
- Add simple joker effects (10-20 most common)
- Add Planet cards (hand leveling)
- Expand from there

**Phase 3 (Advanced):** Full game
- All jokers with complex effects
- All consumables
- Full shop economics
- Deck building

## Implementation Roadmap

### Priority 1: Basic Shops
```python
# Add functional shop with pricing
class Shop:
    def __init__(self, ante: int):
        self.jokers = self._generate_jokers(2)  # 2 random jokers
        self.packs = self._generate_packs(2)     # 2 random packs
        self.cards = self._generate_cards(2)     # 2 random cards
        
    def _generate_jokers(self, count: int):
        # Generate jokers with prices ($4-8)
        pass
```

### Priority 2: Common Jokers (Top 20)
Implement most common jokers:
- Joker (+4 mult)
- Greedy Joker (+3 mult, loses $1 on play)
- Lusty Joker (+3 mult, loses $1 on play)
- Wrathful Joker (+3 mult, loses $1 on play)
- Glutton Joker (+3 mult, loses $1 on play)
- Jolly Joker (+8 mult if played hand is a Pair)
- Zany Joker (+12 mult if played hand is a Three of a Kind)
- Mad Joker (+10 mult if played hand is a Two Pair)
- Crazy Joker (+12 mult if played hand is a Straight)
- Droll Joker (+10 mult if played hand is a Flush)
- Sly Joker (+50 chips if played hand is a Pair)
- Wily Joker (+100 chips if played hand is a Three of a Kind)
- Clever Joker (+80 chips if played hand is a Two Pair)
- Devious Joker (+100 chips if played hand is a Straight)
- Crafty Joker (+80 chips if played hand is a Flush)
- Half Joker (+20 mult if hand contains 5 or fewer cards)
- Stencil Joker (x1 mult per empty joker slot)
- Four Fingers (all Flushes and Straights can be made with 4 cards)
- Mime (Retrigger all cards in hand)
- Credit Card (Go up to -$20 in debt)

### Priority 3: Planet Cards
```python
class PlanetCard:
    def __init__(self, hand_type: HandType):
        self.hand_type = hand_type
        
    def use(self, game_state):
        # Level up the specified hand type
        level = game_state.hand_levels[self.hand_type]
        game_state.hand_levels[self.hand_type] = level + 1
        # Increase chips and mult for that hand
```

### Priority 4: Basic Packs
```python
def open_standard_pack(self):
    # Show 4 cards, player picks 1
    cards = [self._generate_random_card() for _ in range(4)]
    selected = player_choice(cards)
    self.state.deck.append(selected)
```

### Priority 5: More Jokers & Complexity
- Expand to 50+ jokers
- Add triggered effects
- Add joker selling/destroying mechanics

## Estimating Complexity

### Current Environment:
- **States**: ~10^15 possible states
- **Actions per step**: ~256 (8-bit card selections)
- **Episode length**: 50-200 steps
- **Training time**: 2M steps (~3-4 hours on T4)

### Full Environment:
- **States**: ~10^30+ possible states (joker combinations explode this)
- **Actions per step**: ~1000+ (shop has many options)
- **Episode length**: 200-1000 steps (shops between rounds)
- **Training time**: 100M+ steps (weeks on H100)

### State Space Explosion Example:
```
Current: 52 cards + enhancements + editions + seals
≈ 52 * 8 * 4 * 5 = 8,320 card variations

With jokers: + (150 jokers choose 5)
≈ 8,320 + C(150, 5) = 8,320 + 591,600,660 variations

With shop: * (average shop size)
≈ Previous * ~20 shop configurations

With full game: * (all the above + planet levels + vouchers + tags)
≈ Astronomical
```

## Conclusion

**Current Implementation:**
- Good for learning basic gameplay
- Manageable training time
- Proof of concept

**Full Implementation:**
- Would be amazing
- Very complex
- Long training time
- Requires significant development

**Recommendation:**
Start with current, expand incrementally based on results and interest.

The model will learn good poker strategy even without shops/jokers. Adding those later will make it more complete but isn't necessary for initial training.

---

**See also:**
- `ARCHITECTURE.md` - Model architecture details
- `REWARD_GUIDE.md` - Reward function tuning
- `balatro_env.py` - Current environment implementation


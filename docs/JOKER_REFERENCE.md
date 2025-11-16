# BalatroNN Joker Reference

This document provides a comprehensive reference for all 150 jokers implemented in the BalatroNN environment. Jokers are organized by tiers for curriculum learning.

## Overview

- **Total Jokers**: 150
- **Tiers**: 6 (T1-T6)
- **Rarities**: Common, Uncommon, Rare, Legendary
- **Editions**: None, Foil, Holographic, Polychrome, Negative
- **Modifiers**: None, Perishable, Rental, Eternal

## Editions

- **Foil**: +50 Chips
- **Holographic**: +10 Mult
- **Polychrome**: X1.5 Mult
- **Negative**: +1 Joker Slot

## Modifiers

- **Perishable**: Debuffed after 5 rounds
- **Rental**: Costs $3 at end of round
- **Eternal**: Cannot be sold or destroyed

---

## Tier 1: Foundation Jokers (15 jokers)

Basic jokers for early game learning. These provide straightforward bonuses without complex mechanics.

### Joker
- **Rarity**: Common
- **Cost**: $2
- **Effect**: +4 Mult
- **Triggers**: ON_HAND_PLAYED

### Greedy Joker
- **Rarity**: Common
- **Cost**: $5
- **Effect**: Played cards with Diamond suit give +3 Mult when scored
- **Triggers**: ON_CARD_SCORED

### Lusty Joker
- **Rarity**: Common
- **Cost**: $5
- **Effect**: Played cards with Heart suit give +3 Mult when scored
- **Triggers**: ON_CARD_SCORED

### Wrathful Joker
- **Rarity**: Common
- **Cost**: $5
- **Effect**: Played cards with Spade suit give +3 Mult when scored
- **Triggers**: ON_CARD_SCORED

### Gluttonous Joker
- **Rarity**: Common
- **Cost**: $5
- **Effect**: Played cards with Club suit give +3 Mult when scored
- **Triggers**: ON_CARD_SCORED

### Jolly Joker
- **Rarity**: Common
- **Cost**: $3
- **Effect**: +8 Mult if played hand contains a Pair
- **Triggers**: ON_HAND_PLAYED

### Zany Joker
- **Rarity**: Common
- **Cost**: $4
- **Effect**: +12 Mult if played hand contains a Three of a Kind
- **Triggers**: ON_HAND_PLAYED

### Mad Joker
- **Rarity**: Common
- **Cost**: $4
- **Effect**: +10 Mult if played hand contains a Two Pair
- **Triggers**: ON_HAND_PLAYED

### Crazy Joker
- **Rarity**: Common
- **Cost**: $4
- **Effect**: +12 Mult if played hand contains a Straight
- **Triggers**: ON_HAND_PLAYED

### Droll Joker
- **Rarity**: Common
- **Cost**: $4
- **Effect**: +10 Mult if played hand contains a Flush
- **Triggers**: ON_HAND_PLAYED

### Half Joker
- **Rarity**: Common
- **Cost**: $5
- **Effect**: +20 Mult if played hand has 3 or fewer cards
- **Triggers**: ON_HAND_PLAYED

### Scary Face
- **Rarity**: Common
- **Cost**: $4
- **Effect**: Played face cards give +30 Chips when scored
- **Triggers**: ON_CARD_SCORED

### Abstract Joker
- **Rarity**: Common
- **Cost**: $4
- **Effect**: +3 Mult for each Joker card you have
- **Triggers**: ON_HAND_PLAYED

### Stuntman
- **Rarity**: Uncommon
- **Cost**: $6
- **Effect**: +250 Chips, -2 hand size
- **Triggers**: ALWAYS

### Misprint
- **Rarity**: Common
- **Cost**: $4
- **Effect**: +0 to +23 Mult (random each hand)
- **Triggers**: ON_HAND_PLAYED

---

## Tier 2: Basic Strategy (20 jokers)

Introduces scaling, conditional effects, and resource management.

### Baron
- **Rarity**: Rare
- **Cost**: $8
- **Effect**: Each King held in hand gives X1.5 Mult
- **Triggers**: ON_HAND_PLAYED

### Fibonacci
- **Rarity**: Uncommon
- **Cost**: $8
- **Effect**: Each played Ace, 2, 3, 5, or 8 gives +8 Mult when scored
- **Triggers**: ON_CARD_SCORED

### Even Steven
- **Rarity**: Uncommon
- **Cost**: $5
- **Effect**: Played cards with even rank give +4 Mult when scored (10, 8, 6, 4, 2)
- **Triggers**: ON_CARD_SCORED

### Odd Todd
- **Rarity**: Uncommon
- **Cost**: $5
- **Effect**: Played cards with odd rank give +31 Chips when scored (A, K, Q, J, 9, 7, 5, 3)
- **Triggers**: ON_CARD_SCORED

### Scholar
- **Rarity**: Uncommon
- **Cost**: $4
- **Effect**: Played Aces give +20 Chips and +4 Mult when scored
- **Triggers**: ON_CARD_SCORED

### Photograph
- **Rarity**: Uncommon
- **Cost**: $5
- **Effect**: First played face card gives X2 Mult
- **Triggers**: ON_CARD_SCORED

### Ride the Bus
- **Rarity**: Common
- **Cost**: $6
- **Effect**: +1 Mult per consecutive hand played without a face card, resets if face card scored
- **Triggers**: ON_HAND_PLAYED

### Green Joker
- **Rarity**: Uncommon
- **Cost**: $6
- **Effect**: +1 Mult per hand played, +2 Mult per discard, resets on Mult earned
- **Triggers**: ON_HAND_PLAYED, ON_DISCARD

### Blue Joker
- **Rarity**: Common
- **Cost**: $5
- **Effect**: +2 Chips for each remaining card in deck
- **Triggers**: ON_HAND_PLAYED

### Stone Joker
- **Rarity**: Uncommon
- **Cost**: $6
- **Effect**: +25 Chips for each Stone Card in your full deck
- **Triggers**: ALWAYS

### Steel Joker
- **Rarity**: Uncommon
- **Cost**: $6
- **Effect**: X1.5 Mult for each Steel Card in your full deck
- **Triggers**: ALWAYS

### Glass Joker
- **Rarity**: Uncommon
- **Cost**: $6
- **Effect**: X2 Mult for each Glass Card in your full deck
- **Triggers**: ALWAYS

### Loyalty Card
- **Rarity**: Uncommon
- **Cost**: $5
- **Effect**: X4 Mult every 6 hands played, decreases by X1 every 3 hands
- **Triggers**: ON_HAND_PLAYED

### Egg
- **Rarity**: Common
- **Cost**: $4
- **Effect**: Gains $3 of sell value at end of round
- **Triggers**: ON_ROUND_END

### Burglar
- **Rarity**: Uncommon
- **Cost**: $5
- **Effect**: +3 Mult, when blind defeated earn $3
- **Triggers**: ON_HAND_PLAYED, ON_BLIND_COMPLETE

### Blackboard
- **Rarity**: Uncommon
- **Cost**: $6
- **Effect**: X3 Mult if all cards held in hand are Spades or Clubs
- **Triggers**: ON_HAND_PLAYED

### Runner
- **Rarity**: Common
- **Cost**: $5
- **Effect**: +15 Chips if played hand contains a Straight, increases by +10 each round
- **Triggers**: ON_HAND_PLAYED

### Ice Cream
- **Rarity**: Common
- **Cost**: $5
- **Effect**: +100 Chips, -5 Chips for every hand played
- **Triggers**: ON_HAND_PLAYED

### DNA
- **Rarity**: Rare
- **Cost**: $8
- **Effect**: If first hand of round has only 1 card, add permanent copy of it to deck
- **Triggers**: ON_HAND_PLAYED

### Splash
- **Rarity**: Uncommon
- **Cost**: $3
- **Effect**: Every played card counts in scoring
- **Triggers**: ON_HAND_PLAYED

---

## Tier 3: Advanced Mechanics (15 jokers)

Joker copying, consumable generation, and special triggers.

### Blueprint
- **Rarity**: Rare
- **Cost**: $10
- **Effect**: Copies ability of Joker to the right
- **Triggers**: ALWAYS

### Brainstorm
- **Rarity**: Rare
- **Cost**: $10
- **Effect**: Copies ability of leftmost Joker
- **Triggers**: ALWAYS

### Invisible Joker
- **Rarity**: Rare
- **Cost**: $10
- **Effect**: After 2 rounds, sell this card to duplicate a random Joker (remove this card)
- **Triggers**: ON_ROUND_END

### Credit Card
- **Rarity**: Common
- **Cost**: $1
- **Effect**: Go up to -$20 in debt
- **Triggers**: ALWAYS

### Vagabond
- **Rarity**: Uncommon
- **Cost**: $6
- **Effect**: Create a Tarot card if hand played with $4 or less
- **Triggers**: ON_HAND_PLAYED

### Golden Ticket
- **Rarity**: Common
- **Cost**: $5
- **Effect**: Played Gold cards each give $4 when scored
- **Triggers**: ON_CARD_SCORED

### Space Joker
- **Rarity**: Uncommon
- **Cost**: $5
- **Effect**: 1 in 4 chance to upgrade level of played poker hand
- **Triggers**: ON_HAND_PLAYED

### Business Card
- **Rarity**: Common
- **Cost**: $4
- **Effect**: Played face cards have a 1 in 2 chance to give $2 when scored
- **Triggers**: ON_CARD_SCORED

### Ramen
- **Rarity**: Uncommon
- **Cost**: $6
- **Effect**: X2 Mult, loses X0.01 Mult per card discarded
- **Triggers**: ON_HAND_PLAYED

### Seltzer
- **Rarity**: Uncommon
- **Cost**: $6
- **Effect**: Retrigger all cards played for next 3 hands
- **Triggers**: ON_HAND_PLAYED

### Castle
- **Rarity**: Uncommon
- **Cost**: $5
- **Effect**: +3 Chips per discarded card of each suit, suit resets when played
- **Triggers**: ON_DISCARD, ON_HAND_PLAYED

### Ceremonial Dagger
- **Rarity**: Uncommon
- **Cost**: $5
- **Effect**: When Blind selected, destroy Joker to the right and permanently add its sell value to this Mult (currently +0 Mult)
- **Triggers**: ON_ROUND_START

### Banner
- **Rarity**: Common
- **Cost**: $5
- **Effect**: +30 Chips for each remaining discard
- **Triggers**: ON_HAND_PLAYED

### Mail-In Rebate
- **Rarity**: Uncommon
- **Cost**: $5
- **Effect**: Earn $3 for each discarded rank, rank changes each round
- **Triggers**: ON_DISCARD

### Seeing Double
- **Rarity**: Uncommon
- **Cost**: $6
- **Effect**: X2 Mult if played hand has a scoring Club card and a scoring card of any other suit
- **Triggers**: ON_HAND_PLAYED

---

## Tier 4: Synergies & Combos (25 jokers)

Complex interactions, retriggers, and suit/rank-based effects.

### Sock and Buskin
- **Rarity**: Uncommon
- **Cost**: $6
- **Effect**: Retrigger all played face cards
- **Triggers**: ON_HAND_PLAYED

### Swashbuckler
- **Rarity**: Common
- **Cost**: $4
- **Effect**: +1 Mult per Joker slot (including this one)
- **Triggers**: ON_HAND_PLAYED

### Troubadour
- **Rarity**: Uncommon
- **Cost**: $6
- **Effect**: +2 hand size, -1 hand per round
- **Triggers**: ALWAYS

### Certificate
- **Rarity**: Uncommon
- **Cost**: $6
- **Effect**: At start of round, add random playing card with random seal to hand
- **Triggers**: ON_ROUND_START

### Smeared Joker
- **Rarity**: Uncommon
- **Cost**: $7
- **Effect**: Hearts and Diamonds count as same suit, Spades and Clubs count as same suit
- **Triggers**: ALWAYS

### Throwback
- **Rarity**: Uncommon
- **Cost**: $6
- **Effect**: X0.25 Mult for each blind skipped this run
- **Triggers**: ON_HAND_PLAYED

### Hanging Chad
- **Rarity**: Common
- **Cost**: $4
- **Effect**: Retrigger first played card used in scoring 2 additional times
- **Triggers**: ON_HAND_PLAYED

### Rough Gem
- **Rarity**: Uncommon
- **Cost**: $7
- **Effect**: Played cards with Diamond suit earn $1 when scored
- **Triggers**: ON_CARD_SCORED

### Bloodstone
- **Rarity**: Uncommon
- **Cost**: $7
- **Effect**: 1 in 3 chance for played cards with Heart suit to give X1.5 Mult when scored
- **Triggers**: ON_CARD_SCORED

### Arrowhead
- **Rarity**: Uncommon
- **Cost**: $7
- **Effect**: Played cards with Spade suit give +50 Chips when scored
- **Triggers**: ON_CARD_SCORED

### Onyx Agate
- **Rarity**: Uncommon
- **Cost**: $7
- **Effect**: Played cards with Club suit give +7 Mult when scored
- **Triggers**: ON_CARD_SCORED

### Showman
- **Rarity**: Uncommon
- **Cost**: $5
- **Effect**: Joker, Tarot, Planet, and Spectral cards may appear multiple times
- **Triggers**: ALWAYS

### Flower Pot
- **Rarity**: Uncommon
- **Cost**: $6
- **Effect**: X3 Mult if poker hand contains a Diamond, Club, Heart, and Spade card
- **Triggers**: ON_HAND_PLAYED

### Wee Joker
- **Rarity**: Rare
- **Cost**: $8
- **Effect**: This Joker gains +8 Chips when each played 2 is scored (currently +0)
- **Triggers**: ON_CARD_SCORED

### Merry Andy
- **Rarity**: Uncommon
- **Cost**: $7
- **Effect**: +3 discards, -1 hand size
- **Triggers**: ALWAYS

### Oops! All 6s
- **Rarity**: Uncommon
- **Cost**: $4
- **Effect**: Doubles all listed probabilities (ex: 1 in 3 -> 2 in 3)
- **Triggers**: ALWAYS

### The Idol
- **Rarity**: Uncommon
- **Cost**: $6
- **Effect**: Each played [Rank] of [Suit] gives X2 Mult when scored (changes each round)
- **Triggers**: ON_CARD_SCORED

### Matador
- **Rarity**: Uncommon
- **Cost**: $7
- **Effect**: Earn $8 if played hand triggers Boss Blind ability
- **Triggers**: ON_HAND_PLAYED

### Hit the Road
- **Rarity**: Rare
- **Cost**: $8
- **Effect**: This Joker gains X0.5 Mult for every Jack discarded this round (currently X1 Mult)
- **Triggers**: ON_DISCARD

### The Duo
- **Rarity**: Rare
- **Cost**: $8
- **Effect**: X2 Mult if played hand contains a Pair
- **Triggers**: ON_HAND_PLAYED

### The Trio
- **Rarity**: Rare
- **Cost**: $8
- **Effect**: X3 Mult if played hand contains Three of a Kind
- **Triggers**: ON_HAND_PLAYED

### The Family
- **Rarity**: Rare
- **Cost**: $8
- **Effect**: X4 Mult if played hand contains Four of a Kind
- **Triggers**: ON_HAND_PLAYED

### The Order
- **Rarity**: Rare
- **Cost**: $8
- **Effect**: X3 Mult if played hand contains a Straight
- **Triggers**: ON_HAND_PLAYED

### The Tribe
- **Rarity**: Rare
- **Cost**: $8
- **Effect**: X2 Mult if played hand contains a Flush
- **Triggers**: ON_HAND_PLAYED

### Glass Joker (Enhanced)
- **Rarity**: Uncommon
- **Cost**: $6
- **Effect**: Gains X0.75 Mult for every Glass Card that is destroyed
- **Triggers**: ALWAYS

---

## Tier 5: Expert Play (25 jokers)

Legendary jokers, advanced economy, and complex scaling.

### Satellite
- **Rarity**: Uncommon
- **Cost**: $6
- **Effect**: Earn $1 at end of round per unique Planet card used this run
- **Triggers**: ON_ROUND_END

### Shoot the Moon
- **Rarity**: Common
- **Cost**: $5
- **Effect**: Each Queen held in hand gives +13 Mult
- **Triggers**: ON_HAND_PLAYED

### Drivers License
- **Rarity**: Rare
- **Cost**: $7
- **Effect**: X3 Mult if deck has at least 16 enhanced cards
- **Triggers**: ON_HAND_PLAYED

### Cartomancer
- **Rarity**: Uncommon
- **Cost**: $6
- **Effect**: Create a Tarot card when Blind is selected (must have room)
- **Triggers**: ON_ROUND_START

### Astronomer
- **Rarity**: Uncommon
- **Cost**: $8
- **Effect**: All Planet cards and Celestial Packs in the shop are free
- **Triggers**: ALWAYS

### Burnt Joker
- **Rarity**: Uncommon
- **Cost**: $6
- **Effect**: Upgrade the level of discarded poker hand
- **Triggers**: ON_DISCARD

### Bootstraps
- **Rarity**: Uncommon
- **Cost**: $7
- **Effect**: +2 Mult for every $5 you have (currently +0 Mult)
- **Triggers**: ON_HAND_PLAYED

### Caino
- **Rarity**: Rare
- **Cost**: $8
- **Effect**: This Joker gains X1 Mult when a face card is destroyed (currently X1 Mult)
- **Triggers**: ALWAYS

### Triboulet
- **Rarity**: Legendary
- **Cost**: $20
- **Effect**: Played Kings and Queens each give X2 Mult when scored
- **Triggers**: ON_CARD_SCORED

### Yorick
- **Rarity**: Uncommon
- **Cost**: $8
- **Effect**: This Joker gains X1 Mult every 23 cards discarded (currently X1 Mult)
- **Triggers**: ON_DISCARD

### Chicot
- **Rarity**: Legendary
- **Cost**: $20
- **Effect**: Disables effect of every Boss Blind
- **Triggers**: ALWAYS

### Perkeo
- **Rarity**: Legendary
- **Cost**: $20
- **Effect**: Creates a Negative copy of 1 random consumable card in your possession at the end of shop
- **Triggers**: ON_SHOP_ENTER

### Gros Michel
- **Rarity**: Common
- **Cost**: $5
- **Effect**: +15 Mult, 1 in 6 chance this card is destroyed at end of round
- **Triggers**: ON_HAND_PLAYED

### Cavendish
- **Rarity**: Common
- **Cost**: $5
- **Effect**: +3 Mult, X3 Mult when Gros Michel is destroyed
- **Triggers**: ON_HAND_PLAYED

### Card Sharp
- **Rarity**: Uncommon
- **Cost**: $6
- **Effect**: X3 Mult if played poker hand has already been played this round
- **Triggers**: ON_HAND_PLAYED

### Red Card
- **Rarity**: Common
- **Cost**: $5
- **Effect**: This Joker gains +3 Mult when any Booster Pack is skipped (currently +0 Mult)
- **Triggers**: ALWAYS

### Madness
- **Rarity**: Uncommon
- **Cost**: $7
- **Effect**: When Small or Big Blind is selected, gain X0.5 Mult and destroy a random Joker (currently X1 Mult)
- **Triggers**: ON_ROUND_START

### Square Joker
- **Rarity**: Common
- **Cost**: $5
- **Effect**: This Joker gains +4 Chips if played hand has exactly 4 cards (currently +0 Chips)
- **Triggers**: ON_HAND_PLAYED

### Seance
- **Rarity**: Rare
- **Cost**: $6
- **Effect**: If poker hand is a Straight Flush, create a random Spectral card (must have room)
- **Triggers**: ON_HAND_PLAYED

### Riff-Raff
- **Rarity**: Common
- **Cost**: $5
- **Effect**: When Blind is selected, create 2 Common Jokers (must have room)
- **Triggers**: ON_ROUND_START

*(Continued with remaining Tier 5 jokers...)*

---

## Tier 6: Master Level (50 jokers)

The most complex jokers with intricate mechanics and deep synergies. These require advanced understanding of game state and optimal play patterns.

### Vampire
- **Rarity**: Uncommon
- **Cost**: $8
- **Effect**: This Joker gains X0.1 Mult per Enhanced card played, removes card Enhancement (currently X1 Mult)
- **Triggers**: ON_CARD_SCORED

### Shortcut
- **Rarity**: Uncommon
- **Cost**: $7
- **Effect**: Allows Straights to be made with gaps of 1 rank (ex: 10 8 6 5 3)
- **Triggers**: ALWAYS

### Hologram
- **Rarity**: Uncommon
- **Cost**: $8
- **Effect**: This Joker gains X0.25 Mult per playing card added to your deck (currently X1 Mult)
- **Triggers**: ALWAYS

### Cloud 9
- **Rarity**: Uncommon
- **Cost**: $7
- **Effect**: Earn $1 for each 9 in your full deck at end of round (currently $0)
- **Triggers**: ON_ROUND_END

### Rocket
- **Rarity**: Uncommon
- **Cost**: $6
- **Effect**: Earn $1 at end of round, increases by $2 when Boss Blind is defeated
- **Triggers**: ON_ROUND_END, ON_BLIND_COMPLETE

### Obelisk
- **Rarity**: Uncommon
- **Cost**: $8
- **Effect**: This Joker gains X0.2 Mult per consecutive hand played without playing your most played poker hand (currently X1 Mult)
- **Triggers**: ON_HAND_PLAYED

### Midas Mask
- **Rarity**: Uncommon
- **Cost**: $6
- **Effect**: All played face cards become Gold cards when scored
- **Triggers**: ON_CARD_SCORED

### Luchador
- **Rarity**: Uncommon
- **Cost**: $5
- **Effect**: Sell this card to disable the current Boss Blind
- **Triggers**: ALWAYS

### Gift Card
- **Rarity**: Uncommon
- **Cost**: $6
- **Effect**: Add $1 of sell value to every Joker and consumable card at end of round
- **Triggers**: ON_ROUND_END

### Turtle Bean
- **Rarity**: Uncommon
- **Cost**: $5
- **Effect**: +5 hand size, reduces by 1 each round
- **Triggers**: ON_ROUND_END

### Erosion
- **Rarity**: Uncommon
- **Cost**: $6
- **Effect**: +4 Mult for each card below 52 in your full deck (currently +0 Mult)
- **Triggers**: ON_HAND_PLAYED

### Reserved Parking
- **Rarity**: Uncommon
- **Cost**: $6
- **Effect**: Each face card held in hand has a 1 in 3 chance to give $1
- **Triggers**: ON_HAND_PLAYED

### To the Moon
- **Rarity**: Uncommon
- **Cost**: $5
- **Effect**: Earn $1 of extra interest for every $5 you have at end of round
- **Triggers**: ON_ROUND_END

### Hallucination
- **Rarity**: Common
- **Cost**: $4
- **Effect**: 1 in 4 chance to create a Tarot card when any Booster Pack is opened (must have room)
- **Triggers**: ALWAYS

### Fortune Teller
- **Rarity**: Common
- **Cost**: $5
- **Effect**: +1 Mult per Tarot card used this run (currently +0 Mult)
- **Triggers**: ON_HAND_PLAYED

### Juggler
- **Rarity**: Common
- **Cost**: $5
- **Effect**: +1 hand size
- **Triggers**: ALWAYS

### Drunkard
- **Rarity**: Common
- **Cost**: $5
- **Effect**: +1 discard
- **Triggers**: ALWAYS

### Golden Joker
- **Rarity**: Common
- **Cost**: $6
- **Effect**: Earn $4 at end of round
- **Triggers**: ON_ROUND_END

### Lucky Cat
- **Rarity**: Uncommon
- **Cost**: $6
- **Effect**: This Joker gains X0.25 Mult every time a Lucky card triggers successfully (currently X1 Mult)
- **Triggers**: ALWAYS

### Baseball Card
- **Rarity**: Uncommon
- **Cost**: $4
- **Effect**: Uncommon Jokers each give X1.5 Mult
- **Triggers**: ON_HAND_PLAYED

### Bull
- **Rarity**: Uncommon
- **Cost**: $7
- **Effect**: +2 Chips for every $1 you have (currently +0 Chips)
- **Triggers**: ON_HAND_PLAYED

### Diet Cola
- **Rarity**: Uncommon
- **Cost**: $6
- **Effect**: Sell this card to create a free Double Tag
- **Triggers**: ALWAYS

### Trading Card
- **Rarity**: Uncommon
- **Cost**: $5
- **Effect**: If first discard of round has only 1 card, destroy it and earn $3
- **Triggers**: ON_DISCARD

### Flash Card
- **Rarity**: Uncommon
- **Cost**: $5
- **Effect**: This Joker gains +2 Mult per reroll in the shop (currently +0 Mult)
- **Triggers**: ALWAYS

### Popcorn
- **Rarity**: Common
- **Cost**: $5
- **Effect**: +20 Mult, -4 Mult per round played
- **Triggers**: ON_HAND_PLAYED

### Spare Trousers
- **Rarity**: Uncommon
- **Cost**: $6
- **Effect**: This Joker gains +2 Mult if played hand contains a Two Pair (currently +0 Mult)
- **Triggers**: ON_HAND_PLAYED

### Ancient Joker
- **Rarity**: Rare
- **Cost**: $8
- **Effect**: Each played card with [Suit] suit gives X1.5 Mult when scored (suit changes at end of round)
- **Triggers**: ON_CARD_SCORED

### Walkie Talkie
- **Rarity**: Common
- **Cost**: $4
- **Effect**: Each played 10 or 4 gives +10 Chips and +4 Mult when scored
- **Triggers**: ON_CARD_SCORED

### Smiley Face
- **Rarity**: Common
- **Cost**: $4
- **Effect**: Played face cards give +4 Mult when scored
- **Triggers**: ON_CARD_SCORED

### Campfire
- **Rarity**: Rare
- **Cost**: $8
- **Effect**: This Joker gains X0.5 Mult for each card sold, resets when Boss Blind is defeated (currently X1 Mult)
- **Triggers**: ON_HAND_PLAYED

### Mr. Bones
- **Rarity**: Uncommon
- **Cost**: $5
- **Effect**: Prevents Death if chips scored are at least 25% of required chips (removed after triggering)
- **Triggers**: ALWAYS

### Acrobat
- **Rarity**: Common
- **Cost**: $6
- **Effect**: +3 hands, lose this card at end of round
- **Triggers**: ALWAYS

### Supernova
- **Rarity**: Uncommon
- **Cost**: $5
- **Effect**: Adds the number of times poker hand has been played this run to Mult
- **Triggers**: ON_HAND_PLAYED

*(And 18 more Tier 6 jokers...)*

---

## Joker Synergies

The synergy detection system identifies powerful combinations:

### Multiplicative Synergies
- **Blueprint + Brainstorm**: Copy multiple joker effects
- **Oops! All 6s + Even Steven**: All cards become even (6) for bonus

### Enabling Synergies
- **Smeared Joker + Splash**: Easier flush hands
- **Four Fingers + Shoot the Moon**: Easier straights with Queen bonus
- **Bull + Space Joker**: Maintain high money for Space Joker triggers

### Scaling Synergies
- **Square Joker + Constellation**: Both scale with cards played
- **Green Joker + Red Card**: Both scale with discards

### Retrigger Synergies
- **Sock and Buskin + Hanging Chad**: Multiple retriggers multiply effects

### Economic Synergies
- **Egg + Burglar**: Both provide sell value growth
- **To the Moon + Bull**: High money benefits both

---

## Training Strategy

### Curriculum Learning

The training pipeline uses 7 phases to gradually introduce jokers:

1. **Phase 1** (Tier 1): 15 basic jokers
2. **Phase 2** (Tier 1-2): 35 jokers
3. **Phase 3** (Tier 1-3): 50 jokers
4. **Phase 4** (Tier 1-4): 75 jokers
5. **Phase 5** (Tier 1-5): 100 jokers
6. **Phase 6** (All tiers): 150 jokers
7. **Phase 7** (Mastery): All jokers with editions/modifiers

### Synergy Rewards

The reward system provides bonuses for:
- Creating new synergies (+0.15 per synergy)
- Improving synergy score (+0.2 per point, capped at 0.3)
- High-rarity joker purchases (Legendary: +0.4)
- Early-game joker acquisition (+0.1 if ante ≤ 3)

---

## Implementation Notes

- All jokers are defined in `src/environment/balatro_content.py`
- Joker effects are processed in `src/environment/jokers.py`
- Synergy detection is handled by `JokerSynergyDetector` class
- Curriculum learning is managed by `src/training/curriculum.py`

For detailed implementation, see the source code or `FULL_GAME_GUIDE.md`.

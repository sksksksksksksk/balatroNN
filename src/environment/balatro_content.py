"""
Balatro Game Content Database

Contains all game content definitions including jokers, consumables, vouchers, and packs.
Based on Balatro by LocalThunk.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Callable, Any
from enum import Enum


class Rarity(Enum):
    """Item rarity levels"""
    COMMON = 0
    UNCOMMON = 1
    RARE = 2
    LEGENDARY = 3


class JokerEdition(Enum):
    """Joker editions (visual/effect modifiers)"""
    NONE = 0
    FOIL = 1  # +50 chips
    HOLOGRAPHIC = 2  # +10 mult
    POLYCHROME = 3  # X1.5 mult
    NEGATIVE = 4  # +1 joker slot


class JokerModifier(Enum):
    """Special joker modifiers"""
    NONE = 0
    PERISHABLE = 1  # Debuffed after 5 rounds
    RENTAL = 2  # Costs $3 at end of round
    ETERNAL = 3  # Cannot be sold or destroyed


class JokerTrigger(Enum):
    """When jokers activate"""
    ON_HAND_PLAYED = "on_hand_played"
    ON_CARD_SCORED = "on_card_scored"
    ON_DISCARD = "on_discard"
    ON_BLIND_COMPLETE = "on_blind_complete"
    ON_SHOP_ENTER = "on_shop_enter"
    ON_JOKER_SOLD = "on_joker_sold"
    ON_ROUND_END = "on_round_end"
    ON_ROUND_START = "on_round_start"
    ALWAYS = "always"


@dataclass
class JokerData:
    """Data for a joker"""
    name: str
    description: str
    rarity: Rarity
    base_cost: int
    effect_type: str  # 'chips', 'mult', 'xmult', 'money', 'special'
    triggers: List[JokerTrigger]
    tier: int  # 1=simple, 2=conditional, 3=complex
    
    # Effect parameters
    chips_bonus: int = 0
    mult_bonus: int = 0
    xmult_bonus: float = 1.0
    money_bonus: int = 0
    
    # Condition checking
    requires_condition: bool = False
    persistent_state: bool = False  # Tracks state across rounds


# ============================================================================
# TIER 1 JOKERS: Simple Stat Boosts (15 jokers)
# ============================================================================

TIER1_JOKERS = [
    JokerData(
        name="Joker",
        description="+4 Mult",
        rarity=Rarity.COMMON,
        base_cost=2,
        effect_type="mult",
        triggers=[JokerTrigger.ON_HAND_PLAYED],
        tier=1,
        mult_bonus=4
    ),
    JokerData(
        name="Greedy Joker",
        description="Played cards with Diamond suit give +3 Mult when scored",
        rarity=Rarity.COMMON,
        base_cost=5,
        effect_type="mult",
        triggers=[JokerTrigger.ON_CARD_SCORED],
        tier=1,
        mult_bonus=3,
        requires_condition=True  # Must be diamond
    ),
    JokerData(
        name="Lusty Joker",
        description="Played cards with Heart suit give +3 Mult when scored",
        rarity=Rarity.COMMON,
        base_cost=5,
        effect_type="mult",
        triggers=[JokerTrigger.ON_CARD_SCORED],
        tier=1,
        mult_bonus=3,
        requires_condition=True  # Must be heart
    ),
    JokerData(
        name="Wrathful Joker",
        description="Played cards with Spade suit give +3 Mult when scored",
        rarity=Rarity.COMMON,
        base_cost=5,
        effect_type="mult",
        triggers=[JokerTrigger.ON_CARD_SCORED],
        tier=1,
        mult_bonus=3,
        requires_condition=True  # Must be spade
    ),
    JokerData(
        name="Gluttonous Joker",
        description="Played cards with Club suit give +3 Mult when scored",
        rarity=Rarity.COMMON,
        base_cost=5,
        effect_type="mult",
        triggers=[JokerTrigger.ON_CARD_SCORED],
        tier=1,
        mult_bonus=3,
        requires_condition=True  # Must be club
    ),
    JokerData(
        name="Jolly Joker",
        description="+8 Mult if played hand contains a Pair",
        rarity=Rarity.COMMON,
        base_cost=3,
        effect_type="mult",
        triggers=[JokerTrigger.ON_HAND_PLAYED],
        tier=1,
        mult_bonus=8,
        requires_condition=True  # Pair
    ),
    JokerData(
        name="Zany Joker",
        description="+12 Mult if played hand contains a Three of a Kind",
        rarity=Rarity.COMMON,
        base_cost=4,
        effect_type="mult",
        triggers=[JokerTrigger.ON_HAND_PLAYED],
        tier=1,
        mult_bonus=12,
        requires_condition=True  # Three of a Kind
    ),
    JokerData(
        name="Mad Joker",
        description="+10 Mult if played hand contains a Two Pair",
        rarity=Rarity.COMMON,
        base_cost=4,
        effect_type="mult",
        triggers=[JokerTrigger.ON_HAND_PLAYED],
        tier=1,
        mult_bonus=10,
        requires_condition=True  # Two Pair
    ),
    JokerData(
        name="Crazy Joker",
        description="+12 Mult if played hand contains a Straight",
        rarity=Rarity.COMMON,
        base_cost=4,
        effect_type="mult",
        triggers=[JokerTrigger.ON_HAND_PLAYED],
        tier=1,
        mult_bonus=12,
        requires_condition=True  # Straight
    ),
    JokerData(
        name="Droll Joker",
        description="+10 Mult if played hand contains a Flush",
        rarity=Rarity.COMMON,
        base_cost=4,
        effect_type="mult",
        triggers=[JokerTrigger.ON_HAND_PLAYED],
        tier=1,
        mult_bonus=10,
        requires_condition=True  # Flush
    ),
    JokerData(
        name="Half Joker",
        description="+20 Mult if played hand has 3 or fewer cards",
        rarity=Rarity.COMMON,
        base_cost=5,
        effect_type="mult",
        triggers=[JokerTrigger.ON_HAND_PLAYED],
        tier=1,
        mult_bonus=20,
        requires_condition=True
    ),
    JokerData(
        name="Scary Face",
        description="Played face cards give +30 Chips when scored",
        rarity=Rarity.COMMON,
        base_cost=4,
        effect_type="chips",
        triggers=[JokerTrigger.ON_CARD_SCORED],
        tier=1,
        chips_bonus=30,
        requires_condition=True  # Face card
    ),
    JokerData(
        name="Abstract Joker",
        description="+3 Mult for each Joker card you have",
        rarity=Rarity.COMMON,
        base_cost=4,
        effect_type="mult",
        triggers=[JokerTrigger.ON_HAND_PLAYED],
        tier=1,
        mult_bonus=3,  # Multiplied by joker count
        requires_condition=True
    ),
    JokerData(
        name="Stuntman",
        description="+250 Chips, -2 hand size",
        rarity=Rarity.UNCOMMON,
        base_cost=6,
        effect_type="chips",
        triggers=[JokerTrigger.ALWAYS],
        tier=1,
        chips_bonus=250
    ),
    JokerData(
        name="Misprint",
        description="+0 to +23 Mult (random each hand)",
        rarity=Rarity.COMMON,
        base_cost=4,
        effect_type="mult",
        triggers=[JokerTrigger.ON_HAND_PLAYED],
        tier=1,
        mult_bonus=11,  # Average
        requires_condition=True
    ),
]

# ============================================================================
# TIER 2 JOKERS: Conditional Effects (20 jokers)
# ============================================================================

TIER2_JOKERS = [
    JokerData(
        name="Baron",
        description="Each King held in hand gives X1.5 Mult",
        rarity=Rarity.RARE,
        base_cost=8,
        effect_type="xmult",
        triggers=[JokerTrigger.ON_HAND_PLAYED],
        tier=2,
        xmult_bonus=1.5,
        requires_condition=True  # Per King
    ),
    JokerData(
        name="Fibonacci",
        description="Each played Ace, 2, 3, 5, or 8 gives +8 Mult when scored",
        rarity=Rarity.UNCOMMON,
        base_cost=8,
        effect_type="mult",
        triggers=[JokerTrigger.ON_CARD_SCORED],
        tier=2,
        mult_bonus=8,
        requires_condition=True
    ),
    JokerData(
        name="Even Steven",
        description="Played cards with even rank give +4 Mult when scored (10, 8, 6, 4, 2)",
        rarity=Rarity.UNCOMMON,
        base_cost=5,
        effect_type="mult",
        triggers=[JokerTrigger.ON_CARD_SCORED],
        tier=2,
        mult_bonus=4,
        requires_condition=True
    ),
    JokerData(
        name="Odd Todd",
        description="Played cards with odd rank give +31 Chips when scored (A, K, Q, J, 9, 7, 5, 3)",
        rarity=Rarity.UNCOMMON,
        base_cost=5,
        effect_type="chips",
        triggers=[JokerTrigger.ON_CARD_SCORED],
        tier=2,
        chips_bonus=31,
        requires_condition=True
    ),
    JokerData(
        name="Scholar",
        description="Played Aces give +20 Chips and +4 Mult when scored",
        rarity=Rarity.UNCOMMON,
        base_cost=4,
        effect_type="mult",
        triggers=[JokerTrigger.ON_CARD_SCORED],
        tier=2,
        chips_bonus=20,
        mult_bonus=4,
        requires_condition=True
    ),
    JokerData(
        name="Photograph",
        description="First played face card gives X2 Mult",
        rarity=Rarity.UNCOMMON,
        base_cost=5,
        effect_type="xmult",
        triggers=[JokerTrigger.ON_CARD_SCORED],
        tier=2,
        xmult_bonus=2.0,
        requires_condition=True,
        persistent_state=True  # Track first face card
    ),
    JokerData(
        name="Ride the Bus",
        description="+1 Mult per consecutive hand played without a face card, resets if face card scored",
        rarity=Rarity.COMMON,
        base_cost=6,
        effect_type="mult",
        triggers=[JokerTrigger.ON_HAND_PLAYED],
        tier=2,
        mult_bonus=1,  # Per hand
        requires_condition=True,
        persistent_state=True
    ),
    JokerData(
        name="Green Joker",
        description="+1 Mult per hand played, +2 Mult per discard, resets on Mult earned",
        rarity=Rarity.UNCOMMON,
        base_cost=6,
        effect_type="mult",
        triggers=[JokerTrigger.ON_HAND_PLAYED, JokerTrigger.ON_DISCARD],
        tier=2,
        mult_bonus=1,
        requires_condition=True,
        persistent_state=True
    ),
    JokerData(
        name="Blue Joker",
        description="+2 Chips for each remaining card in deck",
        rarity=Rarity.COMMON,
        base_cost=5,
        effect_type="chips",
        triggers=[JokerTrigger.ON_HAND_PLAYED],
        tier=2,
        chips_bonus=2,  # Per card in deck
        requires_condition=True
    ),
    JokerData(
        name="Stone Joker",
        description="+25 Chips for each Stone Card in your full deck",
        rarity=Rarity.UNCOMMON,
        base_cost=6,
        effect_type="chips",
        triggers=[JokerTrigger.ALWAYS],
        tier=2,
        chips_bonus=25,  # Per stone card
        requires_condition=True
    ),
    JokerData(
        name="Steel Joker",
        description="X1.5 Mult for each Steel Card in your full deck",
        rarity=Rarity.UNCOMMON,
        base_cost=6,
        effect_type="xmult",
        triggers=[JokerTrigger.ALWAYS],
        tier=2,
        xmult_bonus=1.5,  # Per steel card
        requires_condition=True
    ),
    JokerData(
        name="Glass Joker",
        description="X2 Mult for each Glass Card in your full deck",
        rarity=Rarity.UNCOMMON,
        base_cost=6,
        effect_type="xmult",
        triggers=[JokerTrigger.ALWAYS],
        tier=2,
        xmult_bonus=2.0,  # Per glass card
        requires_condition=True
    ),
    JokerData(
        name="Loyalty Card",
        description="X4 Mult every 6 hands played, decreases by X1 every 3 hands",
        rarity=Rarity.UNCOMMON,
        base_cost=5,
        effect_type="xmult",
        triggers=[JokerTrigger.ON_HAND_PLAYED],
        tier=2,
        xmult_bonus=4.0,
        requires_condition=True,
        persistent_state=True
    ),
    JokerData(
        name="Egg",
        description="Gains $3 of sell value at end of round",
        rarity=Rarity.COMMON,
        base_cost=4,
        effect_type="money",
        triggers=[JokerTrigger.ON_ROUND_END],
        tier=2,
        money_bonus=3,
        persistent_state=True
    ),
    JokerData(
        name="Burglar",
        description="+3 Mult, when blind defeated earn $3",
        rarity=Rarity.UNCOMMON,
        base_cost=5,
        effect_type="mult",
        triggers=[JokerTrigger.ON_HAND_PLAYED, JokerTrigger.ON_BLIND_COMPLETE],
        tier=2,
        mult_bonus=3,
        money_bonus=3
    ),
    JokerData(
        name="Blackboard",
        description="X3 Mult if all cards held in hand are Spades or Clubs",
        rarity=Rarity.UNCOMMON,
        base_cost=6,
        effect_type="xmult",
        triggers=[JokerTrigger.ON_HAND_PLAYED],
        tier=2,
        xmult_bonus=3.0,
        requires_condition=True
    ),
    JokerData(
        name="Runner",
        description="+15 Chips if played hand contains a Straight, increases by +10 each round",
        rarity=Rarity.COMMON,
        base_cost=5,
        effect_type="chips",
        triggers=[JokerTrigger.ON_HAND_PLAYED],
        tier=2,
        chips_bonus=15,
        requires_condition=True,
        persistent_state=True
    ),
    JokerData(
        name="Ice Cream",
        description="+100 Chips, -5 Chips for every hand played",
        rarity=Rarity.COMMON,
        base_cost=5,
        effect_type="chips",
        triggers=[JokerTrigger.ON_HAND_PLAYED],
        tier=2,
        chips_bonus=100,
        persistent_state=True
    ),
    JokerData(
        name="DNA",
        description="If first hand of round has only 1 card, add permanent copy of it to deck",
        rarity=Rarity.RARE,
        base_cost=8,
        effect_type="special",
        triggers=[JokerTrigger.ON_HAND_PLAYED],
        tier=2,
        requires_condition=True,
        persistent_state=True
    ),
    JokerData(
        name="Splash",
        description="Every played card counts in scoring",
        rarity=Rarity.UNCOMMON,
        base_cost=3,
        effect_type="special",
        triggers=[JokerTrigger.ON_HAND_PLAYED],
        tier=2
    ),
]

# ============================================================================
# TIER 3 JOKERS: Complex/Meta Effects (15 jokers)
# ============================================================================

TIER3_JOKERS = [
    JokerData(
        name="Blueprint",
        description="Copies ability of Joker to the right",
        rarity=Rarity.RARE,
        base_cost=10,
        effect_type="special",
        triggers=[JokerTrigger.ALWAYS],
        tier=3,
        requires_condition=True
    ),
    JokerData(
        name="Brainstorm",
        description="Copies ability of leftmost Joker",
        rarity=Rarity.RARE,
        base_cost=10,
        effect_type="special",
        triggers=[JokerTrigger.ALWAYS],
        tier=3,
        requires_condition=True
    ),
    JokerData(
        name="Invisible Joker",
        description="After 2 rounds, sell this card to duplicate a random Joker (remove this card)",
        rarity=Rarity.RARE,
        base_cost=10,
        effect_type="special",
        triggers=[JokerTrigger.ON_ROUND_END],
        tier=3,
        persistent_state=True
    ),
    JokerData(
        name="Credit Card",
        description="Go up to -$20 in debt",
        rarity=Rarity.COMMON,
        base_cost=1,
        effect_type="special",
        triggers=[JokerTrigger.ALWAYS],
        tier=3
    ),
    JokerData(
        name="Vagabond",
        description="Create a Tarot card if hand played with $4 or less",
        rarity=Rarity.UNCOMMON,
        base_cost=6,
        effect_type="special",
        triggers=[JokerTrigger.ON_HAND_PLAYED],
        tier=3,
        requires_condition=True
    ),
    JokerData(
        name="Golden Ticket",
        description="Played Gold cards each give $4 when scored",
        rarity=Rarity.COMMON,
        base_cost=5,
        effect_type="money",
        triggers=[JokerTrigger.ON_CARD_SCORED],
        tier=3,
        money_bonus=4,
        requires_condition=True
    ),
    JokerData(
        name="Space Joker",
        description="1 in 4 chance to upgrade level of played poker hand",
        rarity=Rarity.UNCOMMON,
        base_cost=5,
        effect_type="special",
        triggers=[JokerTrigger.ON_HAND_PLAYED],
        tier=3,
        requires_condition=True
    ),
    JokerData(
        name="Business Card",
        description="Played face cards have a 1 in 2 chance to give $2 when scored",
        rarity=Rarity.COMMON,
        base_cost=4,
        effect_type="money",
        triggers=[JokerTrigger.ON_CARD_SCORED],
        tier=3,
        money_bonus=2,
        requires_condition=True
    ),
    JokerData(
        name="Ramen",
        description="X2 Mult, loses X0.01 Mult per card discarded",
        rarity=Rarity.UNCOMMON,
        base_cost=6,
        effect_type="xmult",
        triggers=[JokerTrigger.ON_HAND_PLAYED],
        tier=3,
        xmult_bonus=2.0,
        persistent_state=True
    ),
    JokerData(
        name="Seltzer",
        description="Retrigger all cards played for next 3 hands",
        rarity=Rarity.UNCOMMON,
        base_cost=6,
        effect_type="special",
        triggers=[JokerTrigger.ON_HAND_PLAYED],
        tier=3,
        persistent_state=True
    ),
    JokerData(
        name="Castle",
        description="+3 Chips per discarded card of each suit, suit resets when played",
        rarity=Rarity.UNCOMMON,
        base_cost=5,
        effect_type="chips",
        triggers=[JokerTrigger.ON_DISCARD, JokerTrigger.ON_HAND_PLAYED],
        tier=3,
        chips_bonus=3,
        persistent_state=True
    ),
    JokerData(
        name="Ceremonial Dagger",
        description="When Blind selected, destroy Joker to the right and permanently add its sell value to this Mult (currently +0 Mult)",
        rarity=Rarity.UNCOMMON,
        base_cost=5,
        effect_type="mult",
        triggers=[JokerTrigger.ON_ROUND_START],
        tier=3,
        mult_bonus=0,  # Grows
        persistent_state=True
    ),
    JokerData(
        name="Banner",
        description="+30 Chips for each remaining discard",
        rarity=Rarity.COMMON,
        base_cost=5,
        effect_type="chips",
        triggers=[JokerTrigger.ON_HAND_PLAYED],
        tier=3,
        chips_bonus=30,
        requires_condition=True
    ),
    JokerData(
        name="Mail-In Rebate",
        description="Earn $3 for each discarded rank, rank changes each round",
        rarity=Rarity.UNCOMMON,
        base_cost=5,
        effect_type="money",
        triggers=[JokerTrigger.ON_DISCARD],
        tier=3,
        money_bonus=3,
        persistent_state=True
    ),
    JokerData(
        name="Seeing Double",
        description="X2 Mult if played hand has a scoring Club card and a scoring card of any other suit",
        rarity=Rarity.UNCOMMON,
        base_cost=6,
        effect_type="xmult",
        triggers=[JokerTrigger.ON_HAND_PLAYED],
        tier=3,
        xmult_bonus=2.0,
        requires_condition=True
    ),
]

# ============================================================================
# TIER 4 JOKERS: Economy and Scaling (25 jokers)
# ============================================================================

TIER4_JOKERS = [
    JokerData(
        name="Sock and Buskin",
        description="Retrigger all played face cards",
        rarity=Rarity.UNCOMMON,
        base_cost=6,
        effect_type="special",
        triggers=[JokerTrigger.ON_HAND_PLAYED],
        tier=4,
        requires_condition=True
    ),
    JokerData(
        name="Swashbuckler",
        description="+1 Mult per Joker slot (including this one)",
        rarity=Rarity.COMMON,
        base_cost=4,
        effect_type="mult",
        triggers=[JokerTrigger.ON_HAND_PLAYED],
        tier=4,
        mult_bonus=1,
        requires_condition=True
    ),
    JokerData(
        name="Troubadour",
        description="+2 hand size, -1 hand per round",
        rarity=Rarity.UNCOMMON,
        base_cost=6,
        effect_type="special",
        triggers=[JokerTrigger.ALWAYS],
        tier=4
    ),
    JokerData(
        name="Certificate",
        description="At start of round, add random playing card with random seal to hand",
        rarity=Rarity.UNCOMMON,
        base_cost=6,
        effect_type="special",
        triggers=[JokerTrigger.ON_ROUND_START],
        tier=4
    ),
    JokerData(
        name="Smeared Joker",
        description="Hearts and Diamonds count as same suit, Spades and Clubs count as same suit",
        rarity=Rarity.UNCOMMON,
        base_cost=7,
        effect_type="special",
        triggers=[JokerTrigger.ALWAYS],
        tier=4
    ),
    JokerData(
        name="Throwback",
        description="X0.25 Mult for each blind skipped this run",
        rarity=Rarity.UNCOMMON,
        base_cost=6,
        effect_type="xmult",
        triggers=[JokerTrigger.ON_HAND_PLAYED],
        tier=4,
        xmult_bonus=0.25,
        persistent_state=True
    ),
    JokerData(
        name="Hanging Chad",
        description="Retrigger first played card used in scoring 2 additional times",
        rarity=Rarity.COMMON,
        base_cost=4,
        effect_type="special",
        triggers=[JokerTrigger.ON_HAND_PLAYED],
        tier=4
    ),
    JokerData(
        name="Rough Gem",
        description="Played cards with Diamond suit earn $1 when scored",
        rarity=Rarity.UNCOMMON,
        base_cost=7,
        effect_type="money",
        triggers=[JokerTrigger.ON_CARD_SCORED],
        tier=4,
        money_bonus=1,
        requires_condition=True
    ),
    JokerData(
        name="Bloodstone",
        description="1 in 3 chance for played cards with Heart suit to give X1.5 Mult when scored",
        rarity=Rarity.UNCOMMON,
        base_cost=7,
        effect_type="xmult",
        triggers=[JokerTrigger.ON_CARD_SCORED],
        tier=4,
        xmult_bonus=1.5,
        requires_condition=True
    ),
    JokerData(
        name="Arrowhead",
        description="Played cards with Spade suit give +50 Chips when scored",
        rarity=Rarity.UNCOMMON,
        base_cost=7,
        effect_type="chips",
        triggers=[JokerTrigger.ON_CARD_SCORED],
        tier=4,
        chips_bonus=50,
        requires_condition=True
    ),
    JokerData(
        name="Onyx Agate",
        description="Played cards with Club suit give +7 Mult when scored",
        rarity=Rarity.UNCOMMON,
        base_cost=7,
        effect_type="mult",
        triggers=[JokerTrigger.ON_CARD_SCORED],
        tier=4,
        mult_bonus=7,
        requires_condition=True
    ),
    JokerData(
        name="Glass Joker",
        description="Gains X0.75 Mult for every Glass Card that is destroyed",
        rarity=Rarity.UNCOMMON,
        base_cost=6,
        effect_type="xmult",
        triggers=[JokerTrigger.ALWAYS],
        tier=4,
        xmult_bonus=1.0,
        persistent_state=True
    ),
    JokerData(
        name="Showman",
        description="Joker, Tarot, Planet, and Spectral cards may appear multiple times",
        rarity=Rarity.UNCOMMON,
        base_cost=5,
        effect_type="special",
        triggers=[JokerTrigger.ALWAYS],
        tier=4
    ),
    JokerData(
        name="Flower Pot",
        description="X3 Mult if poker hand contains a Diamond, Club, Heart, and Spade card",
        rarity=Rarity.UNCOMMON,
        base_cost=6,
        effect_type="xmult",
        triggers=[JokerTrigger.ON_HAND_PLAYED],
        tier=4,
        xmult_bonus=3.0,
        requires_condition=True
    ),
    JokerData(
        name="Blueprint",
        description="Copies ability of Joker to right",
        rarity=Rarity.RARE,
        base_cost=10,
        effect_type="special",
        triggers=[JokerTrigger.ALWAYS],
        tier=4,
        requires_condition=True
    ),
    JokerData(
        name="Wee Joker",
        description="This Joker gains +8 Chips when each played 2 is scored (currently +0)",
        rarity=Rarity.RARE,
        base_cost=8,
        effect_type="chips",
        triggers=[JokerTrigger.ON_CARD_SCORED],
        tier=4,
        chips_bonus=0,
        persistent_state=True,
        requires_condition=True
    ),
    JokerData(
        name="Merry Andy",
        description="+3 discards, -1 hand size",
        rarity=Rarity.UNCOMMON,
        base_cost=7,
        effect_type="special",
        triggers=[JokerTrigger.ALWAYS],
        tier=4
    ),
    JokerData(
        name="Oops! All 6s",
        description="Doubles all listed probabilities (ex: 1 in 3 -> 2 in 3)",
        rarity=Rarity.UNCOMMON,
        base_cost=4,
        effect_type="special",
        triggers=[JokerTrigger.ALWAYS],
        tier=4
    ),
    JokerData(
        name="The Idol",
        description="Each played [Rank] of [Suit] gives X2 Mult when scored (changes each round)",
        rarity=Rarity.UNCOMMON,
        base_cost=6,
        effect_type="xmult",
        triggers=[JokerTrigger.ON_CARD_SCORED],
        tier=4,
        xmult_bonus=2.0,
        persistent_state=True,
        requires_condition=True
    ),
    JokerData(
        name="Seeing Double",
        description="X2 Mult if played hand has a Club card and a card of any other suit",
        rarity=Rarity.UNCOMMON,
        base_cost=6,
        effect_type="xmult",
        triggers=[JokerTrigger.ON_HAND_PLAYED],
        tier=4,
        xmult_bonus=2.0,
        requires_condition=True
    ),
    JokerData(
        name="Matador",
        description="Earn $8 if played hand triggers Boss Blind ability",
        rarity=Rarity.UNCOMMON,
        base_cost=7,
        effect_type="money",
        triggers=[JokerTrigger.ON_HAND_PLAYED],
        tier=4,
        money_bonus=8,
        requires_condition=True
    ),
    JokerData(
        name="Hit the Road",
        description="This Joker gains X0.5 Mult for every Jack discarded this round (currently X1 Mult)",
        rarity=Rarity.RARE,
        base_cost=8,
        effect_type="xmult",
        triggers=[JokerTrigger.ON_DISCARD],
        tier=4,
        xmult_bonus=1.0,
        persistent_state=True
    ),
    JokerData(
        name="The Duo",
        description="X2 Mult if played hand contains a Pair",
        rarity=Rarity.RARE,
        base_cost=8,
        effect_type="xmult",
        triggers=[JokerTrigger.ON_HAND_PLAYED],
        tier=4,
        xmult_bonus=2.0,
        requires_condition=True
    ),
    JokerData(
        name="The Trio",
        description="X3 Mult if played hand contains Three of a Kind",
        rarity=Rarity.RARE,
        base_cost=8,
        effect_type="xmult",
        triggers=[JokerTrigger.ON_HAND_PLAYED],
        tier=4,
        xmult_bonus=3.0,
        requires_condition=True
    ),
    JokerData(
        name="The Family",
        description="X4 Mult if played hand contains Four of a Kind",
        rarity=Rarity.RARE,
        base_cost=8,
        effect_type="xmult",
        triggers=[JokerTrigger.ON_HAND_PLAYED],
        tier=4,
        xmult_bonus=4.0,
        requires_condition=True
    ),
]

# ============================================================================
# TIER 5 JOKERS: Advanced Combos and Scaling (25 jokers)
# ============================================================================

TIER5_JOKERS = [
    JokerData(
        name="The Order",
        description="X3 Mult if played hand contains a Straight",
        rarity=Rarity.RARE,
        base_cost=8,
        effect_type="xmult",
        triggers=[JokerTrigger.ON_HAND_PLAYED],
        tier=5,
        xmult_bonus=3.0,
        requires_condition=True
    ),
    JokerData(
        name="The Tribe",
        description="X2 Mult if played hand contains a Flush",
        rarity=Rarity.RARE,
        base_cost=8,
        effect_type="xmult",
        triggers=[JokerTrigger.ON_HAND_PLAYED],
        tier=5,
        xmult_bonus=2.0,
        requires_condition=True
    ),
    JokerData(
        name="Stuntman",
        description="+250 Chips, -2 hand size",
        rarity=Rarity.UNCOMMON,
        base_cost=6,
        effect_type="chips",
        triggers=[JokerTrigger.ALWAYS],
        tier=5,
        chips_bonus=250
    ),
    JokerData(
        name="Invisible Joker",
        description="After 3 rounds, sell this card to duplicate a random Joker",
        rarity=Rarity.RARE,
        base_cost=10,
        effect_type="special",
        triggers=[JokerTrigger.ON_ROUND_END],
        tier=5,
        persistent_state=True
    ),
    JokerData(
        name="Brainstorm",
        description="Copies ability of leftmost Joker",
        rarity=Rarity.RARE,
        base_cost=10,
        effect_type="special",
        triggers=[JokerTrigger.ALWAYS],
        tier=5,
        requires_condition=True
    ),
    JokerData(
        name="Satellite",
        description="Earn $1 at end of round per unique Planet card used this run",
        rarity=Rarity.UNCOMMON,
        base_cost=6,
        effect_type="money",
        triggers=[JokerTrigger.ON_ROUND_END],
        tier=5,
        money_bonus=1,
        persistent_state=True
    ),
    JokerData(
        name="Shoot the Moon",
        description="Each Queen held in hand gives +13 Mult",
        rarity=Rarity.COMMON,
        base_cost=5,
        effect_type="mult",
        triggers=[JokerTrigger.ON_HAND_PLAYED],
        tier=5,
        mult_bonus=13,
        requires_condition=True
    ),
    JokerData(
        name="Drivers License",
        description="X3 Mult if deck has at least 16 enhanced cards",
        rarity=Rarity.RARE,
        base_cost=7,
        effect_type="xmult",
        triggers=[JokerTrigger.ON_HAND_PLAYED],
        tier=5,
        xmult_bonus=3.0,
        requires_condition=True
    ),
    JokerData(
        name="Cartomancer",
        description="Create a Tarot card when Blind is selected (must have room)",
        rarity=Rarity.UNCOMMON,
        base_cost=6,
        effect_type="special",
        triggers=[JokerTrigger.ON_ROUND_START],
        tier=5
    ),
    JokerData(
        name="Astronomer",
        description="All Planet cards and Celestial Packs in the shop are free",
        rarity=Rarity.UNCOMMON,
        base_cost=8,
        effect_type="special",
        triggers=[JokerTrigger.ALWAYS],
        tier=5
    ),
    JokerData(
        name="Burnt Joker",
        description="Upgrade the level of discarded poker hand",
        rarity=Rarity.UNCOMMON,
        base_cost=6,
        effect_type="special",
        triggers=[JokerTrigger.ON_DISCARD],
        tier=5
    ),
    JokerData(
        name="Bootstraps",
        description="+2 Mult for every $5 you have (currently +0 Mult)",
        rarity=Rarity.UNCOMMON,
        base_cost=7,
        effect_type="mult",
        triggers=[JokerTrigger.ON_HAND_PLAYED],
        tier=5,
        mult_bonus=2,
        requires_condition=True
    ),
    JokerData(
        name="Caino",
        description="This Joker gains X1 Mult when a face card is destroyed (currently X1 Mult)",
        rarity=Rarity.RARE,
        base_cost=8,
        effect_type="xmult",
        triggers=[JokerTrigger.ALWAYS],
        tier=5,
        xmult_bonus=1.0,
        persistent_state=True
    ),
    JokerData(
        name="Triboulet",
        description="Played Kings and Queens each give X2 Mult when scored",
        rarity=Rarity.LEGENDARY,
        base_cost=20,
        effect_type="xmult",
        triggers=[JokerTrigger.ON_CARD_SCORED],
        tier=5,
        xmult_bonus=2.0,
        requires_condition=True
    ),
    JokerData(
        name="Yorick",
        description="This Joker gains X1 Mult every 23 cards discarded (currently X1 Mult)",
        rarity=Rarity.UNCOMMON,
        base_cost=8,
        effect_type="xmult",
        triggers=[JokerTrigger.ON_DISCARD],
        tier=5,
        xmult_bonus=1.0,
        persistent_state=True
    ),
    JokerData(
        name="Chicot",
        description="Disables effect of every Boss Blind",
        rarity=Rarity.LEGENDARY,
        base_cost=20,
        effect_type="special",
        triggers=[JokerTrigger.ALWAYS],
        tier=5
    ),
    JokerData(
        name="Perkeo",
        description="Creates a Negative copy of 1 random consumable card in your possession at the end of shop",
        rarity=Rarity.LEGENDARY,
        base_cost=20,
        effect_type="special",
        triggers=[JokerTrigger.ON_SHOP_ENTER],
        tier=5
    ),
    JokerData(
        name="Gros Michel",
        description="+15 Mult, 1 in 6 chance this card is destroyed at end of round",
        rarity=Rarity.COMMON,
        base_cost=5,
        effect_type="mult",
        triggers=[JokerTrigger.ON_HAND_PLAYED],
        tier=5,
        mult_bonus=15,
        persistent_state=True
    ),
    JokerData(
        name="Cavendish",
        description="+3 Mult, X3 Mult when Gros Michel is destroyed",
        rarity=Rarity.COMMON,
        base_cost=5,
        effect_type="mult",
        triggers=[JokerTrigger.ON_HAND_PLAYED],
        tier=5,
        mult_bonus=3,
        persistent_state=True
    ),
    JokerData(
        name="Card Sharp",
        description="X3 Mult if played poker hand has already been played this round",
        rarity=Rarity.UNCOMMON,
        base_cost=6,
        effect_type="xmult",
        triggers=[JokerTrigger.ON_HAND_PLAYED],
        tier=5,
        xmult_bonus=3.0,
        persistent_state=True,
        requires_condition=True
    ),
    JokerData(
        name="Red Card",
        description="This Joker gains +3 Mult when any Booster Pack is skipped (currently +0 Mult)",
        rarity=Rarity.COMMON,
        base_cost=5,
        effect_type="mult",
        triggers=[JokerTrigger.ALWAYS],
        tier=5,
        mult_bonus=0,
        persistent_state=True
    ),
    JokerData(
        name="Madness",
        description="When Small or Big Blind is selected, gain X0.5 Mult and destroy a random Joker (currently X1 Mult)",
        rarity=Rarity.UNCOMMON,
        base_cost=7,
        effect_type="xmult",
        triggers=[JokerTrigger.ON_ROUND_START],
        tier=5,
        xmult_bonus=1.0,
        persistent_state=True
    ),
    JokerData(
        name="Square Joker",
        description="This Joker gains +4 Chips if played hand has exactly 4 cards (currently +0 Chips)",
        rarity=Rarity.COMMON,
        base_cost=5,
        effect_type="chips",
        triggers=[JokerTrigger.ON_HAND_PLAYED],
        tier=5,
        chips_bonus=0,
        persistent_state=True,
        requires_condition=True
    ),
    JokerData(
        name="Seance",
        description="If poker hand is a Straight Flush, create a random Spectral card (must have room)",
        rarity=Rarity.RARE,
        base_cost=6,
        effect_type="special",
        triggers=[JokerTrigger.ON_HAND_PLAYED],
        tier=5,
        requires_condition=True
    ),
    JokerData(
        name="Riff-Raff",
        description="When Blind is selected, create 2 Common Jokers (must have room)",
        rarity=Rarity.COMMON,
        base_cost=5,
        effect_type="special",
        triggers=[JokerTrigger.ON_ROUND_START],
        tier=5
    ),
]

# ============================================================================
# TIER 6 JOKERS: Legendary and Ultra-Rare (50 jokers)
# ============================================================================

TIER6_JOKERS = [
    JokerData(
        name="Vampire",
        description="This Joker gains X0.1 Mult per Enhanced card played, removes card Enhancement (currently X1 Mult)",
        rarity=Rarity.UNCOMMON,
        base_cost=8,
        effect_type="xmult",
        triggers=[JokerTrigger.ON_CARD_SCORED],
        tier=6,
        xmult_bonus=1.0,
        persistent_state=True
    ),
    JokerData(
        name="Shortcut",
        description="Allows Straights to be made with gaps of 1 rank (ex: 10 8 6 5 3)",
        rarity=Rarity.UNCOMMON,
        base_cost=7,
        effect_type="special",
        triggers=[JokerTrigger.ALWAYS],
        tier=6
    ),
    JokerData(
        name="Hologram",
        description="This Joker gains X0.25 Mult per playing card added to your deck (currently X1 Mult)",
        rarity=Rarity.UNCOMMON,
        base_cost=8,
        effect_type="xmult",
        triggers=[JokerTrigger.ALWAYS],
        tier=6,
        xmult_bonus=1.0,
        persistent_state=True
    ),
    JokerData(
        name="Vagabond",
        description="Create a Tarot card if hand is played with $4 or less",
        rarity=Rarity.UNCOMMON,
        base_cost=6,
        effect_type="special",
        triggers=[JokerTrigger.ON_HAND_PLAYED],
        tier=6,
        requires_condition=True
    ),
    JokerData(
        name="Baron",
        description="Each King held in hand gives X1.5 Mult",
        rarity=Rarity.RARE,
        base_cost=8,
        effect_type="xmult",
        triggers=[JokerTrigger.ON_HAND_PLAYED],
        tier=6,
        xmult_bonus=1.5,
        requires_condition=True
    ),
    JokerData(
        name="Cloud 9",
        description="Earn $1 for each 9 in your full deck at end of round (currently $0)",
        rarity=Rarity.UNCOMMON,
        base_cost=7,
        effect_type="money",
        triggers=[JokerTrigger.ON_ROUND_END],
        tier=6,
        money_bonus=1,
        requires_condition=True
    ),
    JokerData(
        name="Rocket",
        description="Earn $1 at end of round, increases by $2 when Boss Blind is defeated",
        rarity=Rarity.UNCOMMON,
        base_cost=6,
        effect_type="money",
        triggers=[JokerTrigger.ON_ROUND_END, JokerTrigger.ON_BLIND_COMPLETE],
        tier=6,
        money_bonus=1,
        persistent_state=True
    ),
    JokerData(
        name="Obelisk",
        description="This Joker gains X0.2 Mult per consecutive hand played without playing your most played poker hand (currently X1 Mult)",
        rarity=Rarity.UNCOMMON,
        base_cost=8,
        effect_type="xmult",
        triggers=[JokerTrigger.ON_HAND_PLAYED],
        tier=6,
        xmult_bonus=1.0,
        persistent_state=True
    ),
    JokerData(
        name="Midas Mask",
        description="All played face cards become Gold cards when scored",
        rarity=Rarity.UNCOMMON,
        base_cost=6,
        effect_type="special",
        triggers=[JokerTrigger.ON_CARD_SCORED],
        tier=6
    ),
    JokerData(
        name="Luchador",
        description="Sell this card to disable the current Boss Blind",
        rarity=Rarity.UNCOMMON,
        base_cost=5,
        effect_type="special",
        triggers=[JokerTrigger.ALWAYS],
        tier=6
    ),
    JokerData(
        name="Photograph",
        description="First played face card gives X2 Mult",
        rarity=Rarity.COMMON,
        base_cost=5,
        effect_type="xmult",
        triggers=[JokerTrigger.ON_CARD_SCORED],
        tier=6,
        xmult_bonus=2.0,
        persistent_state=True,
        requires_condition=True
    ),
    JokerData(
        name="Gift Card",
        description="Add $1 of sell value to every Joker and consumable card at end of round",
        rarity=Rarity.UNCOMMON,
        base_cost=6,
        effect_type="special",
        triggers=[JokerTrigger.ON_ROUND_END],
        tier=6
    ),
    JokerData(
        name="Turtle Bean",
        description="+5 hand size, reduces by 1 each round",
        rarity=Rarity.UNCOMMON,
        base_cost=5,
        effect_type="special",
        triggers=[JokerTrigger.ON_ROUND_END],
        tier=6,
        persistent_state=True
    ),
    JokerData(
        name="Erosion",
        description="+4 Mult for each card below 52 in your full deck (currently +0 Mult)",
        rarity=Rarity.UNCOMMON,
        base_cost=6,
        effect_type="mult",
        triggers=[JokerTrigger.ON_HAND_PLAYED],
        tier=6,
        mult_bonus=4,
        requires_condition=True
    ),
    JokerData(
        name="Reserved Parking",
        description="Each face card held in hand has a 1 in 3 chance to give $1",
        rarity=Rarity.UNCOMMON,
        base_cost=6,
        effect_type="money",
        triggers=[JokerTrigger.ON_HAND_PLAYED],
        tier=6,
        money_bonus=1,
        requires_condition=True
    ),
    JokerData(
        name="Mail-In Rebate",
        description="Earn $3 for each discarded rank, rank changes every round",
        rarity=Rarity.UNCOMMON,
        base_cost=5,
        effect_type="money",
        triggers=[JokerTrigger.ON_DISCARD],
        tier=6,
        money_bonus=3,
        persistent_state=True
    ),
    JokerData(
        name="To the Moon",
        description="Earn $1 of extra interest for every $5 you have at end of round",
        rarity=Rarity.UNCOMMON,
        base_cost=5,
        effect_type="money",
        triggers=[JokerTrigger.ON_ROUND_END],
        tier=6,
        money_bonus=1
    ),
    JokerData(
        name="Hallucination",
        description="1 in 4 chance to create a Tarot card when any Booster Pack is opened (must have room)",
        rarity=Rarity.COMMON,
        base_cost=4,
        effect_type="special",
        triggers=[JokerTrigger.ALWAYS],
        tier=6
    ),
    JokerData(
        name="Fortune Teller",
        description="+1 Mult per Tarot card used this run (currently +0 Mult)",
        rarity=Rarity.COMMON,
        base_cost=5,
        effect_type="mult",
        triggers=[JokerTrigger.ON_HAND_PLAYED],
        tier=6,
        mult_bonus=1,
        persistent_state=True
    ),
    JokerData(
        name="Juggler",
        description="+1 hand size",
        rarity=Rarity.COMMON,
        base_cost=5,
        effect_type="special",
        triggers=[JokerTrigger.ALWAYS],
        tier=6
    ),
    JokerData(
        name="Drunkard",
        description="+1 discard",
        rarity=Rarity.COMMON,
        base_cost=5,
        effect_type="special",
        triggers=[JokerTrigger.ALWAYS],
        tier=6
    ),
    JokerData(
        name="Stone Joker",
        description="+25 Chips for each Stone Card in your full deck (currently +0 Chips)",
        rarity=Rarity.UNCOMMON,
        base_cost=6,
        effect_type="chips",
        triggers=[JokerTrigger.ALWAYS],
        tier=6,
        chips_bonus=25,
        requires_condition=True
    ),
    JokerData(
        name="Golden Joker",
        description="Earn $4 at end of round",
        rarity=Rarity.COMMON,
        base_cost=6,
        effect_type="money",
        triggers=[JokerTrigger.ON_ROUND_END],
        tier=6,
        money_bonus=4
    ),
    JokerData(
        name="Lucky Cat",
        description="This Joker gains X0.25 Mult every time a Lucky card triggers successfully (currently X1 Mult)",
        rarity=Rarity.UNCOMMON,
        base_cost=6,
        effect_type="xmult",
        triggers=[JokerTrigger.ALWAYS],
        tier=6,
        xmult_bonus=1.0,
        persistent_state=True
    ),
    JokerData(
        name="Baseball Card",
        description="Uncommon Jokers each give X1.5 Mult",
        rarity=Rarity.UNCOMMON,
        base_cost=4,
        effect_type="xmult",
        triggers=[JokerTrigger.ON_HAND_PLAYED],
        tier=6,
        xmult_bonus=1.5,
        requires_condition=True
    ),
    JokerData(
        name="Bull",
        description="+2 Chips for every $1 you have (currently +0 Chips)",
        rarity=Rarity.UNCOMMON,
        base_cost=7,
        effect_type="chips",
        triggers=[JokerTrigger.ON_HAND_PLAYED],
        tier=6,
        chips_bonus=2,
        requires_condition=True
    ),
    JokerData(
        name="Diet Cola",
        description="Sell this card to create a free Double Tag",
        rarity=Rarity.UNCOMMON,
        base_cost=6,
        effect_type="special",
        triggers=[JokerTrigger.ALWAYS],
        tier=6
    ),
    JokerData(
        name="Trading Card",
        description="If first discard of round has only 1 card, destroy it and earn $3",
        rarity=Rarity.UNCOMMON,
        base_cost=5,
        effect_type="money",
        triggers=[JokerTrigger.ON_DISCARD],
        tier=6,
        money_bonus=3,
        persistent_state=True
    ),
    JokerData(
        name="Flash Card",
        description="This Joker gains +2 Mult per reroll in the shop (currently +0 Mult)",
        rarity=Rarity.UNCOMMON,
        base_cost=5,
        effect_type="mult",
        triggers=[JokerTrigger.ALWAYS],
        tier=6,
        mult_bonus=2,
        persistent_state=True
    ),
    JokerData(
        name="Popcorn",
        description="+20 Mult, -4 Mult per round played",
        rarity=Rarity.COMMON,
        base_cost=5,
        effect_type="mult",
        triggers=[JokerTrigger.ON_HAND_PLAYED],
        tier=6,
        mult_bonus=20,
        persistent_state=True
    ),
    JokerData(
        name="Spare Trousers",
        description="This Joker gains +2 Mult if played hand contains a Two Pair (currently +0 Mult)",
        rarity=Rarity.UNCOMMON,
        base_cost=6,
        effect_type="mult",
        triggers=[JokerTrigger.ON_HAND_PLAYED],
        tier=6,
        mult_bonus=2,
        persistent_state=True,
        requires_condition=True
    ),
    JokerData(
        name="Ancient Joker",
        description="Each played card with [Suit] suit gives X1.5 Mult when scored (suit changes at end of round)",
        rarity=Rarity.RARE,
        base_cost=8,
        effect_type="xmult",
        triggers=[JokerTrigger.ON_CARD_SCORED],
        tier=6,
        xmult_bonus=1.5,
        persistent_state=True,
        requires_condition=True
    ),
    JokerData(
        name="Ramen",
        description="X2 Mult, loses X0.01 Mult per card discarded",
        rarity=Rarity.UNCOMMON,
        base_cost=6,
        effect_type="xmult",
        triggers=[JokerTrigger.ON_HAND_PLAYED],
        tier=6,
        xmult_bonus=2.0,
        persistent_state=True
    ),
    JokerData(
        name="Walkie Talkie",
        description="Each played 10 or 4 gives +10 Chips and +4 Mult when scored",
        rarity=Rarity.COMMON,
        base_cost=4,
        effect_type="mult",
        triggers=[JokerTrigger.ON_CARD_SCORED],
        tier=6,
        chips_bonus=10,
        mult_bonus=4,
        requires_condition=True
    ),
    JokerData(
        name="Seltzer",
        description="Retrigger all cards played for next 3 hands",
        rarity=Rarity.UNCOMMON,
        base_cost=6,
        effect_type="special",
        triggers=[JokerTrigger.ON_HAND_PLAYED],
        tier=6,
        persistent_state=True
    ),
    JokerData(
        name="Castle",
        description="This Joker gains +3 Chips per discarded [Suit] card, suit resets every round (currently +0 Chips)",
        rarity=Rarity.UNCOMMON,
        base_cost=5,
        effect_type="chips",
        triggers=[JokerTrigger.ON_DISCARD],
        tier=6,
        chips_bonus=3,
        persistent_state=True
    ),
    JokerData(
        name="Smiley Face",
        description="Played face cards give +4 Mult when scored",
        rarity=Rarity.COMMON,
        base_cost=4,
        effect_type="mult",
        triggers=[JokerTrigger.ON_CARD_SCORED],
        tier=6,
        mult_bonus=4,
        requires_condition=True
    ),
    JokerData(
        name="Campfire",
        description="This Joker gains X0.5 Mult for each card sold, resets when Boss Blind is defeated (currently X1 Mult)",
        rarity=Rarity.RARE,
        base_cost=8,
        effect_type="xmult",
        triggers=[JokerTrigger.ON_HAND_PLAYED],
        tier=6,
        xmult_bonus=1.0,
        persistent_state=True
    ),
    JokerData(
        name="Golden Ticket",
        description="Played Gold cards each give $4 when scored",
        rarity=Rarity.COMMON,
        base_cost=5,
        effect_type="money",
        triggers=[JokerTrigger.ON_CARD_SCORED],
        tier=6,
        money_bonus=4,
        requires_condition=True
    ),
    JokerData(
        name="Mr. Bones",
        description="Prevents Death if chips scored are at least 25% of required chips (removed after triggering)",
        rarity=Rarity.UNCOMMON,
        base_cost=5,
        effect_type="special",
        triggers=[JokerTrigger.ALWAYS],
        tier=6,
        persistent_state=True
    ),
    JokerData(
        name="Acrobat",
        description="+3 hands, lose this card at end of round",
        rarity=Rarity.COMMON,
        base_cost=6,
        effect_type="special",
        triggers=[JokerTrigger.ALWAYS],
        tier=6,
        persistent_state=True
    ),
    JokerData(
        name="Sock and Buskin",
        description="Retrigger all played face cards",
        rarity=Rarity.UNCOMMON,
        base_cost=6,
        effect_type="special",
        triggers=[JokerTrigger.ON_HAND_PLAYED],
        tier=6
    ),
    JokerData(
        name="Swashbuckler",
        description="+1 Mult per Joker slot, adds sell value of other Jokers when sold",
        rarity=Rarity.COMMON,
        base_cost=4,
        effect_type="mult",
        triggers=[JokerTrigger.ON_HAND_PLAYED],
        tier=6,
        mult_bonus=1,
        requires_condition=True
    ),
    JokerData(
        name="Troubadour",
        description="+2 hand size, -1 hand per round",
        rarity=Rarity.UNCOMMON,
        base_cost=6,
        effect_type="special",
        triggers=[JokerTrigger.ALWAYS],
        tier=6
    ),
    JokerData(
        name="Certificate",
        description="At start of round, add random playing card with random Seal to your hand",
        rarity=Rarity.UNCOMMON,
        base_cost=6,
        effect_type="special",
        triggers=[JokerTrigger.ON_ROUND_START],
        tier=6
    ),
    JokerData(
        name="Smeared Joker",
        description="Hearts and Diamonds count as same suit, Spades and Clubs count as same suit",
        rarity=Rarity.UNCOMMON,
        base_cost=7,
        effect_type="special",
        triggers=[JokerTrigger.ALWAYS],
        tier=6
    ),
    JokerData(
        name="Throwback",
        description="X0.25 Mult for each Blind skipped this run",
        rarity=Rarity.UNCOMMON,
        base_cost=6,
        effect_type="xmult",
        triggers=[JokerTrigger.ON_HAND_PLAYED],
        tier=6,
        xmult_bonus=0.25,
        persistent_state=True
    ),
    JokerData(
        name="Hanging Chad",
        description="Retrigger first played card 2 additional times",
        rarity=Rarity.COMMON,
        base_cost=4,
        effect_type="special",
        triggers=[JokerTrigger.ON_HAND_PLAYED],
        tier=6
    ),
    JokerData(
        name="Rough Gem",
        description="Played cards with Diamond suit earn $1 when scored",
        rarity=Rarity.UNCOMMON,
        base_cost=7,
        effect_type="money",
        triggers=[JokerTrigger.ON_CARD_SCORED],
        tier=6,
        money_bonus=1,
        requires_condition=True
    ),
    JokerData(
        name="Supernova",
        description="Adds the number of times poker hand has been played this run to Mult",
        rarity=Rarity.UNCOMMON,
        base_cost=5,
        effect_type="mult",
        triggers=[JokerTrigger.ON_HAND_PLAYED],
        tier=6,
        mult_bonus=1,
        persistent_state=True,
        requires_condition=True
    ),
]

# Combine all jokers
ALL_JOKERS = TIER1_JOKERS + TIER2_JOKERS + TIER3_JOKERS + TIER4_JOKERS + TIER5_JOKERS + TIER6_JOKERS


# ============================================================================
# TAROT CARDS (22 total)
# ============================================================================

@dataclass
class TarotData:
    """Data for a tarot card"""
    name: str
    description: str
    effect_type: str
    target_required: bool = False  # Needs to target specific cards


TAROT_CARDS = [
    TarotData("The Fool", "Creates the last Tarot or Planet card used during this run", "create_last_consumable", False),
    TarotData("The Magician", "Enhances 2 selected cards to Lucky Cards", "enhance_to_lucky", True),
    TarotData("The High Priestess", "Creates up to 2 random Planet cards", "create_planet", False),
    TarotData("The Empress", "Enhances 2 selected cards to Mult Cards", "enhance_to_mult", True),
    TarotData("The Emperor", "Creates up to 2 random Tarot cards", "create_tarot", False),
    TarotData("The Hierophant", "Enhances 2 selected cards to Bonus Cards", "enhance_to_bonus", True),
    TarotData("The Lovers", "Enhances 1 selected card into a Wild Card", "enhance_to_wild", True),
    TarotData("The Chariot", "Enhances 1 selected card into a Steel Card", "enhance_to_steel", True),
    TarotData("Justice", "Enhances 1 selected card into a Glass Card", "enhance_to_glass", True),
    TarotData("The Hermit", "Doubles money (max of $20)", "double_money", False),
    TarotData("The Wheel of Fortune", "1 in 4 chance to add Foil, Holographic, or Polychrome edition to a random Joker", "add_edition_joker", False),
    TarotData("Strength", "Increases rank of up to 2 selected cards by 1", "increase_rank", True),
    TarotData("The Hanged Man", "Destroys up to 2 selected cards", "destroy_cards", True),
    TarotData("Death", "Select 2 cards, convert left card into right card", "convert_cards", True),
    TarotData("Temperance", "Gives the total sell value of all current Jokers (max of $50)", "joker_sell_value", False),
    TarotData("The Devil", "Enhances 1 selected card into a Gold Card", "enhance_to_gold", True),
    TarotData("The Tower", "Enhances 1 selected card into a Stone Card", "enhance_to_stone", True),
    TarotData("The Star", "Converts up to 3 selected cards to Diamonds", "convert_to_diamonds", True),
    TarotData("The Moon", "Converts up to 3 selected cards to Clubs", "convert_to_clubs", True),
    TarotData("The Sun", "Converts up to 3 selected cards to Hearts", "convert_to_hearts", True),
    TarotData("Judgement", "Creates a random Joker card (must have room)", "create_joker", False),
    TarotData("The World", "Converts up to 3 selected cards to Spades", "convert_to_spades", True),
]


# ============================================================================
# PLANET CARDS (11 total - one per hand type)
# ============================================================================

@dataclass
class PlanetData:
    """Data for a planet card"""
    name: str
    hand_type: str  # Which poker hand it levels up
    description: str
    chip_increase: int = 30
    mult_increase: int = 3


PLANET_CARDS = [
    PlanetData("Pluto", "HIGH_CARD", "Level up High Card (+30 chips, +3 mult)"),
    PlanetData("Mercury", "PAIR", "Level up Pair (+30 chips, +3 mult)"),
    PlanetData("Uranus", "TWO_PAIR", "Level up Two Pair (+30 chips, +3 mult)"),
    PlanetData("Venus", "THREE_OF_A_KIND", "Level up Three of a Kind (+30 chips, +3 mult)"),
    PlanetData("Saturn", "STRAIGHT", "Level up Straight (+30 chips, +3 mult)"),
    PlanetData("Jupiter", "FLUSH", "Level up Flush (+30 chips, +3 mult)"),
    PlanetData("Earth", "FULL_HOUSE", "Level up Full House (+30 chips, +3 mult)"),
    PlanetData("Mars", "FOUR_OF_A_KIND", "Level up Four of a Kind (+30 chips, +3 mult)"),
    PlanetData("Neptune", "STRAIGHT_FLUSH", "Level up Straight Flush (+30 chips, +3 mult)"),
    PlanetData("Planet X", "FLUSH_FIVE", "Level up Flush Five (+30 chips, +3 mult)"),
    PlanetData("Ceres", "FLUSH_HOUSE", "Level up Flush House (+30 chips, +3 mult)"),
]


# ============================================================================
# SPECTRAL CARDS (15 total - powerful single-use)
# ============================================================================

@dataclass
class SpectralData:
    """Data for a spectral card"""
    name: str
    description: str
    effect_type: str
    target_required: bool = False


SPECTRAL_CARDS = [
    SpectralData("Familiar", "Destroy 1 random card in your hand, add 3 random Enhanced face cards to your hand", "destroy_add_faces", False),
    SpectralData("Grim", "Destroy 1 random card in your hand, add 2 random Enhanced Aces to your hand", "destroy_add_aces", False),
    SpectralData("Incantation", "Destroy 1 random card in your hand, add 4 random Enhanced numbered cards (2-10) to your hand", "destroy_add_numbered", False),
    SpectralData("Talisman", "Add a Gold Seal to 1 selected card in your hand", "add_gold_seal", True),
    SpectralData("Aura", "Add a Foil, Holographic, or Polychrome effect to 1 selected card in your hand", "add_edition", True),
    SpectralData("Wraith", "Creates a random Rare Joker, sets money to $0", "create_rare_joker", False),
    SpectralData("Sigil", "Converts all cards in hand to a single random suit", "convert_all_suit", False),
    SpectralData("Ouija", "Converts all cards in hand to a single random rank", "convert_all_rank", False),
    SpectralData("Ectoplasm", "Add Negative to a random Joker, -1 hand per round", "add_negative", False),
    SpectralData("Immolate", "Destroys 5 random cards in hand, gain $20", "destroy_5_earn_20", False),
    SpectralData("Ankh", "Create a copy of a random Joker, destroy all other Jokers", "copy_destroy_jokers", False),
    SpectralData("Deja Vu", "Add a Red Seal to 1 selected card in your hand", "add_red_seal", True),
    SpectralData("Hex", "Add Polychrome to a random Joker, destroy all other Jokers", "poly_destroy_jokers", False),
    SpectralData("Trance", "Add a Blue Seal to 1 selected card in your hand", "add_blue_seal", True),
    SpectralData("Medium", "Add a Purple Seal to 1 selected card in your hand", "add_purple_seal", True),
]


# ============================================================================
# VOUCHERS (32 total - permanent upgrades)
# ============================================================================

@dataclass
class VoucherData:
    """Data for a voucher"""
    name: str
    description: str
    base_cost: int
    tier: int  # 1 or 2
    effect_type: str
    tier1_prereq: Optional[str] = None  # For tier 2 vouchers


VOUCHERS = [
    # Tier 1
    VoucherData("Overstock", "+1 card slot available in shop", 10, 1, "shop_slots"),
    VoucherData("Clearance Sale", "All cards and packs in shop are 25% off", 10, 1, "shop_discount"),
    VoucherData("Hone", "Foil, Holographic, Polychrome cards appear 2x more frequently", 10, 1, "edition_rate"),
    VoucherData("Reroll Surplus", "Rerolls cost $2 less", 10, 1, "reroll_cost"),
    VoucherData("Crystal Ball", "+1 consumable slot", 10, 1, "consumable_slots"),
    VoucherData("Telescope", "Celestial Packs always contain the Planet card for your most played poker hand", 10, 1, "planet_selection"),
    VoucherData("Grabber", "Permanently gain +1 hand per round", 10, 1, "hands_per_round"),
    VoucherData("Wasteful", "+1 discard per round", 10, 1, "discards_per_round"),
    VoucherData("Tarot Merchant", "Tarot cards appear 2x more frequently in the shop", 10, 1, "tarot_rate"),
    VoucherData("Planet Merchant", "Planet cards appear 2x more frequently in the shop", 10, 1, "planet_rate"),
    VoucherData("Seed Money", "Raise the cap on interest earned to $10 per round", 10, 1, "interest_cap"),
    VoucherData("Blank", "Does nothing?", 10, 1, "mystery"),
    VoucherData("Magic Trick", "+2 hand size", 10, 1, "hand_size"),
    VoucherData("Good Omen", "+1 consumable slot", 10, 1, "consumable_slots_2"),
    VoucherData("Observatory", "Planet cards in your consumable slots give X1.5 Mult for their specified hand", 10, 1, "planet_xmult"),
    VoucherData("Antenna", "Ante gives +1 hand", 10, 1, "ante_bonus_hands"),
    
    # Tier 2 (upgrades of tier 1)
    VoucherData("Overstock Plus", "+1 card slot available in shop", 10, 2, "shop_slots", "Overstock"),
    VoucherData("Liquidation", "All cards and packs in shop are 50% off", 10, 2, "shop_discount", "Clearance Sale"),
    VoucherData("Glow Up", "Foil, Holographic, Polychrome cards appear 4x more frequently", 10, 2, "edition_rate", "Hone"),
    VoucherData("Reroll Glut", "Rerolls cost $2 less", 10, 2, "reroll_cost", "Reroll Surplus"),
    VoucherData("Omen Globe", "+1 consumable slot", 10, 2, "consumable_slots", "Crystal Ball"),
    VoucherData("Observatory", "Planet cards give X1.5 Mult for hand type", 10, 2, "planet_selection", "Telescope"),
    VoucherData("Nacho Tong", "Permanently gain +1 hand per round", 10, 2, "hands_per_round", "Grabber"),
    VoucherData("Recyclomancy", "+1 discard per round", 10, 2, "discards_per_round", "Wasteful"),
    VoucherData("Tarot Tycoon", "Tarot cards appear 4x more frequently in the shop", 10, 2, "tarot_rate", "Tarot Merchant"),
    VoucherData("Planet Tycoon", "Planet cards appear 4x more frequently in the shop", 10, 2, "planet_rate", "Planet Merchant"),
    VoucherData("Money Tree", "Raise the cap on interest earned to $25 per round", 10, 2, "interest_cap", "Seed Money"),
    VoucherData("Antimatter", "?", 10, 2, "mystery", "Blank"),
    VoucherData("Illusion", "+2 hand size", 10, 2, "hand_size", "Magic Trick"),
    VoucherData("Sixth Sense", "+1 consumable slot", 10, 2, "consumable_slots_2", "Good Omen"),
    VoucherData("Planetarium", "Planet cards give X1.5 Mult", 10, 2, "planet_xmult", "Observatory"),
    VoucherData("Satellite", "Ante gives +1 hand", 10, 2, "ante_bonus_hands", "Antenna"),
]


# ============================================================================
# BOOSTER PACKS
# ============================================================================

@dataclass
class PackData:
    """Data for a booster pack"""
    name: str
    cost: int
    contents_count: int
    content_type: str  # 'tarot', 'planet', 'spectral', 'playing_card', 'joker'
    description: str


PACKS = [
    PackData("Arcana Pack", 4, 2, "tarot", "Choose 1 of 2 Tarot cards"),
    PackData("Celestial Pack", 4, 2, "planet", "Choose 1 of 2 Planet cards"),
    PackData("Spectral Pack", 4, 2, "spectral", "Choose 1 of 2 Spectral cards"),
    PackData("Standard Pack", 4, 4, "playing_card", "Choose 1 of 4 playing cards (may have enhancements)"),
    PackData("Buffoon Pack", 4, 2, "joker", "Choose 1 of 2 Joker cards"),
]


# ============================================================================
# PRICING TABLES
# ============================================================================

def get_joker_price(joker_data: JokerData) -> int:
    """Get shop price for a joker based on rarity"""
    base_prices = {
        Rarity.COMMON: 3,
        Rarity.UNCOMMON: 5,
        Rarity.RARE: 7,
        Rarity.LEGENDARY: 20,
    }
    return base_prices.get(joker_data.rarity, 5)


def get_pack_price(pack_name: str) -> int:
    """Get price for a booster pack"""
    return 4  # All packs cost $4


def get_card_price(has_enhancement: bool, has_edition: bool) -> int:
    """Get price for a playing card"""
    price = 2
    if has_enhancement:
        price += 2
    if has_edition:
        price += 3
    return price


def get_voucher_price(tier: int) -> int:
    """Get price for a voucher"""
    return 10  # All vouchers cost $10


# ============================================================================
# LOOKUP FUNCTIONS
# ============================================================================

def get_joker_by_name(name: str) -> Optional[JokerData]:
    """Get joker data by name"""
    for joker in ALL_JOKERS:
        if joker.name == name:
            return joker
    return None


def get_random_jokers_by_tier(tier: int, count: int) -> List[JokerData]:
    """Get random jokers from a specific tier"""
    import random
    tier_jokers = [j for j in ALL_JOKERS if j.tier == tier]
    return random.sample(tier_jokers, min(count, len(tier_jokers)))


def get_random_jokers_by_rarity(rarity: Rarity, count: int) -> List[JokerData]:
    """Get random jokers of a specific rarity"""
    import random
    rarity_jokers = [j for j in ALL_JOKERS if j.rarity == rarity]
    return random.sample(rarity_jokers, min(count, len(rarity_jokers)))


def get_tarot_by_name(name: str) -> Optional[TarotData]:
    """Get tarot card data by name"""
    for tarot in TAROT_CARDS:
        if tarot.name == name:
            return tarot
    return None


def get_planet_by_hand_type(hand_type: str) -> Optional[PlanetData]:
    """Get planet card for a specific hand type"""
    for planet in PLANET_CARDS:
        if planet.hand_type == hand_type:
            return planet
    return None


def get_spectral_by_name(name: str) -> Optional[SpectralData]:
    """Get spectral card data by name"""
    for spectral in SPECTRAL_CARDS:
        if spectral.name == name:
            return spectral
    return None


def get_voucher_by_name(name: str) -> Optional[VoucherData]:
    """Get voucher data by name"""
    for voucher in VOUCHERS:
        if voucher.name == name:
            return voucher
    return None


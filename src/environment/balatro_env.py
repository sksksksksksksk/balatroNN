"""
Balatro Game Environment for Reinforcement Learning

This module implements a Gymnasium-compatible environment for Balatro,
including game state representation, action space, and reward calculation.
"""

import gymnasium as gym
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from enum import Enum
from dataclasses import dataclass, field
from collections import Counter


class Suit(Enum):
    """Card suits in Balatro"""
    HEARTS = 0
    DIAMONDS = 1
    CLUBS = 2
    SPADES = 3


class Rank(Enum):
    """Card ranks in Balatro"""
    TWO = 2
    THREE = 3
    FOUR = 4
    FIVE = 5
    SIX = 6
    SEVEN = 7
    EIGHT = 8
    NINE = 9
    TEN = 10
    JACK = 11
    QUEEN = 12
    KING = 13
    ACE = 14


class Enhancement(Enum):
    """Card enhancements"""
    NONE = 0
    BONUS = 1      # +30 chips
    MULT = 2       # +4 mult
    WILD = 3       # Can be any suit
    GLASS = 4      # x2 mult, 1/4 chance to destroy
    STEEL = 5      # x1.5 mult while in hand
    STONE = 6      # +50 chips, no rank/suit
    GOLD = 7       # +$3 when held in hand at end of round
    LUCKY = 8      # 1/5 chance for +20 mult, 1/15 for $20


class Edition(Enum):
    """Card editions (foil effects)"""
    NONE = 0
    FOIL = 1       # +50 chips
    HOLOGRAPHIC = 2  # +10 mult
    POLYCHROME = 3   # x1.5 mult


class Seal(Enum):
    """Card seals"""
    NONE = 0
    GOLD = 1       # Earn $3 when card is played and scores
    RED = 2        # Retrigger card
    BLUE = 3       # Creates Planet card if held in hand at end of round
    PURPLE = 4     # Creates Tarot card when discarded


class HandType(Enum):
    """Poker hand types"""
    HIGH_CARD = 0
    PAIR = 1
    TWO_PAIR = 2
    THREE_OF_A_KIND = 3
    STRAIGHT = 4
    FLUSH = 5
    FULL_HOUSE = 6
    FOUR_OF_A_KIND = 7
    STRAIGHT_FLUSH = 8
    FLUSH_FIVE = 9
    FLUSH_HOUSE = 10
    FIVE_OF_A_KIND = 11


@dataclass
class Card:
    """Represents a single card in Balatro"""
    rank: Rank
    suit: Suit
    enhancement: Enhancement = Enhancement.NONE
    edition: Edition = Edition.NONE
    seal: Seal = Seal.NONE
    debuffed: bool = False  # Some blinds debuff cards
    
    @property
    def base_chips(self) -> int:
        """Base chip value of the card"""
        if self.enhancement == Enhancement.STONE:
            return 50
        if self.rank.value <= 10:
            return self.rank.value
        elif self.rank == Rank.JACK:
            return 10
        elif self.rank == Rank.QUEEN:
            return 10
        elif self.rank == Rank.KING:
            return 10
        elif self.rank == Rank.ACE:
            return 11
        return 0
    
    def to_vector(self) -> np.ndarray:
        """Convert card to feature vector"""
        vec = np.zeros(32)  # 32-dimensional card representation
        vec[0] = self.rank.value / 14.0  # Normalized rank
        vec[1 + self.suit.value] = 1.0  # One-hot suit (4 dims)
        vec[5 + self.enhancement.value] = 1.0  # One-hot enhancement (9 dims)
        vec[14 + self.edition.value] = 1.0  # One-hot edition (4 dims)
        vec[18 + self.seal.value] = 1.0  # One-hot seal (5 dims)
        vec[23] = float(self.debuffed)
        vec[24] = self.base_chips / 50.0  # Normalized chips
        return vec


@dataclass
class Joker:
    """Represents a joker card with special effects"""
    name: str
    rarity: str  # "common", "uncommon", "rare", "legendary"
    effect: str  # Description of the effect
    chips_bonus: int = 0
    mult_bonus: int = 0
    xmult_bonus: float = 1.0
    triggers: List[str] = field(default_factory=list)  # When the joker activates
    
    def to_vector(self) -> np.ndarray:
        """Convert joker to feature vector"""
        vec = np.zeros(64)  # 64-dimensional joker representation
        # For now, simplified representation
        # In practice, we'd need embeddings for each unique joker
        rarity_map = {"common": 0, "uncommon": 1, "rare": 2, "legendary": 3}
        vec[0] = rarity_map.get(self.rarity, 0) / 3.0
        vec[1] = np.clip(self.chips_bonus / 100.0, 0, 10)
        vec[2] = np.clip(self.mult_bonus / 20.0, 0, 10)
        vec[3] = np.clip(self.xmult_bonus / 5.0, 0, 10)
        return vec


@dataclass
class Blind:
    """Represents a blind (opponent) in Balatro"""
    name: str
    ante: int  # Which ante (1-8)
    type: str  # "small", "big", "boss"
    chip_requirement: int
    reward: int  # Money earned
    modifier: Optional[str] = None  # Special boss blind modifiers
    
    def to_vector(self) -> np.ndarray:
        """Convert blind to feature vector"""
        vec = np.zeros(16)
        vec[0] = self.ante / 8.0
        type_map = {"small": 0, "big": 1, "boss": 2}
        vec[1 + type_map.get(self.type, 0)] = 1.0
        vec[4] = np.log1p(self.chip_requirement) / 20.0
        vec[5] = self.reward / 100.0
        return vec


@dataclass
class GameState:
    """Complete game state for Balatro"""
    # Current state
    hand: List[Card] = field(default_factory=list)
    deck: List[Card] = field(default_factory=list)
    jokers: List[Joker] = field(default_factory=list)
    consumables: List[Any] = field(default_factory=list)  # Tarot/Planet cards
    
    # Blind information
    current_blind: Optional[Blind] = None
    chips_scored: int = 0
    hands_remaining: int = 4
    discards_remaining: int = 3
    
    # Meta information
    ante: int = 1
    money: int = 4
    round_number: int = 1
    
    # Hand level information (upgrades from planet cards)
    hand_levels: Dict[HandType, Tuple[int, int]] = field(default_factory=lambda: {
        hand_type: (5, 1) for hand_type in HandType  # (base_chips, base_mult)
    })
    
    # Shop state (if in shop)
    in_shop: bool = False
    shop_items: List[Any] = field(default_factory=list)
    shop_rerolls_available: int = 5
    
    def to_observation(self) -> Dict[str, np.ndarray]:
        """Convert game state to observation dict for neural network"""
        obs = {}
        
        # Hand cards (max 8 cards, pad if necessary)
        hand_vectors = np.zeros((8, 32))
        for i, card in enumerate(self.hand[:8]):
            hand_vectors[i] = card.to_vector()
        obs["hand"] = hand_vectors
        
        # Jokers (max 5, pad if necessary)
        joker_vectors = np.zeros((5, 64))
        for i, joker in enumerate(self.jokers[:5]):
            joker_vectors[i] = joker.to_vector()
        obs["jokers"] = joker_vectors
        
        # Blind information
        if self.current_blind:
            obs["blind"] = self.current_blind.to_vector()
        else:
            obs["blind"] = np.zeros(16)
        
        # Scalar game state
        scalar_state = np.zeros(32)
        scalar_state[0] = self.chips_scored / 10000.0  # Normalized
        scalar_state[1] = (self.current_blind.chip_requirement if self.current_blind else 0) / 10000.0
        scalar_state[2] = self.hands_remaining / 4.0
        scalar_state[3] = self.discards_remaining / 3.0
        scalar_state[4] = self.ante / 8.0
        scalar_state[5] = self.money / 100.0
        scalar_state[6] = len(self.deck) / 52.0
        scalar_state[7] = len(self.jokers) / 5.0
        scalar_state[8] = float(self.in_shop)
        obs["scalar"] = scalar_state
        
        return obs


class BalatroEnv(gym.Env):
    """
    Gymnasium environment for Balatro
    
    This is a simplified simulation of Balatro game mechanics.
    For production, this would interface with actual game state.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__()
        
        self.config = config or {}
        self.max_steps = self.config.get("max_steps", 1000)
        self.starting_deck_size = self.config.get("starting_deck_size", 52)
        
        # Action space: Complex multi-discrete space
        # For simplicity, we'll use a flattened action space
        # Actions: [play_hand_mask (8 bits), discard_mask (8 bits), shop_actions, special_actions]
        self.action_space = gym.spaces.Dict({
            "action_type": gym.spaces.Discrete(5),  # 0: play, 1: discard, 2: shop_buy, 3: skip, 4: reroll
            "card_selection": gym.spaces.MultiBinary(8),  # Which cards to select
            "shop_selection": gym.spaces.Discrete(7),  # Which shop item to buy (0-6)
        })
        
        # Observation space
        self.observation_space = gym.spaces.Dict({
            "hand": gym.spaces.Box(low=0, high=1, shape=(8, 32), dtype=np.float32),
            "jokers": gym.spaces.Box(low=0, high=1, shape=(5, 64), dtype=np.float32),
            "blind": gym.spaces.Box(low=0, high=1, shape=(16,), dtype=np.float32),
            "scalar": gym.spaces.Box(low=0, high=1, shape=(32,), dtype=np.float32),
        })
        
        self.state: Optional[GameState] = None
        self.steps = 0
        
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[Dict, Dict]:
        """Reset the environment to initial state"""
        super().reset(seed=seed)
        
        # Initialize a standard deck
        deck = self._create_standard_deck()
        np.random.shuffle(deck)
        
        # Deal initial hand
        hand = deck[:8]
        remaining_deck = deck[8:]
        
        # Create initial blind (Ante 1, Small Blind)
        blind = Blind(
            name="Small Blind",
            ante=1,
            type="small",
            chip_requirement=300,
            reward=3
        )
        
        self.state = GameState(
            hand=hand,
            deck=remaining_deck,
            jokers=[],
            consumables=[],
            current_blind=blind,
            chips_scored=0,
            hands_remaining=4,
            discards_remaining=3,
            ante=1,
            money=4,
            round_number=1,
        )
        
        self.steps = 0
        
        obs = self.state.to_observation()
        info = {"ante": 1, "blind_type": "small"}
        
        return obs, info
    
    def step(self, action: Dict) -> Tuple[Dict, float, bool, bool, Dict]:
        """Execute one step in the environment"""
        if self.state is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")
        
        self.steps += 1
        reward = 0.0
        terminated = False
        truncated = self.steps >= self.max_steps
        
        action_type = action["action_type"]
        card_selection = action["card_selection"]
        
        # Process action based on type
        if action_type == 0:  # Play hand
            reward = self._play_hand(card_selection)
        elif action_type == 1:  # Discard
            reward = self._discard_cards(card_selection)
        elif action_type == 2:  # Shop buy
            reward = self._shop_buy(action["shop_selection"])
        elif action_type == 3:  # Skip/Continue
            reward = self._skip_action()
        elif action_type == 4:  # Reroll shop
            reward = self._reroll_shop()
        
        # Check win/loss conditions
        if self.state.current_blind:
            if self.state.chips_scored >= self.state.current_blind.chip_requirement:
                # Beat the blind!
                reward += 100.0 * self.state.ante  # Scale reward with difficulty
                self.state.money += self.state.current_blind.reward
                self._advance_to_next_blind()
            elif self.state.hands_remaining == 0:
                # Failed the blind
                reward -= 50.0
                terminated = True
        
        # Check if we've beaten all antes
        if self.state.ante > 8:
            reward += 1000.0  # Huge bonus for winning the run
            terminated = True
        
        obs = self.state.to_observation()
        info = {
            "ante": self.state.ante,
            "chips": self.state.chips_scored,
            "money": self.state.money,
            "hands_left": self.state.hands_remaining,
        }
        
        return obs, reward, terminated, truncated, info
    
    def _create_standard_deck(self) -> List[Card]:
        """Create a standard 52-card deck"""
        deck = []
        for suit in Suit:
            for rank in Rank:
                deck.append(Card(rank=rank, suit=suit))
        return deck
    
    def _play_hand(self, card_selection: np.ndarray) -> float:
        """Play selected cards as a poker hand"""
        if self.state.hands_remaining <= 0:
            return -1.0  # Penalty for invalid action
        
        # Get selected cards
        selected_indices = np.where(card_selection[:len(self.state.hand)])[0]
        if len(selected_indices) == 0 or len(selected_indices) > 5:
            return -1.0  # Invalid selection
        
        selected_cards = [self.state.hand[i] for i in selected_indices]
        
        # Evaluate poker hand
        hand_type, scoring_cards = self._evaluate_hand(selected_cards)
        chips, mult = self._calculate_score(hand_type, scoring_cards)
        
        # Apply joker effects
        chips, mult = self._apply_joker_effects(chips, mult, hand_type, scoring_cards)
        
        total_score = chips * mult
        self.state.chips_scored += total_score
        self.state.hands_remaining -= 1
        
        # Remove played cards and draw new ones
        for idx in sorted(selected_indices, reverse=True):
            self.state.hand.pop(idx)
        self._draw_cards(len(selected_indices))
        
        # Reward is proportional to chips scored
        return np.log1p(total_score) / 10.0
    
    def _discard_cards(self, card_selection: np.ndarray) -> float:
        """Discard selected cards and draw new ones"""
        if self.state.discards_remaining <= 0:
            return -1.0
        
        selected_indices = np.where(card_selection[:len(self.state.hand)])[0]
        if len(selected_indices) == 0:
            return -0.5
        
        # Remove discarded cards
        for idx in sorted(selected_indices, reverse=True):
            self.state.hand.pop(idx)
        
        # Draw new cards
        self._draw_cards(len(selected_indices))
        self.state.discards_remaining -= 1
        
        return 0.0  # Neutral reward for discarding
    
    def _draw_cards(self, num_cards: int):
        """Draw cards from deck to hand"""
        drawn = 0
        while drawn < num_cards and len(self.state.deck) > 0 and len(self.state.hand) < 8:
            self.state.hand.append(self.state.deck.pop(0))
            drawn += 1
    
    def _evaluate_hand(self, cards: List[Card]) -> Tuple[HandType, List[Card]]:
        """Evaluate poker hand type"""
        if len(cards) == 0:
            return HandType.HIGH_CARD, cards
        
        # Extract ranks and suits
        ranks = [c.rank.value for c in cards]
        suits = [c.suit for c in cards]
        rank_counts = Counter(ranks)
        suit_counts = Counter(suits)
        
        is_flush = len(suit_counts) == 1
        
        # Check for straight
        sorted_ranks = sorted(set(ranks))
        is_straight = False
        if len(sorted_ranks) == 5:
            if sorted_ranks[-1] - sorted_ranks[0] == 4:
                is_straight = True
            # Check for Ace-low straight (A-2-3-4-5)
            if sorted_ranks == [2, 3, 4, 5, 14]:
                is_straight = True
        
        # Determine hand type
        counts = sorted(rank_counts.values(), reverse=True)
        
        if counts == [5]:
            return HandType.FIVE_OF_A_KIND, cards
        elif is_flush and is_straight:
            return HandType.STRAIGHT_FLUSH, cards
        elif counts == [4, 1]:
            return HandType.FOUR_OF_A_KIND, cards
        elif counts == [3, 2]:
            if is_flush:
                return HandType.FLUSH_HOUSE, cards
            return HandType.FULL_HOUSE, cards
        elif is_flush:
            if counts == [5]:
                return HandType.FLUSH_FIVE, cards
            return HandType.FLUSH, cards
        elif is_straight:
            return HandType.STRAIGHT, cards
        elif counts == [3, 1, 1]:
            return HandType.THREE_OF_A_KIND, cards
        elif counts == [2, 2, 1]:
            return HandType.TWO_PAIR, cards
        elif counts == [2, 1, 1, 1]:
            return HandType.PAIR, cards
        else:
            return HandType.HIGH_CARD, cards
    
    def _calculate_score(self, hand_type: HandType, cards: List[Card]) -> Tuple[int, int]:
        """Calculate base chips and mult for a hand"""
        base_chips, base_mult = self.state.hand_levels[hand_type]
        
        total_chips = base_chips
        total_mult = base_mult
        
        # Add card chip values
        for card in cards:
            if not card.debuffed:
                total_chips += card.base_chips
                
                # Apply enhancements
                if card.enhancement == Enhancement.BONUS:
                    total_chips += 30
                elif card.enhancement == Enhancement.MULT:
                    total_mult += 4
                
                # Apply editions
                if card.edition == Edition.FOIL:
                    total_chips += 50
                elif card.edition == Edition.HOLOGRAPHIC:
                    total_mult += 10
        
        return total_chips, total_mult
    
    def _apply_joker_effects(self, chips: int, mult: int, hand_type: HandType, cards: List[Card]) -> Tuple[int, int]:
        """Apply joker effects to score"""
        for joker in self.state.jokers:
            chips += joker.chips_bonus
            mult += joker.mult_bonus
            # Multiplicative effects would be applied after
        
        return chips, mult
    
    def _shop_buy(self, item_index: int) -> float:
        """Buy an item from the shop"""
        if not self.state.in_shop:
            return -1.0
        
        # Simplified shop logic
        # In real implementation, would handle actual shop items
        return 0.0
    
    def _skip_action(self) -> float:
        """Skip current action"""
        return 0.0
    
    def _reroll_shop(self) -> float:
        """Reroll shop items"""
        if not self.state.in_shop or self.state.shop_rerolls_available <= 0:
            return -1.0
        
        self.state.shop_rerolls_available -= 1
        # Reroll logic would go here
        return -0.1
    
    def _advance_to_next_blind(self):
        """Move to the next blind"""
        # Simplified progression
        self.state.round_number += 1
        
        if self.state.round_number % 3 == 0:
            # Boss blind every 3 rounds
            self.state.ante += 1
            self.state.current_blind = Blind(
                name="Boss Blind",
                ante=self.state.ante,
                type="boss",
                chip_requirement=int(1000 * (1.5 ** self.state.ante)),
                reward=self.state.ante * 5
            )
        else:
            blind_type = "small" if self.state.round_number % 3 == 1 else "big"
            chip_req = 300 if blind_type == "small" else 600
            chip_req = int(chip_req * (1.5 ** (self.state.ante - 1)))
            
            self.state.current_blind = Blind(
                name=f"{blind_type.title()} Blind",
                ante=self.state.ante,
                type=blind_type,
                chip_requirement=chip_req,
                reward=3 if blind_type == "small" else 4
            )
        
        # Reset round state
        self.state.chips_scored = 0
        self.state.hands_remaining = 4
        self.state.discards_remaining = 3
        
        # Shuffle deck and draw new hand
        all_cards = self.state.hand + self.state.deck
        np.random.shuffle(all_cards)
        self.state.hand = all_cards[:8]
        self.state.deck = all_cards[8:]
    
    def render(self):
        """Render the current game state (for debugging)"""
        if self.state is None:
            print("Environment not initialized")
            return
        
        print(f"\n=== Ante {self.state.ante} - {self.state.current_blind.name if self.state.current_blind else 'Shop'} ===")
        print(f"Chips: {self.state.chips_scored} / {self.state.current_blind.chip_requirement if self.state.current_blind else 0}")
        print(f"Hands: {self.state.hands_remaining} | Discards: {self.state.discards_remaining}")
        print(f"Money: ${self.state.money}")
        print(f"\nHand: {[f'{c.rank.name}-{c.suit.name}' for c in self.state.hand]}")
        print(f"Jokers: {[j.name for j in self.state.jokers]}")


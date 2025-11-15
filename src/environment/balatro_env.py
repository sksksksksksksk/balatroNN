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
import random

# Import new systems
try:
    from . import balatro_content
    from . import jokers
    from . import consumables
except ImportError:
    import balatro_content
    import jokers
    import consumables


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
    shop_reroll_cost: int = 5
    
    # Expanded shop tracking
    shop_jokers: List[Tuple[Any, int]] = field(default_factory=list)  # (joker, price)
    shop_packs: List[Tuple[str, int]] = field(default_factory=list)    # (pack_type, price)
    shop_cards: List[Tuple[Card, int]] = field(default_factory=list)   # (card, price)
    shop_vouchers: List[Tuple[Any, int]] = field(default_factory=list) # (voucher, price)
    
    # Consumable slots (separate by type for clarity)
    tarot_cards: List[Any] = field(default_factory=list)     # Max 2
    planet_cards: List[Any] = field(default_factory=list)    # Max 2  
    spectral_cards: List[Any] = field(default_factory=list)  # Max 2
    
    # Permanent upgrades and modifications
    vouchers_owned: List[Any] = field(default_factory=list)
    interest_cap: int = 5  # Base interest cap, raised by vouchers
    max_joker_slots: int = 5
    max_consumable_slots: int = 2  # Per type
    hand_size: int = 8
    base_hands_per_round: int = 4
    base_discards_per_round: int = 3
    shop_slots: int = 2  # Number of items per category
    
    # Deck modifications tracking
    cards_added_to_deck: int = 0
    cards_removed_from_deck: int = 0
    
    # Persistent joker state (for jokers that track history)
    joker_states: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # Round tracking for effects
    hands_played_this_round: int = 0
    discards_used_this_round: int = 0
    
    def to_observation(self, synergy_detector=None) -> Dict[str, np.ndarray]:
        """Convert game state to observation dict for neural network (EXPANDED)"""
        obs = {}
        
        # Hand cards (max 8 cards, pad if necessary)
        hand_vectors = np.zeros((8, 32))
        for i, card in enumerate(self.hand[:8]):
            hand_vectors[i] = card.to_vector()
        obs["hand"] = hand_vectors
        
        # Jokers (max 5, expanded to 128 dims)
        joker_vectors = np.zeros((5, 128))
        for i, joker in enumerate(self.jokers[:5]):
            if hasattr(joker, 'to_vector'):
                joker_vectors[i] = joker.to_vector()
            else:
                # Fallback for old Joker class
                vec = np.zeros(128)
                vec[:64] = joker.to_vector() if hasattr(joker, 'to_vector') else np.zeros(64)
                joker_vectors[i] = vec
        obs["jokers"] = joker_vectors
        
        # Consumables (2 tarot + 2 planet + 2 spectral = 6 slots, 64 dims each)
        consumable_vectors = np.zeros((6, 64))
        # Tarot cards (slots 0-1)
        for i, tarot in enumerate(self.tarot_cards[:2]):
            if hasattr(tarot, 'to_vector'):
                consumable_vectors[i] = tarot.to_vector()
        # Planet cards (slots 2-3)
        for i, planet in enumerate(self.planet_cards[:2]):
            if hasattr(planet, 'to_vector'):
                consumable_vectors[2 + i] = planet.to_vector()
        # Spectral cards (slots 4-5)
        for i, spectral in enumerate(self.spectral_cards[:2]):
            if hasattr(spectral, 'to_vector'):
                consumable_vectors[4 + i] = spectral.to_vector()
        obs["consumables"] = consumable_vectors
        
        # Shop items (10 slots total, 128 dims each)
        # Layout: 4 jokers + 2 packs + 2 cards + 2 vouchers = 10 slots
        shop_vectors = np.zeros((10, 128))
        if self.in_shop:
            # Jokers (slots 0-3)
            for i, (joker, price) in enumerate(self.shop_jokers[:4]):
                if hasattr(joker, 'to_vector'):
                    vec = joker.to_vector()
                    vec[127] = price / 50.0  # Encode price in last dim
                    shop_vectors[i] = vec
            # Packs (slots 4-5)
            for i, (pack_name, price) in enumerate(self.shop_packs[:2]):
                vec = np.zeros(128)
                vec[0] = hash(pack_name) % 100 / 100.0  # Pack type
                vec[127] = price / 20.0  # Price
                shop_vectors[4 + i] = vec
            # Cards (slots 6-7)
            for i, (card, price) in enumerate(self.shop_cards[:2]):
                vec = np.zeros(128)
                vec[:32] = card.to_vector()
                vec[127] = price / 20.0  # Price
                shop_vectors[6 + i] = vec
            # Vouchers (slots 8-9)
            for i, (voucher, price) in enumerate(self.shop_vouchers[:2]):
                vec = np.zeros(128)
                if hasattr(voucher, 'data'):
                    vec[0] = hash(voucher.data.name) % 100 / 100.0
                    vec[1] = voucher.data.tier / 2.0
                vec[127] = price / 50.0  # Price
                shop_vectors[8 + i] = vec
        obs["shop_items"] = shop_vectors
        
        # Vouchers owned (max 5, 32 dims each)
        voucher_vectors = np.zeros((5, 32))
        for i, voucher in enumerate(self.vouchers_owned[:5]):
            if hasattr(voucher, 'data'):
                vec = np.zeros(32)
                vec[0] = hash(voucher.data.name) % 100 / 100.0
                vec[1] = voucher.data.tier / 2.0
                vec[2] = hash(voucher.data.effect_type) % 100 / 100.0
                voucher_vectors[i] = vec
        obs["vouchers"] = voucher_vectors
        
        # Blind information
        if self.current_blind:
            obs["blind"] = self.current_blind.to_vector()
        else:
            obs["blind"] = np.zeros(16)
        
        # Expanded scalar game state (64 dims)
        scalar_state = np.zeros(64)
        # Core gameplay
        scalar_state[0] = self.chips_scored / 10000.0
        scalar_state[1] = (self.current_blind.chip_requirement if self.current_blind else 0) / 10000.0
        scalar_state[2] = self.hands_remaining / 10.0
        scalar_state[3] = self.discards_remaining / 10.0
        scalar_state[4] = self.ante / 8.0
        scalar_state[5] = np.clip(self.money / 100.0, -0.5, 5.0)  # Allow negative money
        scalar_state[6] = len(self.deck) / 100.0
        scalar_state[7] = len(self.jokers) / 5.0
        scalar_state[8] = float(self.in_shop)
        scalar_state[9] = self.round_number / 100.0
        
        # Economy
        scalar_state[10] = self.interest_cap / 25.0
        scalar_state[11] = (self.money // 5) / 25.0  # Potential interest
        scalar_state[12] = self.shop_reroll_cost / 20.0
        
        # Inventory space
        scalar_state[13] = len(self.tarot_cards) / 2.0
        scalar_state[14] = len(self.planet_cards) / 2.0
        scalar_state[15] = len(self.spectral_cards) / 2.0
        scalar_state[16] = (self.max_joker_slots - len(self.jokers)) / 5.0  # Free slots
        scalar_state[17] = (self.max_consumable_slots - len(self.tarot_cards)) / 2.0
        
        # Permanent upgrades
        scalar_state[18] = self.hand_size / 15.0
        scalar_state[19] = self.base_hands_per_round / 10.0
        scalar_state[20] = self.base_discards_per_round / 10.0
        scalar_state[21] = self.shop_slots / 5.0
        
        # Deck modifications
        scalar_state[22] = self.cards_added_to_deck / 50.0
        scalar_state[23] = self.cards_removed_from_deck / 50.0
        
        # Shop availability (if in shop)
        if self.in_shop:
            scalar_state[24] = len(self.shop_jokers) / 4.0
            scalar_state[25] = len(self.shop_packs) / 2.0
            scalar_state[26] = len(self.shop_cards) / 2.0
            scalar_state[27] = len(self.shop_vouchers) / 2.0
        
        # Hand level info (sample a few important hands)
        for i, hand_type in enumerate([HandType.PAIR, HandType.TWO_PAIR, HandType.FLUSH, HandType.STRAIGHT]):
            if i < 4 and hand_type in self.hand_levels:
                chips, mult = self.hand_levels[hand_type]
                scalar_state[28 + i * 2] = chips / 500.0
                scalar_state[29 + i * 2] = mult / 50.0
        
        obs["scalar"] = scalar_state
        
        # Synergy features (32 dims)
        if synergy_detector is not None:
            obs["synergies"] = synergy_detector.to_feature_vector(self.jokers)
        else:
            obs["synergies"] = np.zeros(32)
        
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
        
        # Expanded Action space for full game
        # Actions:
        # 0: play_hand - play selected cards
        # 1: discard - discard selected cards
        # 2: buy_joker - buy joker from shop
        # 3: buy_pack - buy and open pack from shop
        # 4: buy_card - buy playing card from shop
        # 5: buy_voucher - buy voucher from shop
        # 6: sell_joker - sell a joker
        # 7: use_tarot - use a tarot card
        # 8: use_planet - use a planet card
        # 9: use_spectral - use a spectral card
        # 10: reroll_shop - reroll shop contents
        # 11: skip - skip/continue to next phase
        self.action_space = gym.spaces.Dict({
            "action_type": gym.spaces.Discrete(12),  # Expanded to 12 action types
            "card_selection": gym.spaces.MultiBinary(8),  # Which cards to play/discard/target
            "shop_item_index": gym.spaces.Discrete(7),  # Which item in shop (jokers/packs/cards/vouchers)
            "joker_slot": gym.spaces.Discrete(5),  # Which joker slot (for selling or effects)
            "consumable_slot": gym.spaces.Discrete(2),  # Which consumable slot to use
            "target_card_index": gym.spaces.Discrete(8),  # For tarot cards that target specific cards
        })
        
        # Expanded Observation space for full game
        self.observation_space = gym.spaces.Dict({
            "hand": gym.spaces.Box(low=0, high=1, shape=(8, 32), dtype=np.float32),
            "jokers": gym.spaces.Box(low=0, high=1, shape=(5, 128), dtype=np.float32),  # Expanded for complex jokers
            "consumables": gym.spaces.Box(low=0, high=1, shape=(6, 64), dtype=np.float32),  # 2 each type (tarot/planet/spectral)
            "shop_items": gym.spaces.Box(low=0, high=1, shape=(10, 128), dtype=np.float32),  # All shop slots combined
            "vouchers": gym.spaces.Box(low=0, high=1, shape=(5, 32), dtype=np.float32),  # Owned vouchers
            "blind": gym.spaces.Box(low=0, high=1, shape=(16,), dtype=np.float32),
            "scalar": gym.spaces.Box(low=0, high=1, shape=(64,), dtype=np.float32),  # Extended scalars (money, interest, etc.)
            "synergies": gym.spaces.Box(low=0, high=1, shape=(32,), dtype=np.float32),  # Joker synergy features
        })
        
        # Initialize synergy detector
        self.synergy_detector = jokers.JokerSynergyDetector()
        
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
        
        obs = self.state.to_observation(self.synergy_detector)
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
        
        # Process action based on type (12 action types total)
        if action_type == 0:  # Play hand
            reward = self._play_hand(card_selection)
        elif action_type == 1:  # Discard
            reward = self._discard_cards(card_selection)
        elif action_type == 2:  # Buy joker
            reward = self._buy_joker(action["shop_item_index"])
        elif action_type == 3:  # Buy pack
            reward = self._buy_pack(action["shop_item_index"])
        elif action_type == 4:  # Buy card
            reward = self._buy_card(action["shop_item_index"])
        elif action_type == 5:  # Buy voucher
            reward = self._buy_voucher(action["shop_item_index"])
        elif action_type == 6:  # Sell joker
            reward = self._sell_joker(action["joker_slot"])
        elif action_type == 7:  # Use tarot
            reward = self._use_tarot(action["consumable_slot"], card_selection, action["target_card_index"])
        elif action_type == 8:  # Use planet
            reward = self._use_planet(action["consumable_slot"])
        elif action_type == 9:  # Use spectral
            reward = self._use_spectral(action["consumable_slot"], card_selection, action["target_card_index"])
        elif action_type == 10:  # Reroll shop
            reward = self._reroll_shop()
        elif action_type == 11:  # Skip/Continue
            reward = self._skip_action()
        
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
        
        obs = self.state.to_observation(self.synergy_detector)
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
        """Buy an item from the shop - DEPRECATED, use specific buy methods"""
        if not self.state.in_shop:
            return -1.0
        return 0.0
    
    def _skip_action(self) -> float:
        """Skip current action"""
        return 0.0
    
    def _reroll_shop(self) -> float:
        """Reroll shop items"""
        if not self.state.in_shop:
            return -1.0
        
        # Check money (allow credit card debt)
        can_go_negative = any(j.data.name == "Credit Card" for j in self.state.jokers if hasattr(j, 'data'))
        min_money = -20 if can_go_negative else 0
        
        if self.state.money - self.state.shop_reroll_cost < min_money:
            return -1.0
        
        # Pay for reroll
        self.state.money -= self.state.shop_reroll_cost
        self.state.shop_reroll_cost += 2  # Increases each time
        
        # Regenerate shop
        self._generate_shop()
        return -0.05  # Small penalty for spending
    
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
    
    # ========================================================================
    # SHOP AND ECONOMY SYSTEM
    # ========================================================================
    
    def _generate_shop(self) -> None:
        """Generate shop contents based on ante and vouchers"""
        # Clear existing shop
        self.state.shop_jokers = []
        self.state.shop_packs = []
        self.state.shop_cards = []
        self.state.shop_vouchers = []
        
        # Generate jokers (2 base + voucher bonuses)
        num_joker_slots = self.state.shop_slots
        for voucher in self.state.vouchers_owned:
            if hasattr(voucher, 'data') and voucher.data.effect_type == "shop_slots":
                num_joker_slots += 1
        
        for _ in range(num_joker_slots):
            joker = jokers.get_random_joker()
            price = balatro_content.get_joker_price(joker.data)
            
            # Apply discounts from vouchers
            for voucher in self.state.vouchers_owned:
                if hasattr(voucher, 'data'):
                    if voucher.data.name == "Clearance Sale":
                        price = int(price * 0.75)
                    elif voucher.data.name == "Liquidation":
                        price = int(price * 0.5)
            
            self.state.shop_jokers.append((joker, price))
        
        # Generate booster packs (2 base)
        for _ in range(2):
            pack_data = random.choice(balatro_content.PACKS)
            price = balatro_content.get_pack_price(pack_data.name)
            
            # Apply discounts
            for voucher in self.state.vouchers_owned:
                if hasattr(voucher, 'data'):
                    if voucher.data.name == "Clearance Sale":
                        price = int(price * 0.75)
                    elif voucher.data.name == "Liquidation":
                        price = int(price * 0.5)
            
            self.state.shop_packs.append((pack_data.name, price))
        
        # Generate playing cards (2 base)
        for _ in range(2):
            card = self._generate_shop_card()
            has_enhancement = card.enhancement != Enhancement.NONE
            has_edition = card.edition != Edition.NONE
            price = balatro_content.get_card_price(has_enhancement, has_edition)
            
            # Apply discounts
            for voucher in self.state.vouchers_owned:
                if hasattr(voucher, 'data'):
                    if voucher.data.name == "Clearance Sale":
                        price = int(price * 0.75)
                    elif voucher.data.name == "Liquidation":
                        price = int(price * 0.5)
            
            self.state.shop_cards.append((card, price))
        
        # Generate vouchers (1 if ante >= 2)
        if self.state.ante >= 2:
            # Pick a random voucher the player doesn't have
            available_vouchers = [v for v in balatro_content.VOUCHERS 
                                   if not any(owned.data.name == v.name for owned in self.state.vouchers_owned if hasattr(owned, 'data'))]
            if available_vouchers:
                voucher_data = random.choice(available_vouchers)
                price = balatro_content.get_voucher_price(voucher_data.tier)
                
                # Apply discounts
                for voucher in self.state.vouchers_owned:
                    if hasattr(voucher, 'data'):
                        if voucher.data.name == "Clearance Sale":
                            price = int(price * 0.75)
                        elif voucher.data.name == "Liquidation":
                            price = int(price * 0.5)
                
                # Create voucher instance (simple dataclass wrapper)
                @dataclass
                class VoucherInstance:
                    data: Any
                
                voucher_inst = VoucherInstance(data=voucher_data)
                self.state.shop_vouchers.append((voucher_inst, price))
        
        self.state.in_shop = True
    
    def _generate_shop_card(self) -> Card:
        """Generate a playing card for the shop with possible enhancements"""
        # Random card
        suit = random.choice(list(Suit))
        rank = random.choice(list(Rank))
        
        # Chance for enhancement
        enhancement = Enhancement.NONE
        if random.random() < 0.3:  # 30% chance
            enhancement = random.choice([e for e in Enhancement if e != Enhancement.NONE])
        
        # Chance for edition
        edition = Edition.NONE
        edition_rate = 0.02  # 2% base
        for voucher in self.state.vouchers_owned:
            if hasattr(voucher, 'data'):
                if voucher.data.name == "Hone":
                    edition_rate = 0.04
                elif voucher.data.name == "Glow Up":
                    edition_rate = 0.08
        
        if random.random() < edition_rate:
            edition = random.choice([e for e in Edition if e != Edition.NONE])
        
        return Card(suit=suit, rank=rank, enhancement=enhancement, edition=edition)
    
    def _buy_joker(self, shop_index: int) -> float:
        """Buy a joker from the shop"""
        if not self.state.in_shop or shop_index >= len(self.state.shop_jokers):
            return -1.0
        
        joker, price = self.state.shop_jokers[shop_index]
        
        # Check money
        can_go_negative = any(j.data.name == "Credit Card" for j in self.state.jokers if hasattr(j, 'data'))
        min_money = -20 if can_go_negative else 0
        
        if self.state.money - price < min_money:
            return -1.0  # Can't afford
        
        # Check space
        if len(self.state.jokers) >= self.state.max_joker_slots:
            return -1.0  # No space
        
        # Purchase
        self.state.money -= price
        self.state.jokers.append(joker)
        self.state.shop_jokers.pop(shop_index)
        
        # Reward with quality evaluation
        base_reward = 0.1
        quality_bonus = self._evaluate_purchase_quality("joker", joker)
        return base_reward + quality_bonus
    
    def _buy_pack(self, shop_index: int) -> float:
        """Buy and open a booster pack"""
        if not self.state.in_shop or shop_index >= len(self.state.shop_packs):
            return -1.0
        
        pack_type, price = self.state.shop_packs[shop_index]
        
        # Check money
        can_go_negative = any(j.data.name == "Credit Card" for j in self.state.jokers if hasattr(j, 'data'))
        min_money = -20 if can_go_negative else 0
        
        if self.state.money - price < min_money:
            return -1.0
        
        # Purchase and open
        self.state.money -= price
        reward = self._open_pack(pack_type)
        self.state.shop_packs.pop(shop_index)
        
        return reward
    
    def _buy_card(self, shop_index: int) -> float:
        """Buy a playing card from the shop"""
        if not self.state.in_shop or shop_index >= len(self.state.shop_cards):
            return -1.0
        
        card, price = self.state.shop_cards[shop_index]
        
        # Check money
        can_go_negative = any(j.data.name == "Credit Card" for j in self.state.jokers if hasattr(j, 'data'))
        min_money = -20 if can_go_negative else 0
        
        if self.state.money - price < min_money:
            return -1.0
        
        # Purchase - add to deck
        self.state.money -= price
        self.state.deck.append(card)
        self.state.cards_added_to_deck += 1
        self.state.shop_cards.pop(shop_index)
        
        return 0.05
    
    def _buy_voucher(self, shop_index: int) -> float:
        """Buy a voucher from the shop"""
        if not self.state.in_shop or shop_index >= len(self.state.shop_vouchers):
            return -1.0
        
        voucher, price = self.state.shop_vouchers[shop_index]
        
        # Check money
        can_go_negative = any(j.data.name == "Credit Card" for j in self.state.jokers if hasattr(j, 'data'))
        min_money = -20 if can_go_negative else 0
        
        if self.state.money - price < min_money:
            return -1.0
        
        # Purchase and apply effect
        self.state.money -= price
        self.state.vouchers_owned.append(voucher)
        self._apply_voucher_effect(voucher)
        self.state.shop_vouchers.pop(shop_index)
        
        # Reward with quality evaluation
        base_reward = 0.15
        quality_bonus = self._evaluate_purchase_quality("voucher", voucher)
        return base_reward + quality_bonus
    
    def _sell_joker(self, joker_slot: int) -> float:
        """Sell a joker"""
        if joker_slot >= len(self.state.jokers):
            return -1.0
        
        joker = self.state.jokers[joker_slot]
        sell_value = joker.sell_value if hasattr(joker, 'sell_value') else 1
        
        self.state.money += sell_value
        self.state.jokers.pop(joker_slot)
        
        return 0.0  # Neutral action
    
    def _open_pack(self, pack_type: str) -> float:
        """Open a booster pack and auto-select best option"""
        # Generate pack contents based on pack data
        pack_info = next((p for p in balatro_content.PACKS if p.name == pack_type), None)
        if not pack_info:
            return 0.0
        
        # Generate options
        options = []
        if pack_info.content_type == "tarot":
            options = [consumables.get_random_tarot() for _ in range(pack_info.contents_count)]
        elif pack_info.content_type == "planet":
            options = [consumables.get_random_planet() for _ in range(pack_info.contents_count)]
        elif pack_info.content_type == "spectral":
            options = [consumables.get_random_spectral() for _ in range(pack_info.contents_count)]
        elif pack_info.content_type == "playing_card":
            options = [self._generate_shop_card() for _ in range(pack_info.contents_count)]
        elif pack_info.content_type == "joker":
            options = [jokers.get_random_joker() for _ in range(pack_info.contents_count)]
        
        # Simplified: pick first option (in real game, model would choose)
        if options:
            selected = options[0]
            
            # Add to appropriate inventory
            if pack_info.content_type == "tarot":
                if len(self.state.tarot_cards) < self.state.max_consumable_slots:
                    self.state.tarot_cards.append(selected)
                    return 0.1
            elif pack_info.content_type == "planet":
                if len(self.state.planet_cards) < self.state.max_consumable_slots:
                    self.state.planet_cards.append(selected)
                    return 0.1
            elif pack_info.content_type == "spectral":
                if len(self.state.spectral_cards) < self.state.max_consumable_slots:
                    self.state.spectral_cards.append(selected)
                    return 0.1
            elif pack_info.content_type == "playing_card":
                self.state.deck.append(selected)
                self.state.cards_added_to_deck += 1
                return 0.05
            elif pack_info.content_type == "joker":
                if len(self.state.jokers) < self.state.max_joker_slots:
                    self.state.jokers.append(selected)
                    return 0.1
        
        return 0.0
    
    def _apply_voucher_effect(self, voucher) -> None:
        """Apply the permanent effect of a voucher"""
        if not hasattr(voucher, 'data'):
            return
        
        effect = voucher.data.effect_type
        
        if effect == "shop_slots":
            self.state.shop_slots += 1
        elif effect == "consumable_slots":
            self.state.max_consumable_slots += 1
        elif effect == "hands_per_round":
            self.state.base_hands_per_round += 1
        elif effect == "discards_per_round":
            self.state.base_discards_per_round += 1
        elif effect == "hand_size":
            self.state.hand_size += 2
        elif effect == "interest_cap":
            if voucher.data.name == "Seed Money":
                self.state.interest_cap = 10
            elif voucher.data.name == "Money Tree":
                self.state.interest_cap = 25
        elif effect == "reroll_cost":
            # Reduces reroll cost (handled in _reroll_shop)
            pass
        # Other voucher effects are passive (shop discounts, rates, etc.)
    
    def _calculate_interest(self) -> int:
        """Calculate interest earned ($1 per $5, capped)"""
        # Check for special jokers
        remove_cap = any(j.data.name == "To the Moon" for j in self.state.jokers if hasattr(j, 'data'))
        
        if remove_cap:
            return self.state.money // 5  # No cap
        else:
            return min(self.state.money // 5, self.state.interest_cap)
    
    def _earn_money(self, amount: int) -> None:
        """Add money"""
        self.state.money += amount
    
    def _enter_shop(self) -> None:
        """Enter the shop phase after completing a blind"""
        self._generate_shop()
        self.state.shop_reroll_cost = 5  # Reset reroll cost
        
        # Earn interest
        interest = self._calculate_interest()
        self.state.money += interest
        
        # Trigger shop enter effects for jokers
        # (handled by joker processor in full implementation)
    
    def _evaluate_purchase_quality(self, item_type: str, item: Any) -> float:
        """
        Evaluate the quality of a purchase for reward shaping (ENHANCED with synergy detection)
        Returns bonus reward for good purchases
        """
        bonus = 0.0
        
        if item_type == "joker" and hasattr(item, 'data'):
            # Reward higher rarity jokers
            rarity_bonus = {
                balatro_content.Rarity.COMMON: 0.05,
                balatro_content.Rarity.UNCOMMON: 0.10,
                balatro_content.Rarity.RARE: 0.20,
                balatro_content.Rarity.LEGENDARY: 0.40,
            }
            bonus += rarity_bonus.get(item.data.rarity, 0.0)
            
            # Use synergy detector to evaluate lineup quality
            current_eval = self.synergy_detector.evaluate_joker_lineup(self.state.jokers)
            test_lineup = self.state.jokers + [item]
            new_eval = self.synergy_detector.evaluate_joker_lineup(test_lineup)
            
            # Reward based on synergy improvement
            synergy_delta = new_eval["synergy_score"] - current_eval["synergy_score"]
            if synergy_delta > 0:
                bonus += min(synergy_delta * 0.2, 0.3)  # Cap at 0.3 bonus
            
            # Additional reward for creating new synergies
            new_synergies = new_eval["synergies_active"] - current_eval["synergies_active"]
            if new_synergies > 0:
                bonus += new_synergies * 0.15
            
            # Reward jokers early (more impactful)
            if self.state.ante <= 3:
                bonus += 0.10
        
        elif item_type == "voucher":
            # Vouchers are always good strategic purchases
            bonus += 0.15
            
            # Economy vouchers are especially valuable early
            if hasattr(item, 'data'):
                if item.data.effect_type in ["interest_cap", "shop_discount"]:
                    bonus += 0.10 if self.state.ante <= 2 else 0.05
        
        elif item_type == "pack":
            # Packs provide variety
            bonus += 0.05
            
        elif item_type == "planet":
            # Leveling hands is strategic
            bonus += 0.08
        
        return bonus
    
    def _calculate_interest_reward(self) -> float:
        """Reward maintaining money for interest"""
        money_threshold = self.state.interest_cap * 5  # e.g., 25 for cap of 5
        if self.state.money >= money_threshold:
            return 0.05  # Small bonus for maintaining interest threshold
        return 0.0
    
    # ========================================================================
    # CONSUMABLE USAGE
    # ========================================================================
    
    def _use_tarot(self, slot: int, card_selection: np.ndarray, target_index: int) -> float:
        """Use a tarot card"""
        if slot >= len(self.state.tarot_cards):
            return -1.0
        
        tarot = self.state.tarot_cards[slot]
        
        # Get target cards based on selection
        target_cards = []
        if tarot.data.target_required:
            # Use card_selection to pick cards from hand
            for i, selected in enumerate(card_selection):
                if selected and i < len(self.state.hand):
                    target_cards.append(self.state.hand[i])
        
        # Apply effect
        result = tarot.apply_effect(self.state, target_cards)
        
        if result.get('success'):
            # Remove used tarot
            self.state.tarot_cards.pop(slot)
            
            # Apply effects
            if 'money_change' in result:
                self.state.money += result['money_change']
            
            if 'enhance_cards' in result:
                # Apply enhancement to cards
                enhancement_type = result['enhance_cards']['enhancement']
                for card in result['enhance_cards']['cards']:
                    card.enhancement = Enhancement[enhancement_type]
            
            # Other effects handled in the result dict
            return 0.1  # Reward for using consumable
        
        return -0.5  # Penalty for invalid use
    
    def _use_planet(self, slot: int) -> float:
        """Use a planet card to level up a hand type"""
        if slot >= len(self.state.planet_cards):
            return -1.0
        
        planet = self.state.planet_cards[slot]
        result = planet.apply_effect(self.state)
        
        if result.get('success'):
            # Apply the upgrade
            if 'upgrade_hand' in result:
                hand_type = result['upgrade_hand']['hand_type']
                new_chips = result['upgrade_hand']['chips']
                new_mult = result['upgrade_hand']['mult']
                self.state.hand_levels[hand_type] = (new_chips, new_mult)
            
            # Remove used planet
            self.state.planet_cards.pop(slot)
            return 0.15  # Good reward for strategic hand leveling
        
        return -0.5
    
    def _use_spectral(self, slot: int, card_selection: np.ndarray, target_index: int) -> float:
        """Use a spectral card"""
        if slot >= len(self.state.spectral_cards):
            return -1.0
        
        spectral = self.state.spectral_cards[slot]
        
        # Get target cards
        target_cards = []
        if spectral.data.target_required:
            for i, selected in enumerate(card_selection):
                if selected and i < len(self.state.hand):
                    target_cards.append(self.state.hand[i])
        
        # Apply effect
        result = spectral.apply_effect(self.state, target_cards)
        
        if result.get('success'):
            # Remove used spectral
            self.state.spectral_cards.pop(slot)
            
            # Apply effects
            if 'money_change' in result:
                self.state.money += result['money_change']
            
            if 'create_rare_joker' in result:
                rare_joker = jokers.get_random_joker(balatro_content.Rarity.RARE)
                if len(self.state.jokers) < self.state.max_joker_slots:
                    self.state.jokers.append(rare_joker)
            
            # Other effects handled in result dict
            return 0.2  # High reward for powerful effects
        
        return -0.5
    
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


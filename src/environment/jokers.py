"""
Joker Effect System for Balatro

Implements the activation logic and effects for all 50 priority jokers.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import random

try:
    from . import balatro_content
    from .balatro_content import (
        JokerData, JokerTrigger, Rarity, JokerEdition, JokerModifier,
        ALL_JOKERS, get_joker_by_name, TIER1_JOKERS, TIER2_JOKERS, TIER3_JOKERS
    )
except ImportError:
    import balatro_content
    from balatro_content import (
        JokerData, JokerTrigger, Rarity, JokerEdition, JokerModifier,
        ALL_JOKERS, get_joker_by_name, TIER1_JOKERS, TIER2_JOKERS, TIER3_JOKERS
    )


@dataclass
class JokerInstance:
    """
    Runtime instance of a joker with state, editions, and modifiers
    """
    data: JokerData
    edition: JokerEdition = JokerEdition.NONE
    modifier: JokerModifier = JokerModifier.NONE
    sell_value: int = field(init=False)
    persistent_state: Dict[str, Any] = field(default_factory=dict)
    rounds_remaining: int = 5  # For Perishable jokers
    
    def __post_init__(self):
        """Initialize sell value with edition bonuses"""
        self.sell_value = self.data.base_cost
        
        # Edition increases sell value
        if self.edition == JokerEdition.FOIL:
            self.sell_value += 2
        elif self.edition == JokerEdition.HOLOGRAPHIC:
            self.sell_value += 3
        elif self.edition == JokerEdition.POLYCHROME:
            self.sell_value += 5
        elif self.edition == JokerEdition.NEGATIVE:
            self.sell_value += 10
    
    def get_edition_bonus(self) -> Tuple[int, int, float]:
        """Get edition bonuses (chips, mult, xmult)"""
        chips = 0
        mult = 0
        xmult = 1.0
        
        if self.edition == JokerEdition.FOIL:
            chips = 50
        elif self.edition == JokerEdition.HOLOGRAPHIC:
            mult = 10
        elif self.edition == JokerEdition.POLYCHROME:
            xmult = 1.5
        
        return chips, mult, xmult
    
    def is_eternal(self) -> bool:
        """Check if joker is eternal (cannot be sold/destroyed)"""
        return self.modifier == JokerModifier.ETERNAL
    
    def is_perishable(self) -> bool:
        """Check if joker is perishable"""
        return self.modifier == JokerModifier.PERISHABLE
    
    def is_rental(self) -> bool:
        """Check if joker is rental (costs money each round)"""
        return self.modifier == JokerModifier.RENTAL
    
    def advance_round(self) -> bool:
        """
        Advance round counter for perishable jokers
        Returns True if joker should be debuffed/destroyed
        """
        if self.is_perishable():
            self.rounds_remaining -= 1
            return self.rounds_remaining <= 0
        return False
    
    def to_vector(self) -> np.ndarray:
        """Convert joker to feature vector for neural network"""
        vec = np.zeros(128)  # Expanded from 64 to 128 for more detail
        
        # Rarity encoding (0-3)
        vec[0] = self.data.rarity.value / 3.0
        
        # Tier encoding (1-3)
        vec[1] = self.data.tier / 3.0
        
        # Effect values (normalized)
        vec[2] = np.clip(self.data.chips_bonus / 100.0, 0, 10)
        vec[3] = np.clip(self.data.mult_bonus / 20.0, 0, 10)
        vec[4] = np.clip(self.data.xmult_bonus / 5.0, 0, 10)
        vec[5] = np.clip(self.data.money_bonus / 10.0, 0, 10)
        
        # Sell value
        vec[6] = self.sell_value / 50.0
        
        # Trigger types (one-hot encoded, positions 7-14)
        trigger_map = {
            JokerTrigger.ON_HAND_PLAYED: 7,
            JokerTrigger.ON_CARD_SCORED: 8,
            JokerTrigger.ON_DISCARD: 9,
            JokerTrigger.ON_BLIND_COMPLETE: 10,
            JokerTrigger.ON_SHOP_ENTER: 11,
            JokerTrigger.ON_JOKER_SOLD: 12,
            JokerTrigger.ON_ROUND_END: 13,
            JokerTrigger.ALWAYS: 14,
        }
        for trigger in self.data.triggers:
            if trigger in trigger_map:
                vec[trigger_map[trigger]] = 1.0
        
        # Conditional flag
        vec[15] = 1.0 if self.data.requires_condition else 0.0
        
        # Persistent state flag
        vec[16] = 1.0 if self.data.persistent_state else 0.0
        
        # Persistent state values (positions 17-20 for various counters)
        if self.persistent_state:
            # Encode common persistent values
            vec[17] = self.persistent_state.get('counter', 0) / 100.0
            vec[18] = self.persistent_state.get('mult_accumulated', 0) / 50.0
            vec[19] = self.persistent_state.get('chips_accumulated', 0) / 500.0
            vec[20] = self.persistent_state.get('hands_played', 0) / 50.0
        
        # Edition (one-hot, positions 21-25)
        if self.edition != JokerEdition.NONE:
            vec[21 + self.edition.value - 1] = 1.0
        
        # Modifier (one-hot, positions 26-29)
        if self.modifier != JokerModifier.NONE:
            vec[26 + self.modifier.value - 1] = 1.0
        
        # Rounds remaining (for perishable)
        vec[30] = self.rounds_remaining / 10.0
        
        # Name hash (for unique identification)  
        vec[31] = hash(self.data.name) % 10000 / 10000.0
        
        return vec


class JokerEffectProcessor:
    """
    Processes joker effects based on triggers and game state
    """
    
    def __init__(self):
        """Initialize the effect processor"""
        self.last_hand_type = None
        self.last_scored_cards = []
    
    def process_trigger(
        self, 
        trigger: JokerTrigger, 
        joker_instances: List[JokerInstance],
        game_state: Any,  # GameState object
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Process all jokers for a given trigger
        
        Args:
            trigger: What event triggered
            joker_instances: All active jokers
            game_state: Current game state
            context: Additional context (hand_type, cards, etc.)
            
        Returns:
            Dict with modifications: {'chips': 0, 'mult': 0, 'xmult': 1.0, 'money': 0, 'special': []}
        """
        modifications = {
            'chips': 0,
            'mult': 0,
            'xmult': 1.0,
            'money': 0,
            'special_effects': []
        }
        
        for joker in joker_instances:
            if trigger not in joker.data.triggers and JokerTrigger.ALWAYS not in joker.data.triggers:
                continue
            
            # Check if joker should activate
            if joker.data.requires_condition:
                if not self._check_condition(joker, game_state, context):
                    continue
            
            # Apply the joker effect
            effect = self._apply_joker_effect(joker, game_state, context, trigger)
            
            # Accumulate modifications
            modifications['chips'] += effect.get('chips', 0)
            modifications['mult'] += effect.get('mult', 0)
            modifications['xmult'] *= effect.get('xmult', 1.0)
            modifications['money'] += effect.get('money', 0)
            if effect.get('special_effects'):
                modifications['special_effects'].extend(effect['special_effects'])
            
            # Add edition bonuses
            edition_chips, edition_mult, edition_xmult = joker.get_edition_bonus()
            modifications['chips'] += edition_chips
            modifications['mult'] += edition_mult
            modifications['xmult'] *= edition_xmult
        
        return modifications
    
    def _check_condition(
        self, 
        joker: JokerInstance, 
        game_state: Any, 
        context: Dict[str, Any]
    ) -> bool:
        """Check if conditional joker's conditions are met"""
        name = joker.data.name
        hand_type = context.get('hand_type')
        cards = context.get('cards', [])
        scored_cards = context.get('scored_cards', [])
        
        # TIER 1 Conditionals
        if name == "Greedy Joker":
            card = context.get('card')
            return card and card.suit.name == "DIAMONDS"
        elif name == "Lusty Joker":
            card = context.get('card')
            return card and card.suit.name == "HEARTS"
        elif name == "Wrathful Joker":
            card = context.get('card')
            return card and card.suit.name == "SPADES"
        elif name == "Gluttonous Joker":
            card = context.get('card')
            return card and card.suit.name == "CLUBS"
        elif name == "Jolly Joker":
            return hand_type and hand_type.name == "PAIR"
        elif name == "Zany Joker":
            return hand_type and hand_type.name == "THREE_OF_A_KIND"
        elif name == "Mad Joker":
            return hand_type and hand_type.name == "TWO_PAIR"
        elif name == "Crazy Joker":
            return hand_type and hand_type.name == "STRAIGHT"
        elif name == "Droll Joker":
            return hand_type and hand_type.name == "FLUSH"
        elif name == "Half Joker":
            return len(cards) <= 3
        elif name == "Scary Face":
            card = context.get('card')
            return card and card.rank.value >= 11  # J, Q, K
        elif name == "Abstract Joker":
            return len(game_state.jokers) > 0
        elif name == "Misprint":
            return True  # Always activates with random value
        
        # TIER 2 Conditionals
        elif name == "Baron":
            return any(c.rank.value == 13 for c in cards)  # King
        elif name == "Fibonacci":
            card = context.get('card')
            return card and card.rank.value in [14, 2, 3, 5, 8]  # A, 2, 3, 5, 8
        elif name == "Even Steven":
            card = context.get('card')
            return card and card.rank.value in [2, 4, 6, 8, 10]
        elif name == "Odd Todd":
            card = context.get('card')
            return card and card.rank.value in [14, 13, 12, 11, 9, 7, 5, 3]
        elif name == "Scholar":
            card = context.get('card')
            return card and card.rank.value == 14  # Ace
        elif name == "Photograph":
            # First face card in hand
            card = context.get('card')
            if not (card and card.rank.value >= 11):
                return False
            # Check if it's the first
            return joker.persistent_state.get('first_face_triggered', False) == False
        elif name == "Ride the Bus":
            # Activates if no face cards scored
            return not any(c.rank.value >= 11 for c in scored_cards)
        elif name == "Green Joker":
            return True  # Always activates, accumulates over time
        elif name == "Blue Joker":
            return len(game_state.deck) > 0
        elif name == "Stone Joker":
            from balatro_env import Enhancement
            return any(c.enhancement == Enhancement.STONE for c in game_state.deck)
        elif name == "Steel Joker":
            from balatro_env import Enhancement
            return any(c.enhancement == Enhancement.STEEL for c in game_state.deck)
        elif name == "Glass Joker":
            from balatro_env import Enhancement
            return any(c.enhancement == Enhancement.GLASS for c in game_state.deck)
        elif name == "Loyalty Card":
            hands_played = joker.persistent_state.get('hands_played', 0)
            return hands_played % 6 == 5  # Every 6th hand
        elif name == "Blackboard":
            return all(c.suit.name in ["SPADES", "CLUBS"] for c in cards)
        elif name == "Runner":
            return hand_type and hand_type.name == "STRAIGHT"
        elif name == "DNA":
            return len(cards) == 1 and joker.persistent_state.get('first_hand', True)
        elif name == "Splash":
            return True
        
        # TIER 3 Conditionals
        elif name in ["Blueprint", "Brainstorm"]:
            return len(game_state.jokers) > 1
        elif name == "Vagabond":
            return game_state.money <= 4
        elif name == "Golden Ticket":
            from balatro_env import Enhancement
            card = context.get('card')
            return card and card.enhancement.name == "GOLD"
        elif name == "Space Joker":
            return random.random() < 0.25  # 1 in 4 chance
        elif name == "Business Card":
            card = context.get('card')
            return card and card.rank.value >= 11 and random.random() < 0.5
        elif name == "Seeing Double":
            has_club = any(c.suit.name == "CLUBS" for c in scored_cards)
            has_other = any(c.suit.name != "CLUBS" for c in scored_cards)
            return has_club and has_other
        elif name == "Banner":
            return game_state.discards_remaining > 0
        
        return True
    
    def _apply_joker_effect(
        self, 
        joker: JokerInstance, 
        game_state: Any, 
        context: Dict[str, Any],
        trigger: JokerTrigger
    ) -> Dict[str, Any]:
        """Apply the specific joker's effect"""
        effect = {'chips': 0, 'mult': 0, 'xmult': 1.0, 'money': 0, 'special_effects': []}
        name = joker.data.name
        
        # Simple stat boosts (Tier 1)
        if name == "Joker":
            effect['mult'] = 4
        elif name in ["Greedy Joker", "Lusty Joker", "Wrathful Joker", "Gluttonous Joker"]:
            effect['mult'] = 3
        elif name == "Jolly Joker":
            effect['mult'] = 8
        elif name == "Zany Joker":
            effect['mult'] = 12
        elif name == "Mad Joker":
            effect['mult'] = 10
        elif name == "Crazy Joker":
            effect['mult'] = 12
        elif name == "Droll Joker":
            effect['mult'] = 10
        elif name == "Half Joker":
            effect['mult'] = 20
        elif name == "Scary Face":
            effect['chips'] = 30
        elif name == "Abstract Joker":
            effect['mult'] = 3 * len(game_state.jokers)
        elif name == "Stuntman":
            effect['chips'] = 250
        elif name == "Misprint":
            effect['mult'] = random.randint(0, 23)
        
        # Tier 2 effects
        elif name == "Baron":
            king_count = sum(1 for c in context.get('cards', []) if c.rank.value == 13)
            effect['xmult'] = 1.5 ** king_count
        elif name == "Fibonacci":
            effect['mult'] = 8
        elif name == "Even Steven":
            effect['mult'] = 4
        elif name == "Odd Todd":
            effect['chips'] = 31
        elif name == "Scholar":
            effect['chips'] = 20
            effect['mult'] = 4
        elif name == "Photograph":
            effect['xmult'] = 2.0
            joker.persistent_state['first_face_triggered'] = True
        elif name == "Ride the Bus":
            if trigger == JokerTrigger.ON_HAND_PLAYED:
                counter = joker.persistent_state.get('counter', 0)
                joker.persistent_state['counter'] = counter + 1
                effect['mult'] = counter + 1
        elif name == "Green Joker":
            if trigger == JokerTrigger.ON_HAND_PLAYED:
                counter = joker.persistent_state.get('counter', 0)
                joker.persistent_state['counter'] = counter + 1
                effect['mult'] = counter
            elif trigger == JokerTrigger.ON_DISCARD:
                counter = joker.persistent_state.get('counter', 0)
                joker.persistent_state['counter'] = counter + 2
        elif name == "Blue Joker":
            effect['chips'] = 2 * len(game_state.deck)
        elif name == "Stone Joker":
            from balatro_env import Enhancement
            stone_count = sum(1 for c in game_state.deck if c.enhancement == Enhancement.STONE)
            effect['chips'] = 25 * stone_count
        elif name == "Steel Joker":
            from balatro_env import Enhancement
            steel_count = sum(1 for c in game_state.deck if c.enhancement == Enhancement.STEEL)
            effect['xmult'] = 1.5 ** steel_count
        elif name == "Glass Joker":
            from balatro_env import Enhancement
            glass_count = sum(1 for c in game_state.deck if c.enhancement == Enhancement.GLASS)
            effect['xmult'] = 2.0 ** glass_count
        elif name == "Loyalty Card":
            hands_played = joker.persistent_state.get('hands_played', 0)
            joker.persistent_state['hands_played'] = hands_played + 1
            if hands_played % 6 == 5:
                effect['xmult'] = 4.0
            elif hands_played % 3 == 2:
                effect['xmult'] = max(1.0, 4.0 - ((hands_played % 6) // 3))
        elif name == "Egg":
            if trigger == JokerTrigger.ON_ROUND_END:
                joker.sell_value += 3
        elif name == "Burglar":
            effect['mult'] = 3
            if trigger == JokerTrigger.ON_BLIND_COMPLETE:
                effect['money'] = 3
        elif name == "Blackboard":
            effect['xmult'] = 3.0
        elif name == "Runner":
            counter = joker.persistent_state.get('counter', 15)
            effect['chips'] = counter
            if trigger == JokerTrigger.ON_ROUND_END:
                joker.persistent_state['counter'] = counter + 10
        elif name == "Ice Cream":
            counter = joker.persistent_state.get('counter', 100)
            effect['chips'] = max(0, counter)
            joker.persistent_state['counter'] = counter - 5
        elif name == "DNA":
            if len(context.get('cards', [])) == 1:
                effect['special_effects'].append({'type': 'copy_card', 'card': context['cards'][0]})
            joker.persistent_state['first_hand'] = False
        
        # Tier 3 effects
        elif name == "Blueprint":
            # Copy joker to the right
            joker_idx = game_state.jokers.index(joker) if joker in game_state.jokers else -1
            if joker_idx >= 0 and joker_idx < len(game_state.jokers) - 1:
                right_joker = game_state.jokers[joker_idx + 1]
                # Recursively apply right joker's effect
                copied_effect = self._apply_joker_effect(right_joker, game_state, context, trigger)
                effect['chips'] += copied_effect.get('chips', 0)
                effect['mult'] += copied_effect.get('mult', 0)
                effect['xmult'] *= copied_effect.get('xmult', 1.0)
        elif name == "Brainstorm":
            # Copy leftmost joker
            if len(game_state.jokers) > 0:
                left_joker = game_state.jokers[0]
                if left_joker != joker:
                    copied_effect = self._apply_joker_effect(left_joker, game_state, context, trigger)
                    effect['chips'] += copied_effect.get('chips', 0)
                    effect['mult'] += copied_effect.get('mult', 0)
                    effect['xmult'] *= copied_effect.get('xmult', 1.0)
        elif name == "Credit Card":
            # Allows negative money (handled in shop logic)
            pass
        elif name == "Vagabond":
            effect['special_effects'].append({'type': 'create_tarot'})
        elif name == "Golden Ticket":
            effect['money'] = 4
        elif name == "Space Joker":
            effect['special_effects'].append({'type': 'upgrade_hand_level', 'hand_type': context.get('hand_type')})
        elif name == "Business Card":
            effect['money'] = 2
        elif name == "Ramen":
            mult_reduction = joker.persistent_state.get('mult_reduction', 0)
            effect['xmult'] = max(0.1, 2.0 - mult_reduction)
        elif name == "Seltzer":
            effect['special_effects'].append({'type': 'retrigger_cards'})
        elif name == "Castle":
            if trigger == JokerTrigger.ON_DISCARD:
                # Accumulate chips per suit
                for card in context.get('cards', []):
                    suit_key = f'suit_{card.suit.name}'
                    joker.persistent_state[suit_key] = joker.persistent_state.get(suit_key, 0) + 3
            elif trigger == JokerTrigger.ON_HAND_PLAYED:
                # Add accumulated chips
                total_chips = sum(v for k, v in joker.persistent_state.items() if k.startswith('suit_'))
                effect['chips'] = total_chips
                # Reset suits that were played
                for card in context.get('cards', []):
                    suit_key = f'suit_{card.suit.name}'
                    joker.persistent_state[suit_key] = 0
        elif name == "Ceremonial Dagger":
            effect['mult'] = joker.persistent_state.get('accumulated_mult', 0)
        elif name == "Banner":
            effect['chips'] = 30 * game_state.discards_remaining
        elif name == "Mail-In Rebate":
            if trigger == JokerTrigger.ON_DISCARD:
                target_rank = joker.persistent_state.get('target_rank')
                for card in context.get('cards', []):
                    if card.rank.value == target_rank:
                        effect['money'] = 3
        elif name == "Seeing Double":
            effect['xmult'] = 2.0
        
        return effect


def create_joker_instance(joker_name: str) -> Optional[JokerInstance]:
    """Create a joker instance from name"""
    joker_data = get_joker_by_name(joker_name)
    if joker_data:
        return JokerInstance(data=joker_data)
    return None


def get_random_joker(rarity: Optional[Rarity] = None) -> JokerInstance:
    """Get a random joker, optionally filtered by rarity"""
    if rarity:
        matching = [j for j in ALL_JOKERS if j.rarity == rarity]
        joker_data = random.choice(matching) if matching else random.choice(ALL_JOKERS)
    else:
        # Weighted by rarity
        weights = {
            Rarity.COMMON: 0.7,
            Rarity.UNCOMMON: 0.22,
            Rarity.RARE: 0.07,
            Rarity.LEGENDARY: 0.01
        }
        rarity_choice = random.choices(
            list(weights.keys()), 
            weights=list(weights.values())
        )[0]
        matching = [j for j in ALL_JOKERS if j.rarity == rarity_choice]
        joker_data = random.choice(matching)
    
    return JokerInstance(data=joker_data)


# ============================================================================
# JOKER SYNERGY DETECTION SYSTEM
# ============================================================================

@dataclass
class JokerSynergy:
    """Represents a synergy between jokers"""
    joker_names: List[str]
    synergy_type: str  # "multiplicative", "additive", "complementary", "enabling"
    strength: float  # 0.0 to 1.0
    description: str


class JokerSynergyDetector:
    """
    Detects and evaluates synergies between jokers
    Helps the agent learn which joker combinations are powerful
    """
    
    def __init__(self):
        self.synergies = self._define_synergies()
    
    def _define_synergies(self) -> List[JokerSynergy]:
        """Define known synergies between jokers"""
        return [
            # Multiplier synergies
            JokerSynergy(
                joker_names=["Baron", "Castle"],
                synergy_type="additive",
                strength=0.8,
                description="Both boost Kings specifically"
            ),
            JokerSynergy(
                joker_names=["Blueprint", "Brainstorm"],
                synergy_type="multiplicative",
                strength=0.95,
                description="Blueprint/Brainstorm copy other jokers"
            ),
            JokerSynergy(
                joker_names=["Ride the Bus", "Triboulet"],
                synergy_type="complementary",
                strength=0.7,
                description="Ride the Bus creates Kings/Queens that Triboulet boosts"
            ),
            
            # Chip/Mult synergies
            JokerSynergy(
                joker_names=["Sly Joker", "Crazy Joker", "Sly Joker"],
                synergy_type="additive",
                strength=0.6,
                description="Multiple pair-based bonuses"
            ),
            JokerSynergy(
                joker_names=["Fibonacci", "Even Steven"],
                synergy_type="complementary",
                strength=0.7,
                description="Fibonacci likes specific even cards"
            ),
            
            # Economy synergies
            JokerSynergy(
                joker_names=["Bull", "Space Joker"],
                synergy_type="enabling",
                strength=0.8,
                description="Bull maintains high money for Space Joker"
            ),
            JokerSynergy(
                joker_names=["Egg", "Stuntman"],
                synergy_type="enabling",
                strength=0.6,
                description="Egg adds cards, Stuntman likes small hands"
            ),
            
            # Retrigger synergies
            JokerSynergy(
                joker_names=["Sock and Buskin", "Hanging Chad"],
                synergy_type="multiplicative",
                strength=0.85,
                description="Multiple retriggers multiply effects"
            ),
            JokerSynergy(
                joker_names=["Hack", "Dusk"],
                synergy_type="complementary",
                strength=0.7,
                description="Hack retriggers all cards, Dusk likes final hand"
            ),
            
            # Scaling synergies
            JokerSynergy(
                joker_names=["Square Joker", "Constellation"],
                synergy_type="additive",
                strength=0.75,
                description="Both scale with cards played"
            ),
            JokerSynergy(
                joker_names=["Green Joker", "Red Card"],
                synergy_type="additive",
                strength=0.7,
                description="Both scale with discards"
            ),
            
            # Hand-type synergies
            JokerSynergy(
                joker_names=["Smeared Joker", "Splash"],
                synergy_type="enabling",
                strength=0.9,
                description="Smeared enables flush, Splash boosts flush"
            ),
            JokerSynergy(
                joker_names=["Four Fingers", "Shoot the Moon"],
                synergy_type="enabling",
                strength=0.85,
                description="Four Fingers makes straights easier, Shoot boosted by Queens"
            ),
            
            # Destruction/creation synergies
            JokerSynergy(
                joker_names=["Glass Joker", "Hologram"],
                synergy_type="enabling",
                strength=0.8,
                description="Glass creates/destroys, Hologram scales on destroyed"
            ),
            JokerSynergy(
                joker_names=["Burglar", "Swashbuckler"],
                synergy_type="additive",
                strength=0.6,
                description="Both encourage selling jokers"
            ),
            
            # Probabilistic synergies
            JokerSynergy(
                joker_names=["Oops! All 6s", "Even Steven"],
                synergy_type="multiplicative",
                strength=0.95,
                description="All 6s creates even cards for Even Steven"
            ),
            JokerSynergy(
                joker_names=["Ancient Joker", "Vampire"],
                synergy_type="complementary",
                strength=0.65,
                description="Both have suit-based triggers"
            ),
            
            # Edition synergies
            JokerSynergy(
                joker_names=["Holographic", "Polychrome"],  # These are editions, not jokers, but tracking
                synergy_type="multiplicative",
                strength=0.9,
                description="Multiple mult multipliers stack powerfully"
            ),
        ]
    
    def detect_synergies(self, jokers: List[JokerInstance]) -> List[JokerSynergy]:
        """
        Detect active synergies in current joker lineup
        
        Args:
            jokers: List of active joker instances
        
        Returns:
            List of detected synergies
        """
        if len(jokers) < 2:
            return []
        
        active_names = {j.data.name for j in jokers}
        detected = []
        
        for synergy in self.synergies:
            # Check if all jokers in synergy are present
            if all(name in active_names for name in synergy.joker_names):
                detected.append(synergy)
        
        return detected
    
    def evaluate_joker_lineup(self, jokers: List[JokerInstance]) -> Dict[str, Any]:
        """
        Evaluate overall quality of joker lineup considering synergies
        
        Returns:
            Dictionary with evaluation metrics
        """
        synergies = self.detect_synergies(jokers)
        
        # Base score from individual jokers
        base_score = sum(self._score_individual_joker(j) for j in jokers)
        
        # Synergy bonus
        synergy_score = sum(s.strength for s in synergies)
        
        # Coverage: how many different triggers are covered
        triggers = {j.data.trigger for j in jokers}
        coverage_score = len(triggers) / len(JokerTrigger)
        
        # Diversity: avoid too many similar jokers
        names = [j.data.name for j in jokers]
        diversity_score = len(set(names)) / max(len(names), 1)
        
        total_score = base_score + synergy_score * 2.0 + coverage_score + diversity_score
        
        return {
            "total_score": total_score,
            "base_score": base_score,
            "synergy_score": synergy_score,
            "synergies_active": len(synergies),
            "coverage_score": coverage_score,
            "diversity_score": diversity_score,
            "synergies": synergies
        }
    
    def _score_individual_joker(self, joker: JokerInstance) -> float:
        """Score a single joker based on rarity and edition"""
        rarity_scores = {
            Rarity.COMMON: 1.0,
            Rarity.UNCOMMON: 1.5,
            Rarity.RARE: 2.5,
            Rarity.LEGENDARY: 4.0
        }
        
        edition_bonus = {
            JokerEdition.NONE: 0.0,
            JokerEdition.FOIL: 0.3,
            JokerEdition.HOLOGRAPHIC: 0.5,
            JokerEdition.POLYCHROME: 0.8,
            JokerEdition.NEGATIVE: 1.0
        }
        
        base = rarity_scores.get(joker.data.rarity, 1.0)
        bonus = edition_bonus.get(joker.edition, 0.0)
        
        # Penalty for perishable
        if joker.is_perishable() and joker.rounds_remaining <= 2:
            bonus -= 0.5
        
        return base + bonus
    
    def suggest_next_joker(
        self, 
        current_jokers: List[JokerInstance],
        available_jokers: List[JokerInstance]
    ) -> Optional[JokerInstance]:
        """
        Suggest which joker to buy from shop based on synergies
        
        Args:
            current_jokers: Currently owned jokers
            available_jokers: Jokers available in shop
        
        Returns:
            Best joker to buy, or None if no good options
        """
        if not available_jokers:
            return None
        
        best_joker = None
        best_score = -float('inf')
        
        for candidate in available_jokers:
            # Simulate adding this joker
            test_lineup = current_jokers + [candidate]
            evaluation = self.evaluate_joker_lineup(test_lineup)
            
            # Score improvement
            current_eval = self.evaluate_joker_lineup(current_jokers)
            score_delta = evaluation["total_score"] - current_eval["total_score"]
            
            if score_delta > best_score:
                best_score = score_delta
                best_joker = candidate
        
        return best_joker if best_score > 0 else None
    
    def to_feature_vector(self, jokers: List[JokerInstance]) -> np.ndarray:
        """
        Convert synergy information to a feature vector for neural network
        
        Returns:
            Feature vector encoding synergy information
        """
        vec = np.zeros(32)  # 32-dim synergy feature vector
        
        if len(jokers) < 2:
            return vec
        
        evaluation = self.evaluate_joker_lineup(jokers)
        synergies = evaluation["synergies"]
        
        # Aggregate synergy statistics
        vec[0] = min(evaluation["total_score"] / 20.0, 1.0)  # Normalized total score
        vec[1] = min(evaluation["synergy_score"] / 5.0, 1.0)  # Normalized synergy score
        vec[2] = evaluation["coverage_score"]
        vec[3] = evaluation["diversity_score"]
        vec[4] = len(synergies) / 5.0  # Number of active synergies (normalized)
        
        # Synergy type breakdown
        synergy_types = {"multiplicative": 0, "additive": 0, "complementary": 0, "enabling": 0}
        for s in synergies:
            synergy_types[s.synergy_type] += 1
        
        vec[5] = min(synergy_types["multiplicative"] / 3.0, 1.0)
        vec[6] = min(synergy_types["additive"] / 3.0, 1.0)
        vec[7] = min(synergy_types["complementary"] / 3.0, 1.0)
        vec[8] = min(synergy_types["enabling"] / 3.0, 1.0)
        
        # Trigger coverage breakdown
        triggers_covered = {j.data.trigger for j in jokers}
        for i, trigger in enumerate(JokerTrigger):
            if i < 15:  # First 15 dimensions for triggers
                vec[9 + i] = 1.0 if trigger in triggers_covered else 0.0
        
        return vec


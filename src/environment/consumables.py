"""
Consumable Card System for Balatro

Implements Tarot, Planet, and Spectral cards with their effects.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import random

try:
    from . import balatro_content
    from .balatro_content import (
        TarotData, PlanetData, SpectralData,
        TAROT_CARDS, PLANET_CARDS, SPECTRAL_CARDS,
        get_tarot_by_name, get_planet_by_hand_type, get_spectral_by_name
    )
except ImportError:
    import balatro_content
    from balatro_content import (
        TarotData, PlanetData, SpectralData,
        TAROT_CARDS, PLANET_CARDS, SPECTRAL_CARDS,
        get_tarot_by_name, get_planet_by_hand_type, get_spectral_by_name
    )


@dataclass
class TarotCard:
    """Instance of a tarot card"""
    data: TarotData
    
    def to_vector(self) -> np.ndarray:
        """Convert to neural network feature vector"""
        vec = np.zeros(64)
        
        # Type identifier
        vec[0] = 1.0  # Tarot
        
        # Target required flag
        vec[1] = 1.0 if self.data.target_required else 0.0
        
        # Effect type (hash-based)
        vec[2] = hash(self.data.effect_type) % 100 / 100.0
        
        # Name hash for unique identification
        vec[3] = hash(self.data.name) % 10000 / 10000.0
        
        return vec
    
    def apply_effect(
        self, 
        game_state: Any, 
        target_cards: Optional[List[Any]] = None
    ) -> Dict[str, Any]:
        """
        Apply the tarot card's effect
        
        Returns:
            Dict with effect results
        """
        effect_type = self.data.effect_type
        result = {'success': False, 'message': ''}
        
        if effect_type == "create_last_consumable":
            # The Fool - recreate last tarot/planet
            result['special_action'] = 'create_last_consumable'
            result['success'] = True
            
        elif effect_type in ["enhance_to_lucky", "enhance_to_mult", "enhance_to_bonus", 
                              "enhance_to_wild", "enhance_to_steel", "enhance_to_glass",
                              "enhance_to_gold", "enhance_to_stone"]:
            # Enhancement tarots
            if not target_cards or len(target_cards) == 0:
                result['message'] = 'No cards selected'
                return result
            
            enhancement_map = {
                "enhance_to_lucky": "LUCKY",
                "enhance_to_mult": "MULT",
                "enhance_to_bonus": "BONUS",
                "enhance_to_wild": "WILD",
                "enhance_to_steel": "STEEL",
                "enhance_to_glass": "GLASS",
                "enhance_to_gold": "GOLD",
                "enhance_to_stone": "STONE",
            }
            
            result['enhance_cards'] = {
                'cards': target_cards,
                'enhancement': enhancement_map[effect_type]
            }
            result['success'] = True
            
        elif effect_type == "create_planet":
            # The High Priestess - create 2 random planet cards
            result['special_action'] = 'create_planets'
            result['count'] = 2
            result['success'] = True
            
        elif effect_type == "create_tarot":
            # The Emperor - create 2 random tarot cards
            result['special_action'] = 'create_tarots'
            result['count'] = 2
            result['success'] = True
            
        elif effect_type == "double_money":
            # The Hermit - double money (max $20)
            doubled = min(game_state.money * 2, game_state.money + 20)
            result['money_change'] = doubled - game_state.money
            result['success'] = True
            
        elif effect_type == "add_edition_joker":
            # Wheel of Fortune - random edition to joker
            if len(game_state.jokers) == 0:
                result['message'] = 'No jokers'
                return result
            result['special_action'] = 'add_edition_joker'
            result['success'] = random.random() < 0.25
            
        elif effect_type == "increase_rank":
            # Strength - increase rank by 1
            if not target_cards:
                result['message'] = 'No cards selected'
                return result
            result['rank_change'] = {
                'cards': target_cards,
                'change': +1
            }
            result['success'] = True
            
        elif effect_type == "destroy_cards":
            # The Hanged Man - destroy cards
            if not target_cards:
                result['message'] = 'No cards selected'
                return result
            result['destroy_cards'] = target_cards
            result['success'] = True
            
        elif effect_type == "convert_cards":
            # Death - convert left to right
            if not target_cards or len(target_cards) < 2:
                result['message'] = 'Need 2 cards'
                return result
            result['convert_cards'] = {
                'source': target_cards[0],
                'target': target_cards[1]
            }
            result['success'] = True
            
        elif effect_type == "joker_sell_value":
            # Temperance - give joker sell value as money
            total_value = sum(j.sell_value for j in game_state.jokers)
            result['money_change'] = min(total_value, 50)
            result['success'] = True
            
        elif effect_type in ["convert_to_diamonds", "convert_to_clubs", 
                              "convert_to_hearts", "convert_to_spades"]:
            # Suit conversion tarots
            if not target_cards:
                result['message'] = 'No cards selected'
                return result
            
            suit_map = {
                "convert_to_diamonds": "DIAMONDS",
                "convert_to_clubs": "CLUBS",
                "convert_to_hearts": "HEARTS",
                "convert_to_spades": "SPADES",
            }
            
            result['convert_suit'] = {
                'cards': target_cards,
                'suit': suit_map[effect_type]
            }
            result['success'] = True
            
        elif effect_type == "create_joker":
            # Judgement - create random joker
            if len(game_state.jokers) >= game_state.max_joker_slots:
                result['message'] = 'No joker slots'
                return result
            result['special_action'] = 'create_joker'
            result['success'] = True
        
        return result


@dataclass
class PlanetCard:
    """Instance of a planet card"""
    data: PlanetData
    
    def to_vector(self) -> np.ndarray:
        """Convert to neural network feature vector"""
        vec = np.zeros(64)
        
        # Type identifier
        vec[0] = 2.0  # Planet
        
        # Hand type (hash-based)
        vec[1] = hash(self.data.hand_type) % 100 / 100.0
        
        # Upgrade amounts
        vec[2] = self.data.chip_increase / 100.0
        vec[3] = self.data.mult_increase / 10.0
        
        # Name hash
        vec[4] = hash(self.data.name) % 10000 / 10000.0
        
        return vec
    
    def apply_effect(self, game_state: Any) -> Dict[str, Any]:
        """
        Apply planet card effect - level up a hand type
        
        Returns:
            Dict with effect results
        """
        from balatro_env import HandType
        
        # Find the HandType enum member
        hand_type = None
        for ht in HandType:
            if ht.name == self.data.hand_type:
                hand_type = ht
                break
        
        if not hand_type:
            return {'success': False, 'message': 'Invalid hand type'}
        
        # Get current level
        current_chips, current_mult = game_state.hand_levels.get(hand_type, (5, 1))
        
        # Upgrade
        new_chips = current_chips + self.data.chip_increase
        new_mult = current_mult + self.data.mult_increase
        
        return {
            'success': True,
            'hand_type': hand_type,
            'old_level': (current_chips, current_mult),
            'new_level': (new_chips, new_mult),
            'upgrade_hand': {
                'hand_type': hand_type,
                'chips': new_chips,
                'mult': new_mult
            }
        }


@dataclass
class SpectralCard:
    """Instance of a spectral card"""
    data: SpectralData
    
    def to_vector(self) -> np.ndarray:
        """Convert to neural network feature vector"""
        vec = np.zeros(64)
        
        # Type identifier
        vec[0] = 3.0  # Spectral
        
        # Target required flag
        vec[1] = 1.0 if self.data.target_required else 0.0
        
        # Effect type (hash-based)
        vec[2] = hash(self.data.effect_type) % 100 / 100.0
        
        # Name hash
        vec[3] = hash(self.data.name) % 10000 / 10000.0
        
        return vec
    
    def apply_effect(
        self, 
        game_state: Any, 
        target_cards: Optional[List[Any]] = None
    ) -> Dict[str, Any]:
        """
        Apply spectral card effect
        
        Returns:
            Dict with effect results
        """
        effect_type = self.data.effect_type
        result = {'success': False, 'message': ''}
        
        if effect_type == "destroy_add_faces":
            # Familiar - destroy 1, add 3 enhanced faces
            if len(game_state.hand) == 0:
                result['message'] = 'No cards to destroy'
                return result
            result['destroy_random'] = 1
            result['add_cards'] = {
                'type': 'face',
                'count': 3,
                'enhanced': True
            }
            result['success'] = True
            
        elif effect_type == "destroy_add_aces":
            # Grim - destroy 1, add 2 enhanced aces
            if len(game_state.hand) == 0:
                result['message'] = 'No cards to destroy'
                return result
            result['destroy_random'] = 1
            result['add_cards'] = {
                'type': 'ace',
                'count': 2,
                'enhanced': True
            }
            result['success'] = True
            
        elif effect_type == "destroy_add_numbered":
            # Incantation - destroy 1, add 4 enhanced numbered
            if len(game_state.hand) == 0:
                result['message'] = 'No cards to destroy'
                return result
            result['destroy_random'] = 1
            result['add_cards'] = {
                'type': 'numbered',  # 2-10
                'count': 4,
                'enhanced': True
            }
            result['success'] = True
            
        elif effect_type == "add_gold_seal":
            # Talisman - add gold seal
            if not target_cards or len(target_cards) == 0:
                result['message'] = 'No card selected'
                return result
            result['add_seal'] = {
                'cards': target_cards,
                'seal': 'GOLD'
            }
            result['success'] = True
            
        elif effect_type == "add_edition":
            # Aura - add random edition
            if not target_cards or len(target_cards) == 0:
                result['message'] = 'No card selected'
                return result
            editions = ['FOIL', 'HOLOGRAPHIC', 'POLYCHROME']
            result['add_edition'] = {
                'cards': target_cards,
                'edition': random.choice(editions)
            }
            result['success'] = True
            
        elif effect_type == "create_rare_joker":
            # Wraith - create rare joker, set money to $0
            if len(game_state.jokers) >= game_state.max_joker_slots:
                result['message'] = 'No joker slots'
                return result
            result['create_rare_joker'] = True
            result['money_change'] = -game_state.money
            result['success'] = True
            
        elif effect_type == "convert_all_suit":
            # Sigil - convert all cards to one suit
            if len(game_state.hand) == 0:
                result['message'] = 'No cards in hand'
                return result
            suits = ['DIAMONDS', 'CLUBS', 'HEARTS', 'SPADES']
            result['convert_all_suit'] = random.choice(suits)
            result['success'] = True
            
        elif effect_type == "convert_all_rank":
            # Ouija - convert all cards to one rank
            if len(game_state.hand) == 0:
                result['message'] = 'No cards in hand'
                return result
            result['convert_all_rank'] = random.randint(2, 14)  # 2-A
            result['success'] = True
            
        elif effect_type == "add_negative":
            # Ectoplasm - add negative to joker, -1 hand per round
            if len(game_state.jokers) == 0:
                result['message'] = 'No jokers'
                return result
            result['add_negative_joker'] = True
            result['hands_penalty'] = -1
            result['success'] = True
            
        elif effect_type == "destroy_5_earn_20":
            # Immolate - destroy 5 cards, earn $20
            if len(game_state.hand) < 5:
                result['message'] = 'Need 5 cards in hand'
                return result
            result['destroy_random'] = 5
            result['money_change'] = 20
            result['success'] = True
            
        elif effect_type == "copy_destroy_jokers":
            # Ankh - copy random joker, destroy others
            if len(game_state.jokers) == 0:
                result['message'] = 'No jokers'
                return result
            result['copy_destroy_jokers'] = True
            result['success'] = True
            
        elif effect_type in ["add_red_seal", "add_blue_seal", "add_purple_seal"]:
            # Seal adding spectrals
            if not target_cards or len(target_cards) == 0:
                result['message'] = 'No card selected'
                return result
            
            seal_map = {
                "add_red_seal": "RED",
                "add_blue_seal": "BLUE",
                "add_purple_seal": "PURPLE",
            }
            
            result['add_seal'] = {
                'cards': target_cards,
                'seal': seal_map[effect_type]
            }
            result['success'] = True
            
        elif effect_type == "poly_destroy_jokers":
            # Hex - add polychrome to joker, destroy others
            if len(game_state.jokers) == 0:
                result['message'] = 'No jokers'
                return result
            result['poly_destroy_jokers'] = True
            result['success'] = True
        
        return result


# ============================================================================
# Helper Functions
# ============================================================================

def get_random_tarot() -> TarotCard:
    """Get a random tarot card"""
    tarot_data = random.choice(TAROT_CARDS)
    return TarotCard(data=tarot_data)


def get_random_planet() -> PlanetCard:
    """Get a random planet card"""
    planet_data = random.choice(PLANET_CARDS)
    return PlanetCard(data=planet_data)


def get_random_spectral() -> SpectralCard:
    """Get a random spectral card"""
    spectral_data = random.choice(SPECTRAL_CARDS)
    return SpectralCard(data=spectral_data)


def get_tarot_card(name: str) -> Optional[TarotCard]:
    """Get specific tarot card by name"""
    tarot_data = get_tarot_by_name(name)
    if tarot_data:
        return TarotCard(data=tarot_data)
    return None


def get_planet_card(name: str) -> Optional[PlanetCard]:
    """Get specific planet card by name"""
    for planet_data in PLANET_CARDS:
        if planet_data.name == name:
            return PlanetCard(data=planet_data)
    return None


def get_spectral_card(name: str) -> Optional[SpectralCard]:
    """Get specific spectral card by name"""
    spectral_data = get_spectral_by_name(name)
    if spectral_data:
        return SpectralCard(data=spectral_data)
    return None


def get_planet_for_hand(hand_type_name: str) -> Optional[PlanetCard]:
    """Get planet card that upgrades a specific hand type"""
    planet_data = get_planet_by_hand_type(hand_type_name)
    if planet_data:
        return PlanetCard(data=planet_data)
    return None


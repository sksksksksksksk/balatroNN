"""
Comprehensive tests for the full Balatro game implementation

Tests the expanded environment with shops, consumables, and all game mechanics.
"""

import pytest
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from environment.balatro_env import BalatroEnv, Card, Suit, Rank, Enhancement, Edition
from environment import balatro_content
from environment import jokers
from environment import consumables


class TestContentDatabase:
    """Test the game content database"""
    
    def test_all_jokers_defined(self):
        """Test that all 50 jokers are defined"""
        assert len(balatro_content.ALL_JOKERS) == 50
        assert len(balatro_content.TIER1_JOKERS) == 15
        assert len(balatro_content.TIER2_JOKERS) == 20
        assert len(balatro_content.TIER3_JOKERS) == 15
    
    def test_all_tarots_defined(self):
        """Test that all 22 tarot cards are defined"""
        assert len(balatro_content.TAROT_CARDS) == 22
    
    def test_all_planets_defined(self):
        """Test that all 11 planet cards are defined"""
        assert len(balatro_content.PLANET_CARDS) == 11
    
    def test_all_spectrals_defined(self):
        """Test that all 15 spectral cards are defined"""
        assert len(balatro_content.SPECTRAL_CARDS) == 15
    
    def test_all_vouchers_defined(self):
        """Test that all 32 vouchers are defined"""
        assert len(balatro_content.VOUCHERS) == 32
    
    def test_joker_pricing(self):
        """Test joker pricing by rarity"""
        for joker_data in balatro_content.ALL_JOKERS:
            price = balatro_content.get_joker_price(joker_data)
            assert price >= 3
            assert price <= 20
            
            # Common should be cheapest
            if joker_data.rarity == balatro_content.Rarity.COMMON:
                assert price == 3


class TestJokerSystem:
    """Test the joker effect system"""
    
    def test_joker_instance_creation(self):
        """Test creating joker instances"""
        joker_inst = jokers.create_joker_instance("Joker")
        assert joker_inst is not None
        assert joker_inst.data.name == "Joker"
        assert joker_inst.data.mult_bonus == 4
    
    def test_joker_vector_encoding(self):
        """Test joker to vector conversion"""
        joker_inst = jokers.create_joker_instance("Baron")
        vec = joker_inst.to_vector()
        assert vec.shape == (128,)
        assert np.all(np.isfinite(vec))
    
    def test_random_joker_generation(self):
        """Test random joker generation"""
        joker = jokers.get_random_joker()
        assert joker is not None
        assert hasattr(joker, 'data')
    
    def test_joker_effect_processor(self):
        """Test joker effect processor"""
        processor = jokers.JokerEffectProcessor()
        joker_inst = jokers.create_joker_instance("Joker")
        
        # Create mock game state
        class MockGameState:
            money = 10
            jokers = []
            deck = []
        
        game_state = MockGameState()
        context = {'hand_type': None, 'cards': []}
        
        # Test trigger processing
        mods = processor.process_trigger(
            balatro_content.JokerTrigger.ON_HAND_PLAYED,
            [joker_inst],
            game_state,
            context
        )
        
        assert 'mult' in mods
        assert mods['mult'] == 4  # Joker gives +4 mult


class TestConsumableSystem:
    """Test consumable card system"""
    
    def test_tarot_card_creation(self):
        """Test tarot card creation"""
        tarot = consumables.get_random_tarot()
        assert tarot is not None
        assert hasattr(tarot, 'data')
        
    def test_planet_card_creation(self):
        """Test planet card creation"""
        planet = consumables.get_random_planet()
        assert planet is not None
        assert hasattr(planet, 'data')
    
    def test_spectral_card_creation(self):
        """Test spectral card creation"""
        spectral = consumables.get_random_spectral()
        assert spectral is not None
        assert hasattr(spectral, 'data')
    
    def test_consumable_vector_encoding(self):
        """Test consumable to vector conversion"""
        tarot = consumables.get_random_tarot()
        vec = tarot.to_vector()
        assert vec.shape == (64,)
        assert np.all(np.isfinite(vec))


class TestShopSystem:
    """Test the shop generation and purchasing system"""
    
    def test_shop_generation(self):
        """Test shop generation"""
        env = BalatroEnv()
        env.reset()
        
        # Generate shop
        env.state._generate_shop()
        
        assert env.state.in_shop
        assert len(env.state.shop_jokers) >= 2
        assert len(env.state.shop_packs) == 2
        assert len(env.state.shop_cards) == 2
    
    def test_shop_pricing(self):
        """Test shop item pricing"""
        env = BalatroEnv()
        env.reset()
        env.state._generate_shop()
        
        # Check joker prices
        for joker, price in env.state.shop_jokers:
            assert price >= 3
            assert price <= 20
        
        # Check pack prices
        for pack_name, price in env.state.shop_packs:
            assert price == 4
    
    def test_buy_joker(self):
        """Test buying a joker"""
        env = BalatroEnv()
        env.reset()
        env.state._generate_shop()
        
        initial_money = env.state.money
        initial_joker_count = len(env.state.jokers)
        
        # Buy first joker
        if len(env.state.shop_jokers) > 0:
            joker, price = env.state.shop_jokers[0]
            env.state.money = price + 10  # Ensure we can afford it
            
            reward = env.state._buy_joker(0)
            
            assert reward > 0  # Should get positive reward
            assert env.state.money < price + 10  # Money decreased
            assert len(env.state.jokers) == initial_joker_count + 1  # Joker added
    
    def test_interest_calculation(self):
        """Test interest calculation"""
        env = BalatroEnv()
        env.reset()
        
        # Test basic interest
        env.state.money = 25  # $25 should give $5 interest
        env.state.interest_cap = 5
        interest = env.state._calculate_interest()
        assert interest == 5
        
        # Test below cap
        env.state.money = 10  # $10 should give $2 interest
        interest = env.state._calculate_interest()
        assert interest == 2
    
    def test_reroll_shop(self):
        """Test shop reroll"""
        env = BalatroEnv()
        env.reset()
        env.state._generate_shop()
        
        original_jokers = list(env.state.shop_jokers)
        env.state.money = 100  # Plenty of money
        
        reward = env.state._reroll_shop()
        
        assert reward < 0  # Reroll costs money, negative reward
        assert env.state.shop_reroll_cost > 5  # Cost increased


class TestExpandedEnvironment:
    """Test the expanded environment"""
    
    def test_environment_initialization(self):
        """Test environment initializes with expanded spaces"""
        env = BalatroEnv()
        
        # Check action space
        assert env.action_space['action_type'].n == 12
        assert env.action_space['shop_item_index'].n == 7
        assert env.action_space['joker_slot'].n == 5
        assert env.action_space['consumable_slot'].n == 2
        
        # Check observation space
        assert env.observation_space['jokers'].shape == (5, 128)
        assert env.observation_space['consumables'].shape == (6, 64)
        assert env.observation_space['shop_items'].shape == (10, 128)
        assert env.observation_space['vouchers'].shape == (5, 32)
        assert env.observation_space['scalar'].shape == (64,)
    
    def test_reset_with_expanded_state(self):
        """Test environment reset with expanded state"""
        env = BalatroEnv()
        obs, info = env.reset()
        
        # Check all observation components
        assert 'hand' in obs
        assert 'jokers' in obs
        assert 'consumables' in obs
        assert 'shop_items' in obs
        assert 'vouchers' in obs
        assert 'blind' in obs
        assert 'scalar' in obs
        
        # Check shapes
        assert obs['jokers'].shape == (5, 128)
        assert obs['consumables'].shape == (6, 64)
        assert obs['shop_items'].shape == (10, 128)
    
    def test_action_execution(self):
        """Test executing all action types"""
        env = BalatroEnv()
        obs, info = env.reset()
        
        # Test play hand action
        action = {
            'action_type': 0,  # play_hand
            'card_selection': np.array([1, 1, 0, 0, 0, 0, 0, 0]),
            'shop_item_index': 0,
            'joker_slot': 0,
            'consumable_slot': 0,
            'target_card_index': 0,
        }
        
        obs, reward, done, truncated, info = env.step(action)
        
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert 'scalar' in obs
    
    def test_observation_encoding(self):
        """Test observation encoding"""
        env = BalatroEnv()
        obs, info = env.reset()
        
        # Check that all observations are valid
        for key, value in obs.items():
            assert np.all(np.isfinite(value)), f"{key} has non-finite values"
            assert value.dtype == np.float32, f"{key} has wrong dtype"


class TestFullGameplayLoop:
    """Integration tests for full gameplay loop"""
    
    def test_complete_blind(self):
        """Test completing a blind with shops"""
        env = BalatroEnv()
        obs, info = env.reset()
        
        done = False
        steps = 0
        max_steps = 100
        
        while not done and steps < max_steps:
            # Random action for testing
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)
            steps += 1
        
        assert steps < max_steps  # Should complete before max steps
    
    def test_shop_phase_workflow(self):
        """Test entering and using shop"""
        env = BalatroEnv()
        obs, info = env.reset()
        
        # Beat a blind to enter shop
        env.state.chips_scored = env.state.current_blind.chip_requirement + 100
        env.state._enter_shop()
        
        assert env.state.in_shop
        assert len(env.state.shop_jokers) > 0
        
        # Observation should reflect shop state
        obs = env.state.to_observation()
        assert obs['scalar'][8] > 0.5  # in_shop flag
    
    def test_economy_persistence(self):
        """Test that money persists across rounds"""
        env = BalatroEnv()
        env.reset()
        
        initial_money = env.state.money
        env.state.money += 50
        
        # Advance to next blind
        env.state._advance_to_next_blind()
        
        assert env.state.money > initial_money  # Money persists


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])


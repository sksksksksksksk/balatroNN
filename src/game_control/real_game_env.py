"""
Real game environment wrapper

Connects trained model to actual Balatro game through vision and input control.
"""

import time
import numpy as np
from typing import Dict, Tuple, Any, Optional
import gymnasium as gym

from ..vision.screen_capture import ScreenCapture, ScreenRegions
from ..vision.state_detector import StateDetector
from ..vision.card_detector import CardDetector, detect_card_positions
from .input_controller import InputController


class RealBalatroEnv(gym.Env):
    """
    Gymnasium environment that interfaces with real Balatro game
    
    Uses computer vision to read game state and input automation
    to control the game.
    """
    
    def __init__(self, screen_regions: ScreenRegions, 
                 capture_delay: float = 0.5,
                 action_delay: float = 1.0):
        """
        Initialize real game environment
        
        Args:
            screen_regions: Configured screen regions for game UI
            capture_delay: Delay after each screen capture
            action_delay: Delay after each action
        """
        super().__init__()
        
        self.capture_delay = capture_delay
        self.action_delay = action_delay
        
        # Initialize components
        self.screen_capture = ScreenCapture(regions=screen_regions)
        self.state_detector = StateDetector(self.screen_capture)
        self.input_controller = InputController()
        self.card_detector = CardDetector()
        
        # Define observation and action spaces (same as simulation)
        self.observation_space = gym.spaces.Dict({
            "hand": gym.spaces.Box(low=0, high=1, shape=(8, 32), dtype=np.float32),
            "jokers": gym.spaces.Box(low=0, high=1, shape=(5, 64), dtype=np.float32),
            "blind": gym.spaces.Box(low=0, high=1, shape=(16,), dtype=np.float32),
            "scalar": gym.spaces.Box(low=0, high=1, shape=(32,), dtype=np.float32),
        })
        
        self.action_space = gym.spaces.Dict({
            "action_type": gym.spaces.Discrete(5),
            "card_selection": gym.spaces.MultiBinary(8),
            "shop_selection": gym.spaces.Discrete(7),
        })
        
        self.last_state = None
        self.step_count = 0
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[Dict, Dict]:
        """
        Reset is not applicable for real game - just capture current state
        
        User should manually start a new run in the game.
        """
        print("Capturing initial game state...")
        print("Make sure Balatro is running and visible!")
        
        time.sleep(2)  # Give user time to focus game
        
        obs = self._capture_observation()
        self.last_state = obs
        self.step_count = 0
        
        info = {"message": "Real game connected - awaiting actions"}
        
        return obs, info
    
    def step(self, action: Dict) -> Tuple[Dict, float, bool, bool, Dict]:
        """
        Execute action in real game
        
        Args:
            action: Action dictionary from model
            
        Returns:
            observation, reward, terminated, truncated, info
        """
        self.step_count += 1
        
        # Execute action
        success = self._execute_action(action)
        
        if not success:
            print("Action execution failed!")
            # Return previous state with penalty
            return self.last_state, -10.0, False, False, {"error": "action_failed"}
        
        # Wait for action to complete
        time.sleep(self.action_delay)
        
        # Capture new state
        new_obs = self._capture_observation()
        
        # Calculate reward (simplified)
        reward = self._calculate_reward(self.last_state, new_obs)
        
        self.last_state = new_obs
        
        # Check if game ended (simplified - would need proper detection)
        terminated = False
        truncated = self.step_count >= 1000
        
        info = {
            "step": self.step_count,
            "action_success": success
        }
        
        return new_obs, reward, terminated, truncated, info
    
    def _capture_observation(self) -> Dict[str, np.ndarray]:
        """
        Capture current game state as observation
        
        Returns:
            Observation dictionary
        """
        time.sleep(self.capture_delay)
        
        try:
            # Get state from vision
            obs = self.state_detector.get_state_for_model()
            
            # Update card positions for input controller
            hand_image = self.screen_capture.capture_hand()
            card_positions = detect_card_positions(hand_image)
            self.input_controller.update_card_positions(card_positions)
            
            return obs
        
        except Exception as e:
            print(f"Error capturing observation: {e}")
            # Return zero observation on error
            return {
                "hand": np.zeros((8, 32), dtype=np.float32),
                "jokers": np.zeros((5, 64), dtype=np.float32),
                "blind": np.zeros(16, dtype=np.float32),
                "scalar": np.zeros(32, dtype=np.float32)
            }
    
    def _execute_action(self, action: Dict) -> bool:
        """
        Execute action in game using input controller
        
        Args:
            action: Action dictionary
            
        Returns:
            True if action executed successfully
        """
        action_type = action["action_type"]
        
        if isinstance(action_type, np.ndarray):
            action_type = int(action_type.item())
        
        try:
            if action_type == 0:  # Play hand
                # Get selected cards
                card_selection = action["card_selection"]
                if isinstance(card_selection, np.ndarray):
                    selected_indices = np.where(card_selection > 0.5)[0].tolist()
                else:
                    selected_indices = [i for i, v in enumerate(card_selection) if v > 0.5]
                
                return self.input_controller.play_hand_action(selected_indices)
            
            elif action_type == 1:  # Discard
                card_selection = action["card_selection"]
                if isinstance(card_selection, np.ndarray):
                    selected_indices = np.where(card_selection > 0.5)[0].tolist()
                else:
                    selected_indices = [i for i, v in enumerate(card_selection) if v > 0.5]
                
                return self.input_controller.discard_action(selected_indices)
            
            elif action_type == 2:  # Shop buy
                shop_selection = action["shop_selection"]
                if isinstance(shop_selection, np.ndarray):
                    shop_selection = int(shop_selection.item())
                
                return self.input_controller.click_shop_item(shop_selection)
            
            elif action_type == 3:  # Skip
                return self.input_controller.press_skip()
            
            elif action_type == 4:  # Reroll shop
                return self.input_controller.reroll_shop()
            
            else:
                print(f"Unknown action type: {action_type}")
                return False
        
        except Exception as e:
            print(f"Error executing action: {e}")
            return False
    
    def _calculate_reward(self, old_obs: Dict, new_obs: Dict) -> float:
        """
        Calculate reward based on state change
        
        This is simplified - in practice you'd detect:
        - Chips scored
        - Blind completion
        - Money gained/lost
        - Game over
        
        Args:
            old_obs: Previous observation
            new_obs: New observation
            
        Returns:
            Reward value
        """
        # Very simplified reward
        # Check if chips increased
        old_chips = old_obs["scalar"][0]
        new_chips = new_obs["scalar"][0]
        
        chip_diff = new_chips - old_chips
        
        # Reward for scoring chips
        reward = chip_diff * 100.0
        
        # Small penalty for each action (to encourage efficiency)
        reward -= 0.1
        
        return float(reward)
    
    def render(self):
        """Display current state"""
        state = self.state_detector.detect_current_state()
        self.state_detector.visualize_state(state)
    
    def close(self):
        """Clean up resources"""
        self.screen_capture.close()
    
    def emergency_stop(self):
        """Emergency stop all actions"""
        self.input_controller.emergency_stop()
        print("\n!!! EMERGENCY STOP ACTIVATED !!!")
        print("Move mouse to top-left corner to abort")


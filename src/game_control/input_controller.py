"""
Input controller for automating Balatro gameplay

Provides mouse and keyboard control for interacting with the game.
"""

import time
import pyautogui
from pynput import mouse, keyboard
from typing import Tuple, List, Optional
from dataclasses import dataclass


@dataclass
class CardPosition:
    """Position of a card on screen"""
    x: int
    y: int
    index: int  # Card index (0-7)


class InputController:
    """
    Control mouse and keyboard to interact with Balatro
    
    Provides safe, reliable input automation with configurable delays
    and error handling.
    """
    
    def __init__(self, base_delay: float = 0.1, click_delay: float = 0.05):
        """
        Initialize input controller
        
        Args:
            base_delay: Delay between major actions (seconds)
            click_delay: Delay between clicks (seconds)
        """
        self.base_delay = base_delay
        self.click_delay = click_delay
        
        # Pyautogui safety settings
        pyautogui.PAUSE = click_delay
        pyautogui.FAILSAFE = True  # Move mouse to corner to abort
        
        # Card positions cache
        self.card_positions: List[CardPosition] = []
    
    def update_card_positions(self, positions: List[Tuple[int, int]]):
        """
        Update known card positions
        
        Args:
            positions: List of (x, y) card center positions
        """
        self.card_positions = [
            CardPosition(x=x, y=y, index=i)
            for i, (x, y) in enumerate(positions)
        ]
    
    def click_card(self, card_index: int) -> bool:
        """
        Click on a specific card
        
        Args:
            card_index: Index of card to click (0-7)
            
        Returns:
            True if successful
        """
        if card_index < 0 or card_index >= len(self.card_positions):
            print(f"Invalid card index: {card_index}")
            return False
        
        card_pos = self.card_positions[card_index]
        
        try:
            pyautogui.click(card_pos.x, card_pos.y)
            time.sleep(self.click_delay)
            return True
        except Exception as e:
            print(f"Error clicking card {card_index}: {e}")
            return False
    
    def select_cards(self, card_indices: List[int]) -> bool:
        """
        Select multiple cards by clicking them
        
        Args:
            card_indices: List of card indices to select
            
        Returns:
            True if all selections successful
        """
        success = True
        for idx in card_indices:
            if not self.click_card(idx):
                success = False
        
        time.sleep(self.base_delay)
        return success
    
    def press_play_hand(self, position: Optional[Tuple[int, int]] = None) -> bool:
        """
        Press the 'Play Hand' button
        
        Args:
            position: Optional (x, y) of play button. If None, uses default.
            
        Returns:
            True if successful
        """
        if position is None:
            # Default position (adjust based on your resolution)
            position = (960, 1000)  # Center-bottom for 1920x1080
        
        try:
            pyautogui.click(position[0], position[1])
            time.sleep(self.base_delay)
            return True
        except Exception as e:
            print(f"Error pressing play hand: {e}")
            return False
    
    def press_discard(self, position: Optional[Tuple[int, int]] = None) -> bool:
        """
        Press the 'Discard' button
        
        Args:
            position: Optional (x, y) of discard button
            
        Returns:
            True if successful
        """
        if position is None:
            # Default position (left of play button)
            position = (800, 1000)
        
        try:
            pyautogui.click(position[0], position[1])
            time.sleep(self.base_delay)
            return True
        except Exception as e:
            print(f"Error pressing discard: {e}")
            return False
    
    def press_skip(self) -> bool:
        """
        Press skip/continue button
        
        Returns:
            True if successful
        """
        try:
            # Space bar typically skips in Balatro
            pyautogui.press('space')
            time.sleep(self.base_delay)
            return True
        except Exception as e:
            print(f"Error pressing skip: {e}")
            return False
    
    def play_hand_action(self, card_indices: List[int]) -> bool:
        """
        Complete action: select cards and play hand
        
        Args:
            card_indices: Cards to select and play
            
        Returns:
            True if successful
        """
        if not self.select_cards(card_indices):
            return False
        
        return self.press_play_hand()
    
    def discard_action(self, card_indices: List[int]) -> bool:
        """
        Complete action: select cards and discard
        
        Args:
            card_indices: Cards to discard
            
        Returns:
            True if successful
        """
        if not self.select_cards(card_indices):
            return False
        
        return self.press_discard()
    
    def click_shop_item(self, item_index: int, shop_positions: Optional[List[Tuple[int, int]]] = None) -> bool:
        """
        Click on shop item
        
        Args:
            item_index: Index of shop item (0-6)
            shop_positions: Optional list of shop item positions
            
        Returns:
            True if successful
        """
        if shop_positions is None:
            # Default shop positions for 1920x1080
            shop_positions = [
                (400, 300), (600, 300), (800, 300), (1000, 300),  # Top row
                (400, 500), (600, 500), (800, 500)  # Bottom row
            ]
        
        if item_index < 0 or item_index >= len(shop_positions):
            print(f"Invalid shop item index: {item_index}")
            return False
        
        try:
            pos = shop_positions[item_index]
            pyautogui.click(pos[0], pos[1])
            time.sleep(self.base_delay)
            return True
        except Exception as e:
            print(f"Error clicking shop item: {e}")
            return False
    
    def reroll_shop(self, button_position: Optional[Tuple[int, int]] = None) -> bool:
        """
        Click reroll button in shop
        
        Args:
            button_position: Optional position of reroll button
            
        Returns:
            True if successful
        """
        if button_position is None:
            button_position = (960, 900)  # Default position
        
        try:
            pyautogui.click(button_position[0], button_position[1])
            time.sleep(self.base_delay)
            return True
        except Exception as e:
            print(f"Error rerolling shop: {e}")
            return False
    
    def wait(self, duration: float):
        """Wait for specified duration"""
        time.sleep(duration)
    
    def emergency_stop(self):
        """
        Emergency stop - move mouse to failsafe corner
        
        This triggers pyautogui's failsafe if enabled.
        """
        pyautogui.moveTo(0, 0)
    
    @staticmethod
    def get_mouse_position() -> Tuple[int, int]:
        """
        Get current mouse position
        
        Returns:
            (x, y) mouse position
        """
        return pyautogui.position()
    
    @staticmethod
    def calibrate_button_position(button_name: str) -> Tuple[int, int]:
        """
        Interactive calibration for button position
        
        Args:
            button_name: Name of button to calibrate
            
        Returns:
            (x, y) position where user clicked
        """
        print(f"\nCalibrating {button_name} button position...")
        print("Move your mouse to the button and press 'c' to confirm")
        print("Press 'q' to cancel")
        
        position = None
        
        def on_press(key):
            nonlocal position
            try:
                if key.char == 'c':
                    position = pyautogui.position()
                    print(f"Position captured: {position}")
                    return False  # Stop listener
                elif key.char == 'q':
                    print("Calibration cancelled")
                    return False
            except AttributeError:
                pass
        
        with keyboard.Listener(on_press=on_press) as listener:
            listener.join()
        
        return position if position else (0, 0)


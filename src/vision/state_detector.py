"""
Game state detection from screenshots

Extracts complete game state including cards, counters, and UI elements.
"""

import cv2
import numpy as np
import pytesseract
from typing import Dict, Optional, Any
from dataclasses import dataclass

from .screen_capture import ScreenCapture, ScreenRegions
from .card_detector import CardDetector


@dataclass
class GameStateDetection:
    """Detected game state from screenshot"""
    num_cards_in_hand: int
    chips_scored: Optional[int]
    chips_required: Optional[int]
    money: Optional[int]
    hands_remaining: Optional[int]
    discards_remaining: Optional[int]
    ante: Optional[int]
    num_jokers: Optional[int]
    in_shop: bool
    raw_regions: Dict[str, np.ndarray]  # Raw captured regions


class StateDetector:
    """
    Detect game state from Balatro screenshots
    
    Combines card detection, OCR, and UI analysis to extract
    the complete game state.
    """
    
    def __init__(self, screen_capture: ScreenCapture):
        """
        Initialize state detector
        
        Args:
            screen_capture: Configured ScreenCapture instance
        """
        self.capture = screen_capture
        self.card_detector = CardDetector()
    
    def detect_current_state(self) -> GameStateDetection:
        """
        Detect current game state from screen
        
        Returns:
            Detected game state
        """
        # Capture all regions
        regions = self.capture.capture_all_regions()
        
        # Detect cards in hand
        num_cards = self._count_cards(regions["hand"])
        
        # Extract numeric values using OCR
        chips_scored = self._extract_number(regions["chips"])
        money = self._extract_number(regions["money"])
        hands_remaining = self._extract_number(regions["hands_left"])
        discards_remaining = self._extract_number(regions["discards_left"])
        
        # Extract blind info
        chips_required, ante = self._extract_blind_info(regions["blind_info"])
        
        # Count jokers
        num_jokers = self._count_jokers(regions["jokers"])
        
        # Detect if in shop
        in_shop = self._detect_shop(regions.get("shop"))
        
        return GameStateDetection(
            num_cards_in_hand=num_cards,
            chips_scored=chips_scored,
            chips_required=chips_required,
            money=money,
            hands_remaining=hands_remaining,
            discards_remaining=discards_remaining,
            ante=ante,
            num_jokers=num_jokers,
            in_shop=in_shop,
            raw_regions=regions
        )
    
    def _count_cards(self, hand_image: np.ndarray) -> int:
        """Count cards in hand"""
        try:
            cards = self.card_detector.detect_cards(hand_image)
            return len(cards)
        except Exception as e:
            print(f"Error counting cards: {e}")
            return 0
    
    def _count_jokers(self, joker_image: np.ndarray) -> Optional[int]:
        """Count joker slots (simplified)"""
        # Detect filled joker slots
        # This is a placeholder - would need proper detection
        return None
    
    def _extract_number(self, image: np.ndarray) -> Optional[int]:
        """
        Extract number from image using OCR
        
        Args:
            image: Image containing number
            
        Returns:
            Extracted number or None if failed
        """
        try:
            # Preprocess for better OCR
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # Increase contrast
            gray = cv2.convertScaleAbs(gray, alpha=2.0, beta=0)
            
            # Threshold
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # OCR with number-only config
            custom_config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789'
            text = pytesseract.image_to_string(thresh, config=custom_config)
            
            # Extract number
            text = text.strip().replace(',', '')
            if text and text.isdigit():
                return int(text)
            
            return None
        
        except Exception as e:
            print(f"OCR error: {e}")
            return None
    
    def _extract_blind_info(self, blind_image: np.ndarray) -> tuple[Optional[int], Optional[int]]:
        """
        Extract chips required and ante from blind info
        
        Returns:
            (chips_required, ante)
        """
        try:
            # OCR to extract text
            gray = cv2.cvtColor(blind_image, cv2.COLOR_RGB2GRAY)
            text = pytesseract.image_to_string(gray)
            
            # Parse for numbers
            # Look for patterns like "Ante 3" and "300 chips"
            chips_required = None
            ante = None
            
            lines = text.split('\n')
            for line in lines:
                line_lower = line.lower()
                if 'ante' in line_lower:
                    # Extract ante number
                    numbers = [int(s) for s in line.split() if s.isdigit()]
                    if numbers:
                        ante = numbers[0]
                
                if 'chip' in line_lower or any(c.isdigit() for c in line):
                    # Extract chips
                    numbers = [int(s.replace(',', '')) for s in line.split() 
                              if s.replace(',', '').isdigit()]
                    if numbers:
                        chips_required = numbers[0]
            
            return chips_required, ante
        
        except Exception as e:
            print(f"Error extracting blind info: {e}")
            return None, None
    
    def _detect_shop(self, shop_image: Optional[np.ndarray]) -> bool:
        """Detect if currently in shop"""
        if shop_image is None:
            return False
        
        # Look for shop-specific UI elements
        # This is simplified - would need actual detection
        # Could look for specific colors, text, or patterns unique to shop
        return False
    
    def get_state_for_model(self) -> Dict[str, np.ndarray]:
        """
        Get game state in format expected by neural network
        
        This is a bridge between vision detection and model input.
        Currently returns simplified state - would need enhancement.
        
        Returns:
            Dictionary compatible with BalatroEnv observation space
        """
        state = self.detect_current_state()
        
        # Create dummy observation for now
        # In production, this would:
        # 1. Identify each card's rank/suit
        # 2. Detect joker types
        # 3. Extract all game state
        # 4. Format into model's expected observation format
        
        obs = {
            "hand": np.zeros((8, 32), dtype=np.float32),
            "jokers": np.zeros((5, 64), dtype=np.float32),
            "blind": np.zeros(16, dtype=np.float32),
            "scalar": np.zeros(32, dtype=np.float32)
        }
        
        # Fill in what we can detect
        if state.hands_remaining is not None:
            obs["scalar"][2] = state.hands_remaining / 4.0
        
        if state.discards_remaining is not None:
            obs["scalar"][3] = state.discards_remaining / 3.0
        
        if state.ante is not None:
            obs["scalar"][4] = state.ante / 8.0
        
        if state.money is not None:
            obs["scalar"][5] = state.money / 100.0
        
        if state.chips_scored is not None and state.chips_required is not None:
            obs["scalar"][0] = state.chips_scored / 10000.0
            obs["scalar"][1] = state.chips_required / 10000.0
        
        return obs
    
    def visualize_state(self, state: GameStateDetection):
        """
        Display detected state for debugging
        
        Args:
            state: Detected game state
        """
        print("\n=== Detected Game State ===")
        print(f"Cards in hand: {state.num_cards_in_hand}")
        print(f"Chips: {state.chips_scored} / {state.chips_required}")
        print(f"Money: ${state.money}")
        print(f"Hands: {state.hands_remaining}")
        print(f"Discards: {state.discards_remaining}")
        print(f"Ante: {state.ante}")
        print(f"Jokers: {state.num_jokers}")
        print(f"In Shop: {state.in_shop}")
        print("=" * 30)


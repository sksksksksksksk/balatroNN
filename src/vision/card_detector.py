"""
Card detection from screenshots using computer vision

Detects individual cards, their positions, and attempts to identify
rank and suit using template matching and color detection.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from enum import Enum


@dataclass
class DetectedCard:
    """Detected card information"""
    position: Tuple[int, int]  # (x, y) center position
    bbox: Tuple[int, int, int, int]  # (x, y, width, height) bounding box
    rank: Optional[str] = None  # "A", "2", ..., "K"
    suit: Optional[str] = None  # "Hearts", "Diamonds", "Clubs", "Spades"
    confidence: float = 0.0
    selected: bool = False  # If card appears selected/highlighted


class CardDetector:
    """
    Detect and identify cards from game screenshots
    
    Uses contour detection, template matching, and color analysis
    to identify cards and their properties.
    """
    
    def __init__(self, card_templates: Optional[Dict] = None):
        """
        Initialize card detector
        
        Args:
            card_templates: Optional pre-loaded templates for matching
        """
        self.card_templates = card_templates or {}
        
        # Card size thresholds (adjust based on resolution)
        self.min_card_area = 1000
        self.max_card_area = 50000
        
        # Aspect ratio for cards (typical playing card)
        self.min_aspect_ratio = 0.6
        self.max_aspect_ratio = 0.8
    
    def detect_cards(self, image: np.ndarray, debug: bool = False) -> List[DetectedCard]:
        """
        Detect all cards in an image
        
        Args:
            image: RGB image
            debug: If True, show debug visualization
            
        Returns:
            List of detected cards
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detected_cards = []
        
        for contour in contours:
            # Filter by area
            area = cv2.contourArea(contour)
            if area < self.min_card_area or area > self.max_card_area:
                continue
            
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter by aspect ratio
            aspect_ratio = float(w) / h if h > 0 else 0
            if aspect_ratio < self.min_aspect_ratio or aspect_ratio > self.max_aspect_ratio:
                continue
            
            # Extract card region
            card_roi = image[y:y+h, x:x+w]
            
            # Attempt to identify card
            rank, suit, confidence = self._identify_card(card_roi)
            
            # Check if card is selected (highlighted)
            selected = self._is_card_selected(card_roi)
            
            detected_card = DetectedCard(
                position=(x + w//2, y + h//2),
                bbox=(x, y, w, h),
                rank=rank,
                suit=suit,
                confidence=confidence,
                selected=selected
            )
            
            detected_cards.append(detected_card)
        
        # Sort cards left to right
        detected_cards.sort(key=lambda c: c.position[0])
        
        if debug:
            self._visualize_detections(image, detected_cards)
        
        return detected_cards
    
    def _identify_card(self, card_image: np.ndarray) -> Tuple[Optional[str], Optional[str], float]:
        """
        Identify rank and suit of a card
        
        Args:
            card_image: Cropped card image
            
        Returns:
            (rank, suit, confidence)
        """
        # This is a simplified placeholder
        # In production, you'd use:
        # - Template matching with pre-saved card templates
        # - OCR for rank detection
        # - Color detection for suit
        # - Or a trained CNN classifier
        
        # For now, return unknown
        return None, None, 0.0
    
    def _is_card_selected(self, card_image: np.ndarray) -> bool:
        """
        Check if card appears selected/highlighted
        
        Args:
            card_image: Cropped card image
            
        Returns:
            True if card appears selected
        """
        # Check for highlight color (often yellow/gold in Balatro)
        hsv = cv2.cvtColor(card_image, cv2.COLOR_RGB2HSV)
        
        # Yellow/gold range
        lower_yellow = np.array([20, 100, 100])
        upper_yellow = np.array([35, 255, 255])
        
        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        yellow_ratio = np.sum(mask > 0) / mask.size
        
        # If significant portion is yellow, likely selected
        return yellow_ratio > 0.15
    
    def _visualize_detections(self, image: np.ndarray, cards: List[DetectedCard]):
        """
        Visualize detected cards for debugging
        
        Args:
            image: Original image
            cards: List of detected cards
        """
        vis = image.copy()
        
        for i, card in enumerate(cards):
            x, y, w, h = card.bbox
            
            # Color based on selection
            color = (0, 255, 0) if card.selected else (255, 0, 0)
            
            # Draw bounding box
            cv2.rectangle(vis, (x, y), (x + w, y + h), color, 2)
            
            # Draw card number
            cv2.putText(vis, f"Card {i+1}", (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Draw rank/suit if detected
            if card.rank and card.suit:
                label = f"{card.rank}{card.suit[0]}"
                cv2.putText(vis, label, (x, y + h + 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Display
        cv2.imshow("Card Detection", cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
        cv2.waitKey(1)
    
    def save_card_templates(self, hand_image: np.ndarray, output_dir: str):
        """
        Save detected cards as templates for future matching
        
        Args:
            hand_image: Image containing cards
            output_dir: Directory to save templates
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        cards = self.detect_cards(hand_image)
        
        for i, card in enumerate(cards):
            x, y, w, h = card.bbox
            card_img = hand_image[y:y+h, x:x+w]
            
            filepath = os.path.join(output_dir, f"card_{i:02d}.png")
            cv2.imwrite(filepath, cv2.cvtColor(card_img, cv2.COLOR_RGB2BGR))
            print(f"Saved template: {filepath}")


def detect_card_positions(image: np.ndarray) -> List[Tuple[int, int]]:
    """
    Quick detection of card center positions (without full identification)
    
    Args:
        image: RGB image of hand region
        
    Returns:
        List of (x, y) card center positions
    """
    detector = CardDetector()
    cards = detector.detect_cards(image)
    return [card.position for card in cards]


def count_cards(image: np.ndarray) -> int:
    """
    Count number of cards in image
    
    Args:
        image: RGB image
        
    Returns:
        Number of detected cards
    """
    detector = CardDetector()
    cards = detector.detect_cards(image)
    return len(cards)


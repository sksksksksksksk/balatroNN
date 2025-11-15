"""
Screen capture utilities for reading Balatro game state

Uses MSS for fast screen capture and OpenCV for image processing.
"""

import numpy as np
import mss
import cv2
from PIL import Image
from typing import Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class ScreenRegions:
    """Screen regions for different UI elements"""
    hand: Tuple[int, int, int, int]  # (x, y, width, height)
    jokers: Tuple[int, int, int, int]
    blind_info: Tuple[int, int, int, int]
    chips: Tuple[int, int, int, int]
    money: Tuple[int, int, int, int]
    hands_left: Tuple[int, int, int, int]
    discards_left: Tuple[int, int, int, int]
    shop: Optional[Tuple[int, int, int, int]] = None


class ScreenCapture:
    """
    Fast screen capture for Balatro game
    
    Uses MSS for efficient screen grabbing and provides
    utilities for extracting specific regions.
    """
    
    def __init__(self, monitor_index: int = 1, regions: Optional[ScreenRegions] = None):
        """
        Initialize screen capture
        
        Args:
            monitor_index: Which monitor to capture (1 = primary)
            regions: Pre-defined screen regions for UI elements
        """
        self.sct = mss.mss()
        self.monitor_index = monitor_index
        self.monitor = self.sct.monitors[monitor_index]
        self.regions = regions
        
        # Cache for region coordinates
        self._region_cache = {}
        
    def capture_full_screen(self) -> np.ndarray:
        """
        Capture full screen
        
        Returns:
            RGB image as numpy array (H, W, 3)
        """
        screenshot = self.sct.grab(self.monitor)
        img = np.array(screenshot)
        # Convert BGRA to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        return img
    
    def capture_region(self, region: Tuple[int, int, int, int]) -> np.ndarray:
        """
        Capture specific screen region
        
        Args:
            region: (x, y, width, height) in screen coordinates
            
        Returns:
            RGB image as numpy array
        """
        x, y, width, height = region
        monitor_region = {
            "top": y,
            "left": x,
            "width": width,
            "height": height
        }
        screenshot = self.sct.grab(monitor_region)
        img = np.array(screenshot)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        return img
    
    def capture_hand(self) -> np.ndarray:
        """Capture hand region"""
        if self.regions is None:
            raise ValueError("Screen regions not configured")
        return self.capture_region(self.regions.hand)
    
    def capture_jokers(self) -> np.ndarray:
        """Capture jokers region"""
        if self.regions is None:
            raise ValueError("Screen regions not configured")
        return self.capture_region(self.regions.jokers)
    
    def capture_blind_info(self) -> np.ndarray:
        """Capture blind information region"""
        if self.regions is None:
            raise ValueError("Screen regions not configured")
        return self.capture_region(self.regions.blind_info)
    
    def capture_chips(self) -> np.ndarray:
        """Capture chips counter"""
        if self.regions is None:
            raise ValueError("Screen regions not configured")
        return self.capture_region(self.regions.chips)
    
    def capture_money(self) -> np.ndarray:
        """Capture money counter"""
        if self.regions is None:
            raise ValueError("Screen regions not configured")
        return self.capture_region(self.regions.money)
    
    def capture_all_regions(self) -> Dict[str, np.ndarray]:
        """
        Capture all defined regions
        
        Returns:
            Dictionary mapping region names to images
        """
        if self.regions is None:
            raise ValueError("Screen regions not configured")
        
        captures = {
            "hand": self.capture_hand(),
            "jokers": self.capture_jokers(),
            "blind_info": self.capture_blind_info(),
            "chips": self.capture_chips(),
            "money": self.capture_money(),
            "hands_left": self.capture_region(self.regions.hands_left),
            "discards_left": self.capture_region(self.regions.discards_left)
        }
        
        if self.regions.shop is not None:
            captures["shop"] = self.capture_region(self.regions.shop)
        
        return captures
    
    def save_screenshot(self, filepath: str, region: Optional[Tuple[int, int, int, int]] = None):
        """
        Save screenshot to file
        
        Args:
            filepath: Path to save image
            region: Optional region to capture, None for full screen
        """
        if region is None:
            img = self.capture_full_screen()
        else:
            img = self.capture_region(region)
        
        # Convert RGB to BGR for OpenCV
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(filepath, img_bgr)
    
    def show_regions(self, wait_time: int = 0):
        """
        Display all regions with overlays (for debugging/calibration)
        
        Args:
            wait_time: Time to wait in ms (0 = wait for key press)
        """
        if self.regions is None:
            raise ValueError("Screen regions not configured")
        
        # Capture full screen
        img = self.capture_full_screen()
        img_display = img.copy()
        
        # Draw rectangles for each region
        regions_to_draw = [
            (self.regions.hand, "Hand", (0, 255, 0)),
            (self.regions.jokers, "Jokers", (255, 0, 0)),
            (self.regions.blind_info, "Blind", (0, 0, 255)),
            (self.regions.chips, "Chips", (255, 255, 0)),
            (self.regions.money, "Money", (255, 0, 255)),
            (self.regions.hands_left, "Hands", (0, 255, 255)),
            (self.regions.discards_left, "Discards", (128, 128, 128))
        ]
        
        if self.regions.shop is not None:
            regions_to_draw.append((self.regions.shop, "Shop", (255, 128, 0)))
        
        for region, label, color in regions_to_draw:
            x, y, w, h = region
            # Adjust coordinates relative to monitor
            cv2.rectangle(img_display, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img_display, label, (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        
        # Display
        cv2.imshow("Screen Regions", cv2.cvtColor(img_display, cv2.COLOR_RGB2BGR))
        cv2.waitKey(wait_time)
        cv2.destroyAllWindows()
    
    def close(self):
        """Release resources"""
        self.sct.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def create_default_regions_1920x1080() -> ScreenRegions:
    """
    Create default screen regions for 1920x1080 resolution
    
    These are approximate and should be calibrated for your setup.
    Use calibrate_regions.py to find exact coordinates.
    """
    return ScreenRegions(
        hand=(600, 800, 720, 180),           # Bottom center - hand cards
        jokers=(50, 50, 400, 150),           # Top left - joker slots
        blind_info=(1400, 50, 450, 200),     # Top right - blind info
        chips=(800, 100, 320, 80),           # Top center - chips counter
        money=(50, 950, 150, 50),            # Bottom left - money
        hands_left=(1700, 900, 100, 50),     # Bottom right - hands remaining
        discards_left=(1700, 850, 100, 50),  # Bottom right - discards remaining
        shop=None                             # Set when in shop
    )


def calibrate_regions_interactive() -> ScreenRegions:
    """
    Interactive calibration tool
    
    Click and drag to define regions. Press ESC when done.
    
    Returns:
        Calibrated ScreenRegions
    """
    print("Interactive region calibration")
    print("Instructions:")
    print("1. The game window will be captured")
    print("2. Click and drag to define each region")
    print("3. Press ESC when done with each region")
    print()
    
    capture = ScreenCapture()
    img = capture.capture_full_screen()
    
    def select_region(window_name: str, instruction: str) -> Tuple[int, int, int, int]:
        print(f"\n{instruction}")
        print(f"Select region in '{window_name}' window...")
        
        roi = cv2.selectROI(window_name, cv2.cvtColor(img, cv2.COLOR_RGB2BGR), False, False)
        cv2.destroyAllWindows()
        
        return roi  # Returns (x, y, w, h)
    
    regions = ScreenRegions(
        hand=select_region("Hand Region", "Select the area where your hand cards appear"),
        jokers=select_region("Jokers Region", "Select the area where jokers are displayed"),
        blind_info=select_region("Blind Info", "Select the blind information area"),
        chips=select_region("Chips Counter", "Select the chips counter"),
        money=select_region("Money", "Select the money display"),
        hands_left=select_region("Hands Left", "Select the hands remaining counter"),
        discards_left=select_region("Discards Left", "Select the discards remaining counter")
    )
    
    print("\nCalibration complete!")
    print(f"Regions: {regions}")
    
    capture.close()
    return regions


#!/usr/bin/env python3
"""
Interactive calibration tool for real game interface

Use this to set up screen regions and button positions for your setup.
"""

import sys
import argparse
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from vision.screen_capture import ScreenCapture, ScreenRegions, calibrate_regions_interactive
from game_control.input_controller import InputController


def parse_args():
    parser = argparse.ArgumentParser(description="Calibrate Balatro game interface")
    
    parser.add_argument(
        "--output",
        type=str,
        default="game_calibration.json",
        help="Output file for calibration data"
    )
    
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test existing calibration"
    )
    
    return parser.parse_args()


def calibrate_screen_regions() -> ScreenRegions:
    """Interactive screen region calibration"""
    print("\n" + "="*60)
    print("SCREEN REGION CALIBRATION")
    print("="*60)
    print()
    print("You will be asked to select rectangular regions on screen.")
    print("For each region:")
    print("  1. Click and drag to select the area")
    print("  2. Press SPACE or ENTER to confirm")
    print("  3. Press ESC to cancel and retry")
    print()
    input("Press ENTER when Balatro is running and visible...")
    
    regions = calibrate_regions_interactive()
    
    return regions


def calibrate_buttons() -> dict:
    """Interactive button position calibration"""
    print("\n" + "="*60)
    print("BUTTON POSITION CALIBRATION")
    print("="*60)
    print()
    print("For each button, move your mouse over it and press 'c'")
    print()
    
    controller = InputController()
    
    buttons = {}
    
    print("\n1. PLAY HAND button")
    buttons["play_hand"] = controller.calibrate_button_position("Play Hand")
    
    print("\n2. DISCARD button")
    buttons["discard"] = controller.calibrate_button_position("Discard")
    
    print("\n3. SKIP/CONTINUE button")
    buttons["skip"] = controller.calibrate_button_position("Skip/Continue")
    
    print("\n4. REROLL SHOP button (position mouse and press 'c', or 'q' to skip)")
    buttons["reroll"] = controller.calibrate_button_position("Reroll")
    
    return buttons


def save_calibration(regions: ScreenRegions, buttons: dict, filepath: str):
    """Save calibration to file"""
    calibration = {
        "regions": {
            "hand": list(regions.hand),
            "jokers": list(regions.jokers),
            "blind_info": list(regions.blind_info),
            "chips": list(regions.chips),
            "money": list(regions.money),
            "hands_left": list(regions.hands_left),
            "discards_left": list(regions.discards_left),
            "shop": list(regions.shop) if regions.shop else None
        },
        "buttons": buttons
    }
    
    with open(filepath, 'w') as f:
        json.dump(calibration, f, indent=2)
    
    print(f"\n✓ Calibration saved to {filepath}")


def load_calibration(filepath: str) -> tuple:
    """Load calibration from file"""
    with open(filepath, 'r') as f:
        calibration = json.load(f)
    
    regions_dict = calibration["regions"]
    regions = ScreenRegions(
        hand=tuple(regions_dict["hand"]),
        jokers=tuple(regions_dict["jokers"]),
        blind_info=tuple(regions_dict["blind_info"]),
        chips=tuple(regions_dict["chips"]),
        money=tuple(regions_dict["money"]),
        hands_left=tuple(regions_dict["hands_left"]),
        discards_left=tuple(regions_dict["discards_left"]),
        shop=tuple(regions_dict["shop"]) if regions_dict.get("shop") else None
    )
    
    buttons = calibration["buttons"]
    
    return regions, buttons


def test_calibration(filepath: str):
    """Test loaded calibration"""
    print(f"\nLoading calibration from {filepath}...")
    
    try:
        regions, buttons = load_calibration(filepath)
        
        print("\n✓ Calibration loaded successfully")
        print("\nRegions:")
        print(f"  Hand: {regions.hand}")
        print(f"  Jokers: {regions.jokers}")
        print(f"  Blind Info: {regions.blind_info}")
        print(f"  Chips: {regions.chips}")
        print(f"  Money: {regions.money}")
        print(f"  Hands Left: {regions.hands_left}")
        print(f"  Discards Left: {regions.discards_left}")
        
        print("\nButtons:")
        for name, pos in buttons.items():
            print(f"  {name}: {pos}")
        
        print("\nShowing regions on screen...")
        input("Press ENTER when Balatro is visible...")
        
        capture = ScreenCapture(regions=regions)
        capture.show_regions(wait_time=5000)  # Show for 5 seconds
        capture.close()
        
        print("\n✓ Calibration test complete!")
        print("If regions look correct, you're ready to play!")
        
    except Exception as e:
        print(f"\n✗ Error testing calibration: {e}")
        import traceback
        traceback.print_exc()


def main():
    args = parse_args()
    
    if args.test:
        if not Path(args.output).exists():
            print(f"Error: Calibration file '{args.output}' not found")
            print("Run without --test flag to create new calibration")
            return
        
        test_calibration(args.output)
        return
    
    print("\n" + "="*60)
    print("BALATRO GAME INTERFACE CALIBRATION")
    print("="*60)
    print()
    print("This tool will help you set up the vision system to")
    print("detect game state and control the real Balatro game.")
    print()
    print("Prerequisites:")
    print("  • Balatro running in windowed mode")
    print("  • Game visible on screen")
    print("  • Resolution: 1920x1080 recommended")
    print()
    input("Press ENTER to begin...")
    
    # Step 1: Calibrate screen regions
    regions = calibrate_screen_regions()
    
    # Step 2: Calibrate button positions
    buttons = calibrate_buttons()
    
    # Step 3: Save calibration
    save_calibration(regions, buttons, args.output)
    
    print("\n" + "="*60)
    print("CALIBRATION COMPLETE!")
    print("="*60)
    print()
    print(f"Configuration saved to: {args.output}")
    print()
    print("Next steps:")
    print(f"  1. Test: python calibrate_game.py --test --output {args.output}")
    print(f"  2. Play: python play_real_game.py --calibration {args.output} --checkpoint path/to/model.pt")
    print()


if __name__ == "__main__":
    main()


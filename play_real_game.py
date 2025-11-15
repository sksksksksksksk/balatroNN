#!/usr/bin/env python3
"""
Play real Balatro game with trained model

Uses computer vision and input automation to control the actual game.
"""

import sys
import argparse
import json
import torch
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from vision.screen_capture import ScreenRegions
from game_control import RealBalatroEnv
from models import PolicyValueNetwork


def parse_args():
    parser = argparse.ArgumentParser(description="Play Balatro with trained model")
    
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained model checkpoint"
    )
    
    parser.add_argument(
        "--calibration",
        type=str,
        default="game_calibration.json",
        help="Path to calibration file"
    )
    
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use deterministic policy (no sampling)"
    )
    
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Delay between actions (seconds)"
    )
    
    parser.add_argument(
        "--max-steps",
        type=int,
        default=1000,
        help="Maximum steps before stopping"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run model on"
    )
    
    return parser.parse_args()


def load_calibration(filepath: str) -> ScreenRegions:
    """Load screen regions from calibration file"""
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
    
    return regions


def main():
    args = parse_args()
    
    print("\n" + "="*60)
    print("BALATRO REAL GAME PLAYER")
    print("="*60)
    print()
    
    # Load calibration
    print(f"Loading calibration from {args.calibration}...")
    try:
        regions = load_calibration(args.calibration)
        print("✓ Calibration loaded")
    except FileNotFoundError:
        print(f"✗ Error: Calibration file '{args.calibration}' not found")
        print("Run: python calibrate_game.py")
        return
    except Exception as e:
        print(f"✗ Error loading calibration: {e}")
        return
    
    # Load model
    print(f"\nLoading model from {args.checkpoint}...")
    try:
        checkpoint = torch.load(args.checkpoint, map_location=args.device)
        model = PolicyValueNetwork()
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(args.device)
        model.eval()
        print(f"✓ Model loaded (trained for {checkpoint.get('num_timesteps', '?')} steps)")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return
    
    # Create environment
    print("\nInitializing game interface...")
    try:
        env = RealBalatroEnv(
            screen_regions=regions,
            action_delay=args.delay
        )
        print("✓ Game interface ready")
    except Exception as e:
        print(f"✗ Error initializing environment: {e}")
        return
    
    # Instructions
    print("\n" + "="*60)
    print("READY TO PLAY")
    print("="*60)
    print()
    print("Instructions:")
    print("  • Make sure Balatro is running and visible")
    print("  • Start a new run in the game")
    print("  • The AI will begin making decisions")
    print("  • Press Ctrl+C to stop")
    print("  • Move mouse to top-left corner for emergency stop")
    print()
    print(f"Settings:")
    print(f"  Policy: {'Deterministic' if args.deterministic else 'Stochastic'}")
    print(f"  Action delay: {args.delay}s")
    print(f"  Max steps: {args.max_steps}")
    print(f"  Device: {args.device}")
    print()
    
    countdown = 5
    for i in range(countdown, 0, -1):
        print(f"Starting in {i}...", end='\r')
        time.sleep(1)
    print("\nSTARTING!")
    print()
    
    # Play game
    try:
        obs, info = env.reset()
        
        total_reward = 0
        step_count = 0
        
        print(f"Step {step_count}: Connected to game")
        
        while step_count < args.max_steps:
            step_count += 1
            
            # Convert observation to tensor
            obs_tensor = {
                key: torch.as_tensor(val, dtype=torch.float32, device=args.device).unsqueeze(0)
                for key, val in obs.items()
            }
            
            # Get action from model
            with torch.no_grad():
                action, log_prob, entropy, value = model.get_action_and_value(
                    obs_tensor,
                    deterministic=args.deterministic
                )
            
            # Convert to numpy
            action_np = {
                "action_type": action["action_type"].cpu().numpy()[0],
                "card_selection": action["card_selection"].cpu().numpy()[0],
                "shop_selection": action["shop_selection"].cpu().numpy()[0]
            }
            
            # Decode action for display
            action_names = ["Play Hand", "Discard", "Shop Buy", "Skip", "Reroll"]
            action_name = action_names[int(action_np["action_type"])]
            
            print(f"Step {step_count}: Action = {action_name}, Value = {value.item():.2f}")
            
            # Execute action
            obs, reward, terminated, truncated, info = env.step(action_np)
            
            total_reward += reward
            
            if "error" in info:
                print(f"  ⚠ Warning: {info['error']}")
            
            if terminated or truncated:
                print(f"\nGame ended after {step_count} steps")
                break
            
            # Optional: render state
            if step_count % 10 == 0:
                print(f"  Total reward so far: {total_reward:.2f}")
        
        print("\n" + "="*60)
        print("SESSION COMPLETE")
        print("="*60)
        print(f"Total steps: {step_count}")
        print(f"Total reward: {total_reward:.2f}")
        print(f"Average reward: {total_reward/step_count:.2f}")
        
    except KeyboardInterrupt:
        print("\n\nStopped by user")
    
    except Exception as e:
        print(f"\n✗ Error during gameplay: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        env.close()
        print("\nCleaned up. Goodbye!")


if __name__ == "__main__":
    main()


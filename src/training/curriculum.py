"""
Curriculum Learning for Gradual Joker Introduction

Implements a phased approach to training with increasing complexity of jokers.
"""

import numpy as np
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class CurriculumPhase:
    """Configuration for a single curriculum phase"""
    name: str
    start_step: int
    end_step: int
    enabled_tiers: List[int]
    max_joker_slots: int
    shop_enabled: bool
    consumables_enabled: bool
    vouchers_enabled: bool
    description: str


class JokerCurriculum:
    """
    Manages curriculum learning for gradual joker introduction
    
    Phases:
    1. Core Gameplay: No jokers, learn basic hand playing
    2. Simple Jokers: Tier 1 only, basic stat boosts
    3. Conditional Jokers: Tiers 1-2, conditional effects
    4. Complex Jokers: Tiers 1-3, persistent state
    5. Economy Focus: Tiers 1-4, full shop
    6. Advanced Scaling: Tiers 1-5, all mechanics
    7. Full Game: All tiers, all mechanics
    """
    
    def __init__(self, total_timesteps: int = 50_000_000):
        self.total_timesteps = total_timesteps
        self.phases = self._create_phases()
        self.current_phase_idx = 0
    
    def _create_phases(self) -> List[CurriculumPhase]:
        """Create the curriculum phases"""
        phases = [
            # Phase 1: Core Gameplay (5% of training)
            CurriculumPhase(
                name="Core Gameplay",
                start_step=0,
                end_step=int(self.total_timesteps * 0.05),
                enabled_tiers=[],  # No jokers
                max_joker_slots=0,
                shop_enabled=False,
                consumables_enabled=False,
                vouchers_enabled=False,
                description="Learn basic hand playing and scoring without jokers"
            ),
            
            # Phase 2: Simple Jokers (10% of training)
            CurriculumPhase(
                name="Simple Jokers",
                start_step=int(self.total_timesteps * 0.05),
                end_step=int(self.total_timesteps * 0.15),
                enabled_tiers=[1],  # Tier 1 only
                max_joker_slots=3,
                shop_enabled=True,
                consumables_enabled=False,
                vouchers_enabled=False,
                description="Tier 1 jokers: simple stat boosts"
            ),
            
            # Phase 3: Conditional Jokers (15% of training)
            CurriculumPhase(
                name="Conditional Jokers",
                start_step=int(self.total_timesteps * 0.15),
                end_step=int(self.total_timesteps * 0.30),
                enabled_tiers=[1, 2],  # Tiers 1-2
                max_joker_slots=4,
                shop_enabled=True,
                consumables_enabled=True,
                vouchers_enabled=False,
                description="Tiers 1-2: conditional effects and deck building"
            ),
            
            # Phase 4: Complex Jokers (20% of training)
            CurriculumPhase(
                name="Complex Jokers",
                start_step=int(self.total_timesteps * 0.30),
                end_step=int(self.total_timesteps * 0.50),
                enabled_tiers=[1, 2, 3],  # Tiers 1-3
                max_joker_slots=5,
                shop_enabled=True,
                consumables_enabled=True,
                vouchers_enabled=True,
                description="Tiers 1-3: persistent state and meta effects"
            ),
            
            # Phase 5: Economy & Scaling (20% of training)
            CurriculumPhase(
                name="Economy & Scaling",
                start_step=int(self.total_timesteps * 0.50),
                end_step=int(self.total_timesteps * 0.70),
                enabled_tiers=[1, 2, 3, 4],  # Tiers 1-4
                max_joker_slots=5,
                shop_enabled=True,
                consumables_enabled=True,
                vouchers_enabled=True,
                description="Tiers 1-4: economy jokers and scaling"
            ),
            
            # Phase 6: Advanced Mechanics (15% of training)
            CurriculumPhase(
                name="Advanced Mechanics",
                start_step=int(self.total_timesteps * 0.70),
                end_step=int(self.total_timesteps * 0.85),
                enabled_tiers=[1, 2, 3, 4, 5],  # Tiers 1-5
                max_joker_slots=5,
                shop_enabled=True,
                consumables_enabled=True,
                vouchers_enabled=True,
                description="Tiers 1-5: advanced combos and legendary jokers"
            ),
            
            # Phase 7: Full Game (15% of training)
            CurriculumPhase(
                name="Full Game",
                start_step=int(self.total_timesteps * 0.85),
                end_step=self.total_timesteps,
                enabled_tiers=[1, 2, 3, 4, 5, 6],  # All tiers
                max_joker_slots=5,
                shop_enabled=True,
                consumables_enabled=True,
                vouchers_enabled=True,
                description="All 150 jokers: full game complexity"
            ),
        ]
        
        return phases
    
    def get_phase_for_step(self, step: int) -> CurriculumPhase:
        """Get the curriculum phase for a given training step"""
        for phase in self.phases:
            if phase.start_step <= step < phase.end_step:
                return phase
        # Return last phase if beyond training
        return self.phases[-1]
    
    def update_phase(self, step: int) -> bool:
        """
        Update current phase based on step
        Returns True if phase changed
        """
        new_phase = self.get_phase_for_step(step)
        
        # Check if we've moved to a new phase
        for idx, phase in enumerate(self.phases):
            if phase.name == new_phase.name:
                if idx != self.current_phase_idx:
                    self.current_phase_idx = idx
                    return True
        
        return False
    
    def get_current_phase(self) -> CurriculumPhase:
        """Get current curriculum phase"""
        return self.phases[self.current_phase_idx]
    
    def should_enable_joker(self, joker_tier: int, step: int) -> bool:
        """Check if a joker should be enabled at current step"""
        phase = self.get_phase_for_step(step)
        return joker_tier in phase.enabled_tiers
    
    def get_progress(self, step: int) -> Dict[str, Any]:
        """Get curriculum progress information"""
        phase = self.get_phase_for_step(step)
        phase_progress = (step - phase.start_step) / (phase.end_step - phase.start_step)
        
        return {
            'phase_name': phase.name,
            'phase_number': self.current_phase_idx + 1,
            'total_phases': len(self.phases),
            'phase_progress': phase_progress,
            'enabled_tiers': phase.enabled_tiers,
            'max_joker_slots': phase.max_joker_slots,
            'shop_enabled': phase.shop_enabled,
            'consumables_enabled': phase.consumables_enabled,
            'vouchers_enabled': phase.vouchers_enabled,
        }
    
    def print_curriculum_summary(self):
        """Print a summary of the curriculum"""
        print("\n" + "="*60)
        print("CURRICULUM LEARNING SCHEDULE")
        print("="*60)
        print(f"Total timesteps: {self.total_timesteps:,}")
        print(f"Total phases: {len(self.phases)}\n")
        
        for i, phase in enumerate(self.phases, 1):
            duration = phase.end_step - phase.start_step
            percentage = (duration / self.total_timesteps) * 100
            
            print(f"Phase {i}: {phase.name}")
            print(f"  Steps: {phase.start_step:,} - {phase.end_step:,} ({percentage:.1f}%)")
            print(f"  Tiers: {phase.enabled_tiers if phase.enabled_tiers else 'None'}")
            print(f"  Description: {phase.description}")
            print()
        
        print("="*60 + "\n")


def create_default_curriculum(total_timesteps: int = 50_000_000) -> JokerCurriculum:
    """Create default joker curriculum"""
    return JokerCurriculum(total_timesteps=total_timesteps)


# Example usage
if __name__ == "__main__":
    curriculum = create_default_curriculum(total_timesteps=50_000_000)
    curriculum.print_curriculum_summary()
    
    # Test phase transitions
    test_steps = [0, 1_000_000, 5_000_000, 10_000_000, 25_000_000, 40_000_000, 49_000_000]
    
    print("Phase Transitions:")
    print("-" * 60)
    for step in test_steps:
        phase = curriculum.get_phase_for_step(step)
        print(f"Step {step:,}: {phase.name} (Tiers: {phase.enabled_tiers})")


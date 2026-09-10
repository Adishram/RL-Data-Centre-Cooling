"""
Deterministic rule-based cooling policy (baseline).

Policy logic:
    if any zone is critical  → strong hotspot emphasis (action 6)
    elif any zone is hot     → favour hottest zone    (action 1-4)
    elif any zone is warm    → mild hotspot emphasis  (action 5)
    elif all zones cool      → eco / reduce cooling   (action 7)
    else                     → uniform cooling        (action 0)
"""

import json
from pathlib import Path

import numpy as np

from agents.base_agent import BaseAgent
from config import NUM_ACTIONS


class RuleBasedAgent(BaseAgent):
    """
    Fixed rule-based policy — no learning.
    Conforms to the BaseAgent interface so it can be used interchangeably.
    """

    def __init__(self, n_actions: int = NUM_ACTIONS, name: str = "Rule-Based"):
        super().__init__(n_actions, name)

    def select_action(self, state: tuple) -> int:
        """
        State tuple:
            (z0_temp, z1_temp, z2_temp, z3_temp,
             z0_wl,   z1_wl,   z2_wl,   z3_wl,
             hotspot_zone)

        Temp levels: 0=cool, 1=normal, 2=warm, 3=hot, 4=critical
        """
        n_zones = 4
        temp_levels = state[:n_zones]
        hotspot_zone = state[-1]

        max_temp_level = max(temp_levels)

        if max_temp_level >= 4:  # critical
            return 6  # Strong hotspot emphasis
        elif max_temp_level >= 3:  # hot
            # Favour the hottest zone (action 1-4 maps to zone 0-3)
            return hotspot_zone + 1
        elif max_temp_level >= 2:  # warm
            return 5  # Mild hotspot emphasis
        elif max_temp_level <= 0:  # all cool
            return 7  # Eco mode
        else:
            return 0  # Uniform cooling

    def select_greedy(self, state: tuple) -> int:
        return self.select_action(state)

    def update(self, **kwargs) -> None:
        pass  # No learning

    def decay_epsilon(self) -> None:
        pass

    def reset(self) -> None:
        pass  # Nothing to reset

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump({"name": self.name, "type": "rule_based"}, f)

    def load(self, path: str | Path) -> None:
        pass  # No learned state to load

    def get_info(self) -> dict:
        return {"name": self.name, "type": "rule_based"}

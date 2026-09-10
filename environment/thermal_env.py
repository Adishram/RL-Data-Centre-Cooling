"""
Gymnasium-compatible Data Center Cooling Environment.

Ties together:
    ThermalModel    — physics simulation
    WorkloadManager — real / synthetic workload feed
    StateEncoder    — continuous → discrete state

Provides reset() / step(action) / render() interface.
"""

from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from config import (
    NUM_ACTIONS, ACTION_NAMES,
    ThermalConfig, RewardConfig, DatasetConfig,
    StateEncoderConfig, WorkloadMode,
)
from environment.thermal_model import ThermalModel
from environment.workload_replay import WorkloadManager
from environment.state_encoder import StateEncoder


class DataCenterEnv(gym.Env):
    """
    Observation : discrete state tuple (encoded by StateEncoder)
    Action      : integer 0..7  (cooling allocation pattern)
    Reward      : safety – penalties – energy + hotspot improvement
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        thermal_cfg: ThermalConfig | None = None,
        reward_cfg: RewardConfig | None = None,
        dataset_cfg: DatasetConfig | None = None,
        encoder_cfg: StateEncoderConfig | None = None,
        workload_mode: WorkloadMode = WorkloadMode.DATASET_REPLAY,
        dataset_path: str | Path | None = None,
        seed: int = 42,
    ):
        super().__init__()

        self.t_cfg = thermal_cfg or ThermalConfig()
        self.r_cfg = reward_cfg or RewardConfig()
        self.d_cfg = dataset_cfg or DatasetConfig()
        self.e_cfg = encoder_cfg or StateEncoderConfig()
        self.workload_mode = workload_mode
        self.dataset_path = dataset_path

        self.rows = self.t_cfg.grid_rows
        self.cols = self.t_cfg.grid_cols
        self.n_cells = self.rows * self.cols

        # Random number generator
        self.rng = np.random.default_rng(seed)
        self._seed = seed

        # Sub-components
        self.thermal = ThermalModel(self.t_cfg)
        self.encoder = StateEncoder(self.t_cfg, self.e_cfg)

        # Workload manager (may raise FileNotFoundError if dataset missing)
        self.workload_mgr = WorkloadManager(
            mode=workload_mode,
            thermal_cfg=self.t_cfg,
            dataset_cfg=self.d_cfg,
            dataset_path=dataset_path,
            rng=self.rng,
        )

        # Gym spaces
        self.action_space = spaces.Discrete(NUM_ACTIONS)
        # Observation space is discrete but Gym expects Box for compatibility
        # We'll use the tuple state directly in our agents.
        self.observation_space = spaces.Discrete(self.encoder.state_space_size())

        # Episode state
        self._step_count = 0
        self._prev_max_temp = self.t_cfg.initial_temperature
        self._cumulative_energy = 0.0
        self._workload: np.ndarray = np.zeros((self.rows, self.cols))
        self._cooling: np.ndarray = np.zeros((self.rows, self.cols))

        # Build cooling allocation patterns (action → 6×6 cooling matrix)
        self._cooling_patterns = self._build_cooling_patterns()

    # ── Gymnasium interface ─────────────────────────

    def reset(self, seed: int | None = None, options: dict | None = None):
        """Reset environment to initial state."""
        if seed is not None:
            self._seed = seed
            self.rng = np.random.default_rng(seed)

        self.thermal.reset(self.rng)
        self._workload = self.workload_mgr.reset()
        self._step_count = 0
        self._prev_max_temp = self.t_cfg.initial_temperature
        self._cumulative_energy = 0.0
        self._cooling = np.zeros((self.rows, self.cols))

        state = self.encoder.encode(self.thermal.temperatures, self._workload)
        info = self._build_info()
        return state, info


    # ── Reward ──────────────────────────────────────

    def _compute_reward(self) -> tuple[float, dict]:
        """
        Reward decomposition:
            safety_reward       +1.0 if all cells safe
            overheat_penalty    −0.5 per cell above safe_temp
            critical_penalty    −2.0 per cell above critical_temp
            energy_penalty      proportional to cooling energy used
            hotspot_improvement +bonus if max temp decreased
            variance_penalty    penalise temperature spread
        """
        rc = self.r_cfg
        tc = self.t_cfg
        temps = self.thermal.temperatures

        # Safety
        n_overheated = int(np.sum(temps > tc.safe_temperature))
        n_critical = int(np.sum(temps > tc.critical_temperature))
        safety_reward = rc.safety_reward if n_overheated == 0 else 0.0
        overheat_penalty = rc.overheat_penalty_per_cell * n_overheated
        critical_penalty = rc.critical_penalty_per_cell * n_critical

        # Energy
        total_cooling = float(np.sum(self._cooling))
        max_possible = tc.cooling_capacity * self.n_cells
        energy_fraction = total_cooling / max(max_possible, 1e-6)
        energy_penalty = rc.energy_weight * energy_fraction

        # Hotspot improvement
        current_max = self.thermal.get_max_temperature()
        hotspot_delta = self._prev_max_temp - current_max  # positive = improvement
        hotspot_bonus = rc.hotspot_improvement_weight * np.clip(hotspot_delta, -2, 2)

        # Variance penalty
        variance = self.thermal.get_temperature_variance()
        var_penalty = rc.temperature_variance_penalty * min(variance, 50.0)

        reward = (
            safety_reward
            - overheat_penalty
            - critical_penalty
            - energy_penalty
            + hotspot_bonus
            - var_penalty
        )

        info = {
            "reward_safety": safety_reward,
            "reward_overheat_penalty": -overheat_penalty,
            "reward_critical_penalty": -critical_penalty,
            "reward_energy_penalty": -energy_penalty,
            "reward_hotspot_bonus": hotspot_bonus,
            "reward_variance_penalty": -var_penalty,
            "reward_total": reward,
        }
        return float(reward), info

    # ── Info ────────────────────────────────────────

    def _build_info(self) -> dict:
        hr, hc, ht = self.thermal.get_hotspot()
        return {
            "step": self._step_count,
            "max_temperature": self.thermal.get_max_temperature(),
            "mean_temperature": self.thermal.get_mean_temperature(),
            "temperature_variance": self.thermal.get_temperature_variance(),
            "energy_used": self._cumulative_energy,
            "hotspot_location": (hr, hc),
            "hotspot_temperature": ht,
            "cooling_allocation": self._cooling.copy(),
            "overheated_cells": self.thermal.count_overheated(),
            "workload": self._workload.copy(),
            "temperatures": self.thermal.temperatures.copy(),
        }

    # ── Cooling patterns ────────────────────────────

    def _build_cooling_patterns(self) -> list[np.ndarray]:
        """
        Define 8 concrete cooling allocation patterns.

        Each pattern is a (rows, cols) matrix of cooling power ∈ [0, capacity].

        Design rationale (cap=6.0 → max heat per cell at full load = 5.0):
            - Uniform (0.6×cap=3.6): handles up to ~72% workload; overheats at peak
            - Zone focus (base=0.4, target=0.9): handles peak in one zone
            - Eco (0.15×cap=0.9): saves energy when load is low
            - Hotspot (dynamic): concentrates cooling on hottest cell
        """
        cap = self.t_cfg.cooling_capacity
        base = cap * 0.4    # baseline cooling everywhere
        high = cap * 0.9    # aggressive zone cooling
        low = cap * 0.15    # eco mode

        patterns = []

        # 0: Uniform cooling — moderate everywhere
        patterns.append(np.full((self.rows, self.cols), cap * 0.6))

        # 1-4: Favor each zone — strong in target, moderate elsewhere
        for zone_id in range(4):
            p = np.full((self.rows, self.cols), base)
            rs, re, cs, ce = self.encoder.zones[zone_id]
            p[rs:re, cs:ce] = high
            patterns.append(p)

        # 5: Mild hotspot emphasis (base level — dynamic targeting added in step)
        patterns.append(np.full((self.rows, self.cols), cap * 0.45))

        # 6: Strong hotspot emphasis (lower base — dynamic targeting added in step)
        patterns.append(np.full((self.rows, self.cols), cap * 0.35))

        # 7: Eco / reduce global cooling — saves energy
        patterns.append(np.full((self.rows, self.cols), low))

        return patterns

    def _get_cooling_for_action(self, action: int) -> np.ndarray:
        """Get cooling pattern, with dynamic hotspot adjustment for actions 5-6."""
        pattern = self._cooling_patterns[action].copy()

        if action in (5, 6):
            # Direct extra cooling toward current hotspot
            hr, hc, _ = self.thermal.get_hotspot()
            cap = self.t_cfg.cooling_capacity
            intensity = cap * 0.7 if action == 5 else cap * 0.95

            # Apply to hotspot and its neighbours
            for dr in range(-1, 2):
                for dc in range(-1, 2):
                    rr, cc = hr + dr, hc + dc
                    if 0 <= rr < self.rows and 0 <= cc < self.cols:
                        dist = abs(dr) + abs(dc)
                        pattern[rr, cc] = max(
                            pattern[rr, cc],
                            intensity * (1.0 - 0.3 * dist)
                        )

        return np.clip(pattern, 0, self.t_cfg.cooling_capacity)

    def step(self, action: int):
        """Execute one cooling action and advance the simulation."""
        assert 0 <= action < NUM_ACTIONS, f"Invalid action {action}"

        # 1. Get cooling (with dynamic hotspot targeting for actions 5,6)
        self._cooling = self._get_cooling_for_action(action)

        # 2. Advance workload
        self._workload = self.workload_mgr.advance()

        # 3. Advance thermal simulation
        self.thermal.step(self._workload, self._cooling, self.rng)

        # 4. Compute reward
        reward, reward_info = self._compute_reward()

        # 5. Update bookkeeping
        self._step_count += 1
        energy_this_step = float(np.sum(self._cooling))
        self._cumulative_energy += energy_this_step
        self._prev_max_temp = self.thermal.get_max_temperature()

        # 6. Termination
        max_steps = min(
            self.t_cfg.max_steps_per_episode,
            self.workload_mgr.num_timesteps,
        )
        truncated = self._step_count >= max_steps
        terminated = False

        # 7. Encode state
        state = self.encoder.encode(self.thermal.temperatures, self._workload)

        # 8. Build info
        info = self._build_info()
        info.update(reward_info)
        info["action"] = action
        info["action_name"] = ACTION_NAMES[action]
        info["energy_this_step"] = energy_this_step

        return state, reward, terminated, truncated, info

    # ── Render ──────────────────────────────────────

    def render(self):
        """Print a simple text representation."""
        temps = self.thermal.temperatures
        print(f"\nStep {self._step_count}  "
              f"Max={self.thermal.get_max_temperature():.1f}°C  "
              f"Mean={self.thermal.get_mean_temperature():.1f}°C")
        print("Temperature grid:")
        for r in range(self.rows):
            row_str = " ".join(f"{temps[r, c]:5.1f}" for c in range(self.cols))
            print(f"  {row_str}")

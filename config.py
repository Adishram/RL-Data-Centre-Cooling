"""
Central configuration for the RL Data-Center Cooling project.

Every tunable parameter lives here. No magic numbers elsewhere.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ─────────────────────────────────────────────
# Workload / scenario modes
# ─────────────────────────────────────────────
class WorkloadMode(Enum):
    DATASET_REPLAY = "dataset_replay"
    STATIC_HOTSPOT = "static_hotspot"
    MOVING_HOTSPOT = "moving_hotspot"
    MULTI_HOTSPOT = "multi_hotspot"
    WORKLOAD_BURST = "workload_burst"


# ─────────────────────────────────────────────
# Dataset configuration
# ─────────────────────────────────────────────
@dataclass
class DatasetConfig:
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    mini_csv: str = "data/mini_alibaba.csv"
    machine_count: int = 36
    max_timesteps: int = 1000
    random_seed: int = 42
    # Alibaba server_usage.csv column names (official schema)
    col_timestamp: str = "timestamp"
    col_machine_id: str = "machine_id"
    col_cpu_util: str = "cpu_util"
    col_mem_util: str = "mem_util"
    col_disk_util: str = "disk_util"
    col_load1: str = "load1"
    col_load5: str = "load5"
    col_load15: str = "load15"


# ─────────────────────────────────────────────
# Thermal model configuration
# ─────────────────────────────────────────────
@dataclass
class ThermalConfig:
    grid_rows: int = 6
    grid_cols: int = 6
    # Temperatures (°C)
    initial_temperature: float = 22.0
    ambient_temperature: float = 20.0
    safe_temperature: float = 35.0
    warning_temperature: float = 45.0
    critical_temperature: float = 55.0
    # Physics coefficients
    diffusion_coefficient: float = 0.06     # neighbor heat spread per step
    workload_heat_coefficient: float = 5.0  # max heat from full CPU load
    memory_heat_coefficient: float = 1.0    # smaller secondary heat
    cooling_coefficient: float = 1.0        # multiplier on cooling action
    ambient_pull_coefficient: float = 0.05  # rate of pull toward ambient
    temperature_noise_std: float = 0.15     # small Gaussian noise
    # Cooling
    cooling_capacity: float = 6.0           # max cooling per cell per step
    min_cooling: float = 0.0
    # Simulation
    max_steps_per_episode: int = 500


# ─────────────────────────────────────────────
# Reward configuration
# ─────────────────────────────────────────────
@dataclass
class RewardConfig:
    """
    Reward = safety_reward
             - overheating_penalty
             - energy_penalty
             + hotspot_improvement_bonus

    Scale reasoning:
        ─ Base safety reward is +1.0 when everything is cool.
        ─ Each cell exceeding safe_temp costs −0.15 → at worst −5.4
        ─ Each cell exceeding critical_temp costs an extra −0.5
        ─ Energy penalty scales ∈ [0, ~0.3] relative to max possible cooling.
        ─ Hotspot improvement gives up to +1.0 when the max temp drops.
    """
    safety_reward: float = 1.0
    overheat_penalty_per_cell: float = 0.15
    critical_penalty_per_cell: float = 0.5
    energy_weight: float = 1.5              # penalise energy harder to create tradeoff
    hotspot_improvement_weight: float = 0.5
    temperature_variance_penalty: float = 0.01


# ─────────────────────────────────────────────
# RL hyperparameters
# ─────────────────────────────────────────────
@dataclass
class RLConfig:
    alpha: float = 0.1           # learning rate
    gamma: float = 0.95          # discount factor
    epsilon_start: float = 1.0
    epsilon_decay: float = 0.995
    epsilon_min: float = 0.05
    episodes: int = 300
    seed: int = 42


# ─────────────────────────────────────────────
# State-encoder settings
# ─────────────────────────────────────────────
@dataclass
class StateEncoderConfig:
    # Temperature discretisation thresholds (°C)
    temp_bins: tuple = (30.0, 35.0, 42.0, 50.0)
    # Labels: cool(0), normal(1), warm(2), hot(3), critical(4)
    # Workload discretisation thresholds (fraction 0-1)
    workload_bins: tuple = (0.3, 0.7)
    # Labels: low(0), medium(1), high(2)
    num_temp_levels: int = 5
    num_workload_levels: int = 3
    num_zones: int = 4


# ─────────────────────────────────────────────
# Action space
# ─────────────────────────────────────────────
NUM_ACTIONS = 8
ACTION_NAMES = [
    "Uniform cooling",
    "Favor Zone 0 (top-left)",
    "Favor Zone 1 (top-right)",
    "Favor Zone 2 (bottom-left)",
    "Favor Zone 3 (bottom-right)",
    "Mild hotspot emphasis",
    "Strong hotspot emphasis",
    "Eco / reduce cooling",
]


# ─────────────────────────────────────────────
# Convenience: build a full default config bundle
# ─────────────────────────────────────────────
@dataclass
class ProjectConfig:
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    thermal: ThermalConfig = field(default_factory=ThermalConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    rl: RLConfig = field(default_factory=RLConfig)
    encoder: StateEncoderConfig = field(default_factory=StateEncoderConfig)
    workload_mode: WorkloadMode = WorkloadMode.DATASET_REPLAY


def default_config() -> ProjectConfig:
    return ProjectConfig()

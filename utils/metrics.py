"""
Metric collection and aggregation for evaluation episodes.
"""

from dataclasses import dataclass, field
import numpy as np


@dataclass
class EpisodeMetrics:
    """Collects metrics across one evaluation episode."""
    temperatures: list[float] = field(default_factory=list)
    max_temperatures: list[float] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    energies: list[float] = field(default_factory=list)
    overheated_counts: list[int] = field(default_factory=list)
    actions: list[int] = field(default_factory=list)

    def record_step(self, info: dict, reward: float, action: int):
        self.temperatures.append(info.get("mean_temperature", 0))
        self.max_temperatures.append(info.get("max_temperature", 0))
        self.rewards.append(reward)
        self.energies.append(info.get("energy_this_step", 0))
        self.overheated_counts.append(info.get("overheated_cells", 0))
        self.actions.append(action)

    def summary(self) -> dict:
        """Compute summary statistics."""
        if not self.temperatures:
            return {}

        temps = np.array(self.temperatures)
        max_temps = np.array(self.max_temperatures)
        rewards = np.array(self.rewards)
        energies = np.array(self.energies)
        overheat = np.array(self.overheated_counts)

        return {
            "avg_temperature": float(np.mean(temps)),
            "max_temperature": float(np.max(max_temps)),
            "temperature_variance": float(np.var(temps)),
            "time_above_safe": int(np.sum(overheat > 0)),
            "overheat_events": int(np.sum(overheat)),
            "total_energy": float(np.sum(energies)),
            "avg_cooling_power": float(np.mean(energies)),
            "cumulative_reward": float(np.sum(rewards)),
            "avg_reward": float(np.mean(rewards)),
            "episode_length": len(self.temperatures),
        }

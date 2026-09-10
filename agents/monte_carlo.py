"""
First-Visit Monte Carlo agent.

Collects complete episodes, then updates Q-values using
the discounted return from the first visit to each (s, a) pair.
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from agents.base_agent import BaseAgent
from config import RLConfig


class MonteCarloAgent(BaseAgent):
    """First-visit Monte Carlo control with ε-greedy exploration."""

    def __init__(
        self,
        n_actions: int,
        cfg: RLConfig | None = None,
        name: str = "Monte Carlo",
    ):
        super().__init__(n_actions, name)
        self.cfg = cfg or RLConfig()
        self.gamma = self.cfg.gamma
        self.epsilon = self.cfg.epsilon_start
        self.epsilon_decay = self.cfg.epsilon_decay
        self.epsilon_min = self.cfg.epsilon_min

        self.q_table: dict[tuple, np.ndarray] = defaultdict(
            lambda: np.zeros(self.n_actions, dtype=np.float64)
        )
        # Visit counts for incremental mean
        self.returns_count: dict[tuple, np.ndarray] = defaultdict(
            lambda: np.zeros(self.n_actions, dtype=np.float64)
        )
        self.rng = np.random.default_rng(self.cfg.seed)

        # Episode buffer
        self._episode: list[tuple] = []  # [(state, action, reward), ...]

    # ── interface ───────────────────────────────────

    def select_action(self, state: tuple) -> int:
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(0, self.n_actions))
        q_vals = self.q_table[state]
        max_q = np.max(q_vals)
        best = np.where(q_vals == max_q)[0]
        return int(self.rng.choice(best))

    def select_greedy(self, state: tuple) -> int:
        return int(np.argmax(self.q_table[state]))

    def store_transition(self, state: tuple, action: int, reward: float):
        """Buffer a (s, a, r) transition within the current episode."""
        self._episode.append((state, action, reward))

    def update(self, **kwargs) -> None:
        """
        Process the completed episode buffer using first-visit MC.
        Call this at the end of each episode.
        """
        if not self._episode:
            return

        G = 0.0
        visited: set[tuple[tuple, int]] = set()

        # Walk backwards through the episode
        for state, action, reward in reversed(self._episode):
            G = reward + self.gamma * G
            sa = (state, action)
            if sa not in visited:
                visited.add(sa)
                self.returns_count[state][action] += 1
                n = self.returns_count[state][action]
                # Incremental mean update
                self.q_table[state][action] += (
                    (G - self.q_table[state][action]) / n
                )

        self._episode.clear()

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_min,
                           self.epsilon * self.epsilon_decay)

    def reset(self) -> None:
        self.q_table = defaultdict(
            lambda: np.zeros(self.n_actions, dtype=np.float64)
        )
        self.returns_count = defaultdict(
            lambda: np.zeros(self.n_actions, dtype=np.float64)
        )
        self.epsilon = self.cfg.epsilon_start
        self._episode.clear()
        self.rng = np.random.default_rng(self.cfg.seed)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "q_table": {str(k): v.tolist() for k, v in self.q_table.items()},
            "returns_count": {str(k): v.tolist()
                              for k, v in self.returns_count.items()},
            "epsilon": self.epsilon,
            "n_actions": self.n_actions,
        }
        with open(path, "w") as f:
            json.dump(data, f)

    def load(self, path: str | Path) -> None:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"No saved model at {path}")
        with open(path) as f:
            data = json.load(f)
        self.epsilon = data.get("epsilon", self.cfg.epsilon_min)
        self.q_table = defaultdict(
            lambda: np.zeros(self.n_actions, dtype=np.float64)
        )
        for k_str, v in data["q_table"].items():
            key = tuple(int(x) for x in k_str.strip("()").split(",") if x.strip())
            self.q_table[key] = np.array(v, dtype=np.float64)

        self.returns_count = defaultdict(
            lambda: np.zeros(self.n_actions, dtype=np.float64)
        )
        if "returns_count" in data:
            for k_str, v in data["returns_count"].items():
                key = tuple(int(x) for x in k_str.strip("()").split(",") if x.strip())
                self.returns_count[key] = np.array(v, dtype=np.float64)

    def get_info(self) -> dict:
        info = super().get_info()
        info.update({
            "states_visited": len(self.q_table),
            "epsilon": self.epsilon,
            "episode_buffer_len": len(self._episode),
        })
        return info

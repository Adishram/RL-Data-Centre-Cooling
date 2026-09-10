"""
SARSA agent (on-policy TD(0)).

Update rule:
    Q(s, a) ← Q(s, a) + α [ r + γ Q(s', a') − Q(s, a) ]

where a' is the action *actually taken* in s' (on-policy).
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from agents.base_agent import BaseAgent
from config import RLConfig


class SarsaAgent(BaseAgent):
    """Tabular SARSA with ε-greedy exploration."""

    def __init__(
        self,
        n_actions: int,
        cfg: RLConfig | None = None,
        name: str = "SARSA",
    ):
        super().__init__(n_actions, name)
        self.cfg = cfg or RLConfig()
        self.alpha = self.cfg.alpha
        self.gamma = self.cfg.gamma
        self.epsilon = self.cfg.epsilon_start
        self.epsilon_decay = self.cfg.epsilon_decay
        self.epsilon_min = self.cfg.epsilon_min

        self.q_table: dict[tuple, np.ndarray] = defaultdict(
            lambda: np.zeros(self.n_actions, dtype=np.float64)
        )
        self.rng = np.random.default_rng(self.cfg.seed)

    # ── interface ───────────────────────────────────

    def select_action(self, state: tuple) -> int:
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(0, self.n_actions))
        q_vals = self.q_table[state]
        max_q = np.max(q_vals)
        best_actions = np.where(q_vals == max_q)[0]
        return int(self.rng.choice(best_actions))

    def select_greedy(self, state: tuple) -> int:
        q_vals = self.q_table[state]
        return int(np.argmax(q_vals))

    def update(self, state: tuple, action: int, reward: float,
               next_state: tuple, next_action: int, done: bool,
               **kwargs) -> None:
        """
        On-policy TD(0) update.
        Key difference from Q-Learning: uses Q(s', a') not max_a' Q(s', a').
        """
        q_next = self.q_table[next_state][next_action] if not done else 0.0
        td_target = reward + self.gamma * q_next
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.alpha * td_error

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_min,
                           self.epsilon * self.epsilon_decay)

    def reset(self) -> None:
        self.q_table = defaultdict(
            lambda: np.zeros(self.n_actions, dtype=np.float64)
        )
        self.epsilon = self.cfg.epsilon_start
        self.rng = np.random.default_rng(self.cfg.seed)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "q_table": {str(k): v.tolist() for k, v in self.q_table.items()},
            "epsilon": self.epsilon,
            "alpha": self.alpha,
            "gamma": self.gamma,
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

    def get_info(self) -> dict:
        info = super().get_info()
        info.update({
            "states_visited": len(self.q_table),
            "epsilon": self.epsilon,
        })
        return info

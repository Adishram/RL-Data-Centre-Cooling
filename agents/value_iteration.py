"""
Value Iteration agent using a sampled transition model.

Since our environment is simulated (we have a model), we *can* do
dynamic programming — but the state space (~200K) is large for full
enumeration.

Strategy:
    1. During a "data collection" phase, run the environment with a
       random/ε-greedy policy and record (s, a, r, s') transitions.
    2. Build a sampled transition model: T(s, a) → {(s', r, count)}.
    3. Run Bellman value-iteration sweeps over the *visited* states only.
    4. Extract greedy policy.
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from agents.base_agent import BaseAgent
from config import RLConfig


class ValueIterationAgent(BaseAgent):
    """
    Model-based Value Iteration on a sampled transition model.
    """

    def __init__(
        self,
        n_actions: int,
        cfg: RLConfig | None = None,
        name: str = "Value Iteration",
    ):
        super().__init__(n_actions, name)
        self.cfg = cfg or RLConfig()
        self.gamma = self.cfg.gamma

        # Transition model: (state, action) → list of (next_state, reward, count)
        self._transitions: dict[tuple, dict[tuple, list]] = defaultdict(
            lambda: defaultdict(lambda: [0.0, 0])  # [cum_reward, count]
        )
        # We aggregate: for (s,a) → s', store cumulative reward and count
        # so mean_reward = cum_reward / count
        self._model: dict = defaultdict(lambda: defaultdict(dict))

        # Value function and policy
        self.V: dict[tuple, float] = defaultdict(float)
        self.policy: dict[tuple, int] = {}

        self.rng = np.random.default_rng(self.cfg.seed)
        self.epsilon = self.cfg.epsilon_start
        self.epsilon_decay = self.cfg.epsilon_decay
        self.epsilon_min = self.cfg.epsilon_min

    # ── interface ───────────────────────────────────

    def select_action(self, state: tuple) -> int:
        """During data collection, use ε-greedy. After VI, use policy."""
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(0, self.n_actions))
        if state in self.policy:
            return self.policy[state]
        # Unknown state: random
        return int(self.rng.integers(0, self.n_actions))

    def select_greedy(self, state: tuple) -> int:
        if state in self.policy:
            return self.policy[state]
        return 0  # default

    def update(self, state: tuple, action: int, reward: float,
               next_state: tuple, done: bool, **kwargs) -> None:
        """Record a transition into the model."""
        key = (state, action)
        ns_key = next_state
        entry = self._model[key]
        if ns_key not in entry:
            entry[ns_key] = {"cum_reward": 0.0, "count": 0}
        entry[ns_key]["cum_reward"] += reward
        entry[ns_key]["count"] += 1

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_min,
                           self.epsilon * self.epsilon_decay)

    def run_value_iteration(self, max_iterations: int = 100,
                             theta: float = 1e-4) -> int:
        """
        Run Bellman value-iteration sweeps over visited states.

        Returns the number of iterations until convergence.
        """
        # Collect all visited states
        all_states: set[tuple] = set()
        for (s, a) in self._model:
            all_states.add(s)
            for ns in self._model[(s, a)]:
                all_states.add(ns)

        if not all_states:
            return 0

        # Initialise V
        for s in all_states:
            if s not in self.V:
                self.V[s] = 0.0

        for iteration in range(1, max_iterations + 1):
            delta = 0.0
            for s in all_states:
                if not any((s, a) in self._model for a in range(self.n_actions)):
                    continue
                v_old = self.V[s]
                action_values = []
                for a in range(self.n_actions):
                    key = (s, a)
                    if key not in self._model or not self._model[key]:
                        action_values.append(self.V[s])  # no data → keep current
                        continue
                    # Expected value under sampled transition
                    total_count = sum(d["count"] for d in self._model[key].values())
                    q_sa = 0.0
                    for ns, d in self._model[key].items():
                        prob = d["count"] / total_count
                        mean_r = d["cum_reward"] / d["count"]
                        q_sa += prob * (mean_r + self.gamma * self.V.get(ns, 0.0))
                    action_values.append(q_sa)

                self.V[s] = max(action_values)
                delta = max(delta, abs(v_old - self.V[s]))

            if delta < theta:
                return iteration

        return max_iterations

    def extract_policy(self) -> None:
        """Derive greedy policy from V and the model."""
        for s in self.V:
            best_a = 0
            best_q = -float("inf")
            for a in range(self.n_actions):
                key = (s, a)
                if key not in self._model or not self._model[key]:
                    continue
                total_count = sum(d["count"] for d in self._model[key].values())
                q_sa = 0.0
                for ns, d in self._model[key].items():
                    prob = d["count"] / total_count
                    mean_r = d["cum_reward"] / d["count"]
                    q_sa += prob * (mean_r + self.gamma * self.V.get(ns, 0.0))
                if q_sa > best_q:
                    best_q = q_sa
                    best_a = a
            self.policy[s] = best_a

    def reset(self) -> None:
        self._model = defaultdict(lambda: defaultdict(dict))
        self.V = defaultdict(float)
        self.policy = {}
        self.epsilon = self.cfg.epsilon_start
        self.rng = np.random.default_rng(self.cfg.seed)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Serialise policy and V
        data = {
            "policy": {str(k): v for k, v in self.policy.items()},
            "V": {str(k): v for k, v in self.V.items()},
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
        self.policy = {}
        for k_str, v in data.get("policy", {}).items():
            key = tuple(int(x) for x in k_str.strip("()").split(",") if x.strip())
            self.policy[key] = v
        self.V = defaultdict(float)
        for k_str, v in data.get("V", {}).items():
            key = tuple(int(x) for x in k_str.strip("()").split(",") if x.strip())
            self.V[key] = v

    def get_info(self) -> dict:
        info = super().get_info()
        info.update({
            "model_entries": len(self._model),
            "states_with_V": len(self.V),
            "policy_size": len(self.policy),
            "epsilon": self.epsilon,
        })
        return info

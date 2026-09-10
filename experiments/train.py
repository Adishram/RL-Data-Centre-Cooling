"""
Training pipeline for RL agents.

Usage:
    python -m experiments.train --algorithm q_learning --episodes 300 --seed 42
    python -m experiments.train --algorithm sarsa
    python -m experiments.train --algorithm monte_carlo
    python -m experiments.train --algorithm value_iteration
    python -m experiments.train --algorithm rule_based   # just saves the baseline
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    RLConfig, ThermalConfig, RewardConfig, DatasetConfig,
    StateEncoderConfig, WorkloadMode, NUM_ACTIONS,
)
from environment.thermal_env import DataCenterEnv
from agents.q_learning import QLearningAgent
from agents.sarsa import SarsaAgent
from agents.monte_carlo import MonteCarloAgent
from agents.value_iteration import ValueIterationAgent
from baselines.rule_based import RuleBasedAgent
from experiments.scenarios import get_scenario
from utils.reproducibility import set_global_seed


ALGORITHM_MAP = {
    "q_learning": QLearningAgent,
    "sarsa": SarsaAgent,
    "monte_carlo": MonteCarloAgent,
    "value_iteration": ValueIterationAgent,
    "rule_based": RuleBasedAgent,
}

MODELS_DIR = PROJECT_ROOT / "models"


def create_agent(algo_name: str, rl_cfg: RLConfig) -> object:
    """Instantiate the appropriate agent."""
    if algo_name not in ALGORITHM_MAP:
        raise ValueError(f"Unknown algorithm '{algo_name}'. "
                          f"Available: {list(ALGORITHM_MAP.keys())}")
    cls = ALGORITHM_MAP[algo_name]
    if algo_name == "rule_based":
        return cls(n_actions=NUM_ACTIONS)
    return cls(n_actions=NUM_ACTIONS, cfg=rl_cfg)


def train(
    algo_name: str,
    episodes: int = 300,
    seed: int = 42,
    scenario: str = "alibaba_replay",
    dataset_path: str | None = None,
    alpha: float = 0.1,
    gamma: float = 0.95,
    epsilon_start: float = 1.0,
    epsilon_decay: float = 0.995,
    epsilon_min: float = 0.05,
    max_steps: int = 500,
):
    """Run the full training loop for a given algorithm."""
    set_global_seed(seed)

    # Configuration
    rl_cfg = RLConfig(
        alpha=alpha, gamma=gamma,
        epsilon_start=epsilon_start,
        epsilon_decay=epsilon_decay,
        epsilon_min=epsilon_min,
        episodes=episodes, seed=seed,
    )
    thermal_cfg = ThermalConfig(max_steps_per_episode=max_steps)
    reward_cfg = RewardConfig()
    dataset_cfg = DatasetConfig()
    encoder_cfg = StateEncoderConfig()

    scenario_info = get_scenario(scenario)
    mode = scenario_info["mode"]

    # Resolve dataset path
    if dataset_path is None:
        dp = PROJECT_ROOT / dataset_cfg.mini_csv
    else:
        dp = Path(dataset_path)

    # Create environment
    env = DataCenterEnv(
        thermal_cfg=thermal_cfg,
        reward_cfg=reward_cfg,
        dataset_cfg=dataset_cfg,
        encoder_cfg=encoder_cfg,
        workload_mode=mode,
        dataset_path=dp,
        seed=seed,
    )

    # Create agent
    agent = create_agent(algo_name, rl_cfg)

    if algo_name == "rule_based":
        # No training needed for rule-based
        save_path = MODELS_DIR / f"{algo_name}.json"
        agent.save(save_path)
        print(f"[TRAIN] Rule-based agent saved to {save_path}")
        return [], agent

    print(f"\n{'='*60}")
    print(f"  Training: {agent.name}")
    print(f"  Algorithm: {algo_name}")
    print(f"  Episodes: {episodes}")
    print(f"  Scenario: {scenario_info['name']}")
    print(f"  Seed: {seed}")
    print(f"  α={alpha}, γ={gamma}, ε₀={epsilon_start}")
    print(f"{'='*60}\n")

    episode_rewards = []
    start_time = time.time()

    for ep in range(1, episodes + 1):
        state, info = env.reset(seed=seed + ep)
        episode_reward = 0.0
        done = False

        if algo_name == "sarsa":
            action = agent.select_action(state)

        step = 0
        while not done:
            if algo_name == "sarsa":
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                next_action = agent.select_action(next_state)
                agent.update(
                    state=state, action=action, reward=reward,
                    next_state=next_state, next_action=next_action,
                    done=done,
                )
                state = next_state
                action = next_action

            elif algo_name == "monte_carlo":
                action = agent.select_action(state)
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                agent.store_transition(state, action, reward)
                state = next_state

            elif algo_name in ("q_learning", "value_iteration"):
                action = agent.select_action(state)
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                agent.update(
                    state=state, action=action, reward=reward,
                    next_state=next_state, done=done,
                )
                state = next_state

            episode_reward += reward
            step += 1

        # End-of-episode updates
        if algo_name == "monte_carlo":
            agent.update()  # Process episode buffer

        agent.decay_epsilon()
        episode_rewards.append(episode_reward)

        # Progress
        if ep % max(1, episodes // 10) == 0 or ep == 1:
            avg_last = np.mean(episode_rewards[-20:])
            elapsed = time.time() - start_time
            eps_info = agent.get_info()
            print(f"  Episode {ep:4d}/{episodes}  "
                  f"reward={episode_reward:7.1f}  "
                  f"avg20={avg_last:7.1f}  "
                  f"ε={eps_info.get('epsilon', 0):.3f}  "
                  f"states={eps_info.get('states_visited', eps_info.get('model_entries', 0))}  "
                  f"[{elapsed:.1f}s]")

    # Value Iteration: run VI sweeps after data collection
    if algo_name == "value_iteration":
        print("\n  Running Value Iteration sweeps ...")
        iters = agent.run_value_iteration(max_iterations=200)
        agent.extract_policy()
        print(f"  Converged in {iters} iterations, policy size = {len(agent.policy)}")

    # Save model
    save_path = MODELS_DIR / f"{algo_name}.json"
    agent.save(save_path)
    elapsed = time.time() - start_time
    print(f"\n  Training complete in {elapsed:.1f}s")
    print(f"  Model saved to {save_path}")

    return episode_rewards, agent


def main():
    parser = argparse.ArgumentParser(description="Train an RL agent")
    parser.add_argument("--algorithm", "-a", type=str, default="q_learning",
                        choices=list(ALGORITHM_MAP.keys()))
    parser.add_argument("--episodes", "-e", type=int, default=300)
    parser.add_argument("--seed", "-s", type=int, default=42)
    parser.add_argument("--scenario", type=str, default="alibaba_replay")
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-decay", type=float, default=0.995)
    parser.add_argument("--epsilon-min", type=float, default=0.05)
    parser.add_argument("--max-steps", type=int, default=500)
    args = parser.parse_args()

    train(
        algo_name=args.algorithm,
        episodes=args.episodes,
        seed=args.seed,
        scenario=args.scenario,
        dataset_path=args.dataset,
        alpha=args.alpha,
        gamma=args.gamma,
        epsilon_start=args.epsilon_start,
        epsilon_decay=args.epsilon_decay,
        epsilon_min=args.epsilon_min,
        max_steps=args.max_steps,
    )


if __name__ == "__main__":
    main()

"""
Evaluation pipeline: run all trained agents across scenarios,
collect metrics, save comparison CSV and generate plots.

Usage:
    python -m experiments.evaluate --seed 42
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    RLConfig, ThermalConfig, RewardConfig, DatasetConfig,
    StateEncoderConfig, NUM_ACTIONS,
)
from environment.thermal_env import DataCenterEnv
from agents.q_learning import QLearningAgent
from agents.sarsa import SarsaAgent
from agents.monte_carlo import MonteCarloAgent
from agents.value_iteration import ValueIterationAgent
from baselines.rule_based import RuleBasedAgent
from experiments.scenarios import SCENARIOS
from utils.metrics import EpisodeMetrics
from utils.reproducibility import set_global_seed
from utils.serialization import ensure_dir

MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

ALGORITHMS = {
    "rule_based": (RuleBasedAgent, False),
    "q_learning": (QLearningAgent, True),
    "sarsa": (SarsaAgent, True),
    "monte_carlo": (MonteCarloAgent, True),
    "value_iteration": (ValueIterationAgent, True),
}


def load_agent(algo_name: str) -> object:
    """Load a trained agent from models/."""
    cls, needs_cfg = ALGORITHMS[algo_name]
    if needs_cfg:
        agent = cls(n_actions=NUM_ACTIONS, cfg=RLConfig())
    else:
        agent = cls(n_actions=NUM_ACTIONS)

    model_path = MODELS_DIR / f"{algo_name}.json"
    if model_path.exists():
        agent.load(model_path)
    else:
        print(f"  [WARN] No saved model for {algo_name} at {model_path}")
    return agent


def evaluate_agent(
    agent,
    algo_name: str,
    scenario_name: str,
    seed: int = 42,
    max_steps: int = 500,
    dataset_path: Path | None = None,
) -> dict:
    """Run one evaluation episode and collect metrics."""
    scenario = SCENARIOS[scenario_name]
    thermal_cfg = ThermalConfig(max_steps_per_episode=max_steps)
    dp = dataset_path or (PROJECT_ROOT / DatasetConfig().mini_csv)

    env = DataCenterEnv(
        thermal_cfg=thermal_cfg,
        reward_cfg=RewardConfig(),
        dataset_cfg=DatasetConfig(),
        encoder_cfg=StateEncoderConfig(),
        workload_mode=scenario["mode"],
        dataset_path=dp,
        seed=seed + 10000,  # Different seed from training
    )

    state, info = env.reset(seed=seed + 10000)
    metrics = EpisodeMetrics()
    done = False

    while not done:
        action = agent.select_greedy(state)
        next_state, reward, terminated, truncated, info = env.step(action)
        metrics.record_step(info, reward, action)
        state = next_state
        done = terminated or truncated

    summary = metrics.summary()
    summary["algorithm"] = algo_name
    summary["scenario"] = scenario_name
    summary["seed"] = seed
    return summary, metrics


def generate_plots(results_df: pd.DataFrame, all_trajectories: dict):
    """Generate comparison plots and save to results/."""
    ensure_dir(RESULTS_DIR)

    # Use a clean style
    plt.style.use("seaborn-v0_8-darkgrid")
    colors = ["#2ecc71", "#3498db", "#e74c3c", "#f39c12", "#9b59b6"]

    # ── 1. Algorithm comparison bar chart (Alibaba replay) ──
    replay_df = results_df[results_df["scenario"] == "alibaba_replay"]
    if not replay_df.empty:
        fig, axes = plt.subplots(1, 4, figsize=(18, 5))
        fig.suptitle("Algorithm Comparison — Alibaba Replay", fontsize=14, fontweight="bold")

        metrics_to_plot = [
            ("cumulative_reward", "Cumulative Reward", True),
            ("total_energy", "Total Cooling Energy", False),
            ("max_temperature", "Peak Temperature (°C)", False),
            ("overheat_events", "Overheating Events", False),
        ]

        for ax, (col, title, higher_better) in zip(axes, metrics_to_plot):
            vals = replay_df.set_index("algorithm")[col]
            bar_colors = [colors[i % len(colors)] for i in range(len(vals))]
            vals.plot(kind="bar", ax=ax, color=bar_colors, edgecolor="white", linewidth=0.5)
            ax.set_title(title, fontsize=11)
            ax.set_xlabel("")
            ax.tick_params(axis="x", rotation=45)
            ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "algorithm_comparison.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: algorithm_comparison.png")

    # ── 2. Temperature trajectory ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Temperature Trajectories — Alibaba Replay", fontsize=14, fontweight="bold")

    for i, (algo_name, traj) in enumerate(all_trajectories.items()):
        if "alibaba_replay" not in traj:
            continue
        m = traj["alibaba_replay"]
        c = colors[i % len(colors)]
        axes[0].plot(m.temperatures, label=algo_name, color=c, alpha=0.8, linewidth=1.2)
        axes[1].plot(m.max_temperatures, label=algo_name, color=c, alpha=0.8, linewidth=1.2)

    axes[0].set_title("Mean Temperature")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("°C")
    axes[0].legend(fontsize=8)

    axes[1].set_title("Max Temperature")
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("°C")
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "temperature_trajectories.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: temperature_trajectories.png")

    # ── 3. Cooling energy trajectory ──
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (algo_name, traj) in enumerate(all_trajectories.items()):
        if "alibaba_replay" not in traj:
            continue
        m = traj["alibaba_replay"]
        cumulative = np.cumsum(m.energies)
        ax.plot(cumulative, label=algo_name, color=colors[i % len(colors)],
                alpha=0.8, linewidth=1.2)

    ax.set_title("Cumulative Cooling Energy — Alibaba Replay", fontsize=13, fontweight="bold")
    ax.set_xlabel("Step")
    ax.set_ylabel("Cumulative Energy")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "cooling_energy.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: cooling_energy.png")

    # ── 4. Reward trajectory ──
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (algo_name, traj) in enumerate(all_trajectories.items()):
        if "alibaba_replay" not in traj:
            continue
        m = traj["alibaba_replay"]
        cumulative = np.cumsum(m.rewards)
        ax.plot(cumulative, label=algo_name, color=colors[i % len(colors)],
                alpha=0.8, linewidth=1.2)

    ax.set_title("Cumulative Reward — Alibaba Replay", fontsize=13, fontweight="bold")
    ax.set_xlabel("Step")
    ax.set_ylabel("Cumulative Reward")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "cumulative_reward.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: cumulative_reward.png")


def save_additional_csvs(results_df: pd.DataFrame, all_trajectories: dict):
    """Generate specialized CSV files for IEEE publication reporting and step-level analysis."""
    # 1. IEEE Results Summary Table (Alibaba Replay)
    replay = results_df[results_df["scenario"] == "alibaba_replay"].copy()
    if not replay.empty:
        baseline_row = replay[replay["algorithm"] == "rule_based"]
        base_overheat = baseline_row["overheat_events"].values[0] if not baseline_row.empty else None
        base_energy = baseline_row["total_energy"].values[0] if not baseline_row.empty else None

        ieee_rows = []
        name_map = {
            "rule_based": "Rule-Based Baseline",
            "q_learning": "Q-Learning (Off-Policy TD)",
            "sarsa": "SARSA (On-Policy TD)",
            "monte_carlo": "Monte Carlo (First-Visit)",
            "value_iteration": "Value Iteration (Model-Based DP)",
        }

        for _, row in replay.iterrows():
            algo = row["algorithm"]
            overheat = row["overheat_events"]
            energy = row["total_energy"]
            
            overheat_red = ((base_overheat - overheat) / base_overheat * 100) if base_overheat else 0.0
            energy_diff = ((energy - base_energy) / base_energy * 100) if base_energy else 0.0

            ieee_rows.append({
                "Algorithm": name_map.get(algo, algo),
                "Cumulative Reward": round(row["cumulative_reward"], 2),
                "Peak Temp (°C)": round(row["max_temperature"], 2),
                "Mean Temp (°C)": round(row["avg_temperature"], 2),
                "Temp Variance": round(row["temperature_variance"], 2),
                "Time Above Safe (Steps)": int(row["time_above_safe"]),
                "Overheat Events": int(row["overheat_events"]),
                "Thermal Violation Reduction (%)": f"{overheat_red:+.1f}%" if algo != "rule_based" else "Baseline",
                "Total Cooling Energy": round(row["total_energy"], 1),
                "Energy Delta vs Baseline (%)": f"{energy_diff:+.1f}%" if algo != "rule_based" else "Baseline",
            })

        ieee_df = pd.DataFrame(ieee_rows)
        ieee_path = RESULTS_DIR / "ieee_results_table.csv"
        ieee_df.to_csv(ieee_path, index=False)
        print(f"  Saved: {ieee_path.name}")

    # 2. Scenario Benchmark Matrix
    pivot_cols = ["cumulative_reward", "total_energy", "max_temperature", "overheat_events"]
    scenario_matrix = results_df.pivot_table(index="algorithm", columns="scenario", values=pivot_cols)
    scenario_matrix_path = RESULTS_DIR / "scenario_benchmark_matrix.csv"
    scenario_matrix.to_csv(scenario_matrix_path)
    print(f"  Saved: {scenario_matrix_path.name}")

    # 3. Trajectory Step Log (Step-level telemetry)
    step_rows = []
    for algo, sc_dict in all_trajectories.items():
        for sc_name, metric in sc_dict.items():
            for t in range(len(metric.temperatures)):
                step_rows.append({
                    "step": t + 1,
                    "scenario": sc_name,
                    "algorithm": algo,
                    "mean_temperature": round(metric.temperatures[t], 4),
                    "max_temperature": round(metric.max_temperatures[t], 4),
                    "cooling_energy": round(metric.energies[t], 4),
                    "reward": round(metric.rewards[t], 4),
                    "overheated_cells": metric.overheated_counts[t],
                    "action": metric.actions[t],
                })

    if step_rows:
        trajectory_df = pd.DataFrame(step_rows)
        traj_path = RESULTS_DIR / "trajectories_step_log.csv"
        trajectory_df.to_csv(traj_path, index=False)
        print(f"  Saved: {traj_path.name}")


def print_experiment_summary(results_df: pd.DataFrame):
    """Print the experiment findings."""
    replay = results_df[results_df["scenario"] == "alibaba_replay"]
    if replay.empty:
        return

    print("\n" + "=" * 60)
    print("  EXPERIMENT SUMMARY — Alibaba Replay")
    print("=" * 60)

    best_reward = replay.loc[replay["cumulative_reward"].idxmax()]
    best_thermal = replay.loc[replay["max_temperature"].idxmin()]
    best_energy = replay.loc[replay["total_energy"].idxmin()]
    fewest_overheat = replay.loc[replay["overheat_events"].idxmin()]

    print(f"  Best overall reward   : {best_reward['algorithm']}"
          f"  ({best_reward['cumulative_reward']:.1f})")
    print(f"  Best thermal safety   : {best_thermal['algorithm']}"
          f"  (peak {best_thermal['max_temperature']:.1f}°C)")
    print(f"  Most energy efficient : {best_energy['algorithm']}"
          f"  ({best_energy['total_energy']:.0f} units)")
    print(f"  Fewest overheat events: {fewest_overheat['algorithm']}"
          f"  ({fewest_overheat['overheat_events']} events)")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Evaluate all agents")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--scenarios", nargs="*",
                        default=["alibaba_replay", "workload_burst",
                                 "moving_hotspot", "multi_hotspot"])
    parser.add_argument("--algorithms", nargs="*",
                        default=list(ALGORITHMS.keys()))
    args = parser.parse_args()

    set_global_seed(args.seed)
    ensure_dir(RESULTS_DIR)

    results = []
    all_trajectories = {}  # algo → {scenario → EpisodeMetrics}

    print(f"\n{'='*60}")
    print(f"  EVALUATION")
    print(f"  Algorithms: {args.algorithms}")
    print(f"  Scenarios:  {args.scenarios}")
    print(f"  Seed: {args.seed}")
    print(f"{'='*60}\n")

    for algo_name in args.algorithms:
        print(f"\n  Loading {algo_name} ...")
        agent = load_agent(algo_name)
        all_trajectories[algo_name] = {}

        for scenario_name in args.scenarios:
            print(f"    Evaluating on {scenario_name} ...")
            try:
                summary, metrics = evaluate_agent(
                    agent, algo_name, scenario_name,
                    seed=args.seed, max_steps=args.max_steps,
                )
                results.append(summary)
                all_trajectories[algo_name][scenario_name] = metrics
                print(f"      reward={summary['cumulative_reward']:.1f}  "
                      f"max_T={summary['max_temperature']:.1f}°C  "
                      f"energy={summary['total_energy']:.0f}  "
                      f"overheat={summary['overheat_events']}")
            except Exception as e:
                print(f"      [ERROR] {e}")

    if not results:
        print("\n  No results collected. Check that models exist in models/")
        return

    # Save results
    results_df = pd.DataFrame(results)
    csv_path = RESULTS_DIR / "comparison.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"\n  Results saved to {csv_path}")

    # Export additional structured CSVs for IEEE reporting & analysis
    save_additional_csvs(results_df, all_trajectories)

    # Generate plots
    print("\n  Generating plots ...")
    generate_plots(results_df, all_trajectories)

    # Print summary
    print_experiment_summary(results_df)


if __name__ == "__main__":
    main()

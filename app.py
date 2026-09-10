"""
Streamlit Dashboard — RL Data Center Cooling Optimization

Three tabs:
    1. Live Simulation  — interactive 6×6 thermal heatmap + controls
    2. Training          — launch training, view reward curves
    3. Comparison        — experiment results table + charts

CRITICAL: No RL algorithm logic or thermal physics in this file.
All intelligence is imported from environment/ and agents/.
"""

import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st

# ── Project imports ────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    ThermalConfig, RewardConfig, DatasetConfig, RLConfig,
    StateEncoderConfig, WorkloadMode, NUM_ACTIONS, ACTION_NAMES,
    default_config,
)
from environment.thermal_env import DataCenterEnv
from agents.q_learning import QLearningAgent
from agents.sarsa import SarsaAgent
from agents.monte_carlo import MonteCarloAgent
from agents.value_iteration import ValueIterationAgent
from baselines.rule_based import RuleBasedAgent
from experiments.scenarios import SCENARIOS

MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
DATA_PATH = PROJECT_ROOT / "data" / "mini_alibaba.csv"

ALGO_MAP = {
    "Rule-Based": ("rule_based", RuleBasedAgent),
    "Q-Learning": ("q_learning", QLearningAgent),
    "SARSA": ("sarsa", SarsaAgent),
    "Monte Carlo": ("monte_carlo", MonteCarloAgent),
    "Value Iteration": ("value_iteration", ValueIterationAgent),
}

# ─────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────

st.set_page_config(
    page_title="RL Data Center Cooling",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────
st.markdown("""
<style>
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 1.1rem;
        font-weight: 600;
    }
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 12px;
        padding: 1rem;
        margin: 0.5rem 0;
        border: 1px solid #0f3460;
    }
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #0f0f23 0%, #1a1a3e 100%);
        border-radius: 10px;
        padding: 12px 16px;
        border: 1px solid #2a2a5a;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────────────

@st.cache_data
def check_dataset():
    """Check if mini_alibaba.csv exists."""
    return DATA_PATH.exists()


def load_agent_for_sim(algo_display_name: str):
    """Load a trained agent."""
    algo_key, cls = ALGO_MAP[algo_display_name]
    model_path = MODELS_DIR / f"{algo_key}.json"

    if algo_display_name == "Rule-Based":
        return cls(n_actions=NUM_ACTIONS)

    agent = cls(n_actions=NUM_ACTIONS, cfg=RLConfig())
    if model_path.exists():
        agent.load(model_path)
        agent.epsilon = 0.0  # Greedy during simulation
    return agent


def create_heatmap(data: np.ndarray, title: str, colorscale: str = "RdYlBu_r",
                   zmin=None, zmax=None, show_values: bool = True):
    """Create a Plotly heatmap figure."""
    text = [[f"{data[r][c]:.1f}" for c in range(data.shape[1])]
            for r in range(data.shape[0])] if show_values else None

    fig = go.Figure(data=go.Heatmap(
        z=data,
        text=text,
        texttemplate="%{text}",
        textfont={"size": 11, "color": "white"},
        colorscale=colorscale,
        zmin=zmin,
        zmax=zmax,
        hoverongaps=False,
        colorbar=dict(thickness=15, len=0.9),
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),
        height=320,
        margin=dict(l=30, r=30, t=45, b=30),
        xaxis=dict(title="Column", dtick=1),
        yaxis=dict(title="Row", dtick=1, autorange="reversed"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# ─────────────────────────────────────────────────────
# Session state initialisation
# ─────────────────────────────────────────────────────

def init_session():
    defaults = {
        "sim_running": False,
        "sim_step": 0,
        "sim_env": None,
        "sim_agent": None,
        "sim_history": [],
        "sim_algo": "Rule-Based",
        "sim_scenario": "alibaba_replay",
        "sim_initialized": False,
        "training_log": [],
        "training_rewards": [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_session()


# ─────────────────────────────────────────────────────
# TAB 1: LIVE SIMULATION
# ─────────────────────────────────────────────────────

def render_live_simulation():
    st.header("Live Data Center Simulation")

    if not check_dataset():
        st.error("Dataset not found! Run: `python -m data.prepare_dataset --synthetic`")
        return

    # ── Sidebar controls ──
    with st.sidebar:
        st.subheader("Simulation Controls")

        algo_choice = st.selectbox(
            "Algorithm",
            list(ALGO_MAP.keys()),
            key="algo_select",
        )

        scenario_choice = st.selectbox(
            "Scenario",
            list(SCENARIOS.keys()),
            format_func=lambda x: SCENARIOS[x]["name"],
            key="scenario_select",
        )

        st.divider()

        col1, col2 = st.columns(2)
        with col1:
            start_btn = st.button("Start", use_container_width=True)
            reset_btn = st.button("Reset", use_container_width=True)
        with col2:
            pause_btn = st.button("Pause", use_container_width=True)
            step_btn = st.button("Step", use_container_width=True)

        sim_speed = st.slider("Replay Step", 0, 499, st.session_state.sim_step,
                               key="step_slider")

    # ── Initialise or re-init environment ──
    need_reinit = (
        not st.session_state.sim_initialized or
        algo_choice != st.session_state.sim_algo or
        scenario_choice != st.session_state.sim_scenario
    )

    if need_reinit or reset_btn:
        mode = SCENARIOS[scenario_choice]["mode"]
        env = DataCenterEnv(
            workload_mode=mode,
            dataset_path=DATA_PATH,
            seed=42,
        )
        env.reset(seed=42)
        agent = load_agent_for_sim(algo_choice)

        st.session_state.sim_env = env
        st.session_state.sim_agent = agent
        st.session_state.sim_algo = algo_choice
        st.session_state.sim_scenario = scenario_choice
        st.session_state.sim_step = 0
        st.session_state.sim_history = []
        st.session_state.sim_initialized = True
        st.session_state.sim_running = False

    env = st.session_state.sim_env
    agent = st.session_state.sim_agent

    if env is None:
        st.warning("Click Reset to initialise the simulation.")
        return

    # ── Handle controls ──
    if start_btn:
        st.session_state.sim_running = True
    if pause_btn:
        st.session_state.sim_running = False

    # Jump to slider position
    if sim_speed != st.session_state.sim_step:
        # Re-simulate up to that step
        env.reset(seed=42)
        for s in range(sim_speed):
            state = env.encoder.encode(env.thermal.temperatures, env._workload)
            action = agent.select_greedy(state) if hasattr(agent, 'select_greedy') else agent.select_action(state)
            env.step(action)
        st.session_state.sim_step = sim_speed

    if step_btn or st.session_state.sim_running:
        if st.session_state.sim_step < 499:
            state = env.encoder.encode(env.thermal.temperatures, env._workload)
            action = agent.select_greedy(state) if hasattr(agent, 'select_greedy') else agent.select_action(state)
            _, reward, _, truncated, info = env.step(action)
            st.session_state.sim_step += 1
            st.session_state.sim_history.append({
                "step": st.session_state.sim_step,
                "reward": reward,
                "max_temp": info["max_temperature"],
                "mean_temp": info["mean_temperature"],
                "energy": info["energy_this_step"],
                "action": action,
                "action_name": info["action_name"],
            })

    # ── Get current state for display ──
    info = env._build_info()
    temps = info["temperatures"]
    workload = info["workload"]
    cooling = info["cooling_allocation"]

    # ── Metrics row ──
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Max Temp", f"{info['max_temperature']:.1f}°C")
    m2.metric("Mean Temp", f"{info['mean_temperature']:.1f}°C")
    m3.metric("Cooling Power", f"{np.sum(cooling):.0f}")
    m4.metric("Energy Used", f"{info['energy_used']:.0f}")
    m5.metric("Overheated", f"{info['overheated_cells']}")

    # ── Heatmaps ──
    h1, h2, h3 = st.columns(3)

    with h1:
        fig_temp = create_heatmap(
            temps, "Temperature (°C)",
            colorscale="RdYlBu_r",
            zmin=15, zmax=60,
        )
        st.plotly_chart(fig_temp, use_container_width=True)

    with h2:
        fig_wl = create_heatmap(
            workload * 100, "Workload (%)",
            colorscale="YlOrRd",
            zmin=0, zmax=100,
        )
        st.plotly_chart(fig_wl, use_container_width=True)

    with h3:
        fig_cool = create_heatmap(
            cooling, "Cooling Allocation",
            colorscale="Blues",
            zmin=0, zmax=10,
        )
        st.plotly_chart(fig_cool, use_container_width=True)

    # ── Current action display ──
    st.markdown("---")
    ca1, ca2, ca3 = st.columns([2, 3, 2])
    with ca1:
        hr, hc = info["hotspot_location"]
        st.info(f"**Hotspot**: Cell ({hr},{hc}) at {info['hotspot_temperature']:.1f}°C")
    with ca2:
        if st.session_state.sim_history:
            last = st.session_state.sim_history[-1]
            st.success(f"**RL Action**: {last['action_name']} (reward: {last['reward']:.2f})")
        else:
            st.info("**RL Action**: Waiting for first step")
    with ca3:
        state_tuple = env.encoder.encode(temps, workload)
        labels = env.encoder.decode_state_labels(state_tuple)
        st.info(f"**State**: Temps={labels['zone_temps']}")

    # ── Causal chain display ──
    if st.session_state.sim_history:
        st.markdown("### Simulation History")
        hist_df = pd.DataFrame(st.session_state.sim_history[-100:])
        if not hist_df.empty:
            c1, c2 = st.columns(2)
            with c1:
                fig = px.line(hist_df, x="step", y=["max_temp", "mean_temp"],
                              title="Temperature Over Time",
                              labels={"value": "°C", "step": "Step"})
                fig.update_layout(height=300, margin=dict(l=30, r=30, t=40, b=30))
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                fig = px.line(hist_df, x="step", y="reward",
                              title="Reward Over Time")
                fig.update_layout(height=300, margin=dict(l=30, r=30, t=40, b=30))
                st.plotly_chart(fig, use_container_width=True)

    # Auto-advance
    if st.session_state.sim_running and st.session_state.sim_step < 499:
        time.sleep(0.05)
        st.rerun()


# ─────────────────────────────────────────────────────
# TAB 2: TRAINING
# ─────────────────────────────────────────────────────

def render_training():
    st.header("Training")

    if not check_dataset():
        st.error("Dataset not found! Run: `python -m data.prepare_dataset --synthetic`")
        return

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Configuration")
        algo = st.selectbox("Algorithm", [
            "q_learning", "sarsa", "monte_carlo", "value_iteration", "rule_based"
        ], key="train_algo")
        episodes = st.number_input("Episodes", 50, 2000, 300, step=50, key="train_episodes")
        seed = st.number_input("Seed", 0, 9999, 42, key="train_seed")

        st.markdown("**Hyperparameters**")
        alpha = st.slider("Learning Rate (α)", 0.01, 0.5, 0.1, 0.01, key="train_alpha")
        gamma = st.slider("Discount (γ)", 0.8, 0.999, 0.95, 0.005, key="train_gamma")
        eps_start = st.slider("ε start", 0.1, 1.0, 1.0, 0.1, key="train_eps_start")
        eps_decay = st.slider("ε decay", 0.98, 0.999, 0.995, 0.001, key="train_eps_decay")

        train_btn = st.button("Start Training", type="primary", use_container_width=True)

        # Check for existing models
        st.divider()
        st.subheader("Saved Models")
        for name, (key, _) in ALGO_MAP.items():
            path = MODELS_DIR / f"{key}.json"
            if path.exists():
                size = path.stat().st_size / 1024
                st.success(f"{name} ({size:.0f} KB)")
            else:
                st.warning(f"{name} — not trained")

    with col2:
        st.subheader("Training Output")

        if train_btn:
            cmd = [
                sys.executable, "-m", "experiments.train",
                "--algorithm", algo,
                "--episodes", str(episodes),
                "--seed", str(seed),
                "--alpha", str(alpha),
                "--gamma", str(gamma),
                "--epsilon-start", str(eps_start),
                "--epsilon-decay", str(eps_decay),
            ]

            with st.spinner(f"Training {algo} for {episodes} episodes..."):
                result = subprocess.run(
                    cmd,
                    cwd=str(PROJECT_ROOT),
                    capture_output=True,
                    text=True,
                    timeout=600,
                )

            if result.returncode == 0:
                st.success("Training complete!")
                st.code(result.stdout, language="text")
            else:
                st.error("Training failed!")
                st.code(result.stderr or result.stdout, language="text")

        # Show training reward curve if we have a results file
        reward_files = list(RESULTS_DIR.glob("training_rewards_*.csv"))
        if reward_files:
            st.markdown("### Training Curves")
            for f in reward_files:
                df = pd.read_csv(f)
                fig = px.line(df, title=f.stem.replace("_", " ").title())
                st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────────────
# TAB 3: COMPARISON
# ─────────────────────────────────────────────────────

def render_comparison():
    st.header("Algorithm Comparison")

    csv_path = RESULTS_DIR / "comparison.csv"

    if not csv_path.exists():
        st.warning("No evaluation results found. Run evaluation first:")
        st.code("python -m experiments.evaluate", language="bash")

        if st.button("Run Evaluation Now", type="primary"):
            with st.spinner("Running evaluation..."):
                result = subprocess.run(
                    [sys.executable, "-m", "experiments.evaluate", "--seed", "42"],
                    cwd=str(PROJECT_ROOT),
                    capture_output=True,
                    text=True,
                    timeout=600,
                )
            if result.returncode == 0:
                st.success("Evaluation complete!")
                st.code(result.stdout, language="text")
                st.rerun()
            else:
                st.error("Evaluation failed!")
                st.code(result.stderr or result.stdout, language="text")
        return

    # Load results
    df = pd.read_csv(csv_path)
    replay_df = df[df["scenario"] == "alibaba_replay"].copy()

    # ── Results table ──
    st.subheader("Results Table — Alibaba Replay")
    if not replay_df.empty:
        display_cols = [
            "algorithm", "avg_temperature", "max_temperature",
            "total_energy", "overheat_events", "time_above_safe",
            "cumulative_reward"
        ]
        display_df = replay_df[
            [c for c in display_cols if c in replay_df.columns]
        ].copy()
        display_df.columns = [c.replace("_", " ").title() for c in display_df.columns]

        # Highlight best values
        st.dataframe(
            display_df.style.format(precision=1),
            use_container_width=True,
            hide_index=True,
        )

    # ── Comparison charts ──
    st.subheader("Visual Comparison")

    if not replay_df.empty:
        colors_map = {
            "rule_based": "#95a5a6",
            "q_learning": "#3498db",
            "sarsa": "#2ecc71",
            "monte_carlo": "#e74c3c",
            "value_iteration": "#f39c12",
        }
        chart_colors = [colors_map.get(a, "#666") for a in replay_df["algorithm"]]

        c1, c2 = st.columns(2)

        with c1:
            fig = go.Figure(data=[
                go.Bar(
                    x=replay_df["algorithm"],
                    y=replay_df["cumulative_reward"],
                    marker_color=chart_colors,
                    text=replay_df["cumulative_reward"].round(1),
                    textposition="outside",
                )
            ])
            fig.update_layout(
                title="Cumulative Reward (↑ better)",
                height=400,
                yaxis_title="Reward",
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            fig = go.Figure(data=[
                go.Bar(
                    x=replay_df["algorithm"],
                    y=replay_df["total_energy"],
                    marker_color=chart_colors,
                    text=replay_df["total_energy"].round(0),
                    textposition="outside",
                )
            ])
            fig.update_layout(
                title="Total Cooling Energy (↓ better)",
                height=400,
                yaxis_title="Energy",
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)

        c3, c4 = st.columns(2)

        with c3:
            fig = go.Figure(data=[
                go.Bar(
                    x=replay_df["algorithm"],
                    y=replay_df["max_temperature"],
                    marker_color=chart_colors,
                    text=replay_df["max_temperature"].round(1),
                    textposition="outside",
                )
            ])
            fig.update_layout(
                title="Peak Temperature (↓ better)",
                height=400,
                yaxis_title="°C",
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)

        with c4:
            fig = go.Figure(data=[
                go.Bar(
                    x=replay_df["algorithm"],
                    y=replay_df["overheat_events"],
                    marker_color=chart_colors,
                    text=replay_df["overheat_events"].astype(int),
                    textposition="outside",
                )
            ])
            fig.update_layout(
                title="Overheating Events (↓ better)",
                height=400,
                yaxis_title="Events",
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)

    # ── All scenarios table ──
    if len(df) > len(replay_df):
        st.subheader("All Scenarios")
        st.dataframe(
            df.style.format(precision=1),
            use_container_width=True,
            hide_index=True,
        )

    # ── Experiment summary ──
    if not replay_df.empty:
        st.subheader("Experiment Summary")
        best_reward = replay_df.loc[replay_df["cumulative_reward"].idxmax()]
        best_thermal = replay_df.loc[replay_df["max_temperature"].idxmin()]
        best_energy = replay_df.loc[replay_df["total_energy"].idxmin()]

        s1, s2, s3 = st.columns(3)
        s1.success(f"**Best Reward**: {best_reward['algorithm']}\n\n"
                   f"Score: {best_reward['cumulative_reward']:.1f}")
        s2.info(f"**Best Thermal Safety**: {best_thermal['algorithm']}\n\n"
                f"Peak: {best_thermal['max_temperature']:.1f}°C")
        s3.warning(f"**Most Efficient**: {best_energy['algorithm']}\n\n"
                   f"Energy: {best_energy['total_energy']:.0f}")

    # ── Show saved plots ──
    plot_files = list(RESULTS_DIR.glob("*.png"))
    if plot_files:
        st.subheader("Generated Plots")
        for pf in sorted(plot_files):
            st.image(str(pf), caption=pf.stem.replace("_", " ").title(),
                     use_container_width=True)


# ─────────────────────────────────────────────────────
# Main layout
# ─────────────────────────────────────────────────────

st.title("RL Data Center Cooling Optimization")
st.caption("Real workload telemetry + physics-inspired thermal simulation + classical RL")

tab1, tab2, tab3 = st.tabs([
    "Live Simulation",
    "Training",
    "Comparison",
])

with tab1:
    render_live_simulation()

with tab2:
    render_training()

with tab3:
    render_comparison()

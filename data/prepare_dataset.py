"""
Dataset preparation for the RL Data-Center Cooling project.

Usage:
    # If you have the real Alibaba server_usage.csv:
    python -m data.prepare_dataset

    # To generate a synthetic mini-dataset for demonstration:
    python -m data.prepare_dataset --synthetic

    # Custom sizing:
    python -m data.prepare_dataset --synthetic --machines 36 --timesteps 1000 --seed 42
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ── project root on path ──
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from config import DatasetConfig


# ────────────────────────────────────────────────────────
# Real Alibaba ingestion
# ────────────────────────────────────────────────────────

def _discover_alibaba_file(cfg: DatasetConfig) -> Path | None:
    """Look for server_usage.csv in data/raw/."""
    candidates = [
        Path(cfg.raw_dir) / "server_usage.csv",
        Path(cfg.raw_dir) / "server_usage.csv.gz",
        PROJECT_ROOT / cfg.raw_dir / "server_usage.csv",
        PROJECT_ROOT / cfg.raw_dir / "server_usage.csv.gz",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def ingest_alibaba(cfg: DatasetConfig) -> pd.DataFrame:
    """
    Read the real Alibaba server_usage.csv, validate schema,
    select a subset of machines & timesteps, and return a clean DataFrame.
    """
    path = _discover_alibaba_file(cfg)
    if path is None:
        print("=" * 60)
        print("ERROR: Alibaba dataset not found.")
        print()
        print("Place the official server_usage.csv (or .csv.gz) under:")
        print(f"    {PROJECT_ROOT / cfg.raw_dir}/")
        print()
        print("To obtain it:")
        print("  https://github.com/alibaba/clusterdata/tree/master/cluster-trace-v2017")
        print()
        print("Or generate a synthetic dataset for demonstration:")
        print("  python -m data.prepare_dataset --synthetic")
        print("=" * 60)
        sys.exit(1)

    print(f"[DATA] Reading Alibaba trace from {path} ...")

    # Read in chunks to avoid loading full 33 GB
    expected_cols = [
        cfg.col_timestamp, cfg.col_machine_id,
        cfg.col_cpu_util, cfg.col_mem_util, cfg.col_disk_util,
        cfg.col_load1, cfg.col_load5, cfg.col_load15,
    ]

    chunks = []
    machine_ids_seen: set = set()
    rows_collected = 0
    target_rows = cfg.machine_count * cfg.max_timesteps

    for chunk in pd.read_csv(path, chunksize=50_000, dtype=str,
                             on_bad_lines="skip"):
        # Normalise column names
        chunk.columns = [c.strip().lower() for c in chunk.columns]

        # Handle the Alibaba schema where columns may be unnamed integers
        if chunk.columns.tolist()[:2] == ["0", "1"]:
            chunk.columns = expected_cols[: len(chunk.columns)]

        # Validate required columns exist
        missing = [c for c in expected_cols if c not in chunk.columns]
        if missing:
            print(f"[WARN] Missing columns {missing} — skipping chunk")
            continue

        chunk = chunk[expected_cols]

        # Convert numerics
        for col in expected_cols:
            chunk[col] = pd.to_numeric(chunk[col], errors="coerce")

        chunk.dropna(subset=[cfg.col_machine_id, cfg.col_cpu_util], inplace=True)

        # Collect machine IDs
        if len(machine_ids_seen) < cfg.machine_count:
            new_ids = set(chunk[cfg.col_machine_id].unique())
            machine_ids_seen.update(new_ids)
            if len(machine_ids_seen) > cfg.machine_count:
                # Trim to exact count
                machine_ids_seen = set(
                    sorted(machine_ids_seen)[: cfg.machine_count]
                )

        # Filter to selected machines
        chunk = chunk[chunk[cfg.col_machine_id].isin(machine_ids_seen)]
        chunks.append(chunk)
        rows_collected += len(chunk)

        if rows_collected >= target_rows:
            break

    if not chunks:
        print("[ERROR] No valid data found in the Alibaba trace.")
        sys.exit(1)

    df = pd.concat(chunks, ignore_index=True)
    return _postprocess(df, cfg)


# ────────────────────────────────────────────────────────
# Synthetic data generation
# ────────────────────────────────────────────────────────

def generate_synthetic(cfg: DatasetConfig) -> pd.DataFrame:
    """
    Generate a synthetic mini_alibaba.csv that mimics the Alibaba schema.

    Uses correlated random walks to produce realistic-looking server
    utilisation patterns: diurnal cycles, workload bursts, idle periods,
    and per-machine personality (some machines run hotter than others).
    """
    rng = np.random.default_rng(cfg.random_seed)
    n_machines = cfg.machine_count
    n_steps = cfg.max_timesteps

    machine_ids = list(range(1, n_machines + 1))

    # Each machine has a "personality" — base utilisation level
    base_cpu = rng.uniform(15, 50, size=n_machines)
    base_mem = rng.uniform(20, 60, size=n_machines)

    records = []
    # Random-walk state per machine
    cpu_state = base_cpu.copy()
    mem_state = base_mem.copy()

    for t in range(n_steps):
        timestamp = t * 60  # 60-second intervals

        # Diurnal modulation (peaks around step 300 and 800)
        diurnal = 0.15 * np.sin(2 * np.pi * t / n_steps) + \
                  0.10 * np.sin(4 * np.pi * t / n_steps)

        # Occasional workload bursts (affect random subset)
        burst = np.zeros(n_machines)
        if rng.random() < 0.05:  # 5% chance per step
            burst_machines = rng.choice(n_machines,
                                        size=rng.integers(1, max(2, n_machines // 4)),
                                        replace=False)
            burst[burst_machines] = rng.uniform(15, 40, size=len(burst_machines))

        # Random walk
        cpu_delta = rng.normal(0, 2.0, size=n_machines) + diurnal * base_cpu + burst
        mem_delta = rng.normal(0, 1.0, size=n_machines) + diurnal * base_mem * 0.5

        # Mean-revert toward personality
        cpu_state += cpu_delta
        cpu_state += 0.05 * (base_cpu - cpu_state)
        mem_state += mem_delta
        mem_state += 0.05 * (base_mem - mem_state)

        # Clamp
        cpu_util = np.clip(cpu_state, 0, 100)
        mem_util = np.clip(mem_state, 0, 100)
        disk_util = np.clip(rng.uniform(2, 20, size=n_machines) +
                            0.1 * cpu_util, 0, 100)

        # Load averages correlated with CPU
        load1 = np.clip(cpu_util / 100 * rng.uniform(0.8, 1.5, size=n_machines), 0, 5)
        load5 = np.clip(load1 * rng.uniform(0.85, 1.0, size=n_machines), 0, 5)
        load15 = np.clip(load5 * rng.uniform(0.85, 1.0, size=n_machines), 0, 5)

        for i in range(n_machines):
            records.append({
                "timestamp": timestamp,
                "machine_id": machine_ids[i],
                "cpu_util": round(float(cpu_util[i]), 1),
                "mem_util": round(float(mem_util[i]), 1),
                "disk_util": round(float(disk_util[i]), 1),
                "load1": round(float(load1[i]), 2),
                "load5": round(float(load5[i]), 2),
                "load15": round(float(load15[i]), 2),
            })

    df = pd.DataFrame(records)
    return df


# ────────────────────────────────────────────────────────
# Post-processing
# ────────────────────────────────────────────────────────

def _postprocess(df: pd.DataFrame, cfg: DatasetConfig) -> pd.DataFrame:
    """Normalise timestamps, handle NaN, sort, trim."""
    # Sort by machine + time
    df = df.sort_values([cfg.col_machine_id, cfg.col_timestamp]).reset_index(drop=True)

    # Convert timestamps to ordered step indices (0, 1, 2, ...)
    unique_ts = sorted(df[cfg.col_timestamp].unique())
    ts_map = {ts: i for i, ts in enumerate(unique_ts)}
    df["timestep"] = df[cfg.col_timestamp].map(ts_map)

    # Trim to max timesteps
    max_step = min(cfg.max_timesteps - 1, df["timestep"].max())
    df = df[df["timestep"] <= max_step].copy()

    # Ensure exactly cfg.machine_count machines
    machines = sorted(df[cfg.col_machine_id].unique())
    if len(machines) > cfg.machine_count:
        machines = machines[: cfg.machine_count]
        df = df[df[cfg.col_machine_id].isin(machines)]

    # Fill remaining NaN with column medians
    numeric_cols = [cfg.col_cpu_util, cfg.col_mem_util, cfg.col_disk_util,
                    cfg.col_load1, cfg.col_load5, cfg.col_load15]
    for col in numeric_cols:
        if col in df.columns:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val if pd.notna(median_val) else 0.0)

    # Clamp utilisation to [0, 100]
    for col in [cfg.col_cpu_util, cfg.col_mem_util, cfg.col_disk_util]:
        if col in df.columns:
            df[col] = df[col].clip(0, 100)

    return df


# ────────────────────────────────────────────────────────
# Summary
# ────────────────────────────────────────────────────────

def print_summary(df: pd.DataFrame, cfg: DatasetConfig, synthetic: bool = False):
    """Print dataset statistics."""
    machines = sorted(df[cfg.col_machine_id].unique())
    timesteps = sorted(df["timestep"].unique()) if "timestep" in df.columns else []

    print()
    print("=" * 60)
    print("  DATASET SUMMARY")
    print("=" * 60)
    if synthetic:
        print("  ⚠  Data source : SYNTHETIC (not real Alibaba)")
    else:
        print("  ✓  Data source : Alibaba Cluster Trace 2017")
    print(f"  Machines       : {len(machines)}")
    print(f"  Timesteps      : {len(timesteps)}")
    print(f"  Total rows     : {len(df)}")
    ts_col = cfg.col_timestamp
    if ts_col in df.columns:
        print(f"  Time span      : {df[ts_col].min()} → {df[ts_col].max()}")
    print(f"  Missing values : {df.isna().sum().sum()}")
    print()
    print("  Utilisation statistics:")
    for col in [cfg.col_cpu_util, cfg.col_mem_util]:
        if col in df.columns:
            print(f"    {col:12s}  mean={df[col].mean():.1f}  "
                  f"std={df[col].std():.1f}  "
                  f"min={df[col].min():.1f}  max={df[col].max():.1f}")
    print()
    print(f"  Machine IDs    : {machines[:5]} ... {machines[-3:]}" if len(machines) > 8
          else f"  Machine IDs    : {machines}")
    print("=" * 60)


# ────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Prepare the mini_alibaba.csv dataset")
    parser.add_argument("--synthetic", action="store_true",
                        help="Generate synthetic data when real dataset is unavailable")
    parser.add_argument("--machines", type=int, default=36,
                        help="Number of machines to select (default: 36)")
    parser.add_argument("--timesteps", type=int, default=1000,
                        help="Maximum timesteps (default: 1000)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path (default: data/mini_alibaba.csv)")
    args = parser.parse_args()

    cfg = DatasetConfig(
        machine_count=args.machines,
        max_timesteps=args.timesteps,
        random_seed=args.seed,
    )

    output_path = Path(args.output) if args.output else PROJECT_ROOT / cfg.mini_csv

    if args.synthetic:
        print("[DATA] Generating synthetic dataset ...")
        df = generate_synthetic(cfg)
        synthetic = True
    else:
        # Try real Alibaba first
        alibaba_path = _discover_alibaba_file(cfg)
        if alibaba_path is None:
            print("[DATA] Alibaba dataset not found in data/raw/")
            print("[DATA] Generating synthetic fallback ...")
            print("[DATA] (Use --synthetic flag to suppress this message)")
            df = generate_synthetic(cfg)
            synthetic = True
        else:
            df = ingest_alibaba(cfg)
            synthetic = False

    # Post-process synthetic data too
    if "timestep" not in df.columns:
        df = _postprocess(df, cfg)

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[DATA] Saved to {output_path}")

    print_summary(df, cfg, synthetic=synthetic)

    return df


if __name__ == "__main__":
    main()

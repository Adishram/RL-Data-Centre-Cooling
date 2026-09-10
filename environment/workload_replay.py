"""
Workload manager: feeds real or synthetic utilisation data into the thermal grid.

Supports multiple modes:
    DATASET_REPLAY  — replay mini_alibaba.csv timestep by timestep
    STATIC_HOTSPOT  — fixed hot region for debugging
    MOVING_HOTSPOT  — hot region migrates across the grid
    MULTI_HOTSPOT   — two or more simultaneous hot regions
    WORKLOAD_BURST  — sudden spikes at random intervals
"""

from pathlib import Path

import numpy as np
import pandas as pd

from config import DatasetConfig, ThermalConfig, WorkloadMode


class WorkloadManager:
    """Provides a (rows, cols) workload matrix for each simulation timestep."""

    def __init__(
        self,
        mode: WorkloadMode = WorkloadMode.DATASET_REPLAY,
        thermal_cfg: ThermalConfig | None = None,
        dataset_cfg: DatasetConfig | None = None,
        dataset_path: str | Path | None = None,
        rng: np.random.Generator | None = None,
    ):
        self.mode = mode
        self.t_cfg = thermal_cfg or ThermalConfig()
        self.d_cfg = dataset_cfg or DatasetConfig()
        self.rows = self.t_cfg.grid_rows
        self.cols = self.t_cfg.grid_cols
        self.rng = rng or np.random.default_rng(42)

        # Dataset replay state
        self._df: pd.DataFrame | None = None
        self._machine_to_cell: dict[int, tuple[int, int]] = {}
        self._timesteps: list[int] = []
        self._current_step: int = 0

        if mode == WorkloadMode.DATASET_REPLAY:
            self._load_dataset(dataset_path)

    # ── public API ──────────────────────────────────

    def reset(self) -> np.ndarray:
        self._current_step = 0
        return self.get_workload(0)

    def get_workload(self, step: int | None = None) -> np.ndarray:
        """
        Return a (rows, cols) workload matrix normalised to [0, 1].
        """
        if step is not None:
            self._current_step = step

        if self.mode == WorkloadMode.DATASET_REPLAY:
            return self._dataset_workload()
        elif self.mode == WorkloadMode.STATIC_HOTSPOT:
            return self._static_hotspot()
        elif self.mode == WorkloadMode.MOVING_HOTSPOT:
            return self._moving_hotspot()
        elif self.mode == WorkloadMode.MULTI_HOTSPOT:
            return self._multi_hotspot()
        elif self.mode == WorkloadMode.WORKLOAD_BURST:
            return self._workload_burst()
        else:
            return self._static_hotspot()

    def advance(self) -> np.ndarray:
        """Move to the next timestep and return workload."""
        self._current_step += 1
        return self.get_workload()

    @property
    def num_timesteps(self) -> int:
        if self._timesteps:
            return len(self._timesteps)
        return self.t_cfg.max_steps_per_episode

    @property
    def machine_mapping(self) -> dict:
        return dict(self._machine_to_cell)

    # ── dataset replay ──────────────────────────────

    def _load_dataset(self, path: str | Path | None):
        """Load mini_alibaba.csv and build machine → grid-cell mapping."""
        if path is None:
            path = Path(self.d_cfg.mini_csv)
        path = Path(path)

        if not path.exists():
            # Try relative to project root
            alt = Path(__file__).resolve().parent.parent / path
            if alt.exists():
                path = alt

        if not path.exists():
            raise FileNotFoundError(
                f"Dataset not found at {path}. "
                f"Run `python -m data.prepare_dataset --synthetic` first."
            )

        self._df = pd.read_csv(path)

        # Validate required columns
        required = [self.d_cfg.col_machine_id, self.d_cfg.col_cpu_util]
        missing = [c for c in required if c not in self._df.columns]
        if missing:
            raise ValueError(f"Dataset missing columns: {missing}")

        # Build machine → cell mapping (deterministic)
        machines = sorted(self._df[self.d_cfg.col_machine_id].unique())
        n_cells = self.rows * self.cols
        if len(machines) > n_cells:
            machines = machines[:n_cells]

        self._machine_to_cell = {}
        for idx, mid in enumerate(machines):
            r, c = divmod(idx, self.cols)
            self._machine_to_cell[mid] = (r, c)

        # Build timestep index
        if "timestep" in self._df.columns:
            self._timesteps = sorted(self._df["timestep"].unique())
        elif self.d_cfg.col_timestamp in self._df.columns:
            ts_vals = sorted(self._df[self.d_cfg.col_timestamp].unique())
            ts_map = {v: i for i, v in enumerate(ts_vals)}
            self._df["timestep"] = self._df[self.d_cfg.col_timestamp].map(ts_map)
            self._timesteps = sorted(self._df["timestep"].unique())

    def _dataset_workload(self) -> np.ndarray:
        """Build workload grid from the current dataset timestep."""
        workload = np.full((self.rows, self.cols), 0.1, dtype=np.float64)

        if self._df is None or not self._timesteps:
            return workload

        # Wrap around if we exceed available timesteps
        step_idx = self._current_step % len(self._timesteps)
        ts = self._timesteps[step_idx]

        rows = self._df[self._df["timestep"] == ts]

        for _, row in rows.iterrows():
            mid = row[self.d_cfg.col_machine_id]
            if mid not in self._machine_to_cell:
                continue
            r, c = self._machine_to_cell[mid]

            cpu = float(row.get(self.d_cfg.col_cpu_util, 0))
            cpu_norm = np.clip(cpu / 100.0, 0, 1)

            # Optional memory contribution (smaller weight)
            mem = float(row.get(self.d_cfg.col_mem_util, 0))
            mem_norm = np.clip(mem / 100.0, 0, 1)

            workload[r, c] = 0.8 * cpu_norm + 0.2 * mem_norm

        return np.clip(workload, 0, 1)

    # ── synthetic modes ─────────────────────────────

    def _static_hotspot(self) -> np.ndarray:
        workload = np.full((self.rows, self.cols), 0.15)
        # Fixed hotspot in top-right quadrant
        hr, hc = self.rows // 4, 3 * self.cols // 4
        workload[max(0, hr - 1):hr + 2, max(0, hc - 1):hc + 2] = 0.85
        return workload

    def _moving_hotspot(self) -> np.ndarray:
        workload = np.full((self.rows, self.cols), 0.15)
        period = max(1, self.rows + self.cols - 2)
        pos = self._current_step % (2 * period)
        if pos >= period:
            pos = 2 * period - pos - 1

        # Convert linear position to 2D (snake along rows)
        r = pos // self.cols
        c = pos % self.cols
        r = min(r, self.rows - 1)
        c = min(c, self.cols - 1)

        for dr in range(-1, 2):
            for dc in range(-1, 2):
                rr, cc = r + dr, c + dc
                if 0 <= rr < self.rows and 0 <= cc < self.cols:
                    workload[rr, cc] = 0.8 - 0.15 * (abs(dr) + abs(dc))
        return np.clip(workload, 0, 1)

    def _multi_hotspot(self) -> np.ndarray:
        workload = np.full((self.rows, self.cols), 0.12)
        # Two hotspots in opposite corners
        workload[0:2, 0:2] = 0.8
        workload[-2:, -2:] = 0.75
        # Occasional third hotspot
        if self._current_step % 20 < 10:
            workload[self.rows // 2, self.cols // 2] = 0.9
        return workload

    def _workload_burst(self) -> np.ndarray:
        workload = np.full((self.rows, self.cols), 0.15)
        # Burst every 30 steps lasting 10 steps
        cycle = self._current_step % 50
        if 20 <= cycle < 35:
            # Random burst region (deterministic from step)
            rng = np.random.default_rng(self._current_step)
            burst_r = rng.integers(0, self.rows - 2)
            burst_c = rng.integers(0, self.cols - 2)
            workload[burst_r:burst_r + 3, burst_c:burst_c + 3] = rng.uniform(0.6, 0.95,
                                                                                size=(3, 3))
        return np.clip(workload, 0, 1)

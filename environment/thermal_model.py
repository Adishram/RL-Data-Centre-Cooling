"""
Physics-inspired thermal simulation for a data-center floor grid.

The model is intentionally simple (no CFD / neural ODE), but captures
the key thermal dynamics needed for RL experimentation:

    T_next = T_current
             + workload_heat          (CPU/mem utilisation → heat)
             + neighbor_diffusion     (heat spreads to neighbours)
             + ambient_pull           (environment pulls toward ambient)
             - cooling_effect         (CRAC/cooling action)
             + noise                  (small stochastic perturbation)

All coefficients come from config.ThermalConfig.
"""

import numpy as np
from config import ThermalConfig


class ThermalModel:
    """Stateful 2-D thermal grid simulation."""

    def __init__(self, cfg: ThermalConfig | None = None):
        self.cfg = cfg or ThermalConfig()
        self.rows = self.cfg.grid_rows
        self.cols = self.cfg.grid_cols
        self.temperatures: np.ndarray = np.full(
            (self.rows, self.cols), self.cfg.initial_temperature, dtype=np.float64
        )

    # ── public API ──────────────────────────────────

    def reset(self, rng: np.random.Generator | None = None) -> np.ndarray:
        """Reset temperatures to initial + tiny noise."""
        noise = 0.0
        if rng is not None:
            noise = rng.normal(0, 0.5, size=(self.rows, self.cols))
        self.temperatures = np.full(
            (self.rows, self.cols), self.cfg.initial_temperature, dtype=np.float64
        ) + noise
        self.temperatures = np.clip(self.temperatures, 0, None)
        return self.temperatures.copy()

    def step(
        self,
        workload: np.ndarray,
        cooling: np.ndarray,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """
        Advance the thermal field by one timestep.

        Parameters
        ----------
        workload : ndarray (rows, cols) in [0, 1]
            Normalised workload (0 = idle, 1 = full load).
        cooling : ndarray (rows, cols) in [0, cooling_capacity]
            Cooling power applied to each cell.

        Returns
        -------
        temperatures : ndarray (rows, cols) — new temperature field.
        """
        c = self.cfg
        T = self.temperatures

        # 1. Workload heat generation
        #    Dominant: CPU.  Optional second term: memory.
        #    workload array contains normalised CPU util; we split it
        #    into cpu and mem in the environment layer. Here we treat
        #    the first channel as total heat fraction.
        heat = c.workload_heat_coefficient * workload

        # 2. Neighbour diffusion (von Neumann stencil)
        diffusion = self._compute_diffusion(T) * c.diffusion_coefficient

        # 3. Ambient pull (relaxation toward ambient temperature)
        ambient_pull = c.ambient_pull_coefficient * (c.ambient_temperature - T)

        # 4. Cooling effect
        cooling_clamped = np.clip(cooling, c.min_cooling, c.cooling_capacity)
        cool_effect = c.cooling_coefficient * cooling_clamped

        # 5. Noise
        noise = np.zeros_like(T)
        if rng is not None and c.temperature_noise_std > 0:
            noise = rng.normal(0, c.temperature_noise_std, size=T.shape)

        # Update
        T_new = T + heat + diffusion + ambient_pull - cool_effect + noise

        # Numerical safety: clamp to ≥ 0, protect against NaN/Inf
        T_new = np.nan_to_num(T_new, nan=c.ambient_temperature,
                              posinf=c.critical_temperature + 20,
                              neginf=0.0)
        T_new = np.clip(T_new, 0, c.critical_temperature + 30)

        self.temperatures = T_new
        return self.temperatures.copy()

    # ── internal ────────────────────────────────────

    def _compute_diffusion(self, T: np.ndarray) -> np.ndarray:
        """
        Laplacian-style diffusion: each cell receives heat from
        its von Neumann neighbours (up/down/left/right).

        diffusion[i,j] = Σ (T_neighbour - T[i,j]) / 4
        """
        diffusion = np.zeros_like(T)
        # Shift in each direction and subtract centre
        if self.rows > 1:
            diffusion[1:, :] += T[:-1, :] - T[1:, :]   # from above
            diffusion[:-1, :] += T[1:, :] - T[:-1, :]   # from below
        if self.cols > 1:
            diffusion[:, 1:] += T[:, :-1] - T[:, 1:]    # from left
            diffusion[:, :-1] += T[:, 1:] - T[:, :-1]    # from right
        # Average over number of neighbours (simplification)
        diffusion /= 4.0
        return diffusion

    # ── diagnostics ─────────────────────────────────

    def get_max_temperature(self) -> float:
        return float(np.max(self.temperatures))

    def get_mean_temperature(self) -> float:
        return float(np.mean(self.temperatures))

    def get_temperature_variance(self) -> float:
        return float(np.var(self.temperatures))

    def get_hotspot(self) -> tuple[int, int, float]:
        """Return (row, col, temperature) of the hottest cell."""
        idx = int(np.argmax(self.temperatures))
        r, c = divmod(idx, self.cols)
        return r, c, float(self.temperatures[r, c])

    def count_overheated(self, threshold: float | None = None) -> int:
        """Count cells above the given threshold (default: safe_temperature)."""
        thr = threshold if threshold is not None else self.cfg.safe_temperature
        return int(np.sum(self.temperatures > thr))

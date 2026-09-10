"""
State encoder: converts continuous 6×6 temperatures + workloads into a
discrete tabular state suitable for Q-Learning / SARSA / MC / VI.

Zone layout (4 zones, each 3×3 for a 6×6 grid):
    ┌──────────┬──────────┐
    │  Zone 0  │  Zone 1  │
    ├──────────┼──────────┤
    │  Zone 2  │  Zone 3  │
    └──────────┴──────────┘

State tuple:
    (z0_temp, z1_temp, z2_temp, z3_temp,
     z0_wl,   z1_wl,   z2_wl,   z3_wl,
     hotspot_zone)

Sizes:
    temp levels  = 5  (cool, normal, warm, hot, critical)
    wl levels    = 3  (low, medium, high)
    hotspot zone = 4

Total state space = 5^4 × 3^4 × 4 = 202,500
Total Q-table     = 202,500 × 8 actions = 1.62 M entries  (tractable)
"""

import numpy as np
from config import StateEncoderConfig, ThermalConfig


TEMP_LABELS = ["cool", "normal", "warm", "hot", "critical"]
WORKLOAD_LABELS = ["low", "medium", "high"]


class StateEncoder:
    """Deterministic, unit-testable encoder from continuous to discrete state."""

    def __init__(
        self,
        thermal_cfg: ThermalConfig | None = None,
        encoder_cfg: StateEncoderConfig | None = None,
    ):
        self.t_cfg = thermal_cfg or ThermalConfig()
        self.e_cfg = encoder_cfg or StateEncoderConfig()
        self.rows = self.t_cfg.grid_rows
        self.cols = self.t_cfg.grid_cols

        # Precompute zone boundaries
        self.mid_row = self.rows // 2
        self.mid_col = self.cols // 2

        # Zone slices: (row_start, row_end, col_start, col_end)
        self.zones = [
            (0, self.mid_row, 0, self.mid_col),               # Zone 0: top-left
            (0, self.mid_row, self.mid_col, self.cols),        # Zone 1: top-right
            (self.mid_row, self.rows, 0, self.mid_col),        # Zone 2: bottom-left
            (self.mid_row, self.rows, self.mid_col, self.cols), # Zone 3: bottom-right
        ]

    # ── main API ────────────────────────────────────

    def encode(
        self,
        temperatures: np.ndarray,
        workloads: np.ndarray,
    ) -> tuple:
        """
        Convert continuous temperature + workload grids to a discrete state tuple.

        Returns
        -------
        state : tuple of ints
            (z0_t, z1_t, z2_t, z3_t, z0_w, z1_w, z2_w, z3_w, hotspot_zone)
        """
        temp_levels = []
        wl_levels = []

        for rs, re, cs, ce in self.zones:
            zone_temps = temperatures[rs:re, cs:ce]
            zone_wl = workloads[rs:re, cs:ce]

            # Use mean temperature for discretisation
            mean_t = float(np.mean(zone_temps))
            temp_levels.append(self._discretise_temp(mean_t))

            mean_w = float(np.mean(zone_wl))
            wl_levels.append(self._discretise_workload(mean_w))

        # Identify hotspot zone (zone with highest max temperature)
        zone_maxes = []
        for rs, re, cs, ce in self.zones:
            zone_maxes.append(float(np.max(temperatures[rs:re, cs:ce])))
        hotspot_zone = int(np.argmax(zone_maxes))

        state = tuple(temp_levels) + tuple(wl_levels) + (hotspot_zone,)
        return state

    def state_space_size(self) -> int:
        """Total number of possible discrete states."""
        n = self.e_cfg.num_zones
        return (self.e_cfg.num_temp_levels ** n *
                self.e_cfg.num_workload_levels ** n *
                n)

    # ── discretisation ──────────────────────────────

    def _discretise_temp(self, temp: float) -> int:
        """
        Map temperature to discrete level.
        0=cool, 1=normal, 2=warm, 3=hot, 4=critical
        """
        bins = self.e_cfg.temp_bins
        for i, threshold in enumerate(bins):
            if temp < threshold:
                return i
        return len(bins)  # critical

    def _discretise_workload(self, wl: float) -> int:
        """
        Map normalised workload [0,1] to discrete level.
        0=low, 1=medium, 2=high
        """
        bins = self.e_cfg.workload_bins
        for i, threshold in enumerate(bins):
            if wl < threshold:
                return i
        return len(bins)  # high

    # ── human-readable helpers ──────────────────────

    def decode_state_labels(self, state: tuple) -> dict:
        """Convert a state tuple to human-readable labels."""
        n = self.e_cfg.num_zones
        return {
            "zone_temps": [TEMP_LABELS[state[i]] for i in range(n)],
            "zone_workloads": [WORKLOAD_LABELS[state[n + i]] for i in range(n)],
            "hotspot_zone": state[2 * n],
        }

    def get_zone_cells(self, zone_id: int) -> list[tuple[int, int]]:
        """Return list of (row, col) cells in a given zone."""
        rs, re, cs, ce = self.zones[zone_id]
        return [(r, c) for r in range(rs, re) for c in range(cs, ce)]

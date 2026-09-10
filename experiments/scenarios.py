"""
Named scenario configurations for experiments.
"""

from config import WorkloadMode, ThermalConfig, DatasetConfig


SCENARIOS = {
    "alibaba_replay": {
        "name": "Alibaba Dataset Replay",
        "mode": WorkloadMode.DATASET_REPLAY,
        "description": "Real Alibaba workload trace (or synthetic equivalent)",
    },
    "workload_burst": {
        "name": "Workload Burst",
        "mode": WorkloadMode.WORKLOAD_BURST,
        "description": "Sudden workload spikes at irregular intervals",
    },
    "moving_hotspot": {
        "name": "Moving Hotspot",
        "mode": WorkloadMode.MOVING_HOTSPOT,
        "description": "Hotspot migrates across the data center floor",
    },
    "multi_hotspot": {
        "name": "Multiple Hotspots",
        "mode": WorkloadMode.MULTI_HOTSPOT,
        "description": "Two or more simultaneous hot regions",
    },
    "static_hotspot": {
        "name": "Static Hotspot",
        "mode": WorkloadMode.STATIC_HOTSPOT,
        "description": "Fixed hot region (debugging/baseline)",
    },
}


def get_scenario(name: str) -> dict:
    if name not in SCENARIOS:
        raise ValueError(
            f"Unknown scenario '{name}'. Available: {list(SCENARIOS.keys())}"
        )
    return SCENARIOS[name]


def list_scenarios() -> list[str]:
    return list(SCENARIOS.keys())

"""
Reproducibility utilities: seed management and deterministic setup.
"""

import random
import numpy as np


def set_global_seed(seed: int) -> None:
    """Set seeds for all random sources to ensure reproducibility."""
    random.seed(seed)
    np.random.seed(seed)


def get_rng(seed: int) -> np.random.Generator:
    """Create a dedicated numpy random generator with the given seed."""
    return np.random.default_rng(seed)

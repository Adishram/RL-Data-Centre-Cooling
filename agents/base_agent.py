"""
Abstract base class for all RL agents.

Every agent must implement:
    select_action(state)  → int
    update(...)           → None
    save(path)            → None
    load(path)            → None
    reset()               → None
"""

from abc import ABC, abstractmethod
from pathlib import Path


class BaseAgent(ABC):
    """Common interface for all RL agents."""

    def __init__(self, n_actions: int, name: str = "BaseAgent"):
        self.n_actions = n_actions
        self.name = name

    @abstractmethod
    def select_action(self, state: tuple) -> int:
        """Choose an action given the current state."""
        ...

    @abstractmethod
    def update(self, **kwargs) -> None:
        """Update the agent's value estimates / policy."""
        ...

    @abstractmethod
    def save(self, path: str | Path) -> None:
        """Persist the learned policy / Q-table to disk."""
        ...

    @abstractmethod
    def load(self, path: str | Path) -> None:
        """Restore a previously saved policy / Q-table."""
        ...

    @abstractmethod
    def reset(self) -> None:
        """Clear all learned values (fresh start)."""
        ...

    def get_info(self) -> dict:
        """Return agent-specific diagnostic information."""
        return {"name": self.name, "n_actions": self.n_actions}

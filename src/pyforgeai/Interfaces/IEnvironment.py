from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any
import numpy as np


class IEnvironment(ABC):
    """Gym-style Reinforcement Learning environment interface for step-by-step training (DQN/Q-Learning)."""

    @property
    @abstractmethod
    def state_size(self) -> int:
        """Dimensionality of the state observation vector."""
        pass

    @property
    @abstractmethod
    def action_size(self) -> int:
        """Number of discrete actions available to the agent."""
        pass

    @abstractmethod
    def reset(self) -> np.ndarray:
        """Reset the environment to an initial state and return the initial observation vector."""
        pass

    @abstractmethod
    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
        """Execute one environment step and return (next_state, reward, done, info)."""
        pass

    def should_stop(self) -> bool:
        """Return True if the host application/window requested an immediate exit."""
        return False


class IFitnessEnvironment(ABC):
    """Episode-level evaluation contract for neuroevolution / Genetic Algorithms."""

    @abstractmethod
    def evaluate_model(self, model: Any) -> float:
        """Run one complete episode/trial controlled by `model` and return its scalar fitness score."""
        pass

    def should_stop(self) -> bool:
        """Return True if the host application/window requested an immediate exit."""
        return False

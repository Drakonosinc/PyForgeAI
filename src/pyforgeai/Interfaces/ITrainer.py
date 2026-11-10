from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class TrainingResult:
    """Structured summary produced at the end of a training run."""

    best_model: Any
    best_score: float
    history: list[float] = field(default_factory=list)
    iterations_completed: int = 0
    stopped_early: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


class ITrainingCallback(ABC):
    """Hook interface for observing and controlling training loops (UI updates, logging, checkpoints, early stopping)."""

    def on_training_start(self, trainer: Any) -> None:
        pass

    def on_iteration_end(
        self,
        iteration: int,
        best_score: float,
        mean_score: float,
        trainer: Any,
    ) -> bool | None:
        """Called after each generation or episode. Return True to stop training early."""
        return False

    def on_training_end(self, result: TrainingResult, trainer: Any) -> None:
        pass


class ITrainer(ABC):
    """Unified interface for AI training algorithms (Genetic Algorithm, Deep Q-Learning, etc.)."""

    @abstractmethod
    def train(self, target: Any = None) -> Any:
        """Execute the training loop and return the best trained model."""
        pass

from __future__ import annotations

from typing import Any

from ..Interfaces.ITrainer import ITrainingCallback, TrainingResult
from ..Serialization.Model_Serializer import save_model


class EarlyStoppingCallback(ITrainingCallback):
    """Stop training early when `target_score` is reached or when score stops improving for `patience` iterations."""

    def __init__(
        self,
        patience: int | None = None,
        min_delta: float = 1e-4,
        target_score: float | None = None,
    ) -> None:
        self.patience = patience
        self.min_delta = float(min_delta)
        self.target_score = target_score
        self._best_seen = float("-inf")
        self._stagnant_iters = 0

    def on_training_start(self, trainer: Any) -> None:
        self._best_seen = float("-inf")
        self._stagnant_iters = 0

    def on_iteration_end(
        self,
        iteration: int,
        best_score: float,
        mean_score: float,
        trainer: Any,
    ) -> bool:
        if self.target_score is not None and best_score >= self.target_score:
            return True

        if best_score > self._best_seen + self.min_delta:
            self._best_seen = best_score
            self._stagnant_iters = 0
        else:
            self._stagnant_iters += 1

        if self.patience is not None and self._stagnant_iters >= self.patience:
            return True
        return False


class CheckpointCallback(ITrainingCallback):
    """Automatically save the best model to `save_path` whenever a new record score is achieved."""

    def __init__(self, save_path: str) -> None:
        self.save_path = save_path
        self._best_saved = float("-inf")

    def on_training_start(self, trainer: Any) -> None:
        self._best_saved = float("-inf")

    def on_iteration_end(
        self,
        iteration: int,
        best_score: float,
        mean_score: float,
        trainer: Any,
    ) -> bool:
        if best_score > self._best_saved:
            self._best_saved = best_score
            best_model = getattr(trainer, "best_model", None)
            if best_model is not None:
                save_model(best_model, self.save_path)
        return False


class HistoryCallback(ITrainingCallback):
    """Track per-iteration best and mean scores for plotting or analysis."""

    def __init__(self) -> None:
        self.best_scores: list[float] = []
        self.mean_scores: list[float] = []

    def on_training_start(self, trainer: Any) -> None:
        self.best_scores.clear()
        self.mean_scores.clear()

    def on_iteration_end(
        self,
        iteration: int,
        best_score: float,
        mean_score: float,
        trainer: Any,
    ) -> bool:
        self.best_scores.append(float(best_score))
        self.mean_scores.append(float(mean_score))
        return False

    def on_training_end(self, result: TrainingResult, trainer: Any) -> None:
        pass

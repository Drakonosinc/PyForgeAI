from __future__ import annotations

from typing import Any, Callable, Sequence
import numpy as np

from ..Interfaces.IEnvironment import IEnvironment, IFitnessEnvironment


class FunctionalFitnessEnvironment(IFitnessEnvironment):
    """Domain-agnostic adapter turning pure Python callables into an `IFitnessEnvironment`.

    Supports both single-agent evaluation (`fitness_fn(model) -> float`) and vectorized/multi-agent
    population evaluation (`batch_fitness_fn(population_list) -> list[float]`), without assuming
    any specific application or game object structure.
    """

    def __init__(
        self,
        target: Callable[[Any], float] | IFitnessEnvironment | None = None,
        batch_fitness_fn: Callable[[list[Any]], Sequence[float]] | None = None,
        stop_fn: Callable[[], bool] | None = None,
        on_generation_fn: Callable[[int], None] | None = None,
        on_best_fn: Callable[[Any], None] | None = None,
    ) -> None:
        if target is None and batch_fitness_fn is None:
            raise TypeError(
                "Must provide either `target` (a callable `fn(model) -> float` or `IFitnessEnvironment`) "
                "or `batch_fitness_fn` (`fn(population) -> Sequence[float]`)."
            )
        if target is not None and not isinstance(target, IFitnessEnvironment) and not callable(target):
            raise TypeError(
                f"Expected `target` to be a callable `fn(model) -> float` or an `IFitnessEnvironment`, got {type(target).__name__}."
            )
        self.target = target
        self._batch_fitness_fn = batch_fitness_fn
        self._stop_fn = stop_fn
        self._on_generation_fn = on_generation_fn
        self._on_best_fn = on_best_fn

    def evaluate_model(self, model: Any) -> float:
        if isinstance(self.target, IFitnessEnvironment):
            return float(self.target.evaluate_model(model))
        if callable(self.target):
            return float(self.target(model))
        if self._batch_fitness_fn is not None:
            scores = self._batch_fitness_fn([model])
            return float(scores[0]) if len(scores) > 0 else 0.0
        raise RuntimeError("No single-model evaluation function configured.")

    def is_batch_population_target(self) -> bool:
        """Return True if this environment evaluates an entire population list at once."""
        if self._batch_fitness_fn is not None:
            return True
        if isinstance(self.target, IFitnessEnvironment):
            # Check if subclass overrides evaluate_population
            return type(self.target).evaluate_population is not IFitnessEnvironment.evaluate_population
        return False

    def evaluate_population(self, population: list[Any]) -> list[float] | None:
        if self._batch_fitness_fn is not None:
            scores = self._batch_fitness_fn(list(population))
            return [float(s) for s in scores]
        if isinstance(self.target, IFitnessEnvironment):
            return self.target.evaluate_population(list(population))
        return None

    def evaluate_population_batch(self, population: Sequence[Any]) -> list[float]:
        res = self.evaluate_population(list(population))
        if res is not None:
            return res
        return [self.evaluate_model(m) for m in population]

    def should_stop(self) -> bool:
        if self._stop_fn is not None:
            return bool(self._stop_fn())
        if isinstance(self.target, IFitnessEnvironment):
            return bool(self.target.should_stop())
        return False

    def on_generation_start(self, generation: int) -> None:
        if self._on_generation_fn is not None:
            self._on_generation_fn(int(generation))
        elif isinstance(self.target, IFitnessEnvironment):
            self.target.on_generation_start(int(generation))

    def on_best_model(self, model: Any) -> None:
        if self._on_best_fn is not None:
            self._on_best_fn(model)
        elif isinstance(self.target, IFitnessEnvironment):
            self.target.on_best_model(model)


# Alias kept for backward import compatibility
GameFitnessBridge = FunctionalFitnessEnvironment


class CallbackRLEnvironment(IEnvironment):
    """Creates a step-by-step RL environment from user-supplied callables (`reset_fn`, `step_fn`)."""

    def __init__(
        self,
        state_size: int,
        action_size: int,
        reset_fn: Callable[[], np.ndarray],
        step_fn: Callable[[int], tuple[np.ndarray, float, bool] | tuple[np.ndarray, float, bool, dict[str, Any]]],
        stop_fn: Callable[[], bool] | None = None,
    ) -> None:
        self._state_size = int(state_size)
        self._action_size = int(action_size)
        self._reset_fn = reset_fn
        self._step_fn = step_fn
        self._stop_fn = stop_fn

    @property
    def state_size(self) -> int:
        return self._state_size

    @property
    def action_size(self) -> int:
        return self._action_size

    def reset(self) -> np.ndarray:
        return np.asarray(self._reset_fn(), dtype=np.float32)

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
        result = self._step_fn(int(action))
        if len(result) == 3:
            next_state, reward, done = result
            info: dict[str, Any] = {}
        else:
            next_state, reward, done, info = result
        return np.asarray(next_state, dtype=np.float32), float(reward), bool(done), info

    def should_stop(self) -> bool:
        return bool(self._stop_fn()) if self._stop_fn is not None else False

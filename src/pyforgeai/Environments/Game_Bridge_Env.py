from __future__ import annotations

import inspect
from typing import Any, Callable
import numpy as np

from ..Interfaces.IEnvironment import IEnvironment, IFitnessEnvironment


def _is_plain_callable(obj: Any) -> bool:
    """Return True if `obj` is a function, lambda, or bound method rather than a game object or MagicMock."""
    return inspect.isfunction(obj) or inspect.ismethod(obj) or inspect.isbuiltin(obj)


class GameFitnessBridge(IFitnessEnvironment):
    """Bridges an `IFitnessEnvironment`, a pure function `fn(model) -> float`, or a Pygame `game` instance."""

    def __init__(
        self,
        target: Any,
        stop_fn: Callable[[], bool] | None = None,
    ) -> None:
        self.target = target
        self._stop_fn = stop_fn

    def evaluate_model(self, model: Any) -> float:
        if isinstance(self.target, IFitnessEnvironment):
            return float(self.target.evaluate_model(model))

        if _is_plain_callable(self.target):
            return float(self.target(model))

        game = self.target
        # Support Pong_Game style (`game.model` + `game.ai_handler.set_runtime_model` + `game.run_with_model()`)
        if hasattr(game, "run_with_model"):
            game.model = model
            if hasattr(game, "ai_handler") and hasattr(game.ai_handler, "set_runtime_model"):
                game.ai_handler.set_runtime_model(model)
            return float(game.run_with_model())

        # Support Snake_Game / Flappy_Ghost style (`game.models` + `game.run_with_models()`)
        if hasattr(game, "run_with_models"):
            game.models = model
            return float(game.run_with_models())

        if callable(self.target):
            return float(self.target(model))

        raise TypeError(
            "Target passed to GameFitnessBridge must be a callable `fn(model) -> float`, "
            "an IFitnessEnvironment, or a game instance defining `run_with_model()`."
        )

    def should_stop(self) -> bool:
        if self._stop_fn is not None:
            return bool(self._stop_fn())
        if isinstance(self.target, IFitnessEnvironment):
            return bool(self.target.should_stop())
        if _is_plain_callable(self.target):
            return False
        exit_attr = getattr(self.target, "exit", False)
        return bool(exit_attr) if isinstance(exit_attr, bool) else False

    def set_generation(self, generation: int) -> None:
        if not isinstance(self.target, IFitnessEnvironment) and not _is_plain_callable(self.target):
            try:
                self.target.generation = generation
            except Exception:
                pass

    def assign_best_model(self, model: Any) -> None:
        if not isinstance(self.target, IFitnessEnvironment) and not _is_plain_callable(self.target):
            try:
                self.target.model = model
            except Exception:
                pass


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

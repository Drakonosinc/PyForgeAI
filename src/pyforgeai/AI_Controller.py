from __future__ import annotations

from typing import Any, Callable
import numpy as np

from .Adapters import MockAdapter, PyTorchAdapter, TensorFlowAdapter, wrap_model
from .Interfaces.IAIModel import IAIModel
from .Serialization.Model_Serializer import load_model, save_model
from .Trainers.Q_Learning import QLearningTrainer, get_active_qlearning_trainer


class AIController:
    """Reusable runtime AI controller managing inference adapters and episodic Q-Learning state transitions."""

    def __init__(
        self,
        state_provider: Callable[[], np.ndarray] | None = None,
        action_dispatcher: Callable[[int | np.ndarray], None] | None = None,
        default_backend: str = "Pytorch",
        output_size: int = 2,
    ) -> None:
        self.state_provider = state_provider
        self.action_dispatcher = action_dispatcher
        self.default_backend = default_backend
        self.output_size = int(output_size)

        self.model_adapter: IAIModel | None = None
        self.prev_state: np.ndarray | None = None
        self.prev_action: int | None = None
        self.prev_reward: float = 0.0

    def set_model(self, model_adapter: IAIModel) -> None:
        self.model_adapter = model_adapter

    def clear_model(self) -> None:
        self.model_adapter = None

    def set_runtime_model(self, model: Any, type_model: str | None = None) -> IAIModel | None:
        """Wrap and bind a raw PyTorch/TensorFlow model or IAIModel adapter only when it changes."""
        if model is None:
            return None
        if isinstance(model, IAIModel):
            self.model_adapter = model
            return self.model_adapter

        current_internal = self.model_adapter.get_internal_model() if self.model_adapter is not None else None
        if current_internal is model:
            return self.model_adapter

        backend = (type_model or self.default_backend).strip().capitalize()
        if backend in ("Pytorch", "Torch"):
            self.model_adapter = PyTorchAdapter(model)
        elif backend in ("Tensorflow", "Tf", "Keras"):
            self.model_adapter = TensorFlowAdapter(model)
        else:
            self.model_adapter = wrap_model(model, output_size=self.output_size)
        return self.model_adapter

    def get_state(self) -> np.ndarray:
        if self.state_provider is not None:
            return np.asarray(self.state_provider(), dtype=np.float32)
        if self.prev_state is not None:
            return self.prev_state.copy()
        raise RuntimeError("No state_provider configured and no previous state available.")

    @staticmethod
    def softmax(x: np.ndarray | list[float]) -> np.ndarray:
        arr = np.asarray(x, dtype=np.float32).flatten()
        exp_x = np.exp(arr - np.max(arr))
        total = np.sum(exp_x)
        return exp_x / total if total != 0.0 else np.full_like(arr, 1.0 / max(1, len(arr)))

    def predict(self, state: np.ndarray | list[float] | None = None) -> np.ndarray | None:
        if self.model_adapter is None:
            return None
        obs = np.asarray(state, dtype=np.float32) if state is not None else self.get_state()
        return self.model_adapter.predict(obs)

    def select_action(self, state: np.ndarray | list[float] | None = None, deterministic: bool = True) -> int | None:
        if self.model_adapter is None:
            return None
        obs = np.asarray(state, dtype=np.float32) if state is not None else self.get_state()
        action = self.model_adapter.predict_action(obs, deterministic=deterministic)
        if self.action_dispatcher is not None:
            self.action_dispatcher(action)
        return action

    def step_qlearning(
        self,
        current_state: np.ndarray | list[float],
        current_reward: float,
        done: bool = False,
        trainer: QLearningTrainer | None = None,
        use_reward_delta: bool = True,
    ) -> int | None:
        """Process one Q-Learning frame: record transition from previous frame, optimize, and return next action."""
        active_trainer = trainer or get_active_qlearning_trainer()
        if active_trainer is None:
            return None

        curr_state_arr = np.asarray(current_state, dtype=np.float32)
        if self.prev_state is not None and self.prev_action is not None:
            step_reward = (float(current_reward) - self.prev_reward) if use_reward_delta else float(current_reward)
            active_trainer.store_experience(
                self.prev_state,
                self.prev_action,
                step_reward,
                curr_state_arr,
                bool(done),
            )

        action = active_trainer.get_action(curr_state_arr)
        if self.action_dispatcher is not None:
            self.action_dispatcher(action)

        self.prev_state = curr_state_arr.copy()
        self.prev_action = action
        self.prev_reward = float(current_reward)
        if done:
            self.reset_qlearning_state()
        return action

    def reset_qlearning_state(self) -> None:
        self.prev_state = None
        self.prev_action = None
        self.prev_reward = 0.0

    def manual_save_model(self, path: str) -> None:
        if self.model_adapter is not None:
            internal = self.model_adapter.get_internal_model()
            if internal is not None:
                save_model(internal, path)
            else:
                self.model_adapter.save(path)

    def manual_load_model(
        self,
        path: str,
        input_size: int | None = None,
        output_size: int | None = None,
        hidden_sizes: list[int] | None = None,
    ) -> IAIModel | None:
        loaded = load_model(
            path=path,
            model_or_type_model=self.default_backend,
            input_size=input_size,
            output_size=output_size or self.output_size,
            hidden_sizes=hidden_sizes,
        )
        if loaded is not None:
            return self.set_runtime_model(loaded, type_model=self.default_backend)
        return None


# Alias for backward compatibility with Pong_Game / Snake_Game
AIHandler = AIController

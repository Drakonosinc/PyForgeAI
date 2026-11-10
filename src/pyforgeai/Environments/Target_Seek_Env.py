from __future__ import annotations

from typing import Any
import numpy as np

from ..Adapters import wrap_model
from ..Interfaces.IEnvironment import IEnvironment, IFitnessEnvironment


class TargetSeekEnvironment(IEnvironment, IFitnessEnvironment):
    """Built-in 1D paddle/ball intercept simulation environment for testing and benchmarking Genetic & DQN trainers."""

    def __init__(self, max_steps: int = 80, seed: int | None = None) -> None:
        self.max_steps = int(max_steps)
        self.rng = np.random.default_rng(seed)
        self.agent_y = 0.5
        self.target_y = 0.5
        self.target_vy = 0.02
        self.steps = 0

    @property
    def state_size(self) -> int:
        return 4

    @property
    def action_size(self) -> int:
        return 3  # 0: UP, 1: STAY, 2: DOWN

    def _get_obs(self) -> np.ndarray:
        dy = self.target_y - self.agent_y
        return np.array([self.agent_y, self.target_y, dy, self.target_vy], dtype=np.float32)

    def reset(self) -> np.ndarray:
        self.agent_y = float(self.rng.uniform(0.1, 0.9))
        self.target_y = float(self.rng.uniform(0.1, 0.9))
        self.target_vy = float(self.rng.choice([-0.03, 0.03]))
        self.steps = 0
        return self._get_obs()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
        if action == 0:
            self.agent_y = max(0.0, self.agent_y - 0.05)
        elif action == 2:
            self.agent_y = min(1.0, self.agent_y + 0.05)

        self.target_y += self.target_vy
        if self.target_y <= 0.0 or self.target_y >= 1.0:
            self.target_vy *= -1.0
            self.target_y = float(np.clip(self.target_y, 0.0, 1.0))

        self.steps += 1
        distance = abs(self.target_y - self.agent_y)
        reward = 1.0 - (distance * 2.0)
        if distance < 0.08:
            reward += 1.0

        done = self.steps >= self.max_steps
        return self._get_obs(), float(reward), done, {"distance": distance}

    def evaluate_model(self, model: Any) -> float:
        adapter = wrap_model(model, output_size=self.action_size)
        state = self.reset()
        total_reward = 0.0
        done = False
        while not done:
            action = adapter.predict_action(state, deterministic=True)
            state, reward, done, _ = self.step(action)
            total_reward += reward
        return float(total_reward)

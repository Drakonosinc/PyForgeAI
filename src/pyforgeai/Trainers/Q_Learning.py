from __future__ import annotations

import copy
import random
from collections import deque
from typing import Any, Callable, Sequence
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from ..Interfaces.IEnvironment import IEnvironment
from ..Interfaces.ITrainer import ITrainer, ITrainingCallback, TrainingResult
from ..Models.Neural_Network_Pytorch import SimpleNN_Pytorch
from ..Serialization.Model_Serializer import load_qlearning_model, save_qlearning_model


def _build_qlearning_model(
    type_model: str,
    state_size: int,
    action_size: int,
    hidden_sizes: Sequence[int] | None = None,
) -> SimpleNN_Pytorch:
    if type_model.strip().capitalize() not in ("Pytorch", "Torch"):
        raise NotImplementedError("Q-Learning / DQN currently supports the 'Pytorch' backend.")
    return SimpleNN_Pytorch(state_size, action_size, hidden_sizes=hidden_sizes)


class ReplayMemory:
    """Fixed-capacity experience replay buffer storing (state, action, reward, next_state, done) transitions."""

    def __init__(self, capacity: int = 10000) -> None:
        self.memory: deque[tuple[Any, ...]] = deque(maxlen=int(capacity))

    def push(self, transition: tuple[Any, ...]) -> None:
        self.memory.append(transition)

    def sample(self, batch_size: int) -> list[tuple[Any, ...]]:
        return random.sample(self.memory, int(batch_size))

    def __len__(self) -> int:
        return len(self.memory)


class DQNAgent:
    """Deep Q-Network agent supporting Double-DQN, configurable Huber/MSE loss, and epsilon-greedy exploration."""

    def __init__(
        self,
        type_model: str = "Pytorch",
        state_size: int = 6,
        action_size: int = 2,
        lr: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        memory_size: int = 10000,
        batch_size: int = 32,
        target_update: int = 100,
        hidden_sizes: Sequence[int] | None = None,
        double_dqn: bool = True,
        loss_type: str = "mse",
        grad_clip: float | None = 1.0,
        device: str | torch.device | None = None,
    ) -> None:
        self.type_model = type_model
        self.state_size = int(state_size)
        self.action_size = int(action_size)
        self._hidden_sizes = list(hidden_sizes) if hidden_sizes is not None else None
        self.gamma = float(gamma)
        self.epsilon = float(epsilon_start)
        self.epsilon_min = float(epsilon_end)
        self.epsilon_decay = float(epsilon_decay)
        self.batch_size = int(batch_size)
        self.steps_done = 0
        self.target_update = max(1, int(target_update))
        self.double_dqn = bool(double_dqn)
        self.grad_clip = grad_clip

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.policy_net = _build_qlearning_model(
            type_model,
            self.state_size,
            self.action_size,
            hidden_sizes=self._hidden_sizes,
        ).to(self.device)
        self.target_net = _build_qlearning_model(
            type_model,
            self.state_size,
            self.action_size,
            hidden_sizes=self._hidden_sizes,
        ).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=float(lr))
        self.memory = ReplayMemory(memory_size)
        self.loss_fn = nn.SmoothL1Loss() if loss_type.lower() == "huber" else nn.MSELoss()

    def select_action(self, state: np.ndarray | list[float]) -> int:
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)
        with torch.no_grad():
            tensor_state = torch.as_tensor(
                np.asarray(state, dtype=np.float32),
                dtype=torch.float32,
                device=self.device,
            )
            q_values = self.policy_net(tensor_state)
            return int(torch.argmax(q_values).item())

    def store_transition(
        self,
        state: np.ndarray | list[float],
        action: int,
        reward: float,
        next_state: np.ndarray | list[float],
        done: bool,
    ) -> None:
        self.memory.push((state, int(action), float(reward), next_state, bool(done)))

    def optimize_model(self) -> float | None:
        if len(self.memory) < self.batch_size:
            return None

        transitions = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*transitions)

        states_t = torch.as_tensor(np.asarray(states, dtype=np.float32), dtype=torch.float32, device=self.device)
        actions_t = torch.as_tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(1)
        rewards_t = torch.as_tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states_t = torch.as_tensor(
            np.asarray(next_states, dtype=np.float32),
            dtype=torch.float32,
            device=self.device,
        )
        dones_t = torch.as_tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)

        current_q = self.policy_net(states_t).gather(1, actions_t)

        with torch.no_grad():
            if self.double_dqn:
                best_next_actions = self.policy_net(next_states_t).argmax(dim=1, keepdim=True)
                next_q = self.target_net(next_states_t).gather(1, best_next_actions)
            else:
                next_q = self.target_net(next_states_t).max(1)[0].unsqueeze(1)
            expected_q = rewards_t + (1.0 - dones_t) * self.gamma * next_q

        loss = self.loss_fn(current_q, expected_q)
        self.optimizer.zero_grad()
        loss.backward()
        if self.grad_clip is not None:
            nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=float(self.grad_clip))
        self.optimizer.step()

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.steps_done += 1
        if self.steps_done % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return float(loss.item())


class QLearningTrainer(ITrainer):
    """Deep Q-Learning trainer supporting both Gym-style `IEnvironment` loops and event-driven Pygame loops."""

    def __init__(
        self,
        game: Any = None,
        type_model: str = "Pytorch",
        input_size: int = 6,
        output_size: int = 2,
        episodes: int = 500,
        lr: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        memory_size: int = 10000,
        batch_size: int = 32,
        target_update: int = 100,
        hidden_sizes: Sequence[int] | None = None,
        double_dqn: bool = True,
        save_path: str | None = None,
        callbacks: Sequence[ITrainingCallback] | None = None,
        verbose: bool = True,
    ) -> None:
        self.game = game
        self.type_model = type_model
        self.input_size = int(input_size)
        self.output_size = int(output_size)
        self.episodes = int(episodes)
        self.current_episode = 0
        self.best_reward = float("-inf")
        self.best_model: SimpleNN_Pytorch | None = None
        self.rewards_history: list[float] = []
        self.save_path = save_path
        self.callbacks = list(callbacks) if callbacks else []
        self.verbose = bool(verbose)
        self.last_result: TrainingResult | None = None

        self.agent = DQNAgent(
            type_model=type_model,
            state_size=self.input_size,
            action_size=self.output_size,
            lr=lr,
            gamma=gamma,
            epsilon_start=epsilon_start,
            epsilon_end=epsilon_end,
            epsilon_decay=epsilon_decay,
            memory_size=memory_size,
            batch_size=batch_size,
            target_update=target_update,
            hidden_sizes=hidden_sizes,
            double_dqn=double_dqn,
        )

    def get_action(self, state: np.ndarray | list[float]) -> int:
        return self.agent.select_action(state)

    def store_experience(
        self,
        state: np.ndarray | list[float],
        action: int,
        reward: float,
        next_state: np.ndarray | list[float],
        done: bool,
    ) -> float | None:
        self.agent.store_transition(state, action, reward, next_state, done)
        return self.agent.optimize_model()

    def episode_complete(self, total_reward: float) -> bool:
        self.rewards_history.append(float(total_reward))
        if total_reward > self.best_reward:
            self.best_reward = float(total_reward)
            self.best_model = copy.deepcopy(self.agent.policy_net)

        if self.verbose and self.current_episode % 10 == 0:
            window = self.rewards_history[-10:]
            avg_reward = float(np.mean(window))
            print(
                f"Episode {self.current_episode}/{self.episodes}: "
                f"Avg Reward = {avg_reward:.2f}, Epsilon = {self.agent.epsilon:.3f}"
            )

        self.current_episode += 1
        if self.current_episode >= self.episodes:
            if self.verbose:
                print(f"Q-Learning training completed. Best reward: {self.best_reward:.2f}")
            return True
        return False

    def get_best_model(self) -> SimpleNN_Pytorch:
        return self.best_model if self.best_model is not None else self.agent.policy_net

    def train_environment_with_result(self, env: IEnvironment, max_steps_per_episode: int = 1000) -> TrainingResult:
        """Train directly against a step-based `IEnvironment` (Gym-like interface)."""
        self.current_episode = 0
        self.best_reward = float("-inf")
        self.rewards_history.clear()
        stopped_early = False

        for cb in self.callbacks:
            cb.on_training_start(self)

        for ep in range(self.episodes):
            if env.should_stop():
                stopped_early = True
                break

            state = env.reset()
            total_reward = 0.0
            done = False
            steps = 0

            while not done and steps < max_steps_per_episode:
                if env.should_stop():
                    stopped_early = True
                    break
                action = self.get_action(state)
                next_state, reward, done, _ = env.step(action)
                self.store_experience(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward
                steps += 1

            training_complete = self.episode_complete(total_reward)
            mean_recent = float(np.mean(self.rewards_history[-10:]))

            should_halt = False
            for cb in self.callbacks:
                if cb.on_iteration_end(ep, self.best_reward, mean_recent, self):
                    should_halt = True
            if should_halt or training_complete or stopped_early:
                stopped_early = should_halt or stopped_early
                break

        best = self.get_best_model()
        if self.save_path and best is not None:
            save_qlearning_model(best, self.agent.optimizer, self.save_path)

        result = TrainingResult(
            best_model=best,
            best_score=self.best_reward,
            history=list(self.rewards_history),
            iterations_completed=self.current_episode,
            stopped_early=stopped_early,
        )
        self.last_result = result
        for cb in self.callbacks:
            cb.on_training_end(result, self)
        return result

    def train(self, target: Any = None) -> SimpleNN_Pytorch:
        """Run training against an `IEnvironment` or a game instance defining `run_with_model()`."""
        target_obj = target if target is not None else self.game
        if isinstance(target_obj, IEnvironment):
            return self.train_environment_with_result(target_obj).best_model

        if target_obj is None:
            raise ValueError("A target IEnvironment or game instance must be provided to QLearningTrainer.train().")

        # Event-driven game loop mode (compatible with Pong_Game)
        global _qlearning_trainer
        _qlearning_trainer = self
        target_obj.qlearning_trainer = self
        target_obj._qlearning_state = None
        target_obj._qlearning_prev_reward = 0

        for cb in self.callbacks:
            cb.on_training_start(self)

        while self.current_episode < self.episodes:
            target_obj.generation = self.current_episode
            total_reward = float(target_obj.run_with_model())
            training_complete = self.episode_complete(total_reward)
            mean_recent = float(np.mean(self.rewards_history[-10:]))
            should_halt = any(
                cb.on_iteration_end(self.current_episode - 1, self.best_reward, mean_recent, self)
                for cb in self.callbacks
            )
            if training_complete or should_halt or getattr(target_obj, "exit", False):
                break

        best_model = self.get_best_model()
        for attr in ("_qlearning_state", "_qlearning_prev_reward", "qlearning_trainer"):
            if hasattr(target_obj, attr):
                delattr(target_obj, attr)
        _qlearning_trainer = None
        target_obj.model = best_model

        if self.save_path and best_model is not None:
            save_qlearning_model(best_model, self.agent.optimizer, self.save_path)

        result = TrainingResult(
            best_model=best_model,
            best_score=self.best_reward,
            history=list(self.rewards_history),
            iterations_completed=self.current_episode,
            stopped_early=bool(getattr(target_obj, "exit", False)),
        )
        self.last_result = result
        for cb in self.callbacks:
            cb.on_training_end(result, self)
        return best_model


# Kept for backward compatibility with legacy event-driven Pong_Game imports
_qlearning_trainer: QLearningTrainer | None = None


def get_active_qlearning_trainer(game: Any = None) -> QLearningTrainer | None:
    """Retrieve the active QLearningTrainer from `game.qlearning_trainer` or module fallback."""
    if game is not None and hasattr(game, "qlearning_trainer"):
        return game.qlearning_trainer
    return _qlearning_trainer


def q_learning_algorithm(
    game: Any,
    type_model: str = "Pytorch",
    input_size: int = 6,
    output_size: int = 2,
    episodes: int = 500,
    lr: float = 1e-3,
    gamma: float = 0.99,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.01,
    epsilon_decay: float = 0.995,
    hidden_sizes: Sequence[int] | None = None,
    verbose: bool = True,
) -> SimpleNN_Pytorch:
    """Functional Q-Learning entry point preserving 100% compatibility with Pong_Game."""
    trainer = QLearningTrainer(
        game=game,
        type_model=type_model,
        input_size=input_size,
        output_size=output_size,
        episodes=episodes,
        lr=lr,
        gamma=gamma,
        epsilon_start=epsilon_start,
        epsilon_end=epsilon_end,
        epsilon_decay=epsilon_decay,
        hidden_sizes=hidden_sizes,
        verbose=verbose,
    )
    return trainer.train(game)

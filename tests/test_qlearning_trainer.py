import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock
import numpy as np

SRC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from pyforgeai import (
    DQNAgent,
    QLearningTrainer,
    ReplayMemory,
    TargetSeekEnvironment,
    load_qlearning_model,
    q_learning_algorithm,
    save_qlearning_model,
)


class TestQLearningTrainer(unittest.TestCase):
    def test_replay_memory_fifo(self):
        mem = ReplayMemory(capacity=3)
        self.assertEqual(len(mem), 0)
        mem.push((1, 0, 1.0, 2, False))
        mem.push((2, 1, 0.5, 3, False))
        mem.push((3, 0, -1.0, 4, True))
        mem.push((4, 1, 2.0, 5, False))
        self.assertEqual(len(mem), 3)
        sample = mem.sample(batch_size=2)
        self.assertEqual(len(sample), 2)

    def test_dqn_agent_double_and_standard_optimization(self):
        for double_flag in (True, False):
            agent = DQNAgent(
                type_model="Pytorch",
                state_size=4,
                action_size=3,
                batch_size=4,
                memory_size=50,
                hidden_sizes=[8],
                double_dqn=double_flag,
                loss_type="huber" if double_flag else "mse",
            )
            for _ in range(5):
                s = np.random.randn(4).astype(np.float32)
                s_next = np.random.randn(4).astype(np.float32)
                agent.store_transition(s, 1, 1.0, s_next, False)

            loss = agent.optimize_model()
            self.assertIsNotNone(loss)
            self.assertEqual(agent.steps_done, 1)

    def test_qlearning_trainer_with_environment(self):
        env = TargetSeekEnvironment(max_steps=15, seed=7)
        trainer = QLearningTrainer(
            input_size=env.state_size,
            output_size=env.action_size,
            episodes=4,
            batch_size=8,
            hidden_sizes=[8],
            verbose=False,
        )
        result = trainer.train_environment_with_result(env)
        self.assertEqual(result.iterations_completed, 4)
        self.assertEqual(len(result.history), 4)
        self.assertIsNotNone(result.best_model)

    def test_qlearning_legacy_game_loop_and_save_load(self):
        mock_game = MagicMock()
        mock_game.exit = False
        mock_game.run_with_model.return_value = 12.5

        best = q_learning_algorithm(
            game=mock_game,
            type_model="Pytorch",
            input_size=6,
            output_size=2,
            episodes=2,
            hidden_sizes=[8, 8],
            verbose=False,
        )
        self.assertIsNotNone(best)
        self.assertIs(mock_game.model, best)

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = os.path.join(tmp_dir, "q_model.pth")
            save_qlearning_model(best, optimizer=None, path=path)
            loaded = load_qlearning_model(path, "Pytorch", input_size=6, output_size=2, hidden_sizes=[8, 8])
            self.assertIsNotNone(loaded)
            for p1, p2 in zip(best.parameters(), loaded.parameters()):
                np.testing.assert_allclose(p1.detach().cpu().numpy(), p2.detach().cpu().numpy())


if __name__ == "__main__":
    unittest.main()

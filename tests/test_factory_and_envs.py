import os
import sys
import unittest
import numpy as np

SRC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from pyforgeai import (
    AIFactory,
    CallbackRLEnvironment,
    GeneticTrainer,
    QLearningTrainer,
    TargetSeekEnvironment,
)


class TestFactoryAndEnvironments(unittest.TestCase):
    def test_ai_factory_creation_methods(self):
        factory = AIFactory({"type_model": "Pytorch", "input_size": 4, "output_size": 3, "hidden_sizes": [10]})
        model = factory.create_Model()
        self.assertEqual(model.input_size, 4)
        self.assertEqual(model.output_size, 3)
        self.assertEqual(model.hidden_sizes, [10])

        adapter = factory.create_Adapter(model)
        pred = adapter.predict([0.1, 0.2, 0.3, 0.4])
        self.assertEqual(pred.shape, (3,))

        ga_trainer = factory.create_GeneticTrainer(generations=2, population_size=4)
        self.assertIsInstance(ga_trainer, GeneticTrainer)

        q_trainer = factory.create_QLearningTrainer(episodes=3)
        self.assertIsInstance(q_trainer, QLearningTrainer)

    def test_factory_from_game_config_dict(self):
        factory = AIFactory()
        config_ai = {
            "type_training": {"Genetic": False, "Q-learning": True},
            "type_model": {"Pytorch": True, "Tensorflow": False},
            "nn": {"hidden_layers": 2, "neurons_per_layer": 8},
            "q_learning": {
                "episodes": 5,
                "learning_rate": 0.001,
                "gamma": 0.95,
                "epsilon_start": 1.0,
                "epsilon_end": 0.05,
                "epsilon_decay": 0.99,
            },
            "model_save": False,
        }
        trainer = factory.create_Trainer_from_game_config(config_ai, input_size=6, output_size=2, verbose=False)
        self.assertIsInstance(trainer, QLearningTrainer)
        self.assertEqual(trainer.episodes, 5)

    def test_callback_rl_environment(self):
        step_counter = {"n": 0}

        def reset_fn():
            step_counter["n"] = 0
            return np.zeros(3, dtype=np.float32)

        def step_fn(action: int):
            step_counter["n"] += 1
            done = step_counter["n"] >= 5
            return np.ones(3, dtype=np.float32) * action, 1.5, done

        env = CallbackRLEnvironment(state_size=3, action_size=2, reset_fn=reset_fn, step_fn=step_fn)
        s0 = env.reset()
        self.assertEqual(s0.shape, (3,))
        s1, r, done, _ = env.step(1)
        self.assertEqual(r, 1.5)
        self.assertFalse(done)

    def test_target_seek_environment(self):
        env = TargetSeekEnvironment(max_steps=10, seed=123)
        factory = AIFactory({"input_size": env.state_size, "output_size": env.action_size, "hidden_sizes": [8]})
        model = factory.create_Model()
        score = env.evaluate_model(model)
        self.assertIsInstance(score, float)


if __name__ == "__main__":
    unittest.main()

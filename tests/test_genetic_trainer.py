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
    EarlyStoppingCallback,
    GeneticTrainer,
    HistoryCallback,
    SimpleNN_Pytorch,
    crossover,
    genetic_algorithm,
    get_weights_np,
    initialize_population,
    load_genetic_model,
    mutate,
    save_genetic_model,
    select_parents,
    set_weights_np,
)


class TestGeneticTrainer(unittest.TestCase):
    def setUp(self):
        self.input_size = 6
        self.output_size = 2
        self.hidden_sizes = [8, 8]

    def test_population_crossover_and_mutation(self):
        pop = initialize_population("Pytorch", 4, self.input_size, self.output_size, self.hidden_sizes)
        self.assertEqual(len(pop), 4)

        p1, p2 = pop[0], pop[1]
        set_weights_np(p1, [np.ones_like(w) * 5.0 for w in get_weights_np(p1)])
        set_weights_np(p2, [np.ones_like(w) * 15.0 for w in get_weights_np(p2)])

        c1, c2 = crossover(p1, p2, "Pytorch", self.input_size, self.output_size, self.hidden_sizes)
        for w in get_weights_np(c1):
            self.assertTrue(np.all((w == 5.0) | (w == 15.0)))

        before = [w.copy() for w in get_weights_np(c2)]
        mutate(c2, mutation_rate=1.0, mutation_strength=0.5, strong_mutation_rate=0.5)
        for b, a in zip(before, get_weights_np(c2)):
            self.assertFalse(np.array_equal(b, a))

        parents = select_parents(pop, [5.0, 20.0, 50.0, 10.0], num_parents=2)
        self.assertEqual(len(parents), 2)

    def test_genetic_trainer_with_callable_and_callbacks(self):
        history_cb = HistoryCallback()
        early_cb = EarlyStoppingCallback(patience=2)

        trainer = GeneticTrainer(
            input_size=self.input_size,
            output_size=self.output_size,
            hidden_sizes=self.hidden_sizes,
            generations=10,
            population_size=6,
            num_trials=1,
            adaptive_mutation=True,
            diversity_interval=2,
            callbacks=[history_cb, early_cb],
            verbose=False,
        )

        # Constant fitness triggers adaptive mutation increase and early stopping after patience=2
        result = trainer.train_with_result(lambda model: 10.0)
        self.assertIsInstance(result.best_model, SimpleNN_Pytorch)
        self.assertTrue(result.stopped_early)
        self.assertEqual(len(history_cb.best_scores), result.iterations_completed)
        self.assertGreater(trainer.mutation_rate, trainer.initial_mutation_rate)

    def test_backward_compatible_genetic_algorithm_function(self):
        mock_game = MagicMock()
        mock_game.exit = False
        mock_game.run_with_model.return_value = 25.0

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = os.path.join(tmp_dir, "ga.pth")
            best = genetic_algorithm(
                game=mock_game,
                type_model="Pytorch",
                input_size=self.input_size,
                output_size=self.output_size,
                generations=2,
                population_size=4,
                num_trials=1,
                hidden_sizes=self.hidden_sizes,
                verbose=False,
            )
            self.assertIsNotNone(best)
            self.assertIs(mock_game.model, best)

            save_genetic_model(best, optimizer=None, path=path)
            loaded = load_genetic_model(path, "Pytorch", self.input_size, self.output_size, hidden_sizes=self.hidden_sizes)
            for w1, w2 in zip(get_weights_np(best), get_weights_np(loaded)):
                np.testing.assert_allclose(w1, w2, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()

import os
import sys
import tempfile
import unittest
import numpy as np
import torch

SRC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from pyforgeai import AILoader, SimpleNN_Pytorch, load_model, save_model


class TestSerializationAndLoader(unittest.TestCase):
    def test_auto_architecture_detection_on_load(self):
        net = SimpleNN_Pytorch(input_size=8, output_size=4, hidden_sizes=[24, 12])
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = os.path.join(tmp_dir, "auto_arch.pth")
            save_model(net, path, metadata={"game": "Snake"})

            # Load without specifying input_size, output_size, or hidden_sizes
            loaded = load_model(path, "Pytorch")
            self.assertIsNotNone(loaded)
            self.assertEqual(loaded.input_size, 8)
            self.assertEqual(loaded.output_size, 4)
            self.assertEqual(loaded.hidden_sizes, [24, 12])

    def test_legacy_fc1_fc2_checkpoint_remapping(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = os.path.join(tmp_dir, "legacy_fc.pth")
            legacy_state = {
                "fc1.weight": torch.ones(10, 6),
                "fc1.bias": torch.zeros(10),
                "fc2.weight": torch.ones(2, 10) * 0.5,
                "fc2.bias": torch.zeros(2),
            }
            torch.save(legacy_state, path)

            loaded = load_model(path, "Pytorch", input_size=6, output_size=2)
            self.assertIsNotNone(loaded)
            self.assertEqual(loaded.hidden_sizes, [10])
            np.testing.assert_allclose(loaded.hidden_layers[0].weight.detach().numpy(), np.ones((10, 6)))
            np.testing.assert_allclose(loaded.output_layer.weight.detach().numpy(), np.ones((2, 10)) * 0.5)

    def test_ai_loader_with_config_object(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            ai_dir = os.path.join(tmp_dir, "AI")
            os.makedirs(ai_dir, exist_ok=True)
            model_path = os.path.join(ai_dir, "best_model.pth")

            net = SimpleNN_Pytorch(input_size=6, output_size=2, hidden_sizes=[6, 6])
            save_model(net, model_path)

            class DummyConfig:
                base_dir = tmp_dir
                config_AI = {
                    "type_training": {"Genetic": True, "Q-learning": False},
                    "type_model": {"Pytorch": True, "Tensorflow": False},
                    "nn": {"hidden_layers": 2, "neurons_per_layer": 6},
                }

            loader = AILoader(DummyConfig(), input_size=6, output_size=2)
            res = loader.load_model_result()
            self.assertTrue(res.model_found)
            self.assertIsNotNone(res.model)
            self.assertIsNone(res.error_message)


if __name__ == "__main__":
    unittest.main()

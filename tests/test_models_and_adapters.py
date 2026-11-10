import os
import sys
import tempfile
import unittest
import numpy as np
import torch

SRC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from pyforgeai import (
    AIController,
    MockAdapter,
    PyTorchAdapter,
    SimpleNN,
    SimpleNN_Pytorch,
    TensorFlowAdapter,
    build_model,
    clone_model,
    get_weights_np,
    infer_architecture,
    set_weights_np,
    wrap_model,
)


class TestModelsAndAdapters(unittest.TestCase):
    def test_pytorch_network_forward_and_activations(self):
        net_default = SimpleNN_Pytorch(input_size=6, output_size=2)
        x_default = torch.randn(1, 6)
        out_default = net_default(x_default)
        self.assertEqual(out_default.shape, (1, 2))
        self.assertIsNotNone(net_default.activations)
        self.assertEqual(net_default.activations.shape, (1, 128))
        self.assertIs(net_default.fc1, net_default.hidden_layers[0])

        net_custom = SimpleNN(input_size=6, output_size=3, hidden_sizes=[16, 8, 4], activation="tanh")
        x_custom = torch.randn(4, 6)
        out_custom = net_custom(x_custom)
        self.assertEqual(out_custom.shape, (4, 3))
        self.assertEqual(len(net_custom.hidden_layers), 3)
        act = net_custom.activations
        self.assertIsNotNone(act)
        self.assertTrue(np.all(act >= 0.0) and np.all(act <= 1.0))

    def test_headless_disable_activation_recording(self):
        net_fast = SimpleNN_Pytorch(input_size=4, output_size=2, hidden_sizes=[8], record_activations=False)
        _ = net_fast(torch.randn(1, 4))
        self.assertIsNone(net_fast.activations)

    def test_clone_and_infer_architecture(self):
        orig = build_model("Pytorch", 5, 3, hidden_sizes=[12, 6])
        backend, in_sz, out_sz, hidden = infer_architecture(orig)
        self.assertEqual(backend, "Pytorch")
        self.assertEqual(in_sz, 5)
        self.assertEqual(out_sz, 3)
        self.assertEqual(hidden, [12, 6])

        cloned = clone_model(orig)
        self.assertIsNot(orig, cloned)
        for w1, w2 in zip(get_weights_np(orig), get_weights_np(cloned)):
            np.testing.assert_allclose(w1, w2)

    def test_pytorch_adapter_predict_and_persistence(self):
        model = SimpleNN_Pytorch(input_size=6, output_size=2, hidden_sizes=[8])
        adapter = PyTorchAdapter(model)
        state = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float32)

        pred = adapter.predict(state)
        self.assertEqual(pred.shape, (2,))
        probs = adapter.predict_probabilities(state)
        self.assertAlmostEqual(float(np.sum(probs)), 1.0, places=5)
        action = adapter.predict_action(state, deterministic=True)
        self.assertIn(action, [0, 1])
        self.assertIsNotNone(adapter.get_activations())

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = os.path.join(tmp_dir, "adapter_weights.pth")
            adapter.save(path)

            new_adapter = PyTorchAdapter(SimpleNN_Pytorch(input_size=6, output_size=2, hidden_sizes=[8]))
            new_adapter.load(path)
            np.testing.assert_allclose(adapter.predict(state), new_adapter.predict(state), rtol=1e-5)

    def test_mock_adapter_and_controller(self):
        mock = MockAdapter(output_size=4, fixed_output=[0.1, 0.9, -0.2, 0.0])
        self.assertEqual(mock.predict_action([1.0, 2.0]), 1)

        dispatched = []
        controller = AIController(
            state_provider=lambda: np.array([0.5, -0.5], dtype=np.float32),
            action_dispatcher=lambda act: dispatched.append(act),
            output_size=4,
        )
        controller.set_model(mock)
        chosen = controller.select_action()
        self.assertEqual(chosen, 1)
        self.assertEqual(dispatched, [1])

    def test_tensorflow_network_and_adapter_if_installed(self):
        try:
            import tensorflow as tf
            from pyforgeai import SimpleNN_Tensorflow

            net_tf = SimpleNN_Tensorflow(input_size=6, output_size=2, hidden_sizes=[16, 8])
            adapter = TensorFlowAdapter(net_tf)
            state = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float32)
            out = adapter.predict(state)
            self.assertEqual(out.shape, (2,))
            self.assertIs(wrap_model(net_tf).get_internal_model(), net_tf)
        except ImportError:
            self.skipTest("TensorFlow is not installed in the current environment.")


if __name__ == "__main__":
    unittest.main()

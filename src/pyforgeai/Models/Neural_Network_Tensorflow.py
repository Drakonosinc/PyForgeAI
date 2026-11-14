from __future__ import annotations

from typing import Any, Sequence
import numpy as np


class SimpleNN_Tensorflow:
    """Configurable multi-layer perceptron in TensorFlow/Keras with normalized activation tracking and lazy TF loading."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_sizes: Sequence[int] | None = None,
        activation: str = "relu",
        record_activations: bool = True,
    ) -> None:
        try:
            import tensorflow as tf
            from tensorflow.keras import layers
        except ImportError as exc:
            raise ImportError("TensorFlow is not installed in the current environment.") from exc

        if input_size <= 0 or output_size <= 0:
            raise ValueError("input_size and output_size must be positive integers.")

        if hidden_sizes is None or len(hidden_sizes) == 0:
            hidden_sizes = [128]

        self._tf = tf
        self.input_size = int(input_size)
        self.output_size = int(output_size)
        self.hidden_sizes = [int(h) for h in hidden_sizes]
        self.activation_name = activation.lower()
        self.record_activations = bool(record_activations)

        self.hidden_layers = [layers.Dense(h, activation=self.activation_name) for h in self.hidden_sizes]
        self.output_layer = layers.Dense(self.output_size)
        self.fc1 = self.hidden_layers[0]
        self.activations: np.ndarray | None = None
        self.layer_activations: list[np.ndarray] | None = None

        inputs = tf.keras.Input(shape=(self.input_size,), dtype=tf.float32)
        x = inputs
        for layer in self.hidden_layers:
            x = layer(x)
        outputs = self.output_layer(x)
        self._keras_model = tf.keras.Model(inputs=inputs, outputs=outputs)
        self.built = True

    @staticmethod
    def _normalize_vec(vec: np.ndarray) -> np.ndarray:
        v = np.asarray(vec, dtype=np.float32).flatten()
        if v.size == 0:
            return v
        vmin = float(np.min(v))
        vmax = float(np.max(v))
        span = vmax - vmin
        if span > 1e-6:
            return (v - vmin) / span
        return np.clip(np.abs(v), 0.0, 1.0)

    @property
    def trainable_variables(self) -> Any:
        return self._keras_model.trainable_variables

    @property
    def weights(self) -> Any:
        return self._keras_model.weights

    def get_weights(self) -> list[np.ndarray]:
        return self._keras_model.get_weights()

    def set_weights(self, weights: Sequence[np.ndarray]) -> None:
        self._keras_model.set_weights([np.array(w, copy=False) for w in weights])

    def save(self, path: str) -> None:
        self._keras_model.save(path)

    def forward(self, x: Any) -> Any:
        tf = self._tf
        x_tensor = tf.convert_to_tensor(x, dtype=tf.float32)
        if len(x_tensor.shape) == 1:
            x_tensor = tf.expand_dims(x_tensor, axis=0)

        is_single = x_tensor.shape[0] == 1
        should_record = self.record_activations and (is_single or self.layer_activations is None)
        recorded_layers: list[np.ndarray] = []
        if should_record:
            try:
                recorded_layers.append(self._normalize_vec(x_tensor[0].numpy()))
            except Exception:
                should_record = False

        for i, layer in enumerate(self.hidden_layers):
            x_tensor = layer(x_tensor)
            if should_record:
                try:
                    norm_h = self._normalize_vec(x_tensor[0].numpy())
                    recorded_layers.append(norm_h)
                    if i == 0:
                        self.activations = norm_h.reshape(1, -1)
                except Exception:
                    pass

        out = self.output_layer(x_tensor)
        if should_record:
            try:
                recorded_layers.append(self._normalize_vec(out[0].numpy()))
                self.layer_activations = recorded_layers
            except Exception:
                pass
        return out

    def call(self, x: Any, training: bool = False) -> Any:
        return self.forward(x)

    def __call__(self, x: Any, training: bool = False) -> Any:
        return self.forward(x)

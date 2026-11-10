from __future__ import annotations

from typing import Any
import numpy as np

from ..Interfaces.IAIModel import IAIModel
from ..Models.Model_Builder import get_weights_np, load_tensorflow, set_weights_np


class TensorFlowAdapter(IAIModel):
    """Adapter wrapping a TensorFlow/Keras Model under the unified IAIModel interface."""

    def __init__(self, model: Any) -> None:
        self.tf = load_tensorflow(optional=False)
        self.model = model

    def predict(self, state: np.ndarray | list[float]) -> np.ndarray:
        x = self.tf.convert_to_tensor(np.asarray(state, dtype=np.float32), dtype=self.tf.float32)
        if len(x.shape) == 1:
            x = self.tf.expand_dims(x, axis=0)
        out = self.model(x, training=False)
        return out.numpy().flatten()

    def save(self, path: str) -> None:
        if path.endswith(".npz"):
            weights = get_weights_np(self.model)
            np.savez(path, *weights)
        elif hasattr(self.model, "save"):
            self.model.save(path)

    def load(self, path: str) -> None:
        if path.endswith(".npz"):
            data = np.load(path)
            keys = sorted(data.files, key=lambda k: int(k.replace("arr_", "")))
            set_weights_np(self.model, [data[k] for k in keys])
        else:
            self.model = self.tf.keras.models.load_model(path)

    def get_internal_model(self) -> Any:
        return self.model

    def get_weights(self) -> list[np.ndarray]:
        return get_weights_np(self.model)

    def set_weights(self, weights: list[np.ndarray]) -> None:
        set_weights_np(self.model, weights)

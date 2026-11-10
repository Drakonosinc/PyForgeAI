from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any
import numpy as np


class IAIModel(ABC):
    """Unified contract for AI model adapters across backends (PyTorch, TensorFlow, Mock)."""

    @abstractmethod
    def predict(self, state: np.ndarray | list[float]) -> np.ndarray:
        """Run a forward pass on a 1D or 2D state array and return raw output logits/values."""
        pass

    def predict_probabilities(self, state: np.ndarray | list[float]) -> np.ndarray:
        """Return softmax-normalized action probabilities from raw model predictions."""
        logits = np.asarray(self.predict(state), dtype=np.float32).flatten()
        shifted = logits - np.max(logits)
        exp_scores = np.exp(shifted)
        total = np.sum(exp_scores)
        if total == 0.0 or not np.isfinite(total):
            return np.full_like(logits, 1.0 / max(1, len(logits)))
        return exp_scores / total

    def predict_action(self, state: np.ndarray | list[float], deterministic: bool = True) -> int:
        """Select a discrete action index from the model output."""
        if deterministic:
            raw = np.asarray(self.predict(state), dtype=np.float32).flatten()
            return int(np.argmax(raw))
        probs = self.predict_probabilities(state)
        return int(np.random.choice(len(probs), p=probs))

    @abstractmethod
    def save(self, path: str) -> None:
        """Persist model weights/state to disk."""
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        """Restore model weights/state from disk."""
        pass

    @abstractmethod
    def get_internal_model(self) -> Any:
        """Return the underlying framework model (e.g., torch.nn.Module or tf.keras.Model)."""
        pass

    def get_weights(self) -> list[np.ndarray]:
        """Extract all trainable parameter tensors as a list of detached NumPy arrays."""
        raise NotImplementedError("Weight extraction is not implemented for this adapter.")

    def set_weights(self, weights: list[np.ndarray]) -> None:
        """Inject a list of NumPy parameter arrays into the underlying model."""
        raise NotImplementedError("Weight assignment is not implemented for this adapter.")

    def get_activations(self) -> np.ndarray | None:
        """Return normalized hidden-layer activations (shape: [1, N] in [0, 1]) for UI visualization."""
        model = self.get_internal_model()
        return getattr(model, "activations", None) if model is not None else None

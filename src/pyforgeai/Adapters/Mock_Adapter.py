from __future__ import annotations

import numpy as np

from ..Interfaces.IAIModel import IAIModel


class MockAdapter(IAIModel):
    """Lightweight mock model adapter for testing game loops and UI without neural network overhead."""

    def __init__(self, output_size: int = 2, fixed_output: np.ndarray | list[float] | None = None) -> None:
        self.output_size = int(output_size)
        self.fixed_output = (
            np.asarray(fixed_output, dtype=np.float32).flatten() if fixed_output is not None else None
        )
        self.saved_paths: list[str] = []
        self.loaded_paths: list[str] = []

    def predict(self, state: np.ndarray | list[float]) -> np.ndarray:
        if self.fixed_output is not None:
            return self.fixed_output.copy()
        return np.random.uniform(-1.0, 1.0, self.output_size).astype(np.float32)

    def save(self, path: str) -> None:
        self.saved_paths.append(path)

    def load(self, path: str) -> None:
        self.loaded_paths.append(path)

    def get_internal_model(self) -> None:
        return None

from __future__ import annotations

import numpy as np
import torch

from ..Interfaces.IAIModel import IAIModel
from ..Models.Model_Builder import get_weights_np, set_weights_np


class PyTorchAdapter(IAIModel):
    """Adapter wrapping a PyTorch nn.Module under the unified IAIModel interface."""

    def __init__(self, model: torch.nn.Module, device: str | torch.device | None = None) -> None:
        self.model = model
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        self.model.to(self.device)

    def predict(self, state: np.ndarray | list[float]) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            state_arr = np.asarray(state, dtype=np.float32)
            state_tensor = torch.as_tensor(state_arr, dtype=torch.float32, device=self.device)
            output = self.model(state_tensor)
            return output.detach().cpu().numpy().flatten()

    def save(self, path: str) -> None:
        torch.save(self.model.state_dict(), path)

    def load(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        self.model.load_state_dict(state_dict)

    def get_internal_model(self) -> torch.nn.Module:
        return self.model

    def get_weights(self) -> list[np.ndarray]:
        return get_weights_np(self.model)

    def set_weights(self, weights: list[np.ndarray]) -> None:
        set_weights_np(self.model, weights)

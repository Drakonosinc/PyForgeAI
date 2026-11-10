from __future__ import annotations

from typing import Sequence
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


_ACTIVATION_MAP = {
    "relu": torch.relu,
    "tanh": torch.tanh,
    "sigmoid": torch.sigmoid,
    "leaky_relu": F.leaky_relu,
    "elu": F.elu,
}


class SimpleNN_Pytorch(nn.Module):
    """Configurable multi-layer perceptron in PyTorch with normalized activation tracking for HUD visualization."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_sizes: Sequence[int] | None = None,
        activation: str = "relu",
        record_activations: bool = True,
    ) -> None:
        super().__init__()
        if input_size <= 0 or output_size <= 0:
            raise ValueError("input_size and output_size must be positive integers.")

        if hidden_sizes is None or len(hidden_sizes) == 0:
            hidden_sizes = [128]

        self.input_size = int(input_size)
        self.output_size = int(output_size)
        self.hidden_sizes = [int(h) for h in hidden_sizes]
        self.activation_name = activation.lower()
        if self.activation_name not in _ACTIVATION_MAP:
            raise ValueError(f"Unsupported activation '{activation}'. Choose from {list(_ACTIVATION_MAP.keys())}.")
        self._activation_fn = _ACTIVATION_MAP[self.activation_name]
        self.record_activations = bool(record_activations)

        self.hidden_layers = nn.ModuleList()
        in_features = self.input_size
        for h in self.hidden_sizes:
            self.hidden_layers.append(nn.Linear(in_features, h))
            in_features = h

        self.output_layer = nn.Linear(in_features, self.output_size)
        # Alias preserved for HUD weight inspection compatibility (e.g. Pong_Game / Snake_Game)
        self.fc1 = self.hidden_layers[0]
        self.activations: np.ndarray | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.hidden_layers):
            x = self._activation_fn(layer(x))
            if i == 0 and self.record_activations:
                with torch.no_grad():
                    act = x.detach().cpu().numpy().reshape(1, -1)
                    span = act.max() - act.min()
                    if span != 0.0:
                        self.activations = (act - act.min()) / span
                    else:
                        self.activations = act
        return self.output_layer(x)


# Alias for backward compatibility with projects importing SimpleNN directly
SimpleNN = SimpleNN_Pytorch

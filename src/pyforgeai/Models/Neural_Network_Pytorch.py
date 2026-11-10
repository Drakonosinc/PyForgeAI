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
        output_activation: str | None = None,
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
        self.output_activation_name = output_activation.lower() if output_activation else None
        if self.output_activation_name is not None and self.output_activation_name not in _ACTIVATION_MAP:
            raise ValueError(
                f"Unsupported output_activation '{output_activation}'. Choose from {list(_ACTIVATION_MAP.keys())}."
            )
        self._output_activation_fn = (
            _ACTIVATION_MAP[self.output_activation_name] if self.output_activation_name else None
        )
        self.record_activations = bool(record_activations)

        self.hidden_layers = nn.ModuleList()
        in_features = self.input_size
        for h in self.hidden_sizes:
            self.hidden_layers.append(nn.Linear(in_features, h))
            in_features = h

        self.output_layer = nn.Linear(in_features, self.output_size)
        # Aliases preserved for HUD weight inspection & legacy compatibility
        self.fc1 = self.hidden_layers[0]
        if len(self.hidden_layers) >= 2:
            self.fc2 = self.hidden_layers[1]
            self.fc3 = self.output_layer
        else:
            self.fc2 = self.output_layer
            self.fc3 = self.output_layer
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
        out = self.output_layer(x)
        if self._output_activation_fn is not None:
            out = self._output_activation_fn(out)
        return out


# Alias for backward compatibility with projects importing SimpleNN directly
SimpleNN = SimpleNN_Pytorch

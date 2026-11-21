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
        self.layer_activations: list[np.ndarray] | None = None

    @staticmethod
    def _normalize_vec(vec: np.ndarray) -> np.ndarray:
        v = np.asarray(vec, dtype=np.float32).flatten()
        if v.size == 0:
            return v
        if v.size == 1:
            return 1.0 / (1.0 + np.exp(-np.clip(v, -15.0, 15.0)))
        vmin = float(np.min(v))
        vmax = float(np.max(v))
        span = vmax - vmin
        if span > 1e-6:
            return (v - vmin) / span
        return 1.0 / (1.0 + np.exp(-np.clip(v, -15.0, 15.0)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        is_single = x.dim() == 1 or (x.dim() >= 2 and x.shape[0] == 1)
        should_record = self.record_activations and (is_single or self.layer_activations is None)

        if should_record:
            with torch.no_grad():
                x0 = x if x.dim() == 1 else x[0]
                recorded_layers: list[np.ndarray] = [self._normalize_vec(x0.detach().cpu().numpy())]

        for i, layer in enumerate(self.hidden_layers):
            x = self._activation_fn(layer(x))
            if should_record:
                with torch.no_grad():
                    h0 = x if x.dim() == 1 else x[0]
                    norm_h = self._normalize_vec(h0.detach().cpu().numpy())
                    recorded_layers.append(norm_h)
                    if i == 0:
                        self.activations = norm_h.reshape(1, -1)

        out = self.output_layer(x)
        if self._output_activation_fn is not None:
            out = self._output_activation_fn(out)

        if should_record:
            with torch.no_grad():
                o0 = out if out.dim() == 1 else out[0]
                recorded_layers.append(self._normalize_vec(o0.detach().cpu().numpy()))
                self.layer_activations = recorded_layers

        return out


# Alias for backward compatibility with projects importing SimpleNN directly
SimpleNN = SimpleNN_Pytorch

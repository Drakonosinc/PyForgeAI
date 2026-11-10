from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Sequence
import numpy as np

try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]

from ..Models.Model_Builder import (
    build_model,
    ensure_built_if_tf,
    get_weights_np,
    infer_architecture,
    is_tf_model,
    is_torch_model,
    load_tensorflow,
    set_weights_np,
)


def save_model(
    model: Any,
    path: str,
    optimizer: Any = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Save a PyTorch or TensorFlow model (and optional optimizer/metadata) to disk."""
    if hasattr(model, "get_internal_model"):
        model = model.get_internal_model()

    parent_dir = os.path.dirname(os.path.abspath(path))
    if parent_dir and not os.path.exists(parent_dir):
        os.makedirs(parent_dir, exist_ok=True)

    if is_torch_model(model):
        _, in_size, out_size, hidden = infer_architecture(model)
        payload: dict[str, Any] = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else {},
            "architecture": {
                "input_size": in_size,
                "output_size": out_size,
                "hidden_sizes": hidden,
            },
        }
        if metadata:
            payload["metadata"] = metadata
        torch.save(payload, path)
        return

    if is_tf_model(model):
        weights = get_weights_np(model)
        target_path = path if path.endswith(".npz") else path + ".npz"
        np.savez(target_path, *weights)
        return

    raise TypeError(f"Unsupported model type for saving: {type(model).__name__}")


def _filtered_torch_load(model: Any, state_dict: dict[str, Any]) -> None:
    target_state = model.state_dict()
    filtered = {
        key: value
        for key, value in state_dict.items()
        if key in target_state and target_state[key].shape == value.shape
    }
    model.load_state_dict(filtered, strict=False)


def load_model(
    path: str,
    model_or_type_model: Any = "Pytorch",
    input_size: int | None = None,
    output_size: int | None = None,
    optimizer: Any = None,
    hidden_sizes: Sequence[int] | None = None,
) -> Any | None:
    """Load a saved PyTorch or TensorFlow model with automatic architecture detection & legacy layer remapping."""
    try:
        # Handle pre-instantiated model passed directly
        if not isinstance(model_or_type_model, str):
            model = model_or_type_model
            if is_torch_model(model):
                checkpoint = torch.load(path, map_location="cpu")
                state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
                _filtered_torch_load(model, state_dict)
                if optimizer is not None and isinstance(checkpoint, dict) and checkpoint.get("optimizer_state_dict"):
                    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                return model
            if is_tf_model(model):
                npz_path = path if path.endswith(".npz") else path + ".npz"
                data = np.load(npz_path)
                keys = sorted(data.files, key=lambda v: int(v.replace("arr_", "")))
                set_weights_np(model, [data[k] for k in keys])
                return model

        type_model = str(model_or_type_model).strip().capitalize()

        if type_model in ("Tensorflow", "Tf", "Keras"):
            load_tensorflow(optional=False)
            npz_path = path if path.endswith(".npz") and os.path.exists(path) else (
                path + ".npz" if os.path.exists(path + ".npz") else path
            )
            data = np.load(npz_path)
            keys = sorted(data.files, key=lambda v: int(v.replace("arr_", "")))
            weights = [data[k] for k in keys]
            in_sz = int(input_size or weights[0].shape[0])
            out_sz = int(output_size or weights[-1].shape[0])
            model = build_model("Tensorflow", in_sz, out_sz, hidden_sizes=hidden_sizes)
            ensure_built_if_tf(model, in_sz)
            set_weights_np(model, weights)
            return model

        if torch is None:
            raise ImportError("PyTorch is not installed in the current environment.")

        checkpoint = torch.load(path, map_location="cpu")
        state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        arch_meta = checkpoint.get("architecture", {}) if isinstance(checkpoint, dict) else {}

        has_fc = any(k.startswith("fc1.") or k.startswith("fc2.") for k in state_dict.keys())
        has_hidden = any(k.startswith("hidden_layers.") for k in state_dict.keys())

        resolved_in = int(input_size or arch_meta.get("input_size") or 6)
        resolved_out = int(output_size or arch_meta.get("output_size") or 2)

        if has_hidden:
            indices = sorted({int(k.split(".")[1]) for k in state_dict.keys() if k.startswith("hidden_layers.")})
            sizes: list[int] = []
            for idx in indices:
                w_key = f"hidden_layers.{idx}.weight"
                if w_key in state_dict:
                    sizes.append(int(state_dict[w_key].shape[0]))
                    if idx == 0 and input_size is None:
                        resolved_in = int(state_dict[w_key].shape[1])
            if "output_layer.weight" in state_dict and output_size is None:
                resolved_out = int(state_dict["output_layer.weight"].shape[0])
            if not sizes:
                sizes = list(hidden_sizes) if hidden_sizes else [128]
            model = build_model(type_model, resolved_in, resolved_out, hidden_sizes=sizes)
            _filtered_torch_load(model, state_dict)
        elif has_fc:
            if "fc1.weight" in state_dict:
                first_hidden = int(state_dict["fc1.weight"].shape[0])
                if input_size is None:
                    resolved_in = int(state_dict["fc1.weight"].shape[1])
            else:
                any_w = next((v for k, v in state_dict.items() if k.endswith(".weight")), None)
                first_hidden = int(any_w.shape[0]) if any_w is not None else 128
            model = build_model(type_model, resolved_in, resolved_out, hidden_sizes=[first_hidden])
            remapped: dict[str, Any] = {}
            if "fc1.weight" in state_dict:
                remapped["hidden_layers.0.weight"] = state_dict["fc1.weight"]
            if "fc1.bias" in state_dict:
                remapped["hidden_layers.0.bias"] = state_dict["fc1.bias"]
            out_w = state_dict.get("fc2.weight")
            out_b = state_dict.get("fc2.bias")
            if out_w is not None and out_b is not None:
                if out_w.shape[0] == resolved_out and out_w.shape[1] == first_hidden and out_b.shape[0] == resolved_out:
                    remapped["output_layer.weight"] = out_w
                    remapped["output_layer.bias"] = out_b
            _filtered_torch_load(model, remapped)
        else:
            model = build_model(type_model, resolved_in, resolved_out, hidden_sizes=hidden_sizes)
            _filtered_torch_load(model, state_dict)

        if optimizer is not None and isinstance(checkpoint, dict) and checkpoint.get("optimizer_state_dict"):
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        return model
    except FileNotFoundError:
        return None
    except Exception:
        return None


# Backward-compatible aliases matching Pong_Game / Snake_Game signatures
def save_genetic_model(model: Any, optimizer: Any, path: str) -> None:
    save_model(model=model, path=path, optimizer=optimizer)


def load_genetic_model(
    path: str,
    type_model: str,
    input_size: int,
    output_size: int,
    optimizer: Any = None,
    hidden_sizes: Sequence[int] | None = None,
) -> Any | None:
    return load_model(
        path=path,
        model_or_type_model=type_model,
        input_size=input_size,
        output_size=output_size,
        optimizer=optimizer,
        hidden_sizes=hidden_sizes,
    )


def save_qlearning_model(model: Any, optimizer: Any, path: str) -> None:
    save_model(model=model, path=path, optimizer=optimizer)


def load_qlearning_model(
    path: str,
    model_or_type_model: Any,
    input_size: int | None = None,
    output_size: int | None = None,
    optimizer: Any = None,
    hidden_sizes: Sequence[int] | None = None,
) -> Any | None:
    return load_model(
        path=path,
        model_or_type_model=model_or_type_model,
        input_size=input_size,
        output_size=output_size,
        optimizer=optimizer,
        hidden_sizes=hidden_sizes,
    )


@dataclass(slots=True)
class AIModelLoadResult:
    """Result object returned by AILoader when attempting to restore a trained model."""

    model: Any = None
    model_found: bool = False
    error_message: str | None = None


class AILoader:
    """Configuration-driven model loader compatible with game context/config dictionaries."""

    def __init__(
        self,
        context_or_config: Any,
        input_size: int = 6,
        output_size: int = 2,
        relative_model_path: str = os.path.join("AI", "best_model.pth"),
    ) -> None:
        if hasattr(context_or_config, "base_dir") and hasattr(context_or_config, "config_AI"):
            self.config = context_or_config
        elif hasattr(context_or_config, "config"):
            self.config = context_or_config.config
        else:
            self.config = context_or_config
        self.base_dir = getattr(self.config, "base_dir", ".")
        self.input_size = int(input_size)
        self.output_size = int(output_size)
        self.relative_model_path = relative_model_path

    def _get_config_ai(self) -> dict[str, Any]:
        if isinstance(self.config, dict):
            return self.config.get("config_AI", self.config)
        return getattr(self.config, "config_AI", {})

    def _get_selected_training(self) -> str:
        cfg_ai = self._get_config_ai()
        training = next((k for k, v in cfg_ai.get("type_training", {}).items() if v), None)
        if training is None:
            raise ValueError("No valid training type was specified in config_AI['type_training'].")
        return training

    def _get_selected_model(self) -> str:
        cfg_ai = self._get_config_ai()
        model_type = next((k for k, v in cfg_ai.get("type_model", {}).items() if v), None)
        if model_type is None:
            raise ValueError("No valid model backend was specified in config_AI['type_model'].")
        return model_type

    def _get_hidden_architecture(self) -> list[int]:
        cfg_ai = self._get_config_ai()
        nn_cfg = cfg_ai.get("nn", {"hidden_layers": 2, "neurons_per_layer": 6})
        neurons = int(nn_cfg.get("neurons_per_layer", 6))
        layers = int(nn_cfg.get("hidden_layers", 2))
        return [neurons] * layers

    def load_model_result(self, custom_path: str | None = None) -> AIModelLoadResult:
        model_path = custom_path or os.path.join(self.base_dir, self.relative_model_path)
        try:
            _ = self._get_selected_training()
            model_type = self._get_selected_model()
            arch = self._get_hidden_architecture()
            if not os.path.exists(model_path):
                return AIModelLoadResult(model=None, model_found=False, error_message=None)
            model = load_model(
                path=model_path,
                model_or_type_model=model_type,
                input_size=self.input_size,
                output_size=self.output_size,
                hidden_sizes=arch,
            )
            if model is None:
                return AIModelLoadResult(
                    model=None,
                    model_found=False,
                    error_message="Failed to deserialize the saved model file.",
                )
            return AIModelLoadResult(model=model, model_found=True, error_message=None)
        except Exception as exc:
            return AIModelLoadResult(model=None, model_found=False, error_message=str(exc))

    def load_model(self, custom_path: str | None = None) -> Any | None:
        return self.load_model_result(custom_path=custom_path).model

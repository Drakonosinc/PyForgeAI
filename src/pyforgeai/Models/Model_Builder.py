from __future__ import annotations

from typing import Any, Sequence
import numpy as np

try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]


def load_tensorflow(optional: bool = True):
    """Lazily import TensorFlow, optionally raising an informative error if missing."""
    try:
        import tensorflow as tf
        return tf
    except ImportError:
        if optional:
            return None
        raise ImportError("TensorFlow is not installed in the current environment.")


def is_torch_model(model: Any) -> bool:
    return torch is not None and hasattr(model, "parameters") and isinstance(model, torch.nn.Module)


def is_tf_model(model: Any) -> bool:
    if not hasattr(model, "trainable_variables") and not hasattr(model, "_keras_model"):
        return False
    if hasattr(model, "_keras_model"):
        return True
    tf = load_tensorflow(optional=True)
    return (tf is not None) and isinstance(model, tf.keras.Model)


def ensure_built_if_tf(model: Any, input_size: int) -> None:
    if is_tf_model(model):
        try:
            if not getattr(model, "built", False) or len(model.weights) == 0:
                tf = load_tensorflow(optional=False)
                dummy = tf.zeros((1, int(input_size)), dtype=tf.float32)
                _ = model(dummy, training=False)
        except Exception:
            pass


def build_model(
    type_model: str,
    input_size: int,
    output_size: int,
    hidden_sizes: Sequence[int] | None = None,
    activation: str = "relu",
    record_activations: bool = True,
    output_activation: str | None = None,
) -> Any:
    """Instantiate a neural network for the requested backend ('Pytorch' or 'Tensorflow')."""
    normalized = type_model.strip().capitalize()
    if normalized in ("Pytorch", "Torch"):
        if torch is None:
            raise ImportError("PyTorch is not installed in the current environment.")
        from .Neural_Network_Pytorch import SimpleNN_Pytorch

        return SimpleNN_Pytorch(
            input_size=input_size,
            output_size=output_size,
            hidden_sizes=hidden_sizes,
            activation=activation,
            record_activations=record_activations,
            output_activation=output_activation,
        )
    if normalized in ("Tensorflow", "Tf", "Keras"):
        load_tensorflow(optional=False)
        from .Neural_Network_Tensorflow import SimpleNN_Tensorflow

        model = SimpleNN_Tensorflow(
            input_size=input_size,
            output_size=output_size,
            hidden_sizes=hidden_sizes,
            activation=activation,
            record_activations=record_activations,
        )
        ensure_built_if_tf(model, input_size)
        return model
    raise ValueError(f"Unknown type_model: '{type_model}'. Supported backends: 'Pytorch', 'Tensorflow'.")


def get_weights_np(model: Any) -> list[np.ndarray]:
    """Extract all parameters from a PyTorch or TensorFlow model as a list of NumPy arrays."""
    if hasattr(model, "get_internal_model"):
        model = model.get_internal_model()
    if is_torch_model(model):
        return [p.detach().cpu().numpy().copy() for p in model.parameters()]
    if is_tf_model(model):
        return [w.copy() for w in model.get_weights()]
    raise TypeError(f"Unsupported model type for weight extraction: {type(model).__name__}")


def set_weights_np(model: Any, weights: Sequence[np.ndarray]) -> None:
    """Copy a sequence of NumPy weight arrays into a PyTorch or TensorFlow model in-place."""
    if hasattr(model, "get_internal_model"):
        model = model.get_internal_model()
    if is_torch_model(model):
        with torch.no_grad():
            for p, w in zip(model.parameters(), weights):
                tensor_w = torch.from_numpy(np.array(w, copy=False)).to(p.device, dtype=p.dtype)
                p.copy_(tensor_w)
        return
    if is_tf_model(model):
        model.set_weights([np.array(w, copy=False) for w in weights])
        return
    raise TypeError(f"Unsupported model type for weight assignment: {type(model).__name__}")


def infer_architecture(model: Any) -> tuple[str, int, int, list[int]]:
    """Infer (type_model, input_size, output_size, hidden_sizes) from an existing model instance."""
    if hasattr(model, "get_internal_model"):
        model = model.get_internal_model()
    type_model = "Pytorch" if is_torch_model(model) else ("Tensorflow" if is_tf_model(model) else "Unknown")
    input_size = getattr(model, "input_size", None)
    output_size = getattr(model, "output_size", None)
    hidden_sizes = getattr(model, "hidden_sizes", None)

    if is_torch_model(model):
        if input_size is None and hasattr(model, "hidden_layers") and len(model.hidden_layers) > 0:
            input_size = int(model.hidden_layers[0].in_features)
        if output_size is None and hasattr(model, "output_layer"):
            output_size = int(model.output_layer.out_features)
        if hidden_sizes is None and hasattr(model, "hidden_layers"):
            hidden_sizes = [int(layer.out_features) for layer in model.hidden_layers]

    return (
        type_model,
        int(input_size or 1),
        int(output_size or 1),
        list(hidden_sizes) if hidden_sizes is not None else [128],
    )


def clone_model(
    model: Any,
    type_model: str | None = None,
    input_size: int | None = None,
    output_size: int | None = None,
    hidden_sizes: Sequence[int] | None = None,
) -> Any:
    """Create a deep copy of a model preserving architecture and weights."""
    inferred_type, inferred_in, inferred_out, inferred_hidden = infer_architecture(model)
    target_type = type_model or inferred_type
    target_in = input_size or inferred_in
    target_out = output_size or inferred_out
    target_hidden = list(hidden_sizes) if hidden_sizes is not None else inferred_hidden
    record_act = getattr(model, "record_activations", True)
    activation = getattr(model, "activation_name", "relu")
    output_activation = getattr(model, "output_activation_name", None)

    cloned = build_model(
        target_type,
        target_in,
        target_out,
        hidden_sizes=target_hidden,
        activation=activation,
        record_activations=record_act,
        output_activation=output_activation,
    )
    set_weights_np(cloned, get_weights_np(model))
    return cloned

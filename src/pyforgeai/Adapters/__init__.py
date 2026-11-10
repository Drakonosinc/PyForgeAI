from __future__ import annotations

from typing import Any

from ..Interfaces.IAIModel import IAIModel
from ..Models.Model_Builder import is_tf_model, is_torch_model
from .Mock_Adapter import MockAdapter
from .PyTorch_Adapter import PyTorchAdapter
from .TensorFlow_Adapter import TensorFlowAdapter


def wrap_model(model: Any, output_size: int = 2) -> IAIModel:
    """Automatically wrap a raw PyTorch or TensorFlow model into its corresponding IAIModel adapter."""
    if isinstance(model, IAIModel):
        return model
    if is_torch_model(model):
        return PyTorchAdapter(model)
    if is_tf_model(model):
        return TensorFlowAdapter(model)
    return MockAdapter(output_size=output_size)


__all__ = [
    "MockAdapter",
    "PyTorchAdapter",
    "TensorFlowAdapter",
    "wrap_model",
]

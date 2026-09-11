"""Python launcher and experimental typed facade for the DEEPBOM engine."""

__version__ = "1.96.15"

from .api import (  # noqa: E402
    DeepBomError,
    DeepBomIncompleteBinding,
    DeepBomInvocationError,
    DeepBomOutputTooLarge,
    DeepBomPolicyBlocked,
    DeepBomTimeout,
    audit,
    capabilities,
    tensor_inventory,
    tensors,
)

__all__ = [
    "__version__",
    "audit",
    "capabilities",
    "tensors",
    "tensor_inventory",
    "DeepBomError",
    "DeepBomInvocationError",
    "DeepBomPolicyBlocked",
    "DeepBomIncompleteBinding",
    "DeepBomTimeout",
    "DeepBomOutputTooLarge",
]

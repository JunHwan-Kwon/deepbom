"""Python launcher and experimental typed facade for the DEEPBOM engine."""

__version__ = "1.97.0"

from .api import (  # noqa: E402
    DeepBomError,
    DeepBomIncompleteBinding,
    DeepBomIdentityMismatch,
    DeepBomInvocationError,
    DeepBomOutputTooLarge,
    DeepBomPolicyBlocked,
    DeepBomTimeout,
    audit,
    capture_contract,
    capabilities,
    diff,
    tensor_inventory,
    tensors,
    verify_bom,
    verify_contract,
)

__all__ = [
    "__version__",
    "audit",
    "capabilities",
    "capture_contract",
    "diff",
    "tensors",
    "tensor_inventory",
    "verify_bom",
    "verify_contract",
    "DeepBomError",
    "DeepBomInvocationError",
    "DeepBomPolicyBlocked",
    "DeepBomIncompleteBinding",
    "DeepBomIdentityMismatch",
    "DeepBomTimeout",
    "DeepBomOutputTooLarge",
]

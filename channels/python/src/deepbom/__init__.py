"""Python launcher and experimental typed facade for the DEEPBOM engine."""

__version__ = "1.98.0"

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
    model_ir,
    tensor_inventory,
    tensors,
    visualization_manifest,
    verify_bom,
    verify_contract,
)

__all__ = [
    "__version__",
    "audit",
    "capabilities",
    "capture_contract",
    "diff",
    "model_ir",
    "tensors",
    "tensor_inventory",
    "visualization_manifest",
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

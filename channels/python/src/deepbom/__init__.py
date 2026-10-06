"""Public Python SDK and launcher for the verified DEEPBOM engine."""

__version__ = "2.1.0"

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
    inspect,
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
    "inspect",
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

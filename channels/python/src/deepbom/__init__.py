"""Public Python SDK and launcher for the verified DEEPBOM engine."""

__version__ = "2.5.0"

from .inspection import inspect_native
from .autolog import autolog


def monitor(model, **options):
    from .training.session import monitor as create
    return create(model, **options)

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
    evidence_workflow,
    inspect,
    model_ir,
    optimization_query,
    numerical_evidence,
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
    "evidence_workflow",
    "inspect",
    "inspect_native",
    "monitor",
    "autolog",
    "model_ir",
    "optimization_query",
    "numerical_evidence",
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

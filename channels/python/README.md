# DEEPBOM Python SDK and launcher

This package is a zero-analysis-logic launcher for the same platform-specific
DEEPBOM engine used by the npm CLI release. Each wheel binds its operating
system, architecture, engine SHA-256, and canonical TFLite WASM SHA-256 in an
installed manifest. No parser or numerical rule is reimplemented in Python.

The public SDK invokes that same verified engine and converts its
JSON and exit-code contracts into Python values and exceptions:

The `inspect()` method and supported SDK policy are included from **2.1.0**.
Install the matching wheel for your platform from the release assets, or build
from source. Previously published 2.0.0 wheels do not include these additions. See the
[SDK contract](https://github.com/JunHwan-Kwon/deepbom/blob/main/docs/SDK_CONTRACT.md)
and [three runnable examples](https://github.com/JunHwan-Kwon/deepbom/blob/main/examples/integrations/README.md).

```python
from deepbom import (
    DeepBomPolicyBlocked, audit, inspect, capabilities, capture_contract, diff,
    tensor_inventory, tensors, verify_bom, verify_contract,
)

caps = capabilities()
envelope = audit("model.gguf")
summary = inspect("model.gguf")
selected = audit("model.gguf", sections=["summary", "findings"])
try:
    gated = audit("model.safetensors", gate="defects")
except DeepBomPolicyBlocked as blocked:
    gated = blocked.document
rows = tensors("model.gguf")
inventory = tensor_inventory("model.gguf")
comparison = diff("baseline.gguf", "candidate.gguf", tensors_only=True)
baseline = capture_contract("model.onnx")
contract_result = verify_contract("candidate.onnx", "baseline.interface-contract.json")
bom_result = verify_bom("model.onnx", "supplied.cdx.json")
```

`audit()` defaults to the canonical envelope, or to analysis selection when
`sections` are supplied without an explicit `output`. It accepts the CLI-equivalent
`gate="defects"` and `policy="engineering"|"regulatory"` controls plus explicit
timeout and maximum-output-byte bounds. `gate` and `policy` are mutually exclusive.
Exit 1 raises `DeepBomInvocationError`, exit 2
raises `DeepBomPolicyBlocked`, and exit 3 raises
`DeepBomIncompleteBinding`; exit 4 raises `DeepBomIdentityMismatch`. Policy
exceptions retain any completed JSON result
as `.document`. `tensors()` converts exact counts to Python `int` and decimal
ratios to `decimal.Decimal`; `tensor_inventory()` preserves the raw JSON document.
The SDK follows package semantic versioning from its first supported release;
native CLI/IR schemas retain their independent compatibility contracts.
`inspect()` returns the engine's `deepbom.review_summary.v1` unchanged, including
identity, coverage, separate finding classes and a hash-bound reinspection
command. Python dictionaries preserve the native JSON keys and null values.
The package includes `py.typed`; it does not provide separate parser logic.

From 2.3.0, `evidence_workflow(request, files={sha256: local_path})` connects
Snapshot IR, provenance, evaluation conditions and before/after checks through
the same common engine. `deepbom.evidence.export` explicitly sends the resulting
JSON to an existing MLflow or W&B run, or projects values into TensorBoard.
See the [workflow guide](https://github.com/JunHwan-Kwon/deepbom/blob/main/docs/evidence-ir/SNAPSHOT_WORKFLOW.md)
and [synthetic example](https://github.com/JunHwan-Kwon/deepbom/tree/main/examples/evidence-workflow).
Native model execution remains a separate, explicit local operation.

```console
deepbom audit model.tflite --compact
deepbom audit model.onnx --format cyclonedx
deepbom audit model.onnx --format sarif --output deepbom.sarif --fail-on high
deepbom capabilities --compact
deepbom audit Model.mlpackage --compact
deepbom audit safetensors-repository/ --compact
deepbom audit model.pte --executorch-build deepbom.executorch-build.json --compact
deepbom verify model.onnx --bom supplied.cdx.json --render markdown
deepbom contract capture model.onnx -o baseline.interface-contract.json
deepbom capabilities --format agent-text
```

ONNX external data next to the model is discovered only from safe serialized
references. Use `--external-data-dir` to bind a different explicit root.
ExecuTorch PTE audits can additionally bind a selected source/build and binary
inventory with `--executorch-build`; backend execution remains unobserved.

Set `DEEPBOM_ENGINE` only for a deliberately supplied engine build and bind it
with `DEEPBOM_ENGINE_SHA256`. TensorRT options import configuration or observed
parser evidence; NVIDIA runtime libraries are not bundled.

For the installed version, `deepbom --help` and `deepbom capabilities --compact`
are authoritative. The current source inventory is maintained in the
[generated CLI reference](https://github.com/JunHwan-Kwon/deepbom/blob/main/docs/CLI_REFERENCE.md).

This public channel package is licensed under Apache-2.0. Protected analyzers
and private rulepack-generation sources are not included.

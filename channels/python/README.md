# DEEPBOM Python launcher

This package is a zero-analysis-logic launcher for the same platform-specific
DEEPBOM engine used by the npm CLI release. Each wheel binds its operating
system, architecture, engine SHA-256, and canonical TFLite WASM SHA-256 in an
installed manifest. No parser or numerical rule is reimplemented in Python.

An experimental typed facade invokes that same verified engine and converts its
JSON and exit-code contracts into Python values and exceptions:

```python
from deepbom import audit, capabilities, tensor_inventory, tensors

caps = capabilities()
envelope = audit("model.gguf")
rows = tensors("model.gguf")
inventory = tensor_inventory("model.gguf")
```

`audit()` defaults to the canonical envelope. It accepts explicit timeout and
maximum-output-byte bounds. Exit 1 raises `DeepBomInvocationError`, exit 2
raises `DeepBomPolicyBlocked`, and exit 3 raises
`DeepBomIncompleteBinding`; policy exceptions retain any completed JSON result
as `.document`. The facade is experimental in 1.96.x, and the CLI schemas remain
the compatibility contract.

```console
deepbom audit model.tflite --compact
deepbom audit model.onnx --format cyclonedx
deepbom audit model.onnx --format sarif --output deepbom.sarif --fail-on high
deepbom capabilities --compact
deepbom audit Model.mlpackage --compact
deepbom audit safetensors-repository/ --compact
deepbom audit model.pte --executorch-build deepbom.executorch-build.json --compact
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

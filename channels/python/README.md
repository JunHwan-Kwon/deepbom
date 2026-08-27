# DEEPBOM Python launcher

This package is a zero-analysis-logic launcher for the same platform-specific
DEEPBOM engine used by the npm CLI release. Each wheel binds its operating
system, architecture, engine SHA-256, and canonical TFLite WASM SHA-256 in an
installed manifest. No parser or numerical rule is reimplemented in Python.

```console
deepbom audit model.tflite --compact
deepbom audit model.onnx --format cyclonedx
deepbom audit Model.mlpackage --compact
deepbom audit safetensors-repository/ --compact
```

ONNX external data next to the model is discovered only from safe serialized
references. Use `--external-data-dir` to bind a different explicit root.

Set `DEEPBOM_ENGINE` only for a deliberately supplied engine build and bind it
with `DEEPBOM_ENGINE_SHA256`. TensorRT options import configuration or observed
parser evidence; NVIDIA runtime libraries are not bundled.

This public channel package is licensed under Apache-2.0. Protected analyzers
and private rulepack-generation sources are not included.

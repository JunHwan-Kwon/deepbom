# DEEPBOM CLI

Local deployment-artifact analysis for TFLite, ONNX, GGUF, SafeTensors, Core ML,
and ExecuTorch.

```console
npx deepbom audit model.onnx --format cyclonedx
npx deepbom audit model.onnx --format sarif --output deepbom.sarif --fail-on high
npx deepbom capabilities --compact
npx deepbom gguf model.gguf --context 8192 --memory-mib 8192
npx deepbom audit Model.mlpackage --compact
npx deepbom audit safetensors-repository/ --compact
npx deepbom verify model.tflite --contract production-interface.json
npx deepbom diff baseline.tflite candidate.tflite
npx deepbom explore model.tflite --target-profile target-profile.json
npx deepbom audit model.pte --executorch-build deepbom.executorch-build.json --compact
```

The default output is a bounded human-readable summary. Use `--json` or
`--compact` for the complete analysis document, `--format envelope` for the
canonical cross-format contract, `--format cyclonedx` for CycloneDX 1.7, or
`--format sarif` for OASIS SARIF 2.1.0. `--policy-output` records a deterministic
finding gate when `--fail-on` is selected.

`verify` fails closed on interface contradictions, `diff` preserves the
canonical multi-target TFLite delta ledger, and `explore` exposes deterministic
WASM Pareto candidates. These commands do not add a second analysis engine.

The package contains one generated JavaScript analysis bundle and the canonical
TFLite WebAssembly module. It does not send model bytes or results over the
network. TensorRT parser observations and build profiles can be imported with
`--tensorrt-parser-evidence` and `--tensorrt-profile`.

ONNX external data next to the model is discovered only from safe serialized
references. Use `--external-data-dir` to bind a different explicit root.
ExecuTorch PTE audits use the same option for PTD data; `--executorch-build`
binds a duplicate-key-checked selected-build and binary inventory without
promoting it to observed execution.
Core ML packages and sharded SafeTensors repositories are hashed as canonical
multi-file artifact sets rather than collapsed to one unqualified file hash.

Static provider/delegate placement remains predicted or conditionally eligible
unless an identity-bound runtime or parser observation is imported.

For the installed version, `deepbom --help` and `deepbom capabilities --compact`
are authoritative. The current source inventory is maintained in the
[generated CLI reference](https://github.com/JunHwan-Kwon/deepbom/blob/main/docs/CLI_REFERENCE.md).

This public channel package is licensed under Apache-2.0. Protected analyzers
and private rulepack-generation sources are not included.

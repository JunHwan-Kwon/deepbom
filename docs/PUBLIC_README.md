# DEEPBOM

DEEPBOM is a local static analyzer for deployed AI model artifacts. It audits
serialized graph, tensor, quantization, memory, compatibility, and ML-BOM
evidence without uploading model bytes.

The public source distribution covers TFLite, ONNX, GGUF, SafeTensors, Core ML,
and bounded ExecuTorch artifacts. Findings distinguish observed and derived
artifact facts from predicted compatibility, imported runtime evidence, and
values that cannot be assessed statically.

## Quick start

Run the published CLI without cloning the repository (Node.js 20 or newer):

```bash
npx deepbom audit "https://raw.githubusercontent.com/JunHwan-Kwon/deepbom/main/web/samples/gpu_partition_probe.onnx#sha256=82a2feef00eb6ab03d82f2b30cd17f4d826e2d8307cb059eccd6a0f3120059b2"
```

The pinned expected values and independent verifier are in
[`examples/expected-output`](examples/expected-output/README.md). Source builds
require Rust and, for the Python channel, Python 3.9 or newer; maintainer setup
is documented separately below.

Verified release channels expose the same analysis implementation:

```bash
npx deepbom audit model.onnx --format cyclonedx
npx deepbom audit model.onnx --format sarif --output deepbom.sarif --fail-on high
deepbom capabilities --compact
python -m pip install deepbom
cargo install deepbom
deepbom audit model.gguf --compact
deepbom verify model.tflite --contract production-interface.json
deepbom diff baseline.tflite candidate.tflite
deepbom explore model.tflite
deepbom placement model.tflite --profiles xnnpack_cpu,tflite_coreml_delegate,litert_qualcomm_qnn
deepbom graph model.onnx --format json --output artifact-graph.json
```

The default is a terminal-sized evidence summary. `--json` and `--compact`
expose complete format evidence; `--format envelope` provides the canonical
cross-format contract; CycloneDX 1.7 and OASIS SARIF 2.1.0 are standard
projections. `--policy-output` records a hash-bound gate result when `--fail-on`
is selected. `--review-policy` adds identity-scoped, expiring exceptions and
keeps execution, coverage, and finding-policy states independent. See
[`docs/CLI_AUTOMATION.md`](docs/CLI_AUTOMATION.md). The complete
option inventory is generated from the executable in
[`docs/CLI_REFERENCE.md`](docs/CLI_REFERENCE.md).

The graph JSON output includes the evidence-preserving
`deepbom.artifact_ir.v2` ledger and a deterministic `deepbom.graph_ir.v1`
visualization compatibility projection. Serialized graph, storage topology,
architecture grouping, scoped quantization, static placement, and imported
runtime evidence remain separate. Method `2.1.0` materializes exactly decoded
TFLite subgraphs, ONNX nested graphs/local functions, and ExecuTorch primary
plans without flattening conditional scopes. Runtime-node fusion is reconciled
only from artifact-bound subject references or primary native op indices;
names are never guessed. `graph_ir.v1` remains primary-scope-only for legacy
consumers. The v2 JSON Schema is published at
[`docs/schemas/deepbom-artifact-ir-v2.schema.json`](docs/schemas/deepbom-artifact-ir-v2.schema.json)
and at `https://deepbom.org/schemas/deepbom-artifact-ir-v2.schema.json`.

`verify` compares the serialized external tensor ABI with a supplied,
artifact-bound production declaration. `diff` uses the canonical deterministic
multi-target TFLite deployment-delta ledger, and `explore` exposes the existing
WASM redesign Pareto search without claiming trained-model accuracy. A strict
custom TFLite target can be bound with `--target-profile profile.json`; the CLI
records both the source-file SHA-256 and the resolved Rust profile SHA-256.
Accelerator evidence is separate from that CPU cost profile. Source-pinned
TFLite Core ML and LiteRT Qualcomm QNN profiles, Core ML MLComputePlan, Edge TPU
and Qualcomm compiler reports, TensorRT parser/engine evidence, and NVIDIA host
profiles use a shared staged binding without promoting static or compiled
evidence to observed execution. `placement` compares any available profiles
over one conserved graph ledger without inventing backend priority.

The Cargo launcher downloads only the engine matching its exact package version
and platform from the corresponding immutable GitHub Release. It validates the
release matrix, byte lengths, and SHA-256 digests before caching or execution.

Build the npm, standalone, Python, and Cargo launcher channels:

```bash
npm run build:channels
npm run check:public-package-boundary
npm run check:channels -- --no-build
```

Run the public correctness gates:

```bash
npm run check:cli
npm run check:cli-docs
npm run check:cli-automation
npm run check:formats
npm run check:rust
```

The browser workbench is available at [deepbom.org](https://deepbom.org/).

## Evidence scope

| Format | Public static evidence |
| --- | --- |
| TFLite | FlatBuffer graph and tensor contracts, quantization arithmetic, weight integrity, memory projections, accumulator proofs, redesign candidates, and source-bounded delegation predictions |
| ONNX | Protobuf graph, initializer and external-data contracts, symbolic shape inference, operation cost, Q/DQ and affine quantization, and provider-compatible evidence envelopes |
| GGUF | Container and tensor-directory integrity, quantization encoding inventory, architecture metadata, and bounded LLM memory scenarios |
| SafeTensors | Tensor-directory and sharding integrity, configuration-bound architecture contracts, AWQ/GPTQ/HQQ/compressed-tensors metadata, and bounded LLM memory scenarios |
| Core ML | NeuralNetwork and ML Program serialized graphs, tensor/weight encodings, deployment floor, and imported compute-plan evidence boundaries |
| ExecuTorch | Bounded ET12/FT01 plans, source-bound portable calls and processed payload identities, plus optional selected-build/backend/operator/binary attestation; execution remains external |

Static compatibility does not establish observed execution-provider assignment,
device latency, task accuracy, clinical validity, or release readiness. Runtime
claims require an identity-bound runtime capture.

The detailed format and accelerator boundary is maintained in
[`docs/SUPPORT_MATRIX.md`](docs/SUPPORT_MATRIX.md). Bugs can be reported without
sharing model bytes using
[`docs/MODEL_FREE_BUG_REPORTING.md`](docs/MODEL_FREE_BUG_REPORTING.md).

## Distribution boundary

This repository is generated from an exact reviewed allowlist. Private
rulepack generators, hosted-service infrastructure, and unreleased research
modules are not included. The enforceable boundary and export verification
method are documented in
[`docs/PUBLIC_PRIVATE_BOUNDARY.md`](docs/PUBLIC_PRIVATE_BOUNDARY.md).

## License and citation

The public source and release-channel packages are licensed under the Apache
License 2.0. Third-party model artifacts retain their declared licenses.

Please cite:

> Kwon, J. (2026). DEEPBOM: Browser-Native Static Analysis of On-Device Neural
> Network Deployment Artifacts (Version 1.94.0) [Computer software]. Zenodo.
> https://doi.org/10.5281/zenodo.21834509

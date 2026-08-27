# DEEPBOM

DEEPBOM is a local static analyzer for deployed AI model artifacts. It audits
serialized graph, tensor, quantization, memory, compatibility, and ML-BOM
evidence without uploading model bytes.

The public source distribution covers TFLite, ONNX, GGUF, SafeTensors, Core ML,
and bounded ExecuTorch artifacts. Findings distinguish observed and derived
artifact facts from predicted compatibility, imported runtime evidence, and
values that cannot be assessed statically.

## Quick start

Requirements: Node.js 20 or newer, Rust for Rust-source checks, and Python 3.9
or newer when building the Python channel.

```bash
npm ci
node bin/deepbom.mjs audit web/samples/mobilenet_v2_1.0_224_quant.tflite --compact
```

Verified release channels expose the same analysis implementation:

```bash
npx deepbom audit model.onnx --format cyclonedx
python -m pip install deepbom
cargo install deepbom
deepbom audit model.gguf --compact
```

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
| ExecuTorch | Bounded ET12/FT01 program and segment metadata with fail-closed unsupported contracts |

Static compatibility does not establish observed execution-provider assignment,
device latency, task accuracy, clinical validity, or release readiness. Runtime
claims require an identity-bound runtime capture.

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

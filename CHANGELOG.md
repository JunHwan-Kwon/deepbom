# Changelog

All notable archival releases of DEEPBOM are documented here.

## Unreleased

- Added exact ONNX symbolic-dimension preservation and runtime I/O symbol
  binding without interpreting arbitrary exporter expressions.
- Added immutable SafeTensors Dense/Mixtral/Mamba, GGUF architecture/encoding,
  and Core ML MLProgram/flexible-shape corpus contracts with offline replay.
- Added a four-layer selected runtime backend evidence ledger for QNN, NNAPI,
  Core ML, WebGPU, and WebNN that keeps build inclusion, capability acceptance,
  original-op assignment, and execution independent.
- Added hash-bound representative-dataset capture validation with audited
  external-I/O binding, exact integer endpoint counts, reference-output drift,
  repeated-run nondeterminism, explicit denominators, and JCS/SHA-256 export.
- Rebuilt the residual-coverage ledger from v1.1 symbolic-shape sweeps, counting
  duplicate paths by unique artifact SHA-256 and using node outputs as the
  shape-contract denominator.
- Raised the metric-coverage manifest to v1.47 for the new representative-
  dataset validation family.
- Added source-pinned Mixtral sparse-MoE and Mamba recurrent-SSM contracts,
  including total-versus-active expert ledgers, non-KV recurrent state, bounded
  compute scenarios, and hash-bound runtime residency/offload/paging manifests.
- Raised the public-code source ceiling from 10,647 KiB to 11,000 KiB for the
  new LLM contracts while retaining per-file, private-code, documentation,
  native-tooling, corpus-evidence, and development-check budgets.

## 1.94.6 - 2026-08-27

- Made the default CLI audit output a bounded, human-readable projection while
  preserving complete JSON through `--json`, `--compact`, and `--output`.
- Replaced the accidental exhaustive web deployment gate with a bounded
  production preflight and recorded measured workflow budgets and retry rules.
- Excluded generated Cargo and Python build trees from channel source copies,
  keeping local and CI channel builds deterministic and bounded.
- Documented and mechanically checked the shared public web/CLI boundary, the
  reproducible public corpus, and the private research and source-generation
  boundary.

## 1.94.5 - 2026-08-27

- Hardened the Cargo bootstrap download path and verified a fresh Windows
  installation against the immutable engine manifest and a real ONNX audit.

## 1.94.4 - 2026-08-27

- Added the verified Cargo installation channel and its registry publication
  contract alongside npm and PyPI.

## 1.94.3 - 2026-08-27

- Corrected the npm publication command so the verified tarball is resolved as
  a local file rather than a GitHub package shorthand.

## 1.94.2 - 2026-08-27

- Bound macOS wheels to the architecture of their embedded standalone engine
  and a conservative macOS 14 compatibility floor.

## 1.94.1 - 2026-08-27

- Published the Apache-2.0 public source boundary and installation channels.
- Made npm invocation deterministic across GitHub-hosted Linux, macOS, and
  Windows runners without assuming an npm installation beside the Node binary.
- Added exact public-source manifest verification in a clean Git checkout.

## 1.94.0 - 2026-08-07

Initial DOI-oriented research software release.

- Added bounded static analysis for TFLite, ONNX, GGUF, SafeTensors, Core ML
  model files, Core ML packages, and sharded SafeTensors packages.
- Added graph, tensor, quantization, memory, delegation, execution-provider,
  source-pinned kernel, and deployment-target evidence with explicit evidence
  classes and applicability boundaries.
- Added public-format validation using hash-pinned trained artifacts, actual
  browser workflows, exported reports, machine-readable evidence, and viewer
  captures.
- Added reproducible native TFLite/XNNPACK and ONNX Runtime evidence import
  contracts without presenting unobserved runtime behavior as static fact.
- Added curated Zenodo software and validation-dataset packaging that excludes
  protected modules, production credentials, and local scratch data.

This version number identifies the product release. Date-based analyzer and
rulepack identifiers remain separate provenance fields in generated evidence.

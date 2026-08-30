# Changelog

All notable archival releases of DEEPBOM are documented here.

## Unreleased

- Added immutable Hugging Face, GCS-generation, and SHA-256-pinned HTTPS
  acquisition receipts with a content-addressed cache and no remote code or
  pickle execution.
- Added source-backed ORT CUDA eligibility data, an observed NVIDIA host
  profile collector, conservative LLM VRAM lower-bound binding, and explicit
  separation of host capability, selected build, assignment, and execution.
- Added canonical graph IR plus deterministic SVG, PNG, self-contained HTML,
  Mermaid, DOT, and JSON exports. GGUF and SafeTensors remain architecture or
  storage projections rather than invented executable DAGs.
- Added bounded `auto`, `structure`, `integrity`, and `full` scan policies for
  streamed GGUF/SafeTensors analysis and a fail-closed 1 GiB limit before
  monolithic ONNX, TFLite, or ExecuTorch full-file allocation.
- Raised the handwritten runtime ceiling from 11,000 KiB to 11,048 KiB for the
  separately modularized acquisition, accelerator binding, graph IR,
  deterministic PNG, and ONNX external-data range-ledger surfaces; per-file
  and private-source ceilings remain unchanged.

## 1.95.0 - 2026-08-30

- Added machine-readable `capabilities` discovery and a canonical cross-format
  evidence-envelope output for stable automation across npm, Python, Cargo,
  standalone, and repository CLI surfaces.
- Added OASIS SARIF 2.1.0 Errata 01 projection validated offline against a
  source-pinned schema, with artifact-bound finding fingerprints and no
  invented source-code locations.
- Added deterministic severity gates, structured JSON errors, stable exit
  codes, `SOURCE_DATE_EPOCH`, atomic output, and explicit no-clobber behavior
  for CI and regulated evidence pipelines.
- Added installed-channel parity and regression coverage for capability,
  envelope, SARIF, and policy contracts while retaining the public/private
  package boundary and bounded release workflow.
- Raised the documentation source ceiling from 288 KiB to 296 KiB to retain
  the formal CLI automation and measured delivery-operation contracts without
  relaxing runtime, verification, corpus, or native-tooling budgets.

- Added identity-bound TensorRT 10.x/11.x optimized-engine inspector evidence,
  strict capture tooling, selected tactic-identifier inventory, and shared
  report/CycloneDX projection without deserializing plans in the browser.
- Added bounded Core ML nested-pipeline/package range decoding and strengthened
  ML Program, legacy graph, blob-integrity, shape, MAC, and liveness contracts.
- Added deterministic SafeTensors packed quantization ownership for AWQ, GPTQ,
  HQQ, and compressed-tensors configurations, including exact bit conservation.
- Imported source-pinned ONNX scalar attribute defaults into ORT prechecks and
  bound TFLite XNNPACK/GPU/NNAPI conclusions to explicit selected-build facts.
- Added source-pinned ExecuTorch backend/payload contracts and an optional
  selected-build attestation path shared by the web sidecar selector and CLI,
  while retaining backend initialization and execution as runtime evidence.
- Raised the documentation source ceiling from 280 KiB to 288 KiB to retain
  the current public delivery, corpus, reproducibility, and evidence-boundary
  records without relaxing runtime, verification, native-tooling, or corpus budgets.
- Raised the development-tooling ceiling from 1,280 KiB to 1,296 KiB for the
  web/CLI selected-build sidecars, TensorRT engine-inspector capture, and
  tamper-routing regression cases; runtime, generated-data, verification,
  native-tooling, and corpus budgets are unchanged.
- Added CLI `verify`, `diff`, and `explore` surfaces backed by the existing
  interface-contract, deployment-delta, and redesign-Pareto implementations.
- Added strict, duplicate-key-rejecting custom target-profile files with both
  source-file and resolved-profile SHA-256 provenance.
- Aligned public npm/source repository, homepage, and issue-tracker metadata.
- Added a hash-verified 113-artifact public CLI sweep across ONNX, GGUF,
  SafeTensors, and Core ML, exercising both bounded human and compact JSON
  output in fresh processes.
- Split Core ML ML Program compression validation by source-pinned opset:
  CoreML6 packed-index LUT and packed sparse-mask contracts now remain distinct
  from CoreML8 block/vector LUT and UINT1 sparse contracts. Package files
  outside the selected `.mlpackage` root remain hash-bound as supporting
  evidence instead of being silently omitted.
- Reduced the measured public-PR channel-equivalence gate from 446.3 seconds to
  51.7 seconds by retaining npm format, command, and tamper coverage there while
  reserving native, Python, Cargo, and dual-WASM execution for the manual
  release contract.
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

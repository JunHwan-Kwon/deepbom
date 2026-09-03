# Standardization Evidence

`corpus/standardization-evidence.v1.json` is the machine-readable evidence
index for DEEPBOM standardization proposals. It keeps empirical measurements,
metadata sampling frames, contract fixtures, and pinned-source semantics in
separate populations. The populations overlap and are never summed into an
ecosystem-wide denominator.

## Evidence Classes

| Class | Permitted conclusion |
| --- | --- |
| `EXACT_WITHIN_PREDECLARED_POPULATION` | Exact counts and ratios for the named, predeclared, hash-identified population. |
| `MEASURED_CORPUS` | Exact observations for the named bounded corpus, including explicit residuals. |
| `METADATA_FRAME_ONLY` | Repository/file discovery and a reproducible sampling frame. It is not parser or runtime evidence. |
| `CONTRACT_CONFORMANCE_ONLY` | A parser, adapter, or sidecar accepts and rejects the covered fixtures as specified. |
| `SOURCE_BACKED_RULEPACK` | A condition or registration is present in pinned source. It is not selected-build or runtime assignment evidence. |
| `ARCHIVED_VERIFIED_PROFILE` | A compact result profile retained with source digests. Source bytes must be checked separately before claiming byte-level reconstruction. |

No population is a probability sample. The index therefore emits exact
numerators and denominators but no confidence interval or ecosystem prevalence
estimate. A zero in a bounded corpus is not generalized to a format population.

## Current Empirical Basis

The external-interface population contains 50 predeclared TFLite artifacts in
four workflow/provenance strata. Two isolated passes produced deterministic
rows for all 50 artifacts and 114 external parameters:

- 62 parameters have a complete serialized affine contract.
- 52 parameters are explicitly unquantized.
- 0 parameters are invalid or incomplete.
- 30 artifacts have fully affine external interfaces.
- 18 artifacts have fully unquantized external interfaces.
- 2 artifacts have mixed quantized and unquantized external interfaces.
- 6 direction/dtype/shape signatures occur with more than one affine contract.

These observations support two structural conclusions without requiring a
prevalence claim. First, dtype and shape cannot identify an affine mapping, so
scale and zero point need parameter-level binding. Second, a boundary contract
needs an explicit not-quantized state so FLOAT I/O is not confused with absent
or invalid metadata. They do not establish preprocessing behavior, task
accuracy, clinical performance, or release readiness.

The Hugging Face snapshot enumerates 1,542 repositories, 27,666 files, and
13,852 analyzer-supported filenames across the two named community
organizations. It is a metadata sampling frame only. The measured residual
population now covers all 48 unique-byte ONNX artifacts from 40 revision-pinned
repositories in the public multiformat corpus. It is a purposeful
residual-ranking population, not a random or ecosystem-prevalence sample.

Nine artifacts in that ONNX population contain serialized dtype, rank, or
dimension contradictions. The hash-bound
`corpus/onnx-contract-conflict-corpus.v1.json.gz` preserves six unconditional
roots, 543 condition-bound invalid variants, 1,901 downstream blocked nodes,
and 213 MAC-bearing rows that are deliberately withheld. These are artifact
validity observations, not analyzer residuals. They demonstrate why a portable
evidence vocabulary must distinguish `INVALID_CONTRACT` from an unavailable or
runtime-bound value; they do not establish prevalence outside the named
population.

The separate public cross-format population contains 113 unique primary
artifacts: 48 ONNX, 20 GGUF, 18 SafeTensors, and 27 Core ML. It binds 145 source
files and 2,926,971,262 downloaded bytes; 14 ONNX records include a matching,
hash-bound external-data sidecar. Every artifact is analyzed twice in a fresh
Node process, and publication requires identical analysis and receipt digests.
The [per-artifact review](PUBLIC_MULTIFORMAT_CORPUS.md) and
`corpus/cyclonedx-generalization-evidence.v1.json` and its hash-bound compressed
records in `corpus/cyclonedx-generalization-evidence.records.v1.json.gz` retain
each observation instead of publishing only aggregate format counts.

The ONNX, GGUF, and SafeTensors selections are purposeful and stratified, not
random. The Core ML stratum instead enumerates all assets linked from the public
Apple Developer model catalog snapshot on 2026-08-18. Exact counts therefore
describe parser and representation behavior only within the declared
hash-identified boundary. In particular, tiny/random SafeTensors checkpoints
establish architecture and storage parsing rather than task quality; filename
precision labels are candidates until serialized metadata is read; and static
artifacts do not establish runtime placement.

The earlier SafeTensors, GGUF, and Core ML contract fixtures remain in the
index as a separate conformance class. They test selected positive and negative
contracts and are not relabeled as measured public-artifact evidence. Keeping
the two classes separate prevents generated fixtures from inflating public-file
denominators.

The residual ledger applies the same separation to Core ML. Its 27 public Apple
catalog artifacts, five generated MLProgram contracts, and zero compiled-plan
captures are three independent evidence classes. Serialized graph and
compression evidence cannot establish MLComputePlan device usage, and an
MLComputePlan record would still not establish executed placement without a
runtime observation.

## Taxonomy Readiness, Not Just Format Count

The current evidence supports a format-neutral **schema shape**, but it does not
yet support a closed cross-ecosystem quantization vocabulary. Artifact totals
alone are not a taxonomy denominator. Coverage must be evaluated across
container extension, representation family, granularity, numeric encoding,
scope, axis semantics, packing, and evidence class.

| Measured format boundary | Positive serialized evidence | Important uncovered cells |
| --- | --- | --- |
| TFLite: 50 `.tflite` artifacts | 62 external per-tensor affine parameters and 52 explicitly unquantized parameters; kernel granularity is per-axis in 30 artifacts and per-tensor in 4 | External per-axis affine contracts, affine per-group, and framework-neutral negative-axis normalization |
| ONNX: 48 public `.onnx` artifacts, including 14 `.onnx_data` payloads, plus 4 separately bounded conformance artifacts | Per-tensor affine tensor mappings, operation-scoped integer/scale-graph signals, INT8 and UINT8 storage; source-derived per-axis Q/DQ structure with runtime-supplied values; one complete static per-axis Q/DQ fixture; and exact upstream non-ORT custom-domain fail-closed handling | Public application examples with complete per-axis values, negative-axis cases, QLinear and DynamicQuantizeLinear breadth, packed sub-byte contracts, and repository-stratified contrib/custom-domain coverage |
| GGUF: 20 `.gguf` artifacts | Eleven decoded scalar/block encodings, including Q2_K, Q3_K, Q4_K, Q4_0, Q5_0, Q5_1, Q6_K, Q8_0 and IQ4_NL; observed block sizes 32 and 256 | GGUF block size must not be silently relabeled as affine `per-group`; external interface and execution graph contracts are not serialized here |
| SafeTensors: 18 public architecture checkpoints plus 2 separately bounded public quantization-header contracts | F16, BF16 and F32 storage across 18 architecture families; AutoAWQ and AutoGPTQ single-file config ownership, 322/322 packed modules, 4-bit/group-128 shape conservation, and scale/zero cardinality | Full-payload scale/zero scans, sharded quantized repositories, bitsandbytes, HQQ and compressed-tensors semantics |
| Core ML: 19 `.mlmodel` and 8 `.mlpackage.zip` public artifacts plus 1 source-pinned per-output-channel conformance fixture | Legacy quantized-weight storage, one exact INT4 linear per-output-channel scale/bias contract, and serialized MLProgram quantization/palette transforms including 6-bit and 8-bit palette strata | Public-model linear per-channel application, modern MLProgram per-channel/per-block contracts, and sparse/pruned representations |

These populations are separate and overlap with earlier fixtures, so their
artifact counts are not summed into a prevalence estimate. The machine-readable
`taxonomy_coverage_assessment` records the exact extensions and observed
scheme, granularity, encoding, group-size, and axis sets for each measured
format.

The contribution boundary is therefore explicit:

- **Supported now:** repeatable representations, explicit scope, named external
  parameter binding, representation-family separation, evidence states, and an
  open custom representation mechanism.
- **Not supported yet:** a closed universal `scheme` or `granularity` enum,
  universal axis constraints, canonical cross-format zero-point omission, or
  normative affine per-group semantics.

A vocabulary term should move from candidate to normative only after it has a
primary format/runtime definition, a hash-bound positive artifact, an exact
DEEPBOM extraction record, and valid plus invalid schema fixtures. Semantics
that differ by producer ecosystem remain scoped profiles rather than being
forced into one superficially common term.

## What Generalizes From The Corpora

The following are structural counterexample conclusions. They generalize as
schema capability requirements because one valid counterexample is sufficient
to disprove a universally lossless singleton or single-state representation.
The quoted ratios describe only their named corpora and are not prevalence
estimates.

| Schema requirement | Bounded evidence | Generalizable conclusion |
| --- | --- | --- |
| Repeatable, scoped model representations | 27/27 artifacts with an applicable singleton-losslessness assessment lose information when their extracted tensor/block representation profiles are flattened into one `scheme/granularity/bits/axis` object: ONNX 2/2, GGUF 20/20, and Core ML 5/5. SafeTensors is 0/0 because its current measured files expose storage dtype rather than a decoded quantization contract. | A portable schema must permit multiple representations and bind each to an explicit scope. This supports the structure only; it does not establish a complete scheme/granularity vocabulary, and the ratios are not ecosystem frequencies. |
| Named external-parameter binding | 75/75 non-TFLite artifacts with serialized interfaces expose multiple external parameters (763 parameters total). In the separate TFLite population, 62/114 parameters carry complete affine mappings and six direction/dtype/shape signatures map to multiple affine contracts. | External contracts are a repeatable named input/output collection. Dtype and shape cannot substitute for parameter identity or determine scale and zero point. |
| Explicit field state | The same graph field is `OBSERVED` for 70/113 artifacts, `UNAVAILABLE` for 5/113, and `NOT_APPLICABLE` for 38/113. | Observed zero, unavailable, and not applicable are different values. A null-or-zero default cannot represent the evidence without loss. |
| Compound artifact identity | 14/48 ONNX artifacts reconstruct verified tensors through a separately hashed external payload. | A primary model-file hash is insufficient when external tensor data is required; the artifact content set and each dependency digest need binding. |
| Independent provenance fields | All 113 artifacts are byte-hash identified; 86 carry externally bound source revisions and 44 carry external license metadata. | Artifact digest, source revision, and declared license are independent fields. Missing external provenance is not inferred from bytes and does not erase observed identity. |
| Runtime claim separation | 113/113 static artifacts retain deployment assignment as `RUNTIME_REQUIRED`. | Provider, delegate, backend, and device assignment need a separately bound runtime observation. Static eligibility remains a different claim. |
| Readability is not conformance | 3/113 readable artifacts retain checked serialized-contract issue codes. | Parser success, implemented-rule conformance, external dependency completeness, and runtime compatibility are separate statuses. |
| Storage is not an affine interface | 38/38 GGUF and SafeTensors artifacts retain storage dtype/block encoding without promoting it to an external affine mapping. | Low-bit or integer storage must remain distinct from a named scale/zero-point interface contract and from executable graph placement. |

Two null results are deliberately not generalized. The public multiformat
population contains 0/763 complete extracted external affine mappings, while
the separate TFLite population supplies 62/114 positive mappings; this
establishes optional availability and format-specific extraction, not the
absence of quantization in ONNX or Core ML. Likewise, 0/46 public
cross-format model-representation assessments yielded an explicit axis through
that population's extraction. A separate four-artifact ONNX conformance corpus
now proves that the analyzer preserves source-derived per-axis structure,
distinguishes runtime-supplied values from serialized values, checks static
axis cardinality, and retains an unknown external custom domain without
assigning ORT semantics. Those fixtures establish implementation behavior, not
axis prevalence or application coverage. A separate public SafeTensors
header/config corpus now establishes source-pinned AutoAWQ and AutoGPTQ
per-group packing structure for 322 modules, but it does not establish payload
value quality, sharded package coverage, other producer ecosystems, or
universal per-group semantics.

The CycloneDX 2.0 candidate set contains two valid and three invalid
quantization fixtures validated against PR #990 head `49a945618811`. A separate
12-case probe records eight currently accepted and four rejected combinations.
This is draft-schema conformance evidence only; it does not claim working-group
acceptance or final 2.0 semantics.

## Cross-Format Runtime Terms

The shared `deepbom.runtime_evidence_sidecar.v1` index uses these terms across
TFLite, ONNX, GGUF, and Core ML:

- `artifact_identity`
- `artifact_set_identity`
- `external_parameter_contract`
- `field_state`
- `runtime_build_identity`
- `execution_configuration`
- `placement_evidence`
- `timing_evidence`
- `memory_evidence`
- `claim_boundary`

Source eligibility, configured inclusion, runtime assignment, execution,
timing, and memory are separate claims. A normalized sidecar never promotes a
source rule or static compute plan into observed execution.

## Provenance Caveat

The compact public profile
`quant-policy-boundary-public-50-2026-08-05` declares sweep and review digests
whose exact byte sources are not both present in the current repository state.
The index therefore labels it `DETACHED_DIGEST_PROFILE`. The available repeat-2
sweep is retained as a separate hash-bound measured source and independently
reconstructs the 50-artifact/114-parameter counts. Do not claim byte-for-byte
reconstruction of the detached profile until its declared source bytes are
restored.

## Reproduction

```text
npm run build:standardization-evidence
npm run check:standardization-evidence
```

The generator verifies source schemas, file digests, denominator conservation,
repeat determinism, population strata, metadata snapshot identity, source
rulepack inventories, and the RFC 8785 JCS ledger hash. A stale generated index
fails release and deep checks.

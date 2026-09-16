# DEEPBOM Common Model IR v1

`deepbom.model_ir.v1` is a preview, hash-bound, format-neutral evidence contract. It does not replace `deepbom.artifact_ir.v2` during the migration period. Every Model IR document names and hashes the validated Artifact IR from which it was projected.

## Purpose and non-claims

The IR gives generic analyzers and renderers one vocabulary for serialized programs, logical values, physical storage, bindings, quantization, architecture grouping, static runtime projections, and observed runtime overlays. It does not claim that different native operators are semantically identical, infer a graph from weight names, predict actual runtime order, or decide standards or regulatory conformity.

The evidence layers remain separate:

1. `OBSERVED_SERIALIZED_ARTIFACT`: parsed from the identified artifact or a hash-bound package member.
2. `DERIVED`: deterministic arithmetic, grouping, or projection from observed subjects.
3. `PREDICTED`: conditional static runtime eligibility or resource projection.
4. `OBSERVED_RUNTIME`: imported trace bound to the artifact identity.
5. `DECLARED_UNVERIFIED`: an external declaration whose content is retained but not proven by artifact bytes.
6. `NOT_ASSESSABLE`: information the available evidence cannot establish.

Unknown values are never converted to zero or false. A missing applicable structure is represented in `profiles`, `completeness`, and `loss_ledger`.

## Order contract

- `source_order` preserves the native serialized index where exposed.
- `dependency_order` is a partial order derived only from serialized producer-consumer relationships.
- `display_order` is a deterministic topological projection with native index and stable identifier tie-breaks.
- `runtime_order` remains null until an identity-bound runtime trace supplies an observation.

Display order is suitable for stable documents and diffs. It is not a kernel schedule, fusion decision, device assignment, or latency claim.

## Mandatory and profile-mandatory fields

Core identity, format/role, applicability, evidence vocabulary, completeness, loss ledger, source binding, and digest are mandatory in every document. Program records are mandatory only when the format serializes a program. Storage records are mandatory when serialized tensor or parameter payloads exist. Quantization records are mandatory when the parser observes a quantization contract. Runtime observations are optional and remain a separate overlay.

The machine-readable applicability matrix is [`config/model-ir-applicability.v1.json`](../config/model-ir-applicability.v1.json). Planned or unimplemented semantic coverage is not advertised as present. The PyTorch checkpoint safe envelope is an identity, archive, storage-member, and bounded pickle-risk inventory; it is not a full model analyzer.

## Graphless invariant

GGUF and SafeTensors do not serialize an executable operator graph. Their `program` arrays and execution relationships must be empty. Architecture namespace groups may organize storage for review, but may not contain execution edges. This invariant is enforced by both JSON Schema and semantic validation.

## Binding contract

A weight binding exists only when an observed operation input port resolves to an observed logical value and that value resolves to a serialized storage object. Port position is structural. Semantic labels such as query/key/value or convolution kernel are added only by a separately versioned, capability-gated model profile with its own basis and member trace.

## Upstream sources

Format and runtime facts are generated or checked against immutable upstream revisions and file-level SHA-256 values. [`config/model-ir-upstream-sources.v1.json`](../config/model-ir-upstream-sources.v1.json) is the manifest of source roles; the named verifier scripts contain the file pins and semantic-marker checks. `scripts/verify-model-ir-preview-source-pins.mjs` can check a pinned local clone or create temporary sparse clones. Upstream source bytes and repository clones are not redistributed.

## Bounded preview adapters

GraphDef preserves serialized nodes, referenced data endpoints, control dependencies, function inventory, and bounded Placeholder/Const dtype, shape, and tensor-content facts. SavedModel additionally preserves first-MetaGraph `SignatureDef` tensor contracts and binds every selected package member by path, size, role, and SHA-256, while materializing only the first MetaGraph. Keras and PT2 materialize only declarative graph relationships found in bounded JSON members; they do not construct framework objects, load pickle payloads, lower layers to kernels, or claim runtime order. HDF5 and PyTorch checkpoint inputs remain safe-envelope-only: HDF5 object graphs and checkpoint object semantics are not advertised as analyzed.

## Generic analysis passes

`generic_analysis` contains interface, program topology, tensor/storage, quantization, and static-runtime summaries. These passes accept only Model IR sections, contain no format branch, and may not read native parser ledgers. Model-specific profile passes are separate and retain every assigned source subject. Structural blocks are deterministic, reversible partitions of serialized operations: a repeated contiguous signature is preferred; otherwise the region remains a non-repeating block or a deterministic rank partition for large graphs.

## Format-neutral model summary

`deepbom.model_summary.v1` is the single Keras-summary-like text and table
projection. It is derived only from a validated Model IR document and is bound
to both the Model IR and artifact SHA-256 values. `auto` selects serialized
operation rows when a program is present, serialized storage rows for
graphless weight containers, and an explicit identity-only/not-assessable
result for safe envelopes that expose neither structure.

The summary reports native operation type, output tensor contract, predecessor
subjects, bound serialized storage, exact element and byte counts, static MACs
where assessable, evidence class, completeness, and source identifiers. Block
rows retain every member reference so they are reversible. Storage rows never
gain synthetic graph edges.

Unlike a framework-resident Keras model, a deployment artifact does not in
general serialize trainable/non-trainable state. The projection therefore
reports `trainable_parameter_count: null` and keeps constants, quantization
tables, axes, and weights under serialized storage. Display order is a stable
topological document order, not observed execution order. The public schema is
[`docs/schemas/deepbom-model-summary-v1.schema.json`](schemas/deepbom-model-summary-v1.schema.json).

CLI examples:

```text
deepbom model-summary model.onnx
deepbom model-summary model.gguf --level storage --format markdown
deepbom audit model.tflite --section model_summary --compact
```

## Compatibility and migration

During preview:

```text
native parsers -> artifact_ir.v2 -> model_ir.v1
```

This makes projection loss visible before any analyzer is switched. Once adapter and generic-analysis parity gates pass for all supported formats, the intended direction is:

```text
native facts -> model_ir.v1 -> generic analyses and artifact_ir.v2 compatibility projection
```

No stable `artifact_ir.v2` field or digest rule is changed by introducing Model IR v1.

## Regulatory-use boundary

The IR is designed for traceable engineering evidence: stable subject identifiers, source locators, explicit applicability, reproducible digests, and separation of measured, derived, predicted, declared, and unavailable facts. A regulator, quality system, standards body, or downstream policy determines whether that evidence is sufficient for a particular decision. DeepBOM does not make that decision.

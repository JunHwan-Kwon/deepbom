# WebAssembly and Worker Execution Boundary

Deterministic calculations and portable evidence contracts belong to the shared
Rust/WebAssembly engine. Web, CLI, local MCP and Python consume the same engine.
Host adapters own framework hooks, file/range I/O, decoding into portable values,
worker/process lifetime, rendering and export delivery.

## Current ownership

- Rust/WASM owns TFLite analysis, exact shape/count/cost arithmetic, numerical
  statistics and weight kernels, redesign validation, evidence identity and the
  Snapshot/Evaluation/Deployment contracts described in `WASM_NUMERICAL_ABI.md`.
- ONNX TypeProto admission, nested union/unification and known-type decisions use
  `portable_onnx_type.rs`. The host retains protobuf/source collection, display
  formatting and the pinned enum registry. Unknown dimensions remain unknown;
  union and declared/inferred unification are different operations.
- Native tensor descriptors and imported identities use `portable_native.rs`
  for role/type/layout/alias admission and exact element/byte conservation.
  Tensor payload SHA-256 and numerical-IR document hashes use the same WASM.
- Python canonical content hashes use the native bridge's `content_hash`
  operation, including the Monitor-owned session. Framework collection never
  substitutes a Python numerical implementation. File-byte hashes remain on the
  host for engine bootstrap verification, source-code fingerprints and streaming
  input/output integrity. These do not canonicalize evidence documents; the
  bootstrap checks must also run before the engine is trusted or executed.
- `portable_ir.rs` owns logical value-type admission and exact element counts,
  metric creation/aggregation, logical-inventory bindings, artifact-member range
  bounds/overlap/conservation, and runtime-provenance digest/signature-payload
  binding. Builders and validators call these same rules. Empty half-open byte
  ranges do not overlap; unknown counts remain unknown in complete totals.
- `portable_coreml.rs` owns the observed-feature/declared-version deployment-floor
  decision, consuming source-pinned availability tables. An unmapped MIL opset
  returns `not_assessed_unmapped_feature_floor`, not `assessed`. This is not an
  observed device placement or runtime-execution result.
- Artifact/Model graph admission, Weight/Activation relationships, conversion
  receipts, provenance graphs and imported runtime reconciliation use the
  `portable_*_validation.rs`, `portable_conversion.rs`, `portable_provenance.rs`
  and `portable_runtime_overlay.rs` owners. Host wrappers return their verdicts.
- ONNX standard-node, container, conditional-shape, ORT-contrib and ai.onnx.ml
  contracts and bounded scalar references use `portable_onnx_*.rs`. Ordered
  attributes, exact INT64 decimals and non-finite float tokens cross the ABI
  without JSON rounding. Source-pinned schema tables are compiled into WASM.
  TfIdfVectorizer, linear, SVM and tree reference values retain their source-specific precision
  and explicit work limits; they are not runtime measurements.
- If/Squeeze branch reconciliation uses the same symbolic-dimension predicate
  in Rust. Only a proven matching guard can exclude an impossible branch failure;
  an unrelated guard retains the partial contract and its failure evidence.

## Execution boundary

Heavy public TFLite exports run in `workers/static-audit-worker.js`. The UI may
initialize the module and call `runtime_guard` and `target_profiles`. Common
synchronous APIs require WASM readiness at the host boundary and reject stale
binaries; they have no host calculation fallback.

GGUF, SafeTensors, Core ML, ONNX and ExecuTorch readers run in workers. Host code decodes formats, collects source evidence, resolves files and
lexical graph scopes, orchestrates function/control-flow evaluation and composes
reports. Shape arithmetic, per-node semantic verdicts, type unions, bounded
value evaluation and family admission call WASM. Graph orchestration remains
a shared host adapter used by every channel; it does not substitute another
numerical implementation. Independent conformance reconstruction is deliberately
separate from derivation.

Independent JavaScript quantization, rounding and channel-vitality validators
remain deliberate test oracles. Sharing their implementation with the derivation
would weaken verification. Publication utilities and pre-engine integrity checks
are also distinct from the runtime evidence calculation owner.

## Verification

`npm run check:no-main-thread-heavy-wasm` enforces heavy-export worker ownership.
`node scripts/check-wasm-ir-semantics.mjs` checks interval topology against an
independent all-pairs oracle, exact/partial metrics and cross-document bindings.
`node scripts/check-wasm-owner-migration.mjs` checks tensor conservation, digest
parity, nested type joins and reference producer mismatch rejection. Existing
format, IR, browser and channel tests must also pass against the new binary.

Reference examples record actual producer metadata and WASM SHA-256. Explicit
regeneration creates a new immutable set and rotates its index atomically. The
historical set is checksum-preserved. Dirty-source reference sets are marked
`development_reference_not_released`; they are not clean release evidence.

The streaming screening benchmark remains available as
`npm run benchmark:streaming-wasm-candidates`. It measures the existing host
path, not an established performance improvement from these migrations.

This migration advances Artifact IR method to 2.3.4 and Model IR method to 1.1.4;
wire schema identities remain unchanged. Earlier method identifiers remain
readable for valid evidence. A document that records an empty range as an overlap
or misstates member completeness is rejected instead of being silently rewritten;
retain its original bytes and regenerate derived evidence from the source.

Queue shutdown signals already admitted work without starting new content-hash
engines under its lock. Explicit status/cancel calls still validate the immutable
request through the shared engine, outside that lock. This keeps identity checks
from adding uncontrolled shutdown work; OS scheduling and filesystem stalls are
not a universal hard-real-time guarantee.

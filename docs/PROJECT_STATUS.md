# DEEPBOM Project Status

Last updated: 2026-08-24 KST

Public-safe operating plan for DEEPBOM. Private implementation strategy and
unreleased research notes belong in `LOCAL_PRIVATE_ROADMAP.local.md`, which is
ignored by git.

## Snapshot

DEEPBOM is the primary product name for this browser-local engineering workbench for deployment
artifacts. It covers local model inspection, static on-device audit, graph and
performance views, browser-local runtime benchmarks, controlled advanced modules,
login-free watermarked Engineering and Regulatory Support Reports, profile-based
Evidence Packages, controlled raw ML-BOM/visual exports, and the full
Engineering Bundle.

`/medical` reuses the same shell and adds the Medical AI evidence workspace.
Its watermarked Regulatory Support Report is login-free; the full Regulatory
Bundle remains authorization-controlled. The app does not upload model bytes, weights,
tensors, inputs, outputs, reports, or local filenames. Server scope is auth,
authorization, admin, feedback, research consent, and opt-in structure/timing
metadata. Runtime/backend/delegate signals are assumptions, predictions, or
local measurements; they are not clinical validation, certification, Hessian or
PAC-Bayes evidence, or generalization proof.

Live:

- `https://deepbom.org`
- `https://deepbom.org/medical`
- Repo: `https://github.com/JunHwan-Kwon/deepbom`

## Implemented

| Layer | Scope |
| --- | --- |
| Browser analysis | TFLite Rust/WASM parsing, M001 Model Metadata parsing, explicit input ProcessUnit and normalization/tokenizer contract validation, output-label declaration parsing, exact declared-filename binding to terminal ZIP entries, bounded stored/DEFLATE decoding, size/CRC-32/SHA-256 verification, and UTF-8 label cardinality validation against output axes. Generic TFLite metadata presence is never promoted to a preprocessing or output-semantic contract. The ONNX browser path parses bounded ModelProto/GraphProto doc_string, producer, domain/version, metadata_props, and every external_data key/value record across main, nested, node-attribute, FunctionProto-default TensorProto, and SparseTensorProto value/index scopes while preserving their evidence boundaries. It binds TensorProto DataType IDs 1-26 to pinned ONNX 1.21 source, decodes raw and typed numeric fields including COMPLEX magnitude, FP8, FLOAT4E2M1, FLOAT8E8M0, INT4/UINT4, and INT2/UINT2, and computes packed cardinality as `ceil(elements*bits/8)` without promoting padding to logical elements. Users can select sidecar files or a directory; the analyzer preserves model-relative paths, computes whole-file SHA-256 and SHA-1, rejects unsafe/duplicate paths, malformed decimal ranges, data-location or embedded-payload conflicts, out-of-bounds ranges, dtype-shape byte mismatches, and declared whole-file SHA-1 mismatches, and decodes only verified ranges. Verified external values participate in weight integrity, quantization binding, duplicate-payload, available-initializer, and exact raw-zero-byte calculations; incomplete coverage remains NOT_ASSESSABLE and enters the High action queue. Arbitrary properties are not promoted to executable preprocessing or output-label semantics. The path also includes ONNX protobuf bounds validation, format/header detection, light-read estimate, dynamic shape signatures, I/O contracts, and 152 source-pinned shape rules with bounded exact integer shape-data propagation. A generated 514-version formal-schema ledger resolves the greatest OpSchema version allowed by each imported opset, preserves repeated OperatorSetIdProto imports, resolves each normalized domain to its highest valid referenced version, rejects non-positive OperatorSetIdProto versions, and validates input/output omission and cardinality, attribute names/requiredness/types, discriminator/value-field consistency, and duplicate attributes before inference. Shape evidence conserves supported, unsupported, unresolved, inferred, known, and unknown counts; explicit zero dimensions and rank-0 scalars remain known; declared dtype/rank/dimension contradictions and formal-schema violations fail the shape contract and enter the High action queue. Every nested GraphProto and FunctionProto body is inventoried, FunctionProto default-graph definitions remain distinct from bound invocation scopes, and local functions plus If/Loop/Scan-8/Scan-9+ contracts are recursively evaluated with exact residual-node and unresolved-output counts. Recursive tensor/sequence/map/optional/sparse TypeProto declarations are validated. Direct Sequence/Optional operators additionally derive recursive output TypeProto, exact sequence lengths and bounded element inventories, optional presence, static length/presence scalars, constant-position bounds, and exact split/concat output shapes when the artifact determines them. SequenceMap bodies are recursively inferred with exact first-input length and bounded per-element output inventory expansion; compatible non-dense If branches preserve sequence/optional TypeProto and branch-stable length/presence facts. Loop-13 sequence state and Loop-16+ optional state are source-version validated; bounded exact Loop expansion preserves final container state and scan leading dimensions when artifact-known control values close every reached iteration. The source-pinned `ai.onnx.ml` value pass validates `Binarizer-1`, `Normalizer-1`, `Scaler-1`, `Imputer-1`, `OneHotEncoder-1`, `LabelEncoder-1/2/4`, `LinearClassifier-1`, `LinearRegressor-1`, `ZipMap-1`, `CastMap-1`, `DictVectorizer-1`, `CategoryMapper-1`, `FeatureVectorizer-1`, and `ArrayFeatureExtractor-1` tensor/map/value contracts, exact label/vocabulary/category attributes, conversion modes, key/cardinality contracts, variadic feature widths and batch agreement, exact INT64 selection indices, canonical TypeProto, and exact output cardinality where artifact facts close it. It derives exact Binarizer threshold effects, pinned-ORT Normalizer signed-MAX/L1/L2 row arithmetic, pinned-ORT Scaler affine arithmetic, and pinned-ORT Imputer replacement/NaN-marker arithmetic, and pinned-ORT OneHotEncoder category-axis, numeric-to-INT64 truncation, duplicate-last-write, unknown-policy, and exact one/zero arithmetic. It also derives LinearClassifier/LinearRegressor output shape, coefficient/intercept conservation, label/target contracts, scalar FLOAT32 reference previews, CPU dtype gaps, and post-transform hazards, and retains Normalizer signed-overflow/non-finite risks, Scaler runtime-contract/integer-projection/non-finite risks, Imputer attribute/runtime-dtype/non-finite risks, OneHotEncoder invalid-vocabulary, unreachable-duplicate-column, all-zero unknown, guaranteed-runtime-failure, INT32 kernel-gap, noncanonical-boolean, and unrepresentable-cast risks, plus duplicate ZipMap information-loss risk, duplicate DictVectorizer output-column risk, active-direction CategoryMapper last-write-wins risk, deterministic FeatureVectorizer truncation, and ArrayFeatureExtractor bounds failures. CategoryMapper direction/default/output shape, FeatureVectorizer output width plus per-batch copy/pad/truncate counts, and ArrayFeatureExtractor final-axis replacement plus ORT rank-1 compatibility shape are derived exactly; exact initializer INT64 indices are preserved as decimal strings without JavaScript Number rounding. Separately pinned ORT CPU sources/tests cross-check Binarizer, Normalizer, Scaler, Imputer, OneHotEncoder, LabelEncoder, LinearClassifier, LinearRegressor, ZipMap, DictVectorizer, CategoryMapper, FeatureVectorizer, and ArrayFeatureExtractor behavior without claiming the selected EP was observed; no pinned ORT CPU CastMap implementation is claimed. Scan remains tensor-only under the pinned schemas. Explicit non-dense values remain excluded from dense shape, MAC, activation-memory, liveness, and provider-precheck arithmetic. SparseTensorProto structure and linear/coordinate index contents are validated exactly; valid sparse initializers contribute logical cardinality, stored/implicit-zero weight integrity, dead output slices, quantized-grid metrics, and canonical duplicate-storage arithmetic without densification. Missing or invalid sparse components make dependent totals partial. This bounded engine is not claimed as the complete ONNX checker/reference inference: dynamic Loop control, remaining ai.onnx.ml tensor/map-consuming algebra outside the explicitly enumerated value engine, sparse-value operator algebra, unresolved symbolic/runtime values, unsupported rules, bounded-work expansion overflow, semantic constraints beyond the generated formal schemas, and optimized-runtime rewrites remain unassessed. Domain-isolated shape/MAC/quant/kernel semantics prevent custom-op name collisions from inheriting standard rules, unknown payloads remain null, partial liveness is a lower bound, and standard-domain Q/DQ/QLinear binding covers scale, zero-point, axis, cardinality, implicit bias-scale, accumulator, and quantized-grid checks. |
| Static audit | Op/tensor inventory, quantization state/scope, per-tensor/per-axis metadata, zero-point/scale risk, MACs, traffic, profile-specific L1/L2 row pressure, target roofline, INT8 caveats, channel alignment, packing estimates. The Deployment Frontier computes four-target op distributions, exact per-target cache ratios/watch counts, an exact 80% prefix union, normalized Jensen-Shannon divergence with per-op conservation, rank volatility, signed component shifts, and component-removal upper bounds. The session-local Deployment Delta independently parses two TFLite artifacts, deterministically aligns every op as matched/added/removed, and conserves signed static cost changes across five named components and every requested target. Delegation Repair toggles every predicted TFLite op assignment exactly once, evaluates every complete contiguous CPU island, and ranks deterministic segment/boundary repair, group-only gain, and fragmentation counterfactuals with exact changed-edge payload ledgers. Residual Lattice exhaustively enumerates every legal 8-bit code pair and output-zero-point containment design for eligible TFLite residual ADDs. Accumulator Headroom decodes stored quantized Conv/Depthwise/FC constants and solves every output channel's exact signed integer envelope under the pinned reference algebra. Requantization Fidelity reproduces pinned Q0.31 multiplier/shift preparation for default and single-rounding builds and propagates encoding plus conservative rounding bounds over every exact accumulator channel. Quantized Kernel Witness constructs concrete per-channel legal-code endpoint patterns, proves them against the exact accumulator envelope, and executes both pinned TFLite fixed-point rounding paths with binary term ledgers. Channel Vitality proves constant endpoint outputs by monotonicity, separates constant accumulators from clamp/projection collapse, and inventories build-mode-dependent collapse, sign lock, and exact output interval hulls. Rounding Equivalence partitions every complete post-bias integer interval into exact default/single output-pair segments and inventories every divergent interval state with independent ledger verification. Accumulator Reachability decodes centered weights, proves complete integer/modular lattices or exact endpoint bands by bounded-sum coverage, and partitions every interval and divergent state into certified reachable, residue-excluded, or unresolved classes. Numerical ABI Propagation joins that exact source partition to downstream graph corridors, exposes overlapping exact/residue/unresolved facets, classifies reconvergence, counts acyclic declared-output routes, and inventories predicted domain-crossing payloads with independent graph/rounding/reachability/propagation digest verification. Contract Migration evaluates both residual containment re-export candidates, traces exact direct consumers and reachable downstream ops, regenerates every affected kernel channel's multiplier/shift and bias code, and reproduces pinned TFLite ADD prepare-time parameters. Residual Step Response exhaustively classifies every adjacent legal input-code transition and every joint interior code pair under current and containment output contracts, with independently verified binary transition ledgers. Residual Contract Distortion compares current and both containment contracts over every legal pair, including clamp transitions, signed represented displacement, ideal-error direction, exact percentiles, worst witnesses, tile aggregates, and binary ledgers. |
| Delegate/provider analysis | TFLite XNNPACK chain prediction, exact ModelRuntimeDetails execution-plan/delegate import, capture-bound BenchmarkProfilingData timing, predicted-vs-observed placement, partition totals, graph-edge boundary payloads, fallback traffic, and delegated MAC/fallback byte ratios. ONNX imports raw ORT node profiles or browser-verified pinned native dual-profile envelopes with deterministic original-op/provider mapping, explicit unresolved/conflict counts, duration arithmetic, production transformed-node ledgers, paired-output deltas, and observed provider transitions; no TFLite prediction is applied. Authorization-scoped DEEPBOM WASM adds pinned ORT CPU/WebGPU/WebNN schema-plus-registration evidence, per-op artifact dtype/rank/constant/output/attribute prechecks, separate source and narrowed all-EP intersections, provider gaps/exclusions, and both provider-pair overlap matrices for ONNX/ONNX-ML/com.microsoft, plus a pinned-source GEMM/IGEMM/DWCONV decision ledger for TFLite. |
| Stage/graph | Stage grouping, stage risk, delegated/fallback percentages, mobile block tags, graph explorer, SVG/PNG, and Mermaid. TFLite `deepbom.block_inventory.v1.1` first matches graph semantics, distinguishing depthwise-separable and residual inverted-bottleneck motifs, then assigns every remaining op exactly once through name/shape fallback groups. PAD-only preludes inherit the following compute stage, and stage/block aggregates conserve op count, MACs, steady/cold modeled time, logical traffic, parameters, cache pressure, and predicted boundaries. The primary full-graph viewer lays out the exact operator DAG from top to bottom and labels every serialized tensor edge with tensor identity, shape, and dtype while structure, delegation, quantization, and steady-time overlays reuse the same topology. Deployment mode labels predicted boundary tensors and payloads, outlines exact CPU islands, and previews complete-island repair without changing modeled latency or claiming runtime support. |
| Runtime path | Browser preprocessing, Canvas resize/decode, WASM dtype packing, LiteRT.js TFLite benchmark, and ONNX Runtime Web benchmark. ONNX external-data execution is allowed only after complete static sidecar verification; every required full sidecar is rehashed immediately before being mounted into the ORT `externalData` session option, and the model-relative path/byte length/SHA-256 binding is retained in runtime evidence. Runtime rows separately bind artifact/signature/runtime/executed input and output shapes with exact element counts, reject artifact/runtime dtype and shape conflicts, preserve a true cold invocation before warmup and every exact ordered post-warmup sample, independently reconstruct nearest-rank percentiles and population mean/stddev/CV, fail closed on unresolved dimensions, adapt raw TFLite runtime-plan/profile protobuf and ORT Chrome-trace evidence, and retain execution-node coverage, backend status, compile/setup, first run, output digest, and aggregate-only opt-in timing telemetry. |
| Representative dataset validation | Reports accepts a local `deepbom.representative_dataset_capture.v1` bound to the active artifact SHA-256, dataset manifest, optional preprocessing implementation, runtime identity, and every audited external I/O dtype/rank/static dimension/tensor identity. It independently reconstructs exact bounded-integer input endpoint counts, same-contract reference-output drift, repeated-run nondeterminism, explicit sample/comparison/value denominators, source capture SHA-256, and an RFC8785-JCS ledger SHA-256. Cross-artifact, shape, cardinality, dtype, non-finite, and duplicate-run defects fail closed. The ledger does not promote an external representativeness declaration to calibration quality, task accuracy, clinical validity, production preprocessing identity, or release readiness. See `docs/REPRESENTATIVE_DATASET_CAPTURE.md`. |
| Reports/bundles | Login-free watermarked Engineering Report with a front-loaded Model At A Glance summary, explicit artifact-metadata parser status/schema, TFLite ProcessUnit assessment counts, packed associated-file verification counts, and ONNX typed-evidence limitations. The primary Evidence Package selector emits Public, Engineering review, Regulatory support, or Machine-readable profiles; each profile binds its members through a verification manifest and local-browser package signature without including original model bytes or raw tensor values. Reusable raw JSON, CSV, diagrams, visual exports, and full research bundles retain a separate authorization boundary. Machine-readable metric coverage binds every registered calculation family to status, evidence class, JSON pointers, report sections, viewer tabs, exports, and method, with exact single-family ownership of every public top-level analysis key and a normalized nested-field ledger distinguishing report-consumed paths from raw-evidence-only paths. A front-loaded machine-readable Decision Coverage ledger assigns every format-applicable metric family exactly once to seven decision domains, conserves assessed/partial/not-assessed/not-applicable/suppressed counts, lists residual evidence, and states each domain's claim boundary. Conformance independently reconstructs metric status from the analysis/runtime evidence and rejects both ledger arithmetic drift and self-consistent manifest/summary status tampering; runtime observation and product validation remain not assessed without bound execution or reference-dataset evidence. A separate required-report registry covers emitted decision-critical target and ISA assumptions, denominators, coverage states, quantization totals, boundary payloads, metadata contracts, pinned XNNPACK/ORT source provenance, advanced proof status/schema/source binding, and ONNX shape/Q-DQ/domain/capture-capability counters; omission from the Engineering Report fails conformance while large proof ledgers may remain raw-only. Every authoritative finding is rendered without truncation in deterministic priority order with confidence, source rule, method, and complete evidence JSON Pointers. Bundle conformance resolves every pointer against the exact pre-conformance evidence document and blocks release on stale paths, duplicate IDs/pointers, undeclared confidence, or non-enumerated priority. The context-free triage score is rendered only with its complete emitted penalty arithmetic and an explicit non-readiness boundary. Conformance also fails for missing or multiply-owned keys, unregistered structured calculations, unbound nested fields, incomplete field discovery, unbound assessed evidence, invalid benchmark phases, non-reconstructible benchmark statistics, or null-to-zero promotion. Outputs include artifact integrity posture with ONNX external-data and TFLite tensor constant-buffer posture, runtime reproducibility, cold/warm boundary, exact ordered latency samples, jitter, output digests, controlled Download Raw Data, Roofline CSV, Mermaid, graph SVG, visual PNGs including residual-lattice, accumulator-headroom, requantization-fidelity, kernel-extremum-witness, channel-vitality, rounding-equivalence, numerical-ABI-propagation, input-counterexample, preprocessing-realizability, preprocessing-consequence, contract-migration, residual-step-response, and residual-distortion maps, CycloneDX-style ML-BOM with exact arithmetic, propagation, input-construction, preprocessing-fixture, runtime-consequence, migration, step-response, and distortion properties, SHA-256 binding, raw evidence JSON, and a compact Engineering Bundle that adds one raw constructive input tensor only when certified. Raw Data adds one canonical exact RGB PNG without fragmenting the compact bundle into per-candidate files. |
| Workspace UX | Stable model/session context, workflow rail, separate execution/result panes, output-only report review, and small-screen wrapping contracts. Explorer opens with Blocks, exposes deterministic per-op cache payloads on a W-by-C surface with selected-op-specific L1/L2 contours, execution-order pressure, storage/L1 what-if controls, block coloring, and WASM-produced projection vectors. The TFLite-only Redesign workspace binds source and target hashes, preserves a permanent `PROJECTED_UNTRAINED` boundary, exports JSON/Markdown evidence, and suppresses ONNX projection rather than applying an approximate TFLite rule. Its node view reserves fill for the current/projected contract state (including pre-edit quantization, cache, delegation, alignment, and packing signals) and reserves borders for direct edits and deterministic propagation; every state also has a text label and evidence-linked detail. Large model/session and graph-control surfaces remain in document flow so they cannot cover Explorer evidence while scrolling. |
| Medical surface | `/medical` adds Regulatory Report/Bundle without duplicating the app shell. Regulatory package includes Engineering Report as appendix. |
| Account/admin | Cloudflare Worker auth, password login/signup, Google OAuth, email verification plumbing, D1, account/admin pages, access profiles, requests, feedback, research consent. |

The ONNX recursive engine conserves its top-level status from FunctionProto,
control-flow, SequenceMap, and executed-scope fail/partial rows. Exact scope
execution, definition, fully-assessed, residual-node, and residual-output
totals are decision fields in the Engineering Report, ML-BOM, viewer, and
bundle conformance rather than raw-only calculations.

The artifact-side I/O contract preserves the exact ordered graph input/output
tensor indices and emits `deepbom.input_tensor_contract.v1` for every graph
input. A rank-four tensor is not treated as proof of layout: NHWC or NCHW is
`DERIVED` only when a direct source-defined image operator consumes the graph
input at its activation slot. Scalar quantized real ranges require complete,
finite, positive one-scale/one-zero-point metadata; per-axis, incomplete,
unsupported, or non-quantized source preprocessing ranges remain null with an
explicit status. Engineering Report, ML-BOM, viewer, and independent bundle
conformance preserve and reconstruct the same contract.

Dynamic dimensions use the separate
`deepbom.dynamic_shape_cost_contract.v2.1` evidence contract. ONNX `dim_param`
identity is conserved, anonymous and TFLite signature dimensions remain
independent without an artifact equality statement, and no numeric bound is
invented. Dense tensor payloads, supported convolution/matrix MACs, total MACs,
and graph live-set candidates are exact non-negative integer polynomials after
symbol binding. Sub-byte payloads preserve ceiling expressions when necessary.
A symbolic peak is reported only under coefficient-wise dominance; otherwise
the complete candidate ledger states that runtime dimension binding is
required. TFLite concrete shapes are example projections rather than bounds.
The Engineering Report, Overview, Explorer, finding queue, ML-BOM, metric and
decision coverage, and independent conformance retain the same contract and
reject coefficient or symbol-occurrence tampering.

Cache pressure is profile-bound rather than global. The default Cortex-A55
profile uses 32 KiB L1D and 128 KiB private L2, with selectable 16 KiB and
64 KiB L1D variants. RPi4/A72 binds the Cortex-A72 fixed 32 KiB L1D and
BCM2711 1 MiB shared-L2 selection without presenting 1 MiB as a universal
Cortex-A72 property. The source-bound Zynq UltraScale+ profile uses the AMD
DS891 product configuration: 32 KiB L1I and L1D per Cortex-A53, 1 MiB shared
L2, CG dual-core versus EG/EV quad-core, NEON, and an operating target up to
1.5 GHz. The Cortex-A53 TRM supplies the 64-byte cache lines and 2-way L1I,
4-way L1D, and 16-way L2 organization. x86 and WASM retain explicitly
conservative host-dependent references. Bound hardware records include source
document, revision, page scope, and local PDF SHA-256; the complete record is
included in the target-profile digest. Sustained throughput, memory and packing
bandwidth, thread/runtime configuration, and delegate overhead remain
separately labeled HEURISTIC performance-model inputs rather than hardware
facts.
Row working sets at or above 0.90x are watch conditions, so the bundled INT8
MobileNetV2 sample reports `31.5 KiB / 32 KiB = 0.984375x` and one L1 watch on
the default A55 profile. Deployment Frontier v1.3 exports per-target L1/L2
denominators, maximum ratios, and watch counts. These exact ratios do not add
an uncalibrated cache-miss penalty to latency estimates; executed tiling,
conflicts, miss traffic, and penalties remain runtime-observation requirements.

Explorer reuses the report's `deepbom.model_at_a_glance.v1.2` calculation rather
than maintaining a second summary formula. The first view exposes artifact
shape, quantization, intensity, delegation, cache pressure, target totals,
complete heuristic-score deductions, and a conserved op-time ledger where
roofline time is `max(compute, memory)` and is counted once. Its Cache view can
recompute row-working-set ratios against 16/32/64/128 KiB viewer-only L1D
denominators; leaving that view restores the profile-bound denominator, and the
what-if never mutates the target profile, report, profile hash, or latency.
Delegation Repair actions can project exact changed boundary edges and payloads
onto the graph, but the preview remains a static counterfactual until runtime
placement is observed.

The ONNX-ML engine also source-pins SVMClassifier-1 and SVMRegressor-1 against
ONNX 1.21.0 and ORT 1.26 CPU sources/tests. It independently conserves linear/SVC
mode, class-pair, support-vector, coefficient, rho, probability, output-width,
dtype-gap, and scalar-reference evidence across static JSON, findings, Engineering
Report, ML-BOM, domain viewer, and bundle conformance. Runtime-invalid layouts retain
every deterministically knowable expectation but classify serialized use as unresolved;
scalar references do not claim runtime-bit-exact MLAS or platform-libm execution.

The same source-bound ONNX-ML engine covers legacy
TreeEnsembleClassifier-1/3/5, TreeEnsembleRegressor-1/3/5, and indexed
TreeEnsemble-5. It conserves exact root, tree, node, branch, leaf, maximum-depth,
reachability, orphan, child/feature-reference, class/target weight, MEMBER-set,
and bounded source-order path/score evidence across static JSON, findings,
Engineering Report, ML-BOM, domain viewer, and bundle conformance. Pinned native
ORT fixtures cover ordinary execution and schema-valid CPU registration gaps.
The score preview remains a deterministic source-order reference, not a claim
about an unobserved ORT thread partition, reduction order, optimized graph,
selected EP, microkernel, or platform libm result.

Standard-domain TfIdfVectorizer-9 is now source-pinned independently to ONNX
1.21.0 and ORT 1.26 CPU schema, reference, kernel, and test files. The analyzer
validates the complete serialized constructor contract, derives exact output
shape/coordinate ownership, and evaluates bounded static STRING/INT32/INT64
inputs across every permitted skip distance. It reproduces ORT's repeated
FLOAT32 additions, compares ONNX-reference multiply-once results, preserves
signed zero, and fails closed on unsafe weight coordinates. A separate verifier
reconstructs rows from public tensors and serialized attributes, while findings,
Engineering Report, ML-BOM, domain viewer, and bundle conformance conserve the
same definition, match, output, alias, and divergence ledgers.

Exact XNNPACK selector ownership is confined to
`protected/deepbom_wasm/src/xnnpack_selector.rs`. The public Rust analyzer emits only
profile-level kernel guidance and an explicit advanced-selector-not-loaded
state. Source privacy checks reject exact A55/NEON or WASM SIMD candidate
enumerators in `src/`, while release preflight re-fetches the pinned official
XNNPACK commit, verifies the GEMM/DWCONV file digests, and resolves all protected
source line references before producing deployable bytes.

The pinned native TFLite path is executable end to end on a device-free host:
verified TensorFlow/XNNPACK archives are patched, Bazel builds the instrumented
benchmark and matching PDB, and the collector binds original-op placement,
lowering, exact dispatched XNNPACK symbols, host CPU features, build flags, and
post-commit TFLite arena snapshots to artifact/runtime/profile hashes. Runtime
memory v1 includes complete allocation/alias ledgers and a canonical digest;
the browser independently checks structural invariants and reconciles the final
snapshot against the source-pinned declared-shape ArenaPlanner projection.
This closes host-runtime plumbing and x86 selector ambiguity, but does not turn
host execution into an ARM/NEON device claim or account for delegate-owned
buffers, non-arena scratch, allocator metadata, stacks, constants, or RSS.

Graph low-norm filter evidence is now bound during the primary Rust/WASM model
analysis instead of re-entering the complete analyzer once per kernel during
rendering. INT8 and legacy UINT8 weights use the serialized quantization
contract `scale * (q - zero_point)` with the op-defined output-filter axis;
invalid per-axis cardinality or axis binding remains unassessed. On the bundled
3.41 MiB MobileNetV2 INT8 sample this removed the layer-count-dependent render
pass while retaining all 53 eligible kernel rows; a local headless-Chromium
four-target full-audit check measured 15.2 s versus approximately 30 s before
the change. This is a development-host measurement, not a cross-device SLA.

## Advanced Modules

- DEEPBOM: authorization-scoped WASM for deploy-artifact byte, quantization, and
  topology proxy signals plus exact pinned-source XNNPACK configuration
  enumeration for supported TFLite/A55 and TFLite/WASM-SIMD planning profiles.
  Candidate sets are merged only after schema, method, target-profile digest,
  source commit, source-file digests, op coverage, artifact facts, no-match
  reason, unresolved-gate set, candidate tile/tail arithmetic, and aggregate
  worst-tail ledger validation. They do not identify the executed runtime
  microkernel. The Advanced result summary and Kernel & Runtime Inspector expose
  unique/ambiguous/no-match counts plus direct links to the highest-tail ops.
- Perturbation: Research Beta local TFLite/LiteRT baseline vs perturbed output drift with
  RMS/mean/max, cosine distance, top-1 flip, output count, and timing notes.
  Contracts require distinct `+epsilon`, `-epsilon`, and `2epsilon` probes.
- Loss Landscape: filter-normalized weight directions compare requantized
  LiteRT.js output drift with the WASM floating reference. The displayed
  INT8/f64 gain is the through-origin least-squares coefficient
  `sum(delta_f64 * delta_int8) / sum(delta_f64^2)`. Cross-seed bands and
  Hessian uncertainty use sample SEM `s / sqrt(n)`; INT8 half ties use the
  pinned ties-away-from-zero quantization rule.
- Runtime Backend Compatibility: Research Beta local backend availability, drift, timing, and
  failure evidence, reported as `Runtime Backend Compatibility and Numerical
  Consistency`.
- Deploy Stability: Research Beta finite-difference probes of deployed function
  behavior: directional curvature, second-difference RMS, local Lipschitz proxy,
  basin radius/score, decision margin, and top-1 stability.

All advanced outputs are deployment-artifact or browser-runtime evidence only.
They do not claim training-loss Hessian, PAC-Bayes flatness, clinical
robustness, or generalization proof.

Private optional modules keep capability manifests inside their gated WASM
packages so future UI surfaces can be wired from stable module metadata without
hard-coding private workflow internals into the public app shell.
Local checks use a private WASM module registry so each optional package can
declare its build command, artifact budgets, leak guard, source/export/schema
contract, and runtime smoke contract in one place before it is exposed in
product UI.
Private source budget classification, generated-artifact privacy checks, and
deploy-skip/private-WASM-smoke decisions read the same registry so unreleased
modules stay out of public dist while still receiving release-WASM validation.

## ONNX Runtime Evidence

The public Kernel & Runtime view accepts ONNX Runtime Chrome-trace JSON and
binds the exact input bytes by SHA-256. Mapping is fail-closed:

- exact original graph-node name plus op type is accepted;
- unnamed node index plus op type is accepted only when graph optimization is
  explicitly declared disabled;
- fused, renamed, ambiguous, unmatched, or provider-conflicting runtime nodes
  remain unresolved;
- duration mean, sum, and sample count remain independently recomputable;
- only declared sequential runs with equal sample counts receive additive
  per-original-op duration semantics.

Runtime version, build flags, graph optimization, execution mode, and collection
time are declared metadata. Provider, node identity, and duration rows are
observed profile evidence. An optional runtime-binary SHA-256 is reported when
available; without it, the declared runtime build is not cryptographically
identified.

`npm run capture:pinned-ort -- <model.onnx>` adds the host-native path. It pins
`onnxruntime-node@1.26.0`, records npm integrity plus every host runtime binary
SHA-256, and emits optimization-disabled identity and optimization-all
production envelopes. For models with `external_data`, the collector validates
canonical model-relative paths, regular-file containment, file/aggregate size
limits, ranges, dtype-shape cardinality, and declared SHA-1 before creating an
immutable model-plus-sidecar snapshot. Both sessions execute that snapshot, and
the collector verifies the complete content set again before deleting it. The
envelope records the model hash, sorted external file/range ledger, ledger hash,
and full content-set hash. The browser rehashes the envelope and embedded
profile and reconstructs the active artifact/sidecar content set before import.
The Engineering Report retains the content-set SHA-256 plus external file and
tensor-range counts. The collector-recorded native binary inventory remains
explicitly not rehashed by the browser. The identity profile is mapped to
original ops; the production transformed-node ledger is retained without
unsafe reverse mapping. Output max/mean/RMS, relative-L2, and cosine deltas are
derived across the paired zero-input sessions.

ORT node events do not expose original runtime partition identity or executed
microkernel symbols. The adapter therefore reports contiguous provider segments
separately, leaves runtime partition count unassessed without explicit IDs, and
never emits ONNX placement or boundary mismatches against the TFLite XNNPACK
rulepack. The protected ORT rulepack now resolves standard, ONNX-ML, and 113
`com.microsoft` schema identities before comparing source-documented EP kernel
version ranges. Its pinned extractors preserve 938 EP/operator rows across CPU,
WebGPU, WebNN, DirectML, QNN, CoreML, NNAPI, and XNNPACK; all 611 CPU
registration signatures and type constraints; 540 machine-evaluated artifact
conditions; 432 explicit unresolved source fragments; and 425 informational
source notes. Definite dtype, rank, static-shape, constant-input/value,
output-count, or explicit-attribute failures remove only the affected source
candidate. Passing or unresolved clauses do not prove GetCapability
partitioning or observed assignment. Standard-domain IR/opset runtime floors
are source-backed. For the `com.microsoft` domain, 29 SHA-256-pinned official
release-tag `ContribOperators.md` inventories now derive one first-release row
for every current pinned contrib schema; every used contrib op participates in
the combined parser/schema floor. Unknown contrib identities, non-v1 imports,
and all other external custom domains remain partial. This history does not
establish reduced-build inclusion, EP kernel availability, `GetCapability`
acceptance, or execution assignment.

The pinned native collector separately records the selected
`onnxruntime-node` package's sorted `listSupportedBackends()` inventory and its
SHA-256. The browser rehashes that inventory and cross-references each reported
backend to the protected source profile. This binds package-level backend
presence without inferring `GetCapability` acceptance, transformed-node
ownership, or executed assignment. An optional official-format
`--reduced-op-config` input is now byte-hashed, parsed once through the shared
strict config parser, canonical-hashed, and compared with main, nested, and
FunctionProto-body operator identities. Exact imported-opset inclusion,
definite missing identities, other-opset uncertainty, and unresolved type
reduction are conserved separately. For official prebuilt packages the config
remains `IMPORTED_CONFIG_NOT_BINARY_ATTESTED`. The pinned-source build path
instead hashes the exact ORT checkout and build drivers, supplies the normalized
config through `--include_ops_by_config`, records type-reduction enablement, and
binds the resulting Node package manifest plus sorted binary inventory in a
canonical source-build attestation. That path emits
`BUILD_INPUT_BINARY_ATTESTED`; it establishes configured inclusion intent for
the selected build, not kernel registration, `GetCapability` acceptance,
transformed assignment, successful execution, or correctness.

The public source-pinned ONNX-ML value engine now closes fourteen contracts. Its
Binarizer-1 path derives same-type/same-shape output, the schema-default or
explicit threshold, exact initializer-backed 0/1 distribution, and downstream
static output values. A pinned `onnxruntime-node@1.26.0` regression verifies
schema-default injection and confirms that FLOAT64 Binarizer is rejected by
the CPU execution path even though it is schema-valid; that distinction is a
separate High finding and is independently reconstructed during export.
Normalizer-1 derives FLOAT32 same-shape rank-1/2 output, schema-default MAX or
explicit L1/L2 mode, and exact initializer-backed row arithmetic in pinned ORT
operation order. Zero and negative-MAX rows, integer-to-FLOAT32 changes,
signed abs/square overflow, non-finite outputs, and signed-zero outputs are
conserved through a JSON-safe value array plus exact per-tensor signed-zero
count/index ledger, then propagated through
findings, report, ML-BOM, viewer, independent export reconstruction, and native
ORT fixtures. Overflow-affected outputs are suppressed rather than fabricated.
Scaler-1 derives FLOAT32 same-shape output and separates the optional-by-schema
attribute contract from the stricter pinned ORT CPU implementation. It requires
explicit nonempty equal-length scale/offset arrays, validates scalar or rank-
dependent feature-stride cardinality, reproduces FLOAT64 versus FLOAT32/integer
operation order, and counts exact integer projection changes, zero scales,
non-finite parameters/outputs, and signed-zero outputs. Invalid runtime
contracts and non-finite static results are not propagated. Public attribute
text, tensor signed-zero indices, independent export reconstruction, native ORT
fixtures, report/ML-BOM/viewer bindings, and tamper tests close the evidence path.
Imputer-1 preserves the input dtype and shape while reconstructing pinned ORT
CPU replacement semantics from complete initializer values. The ledger validates
the exclusive FLOAT/INT64 attribute family, schema-default replacement marker,
rank-dependent feature stride, per-feature versus scalar selection, NaN-marker
equality, and the runtime's scalar-first behavior for list lengths outside one
or the feature width. Exact replacement, NaN replacement, unchanged, ignored
configured-value, non-finite output, and signed-zero counts flow through findings,
report, ML-BOM, viewer, and independent export reconstruction. Native FLOAT32 and
INT64 fixtures verify output equality; malformed attributes, FLOAT64/INT32 CPU
kernel gaps, and aggregate/row/attribute/output tampering are rejected explicitly.
OneHotEncoder-1 closes category-axis shape, exact numeric/STRING lookup,
duplicate last-write ownership, all-zero unknown policy, `zeros=0` deterministic
failure, numeric-to-INT64 conversion, and exact one/zero accounting. The source-
pinned LabelEncoder-1/2/4 path derives same-shape output type and exact mapping
effects, preserves INT64 values as decimal text, and distinguishes schema-valid
contracts from the pinned ORT CPU dtype-pair registrations. Serialized native
fixtures verify ordinary mapping, version 4 ONNX last-key versus ORT first-key
duplicate ownership, version 2 NaN defaulting, and mismatched key/value runtime
rejection. Conflicting exact outputs are retained as evidence but never propagated.
The source-
pinned LinearClassifier-1 and LinearRegressor-1 paths derive exact output contracts,
coefficient/intercept conservation, label/target semantics, CPU dtype gaps, and
post-transform hazards. Scalar FLOAT32 references remain explicitly non-bit-exact
until an MLAS microkernel and accumulation order are observed. Native ORT fixtures
confirm multiclass and binary classifier scores/labels, regression values,
FLOAT64 regressor rejection, and ONNX `targets=1` default materialization; the
last check retired a source-reading false positive before release.

## Deployment Frontier

For TFLite, the public Rust/WASM engine evaluates the same artifact against
four pinned planning profiles. Each op's non-negative modeled compute, memory,
packing, predicted-boundary, and predicted-fallback components are normalized
to a per-target contribution distribution. A deterministic descending prefix
first reaching 80% is selected for each target; the union is the robust
hotspot set. Pair distance is Jensen-Shannon divergence divided by `ln(2)`, and
hotspot overlap is set Jaccard. Every pair distance is decomposed exactly into
non-negative per-op JSD terms, then sorted into a deterministic 80% explanation
prefix. Each driver also carries rank change, bound/component transitions, and
the signed change of each modeled component divided by its target total.
Counterfactuals remove exactly one named modeled component and are labeled
upper bounds. They are not device timing and must not be added together.

For ONNX, protected WASM preserves the pinned CPU, WebGPU, WebNN, DirectML,
QNN, CoreML, NNAPI, and XNNPACK schema-plus-kernel-version match sets and the
narrower sets remaining after artifact-visible definite exclusions. The
frontier reports source gaps,
definite exclusions, unresolved condition ledgers, source and narrowed
all-provider intersections, assessed-MAC conservation, and both pairwise
Jaccard matrices. These rows are candidates only. Uncompiled source clauses,
ORT Web/WASM reduced-build inclusion, device/browser capability, GetCapability
partitioning, graph transforms, and runtime placement remain unresolved until
separate runtime evidence is bound.

## Deployment Delta

The browser can pin the current TFLite audit as a baseline in active-tab memory
and compare it with a subsequently audited TFLite candidate. Neither artifact's
bytes are persisted or uploaded. Rust/WASM independently parses both artifacts
for each requested pinned planning profile and hashes both byte arrays inside
the same WASM boundary.

Alignment is deterministic and linear-space. Pass one uses Hirschberg LCS over
op type plus input/output dtype, shape, constant role, and quantization
presence. Pass two uses op type only inside unmatched exact-anchor gaps. Ties
select the earliest candidate coordinate. Every baseline and candidate op is
accounted for exactly once; unmatched operations remain explicit additions or
removals. The coordinate does not establish semantic layer identity, model
lineage, or training equivalence.

For every target, candidate minus baseline static time is decomposed over all
alignment entities and into compute, memory, packing, predicted-boundary, and
predicted-fallback terms. The roofline base uses `max(compute, memory)` and is
assigned exactly once to the active bound component. The browser independently verifies artifact binding,
complete op coverage, target-profile digests, graph arithmetic, component
arithmetic, positive/negative driver totals, exact signed conservation, and
cross-target consistency. The viewer exposes overview, targets, drivers, and
alignment ledgers; the Engineering Report, static evidence JSON, and bundle
retain the same evidence and interpretation boundary. These values remain
static planning estimates rather than runtime measurements.

## Delegation Repair

The TFLite XNNPACK workspace includes a public delegation counterfactual lab.
For every graph op, Rust/WASM flips only that op between the predicted delegate
and CPU domains, recomputes contiguous execution-plan segments, and compares
the complete producer-to-consumer boundary-edge set. Repair candidates rank
predicted CPU ops whose hypothetical support removes boundaries or merges
segments. Fragility rows rank delegated ops whose hypothetical support loss
adds boundaries or splits segments. The edge ledger preserves tensor identity,
shape, dtype, payload status, direction, and exact bytes when deterministically
assessable.

Every maximal contiguous predicted-CPU execution-plan run is then toggled as
one complete island intervention while all assignments outside that run remain
fixed. The result records the best beneficial single member and exact additional
edge/payload reduction achieved by the full set. A group-only repair therefore
means the complete run reduces fragmentation while no member does so alone; it
does not mean the set is a minimal or implementable source patch.

Canonical boundary payloads distinguish fixed static shapes from a narrowly
defined serialized-batch-one projection. The latter is emitted only when the
shape signature leaves batch dynamic, the serialized shape binds batch to one,
all remaining dimensions match and are non-negative, and dtype width is fixed.
Every edge retains that binding; runtime copy materialization and latency remain
unassessed.

Source-qualified scenario rows are separate from single-op support toggles. The
current SE export intervention requires an exact repeated
`MEAN -> FULLY_CONNECTED -> FULLY_CONNECTED -> EXPAND_DIMS -> EXPAND_DIMS -> ADD`
motif, rank and channel agreement, pinned MEAN rank-condition evidence, and a
pinned quantized-FC rule with no rank predicate. It reports a fixed-graph
assignment proxy for a `keepdims` re-export, not the assignment of an artifact
that has not been generated and re-audited. Runtime-build scenarios group every
currently predicted-delegated op by its emitted XNNPACK build requirement and
report the conditional coverage collapse if that requirement is absent.
One-op delegate segments are inventoried separately because profitability
requires runtime measurement.

The WASM result is independently reconstructed and validated in the browser,
bound to artifact SHA-256 and target-profile SHA-256, retained in static JSON,
Engineering Report, evidence and public-share exports, and checked by bundle
conformance. The method never promotes hypothetical support to source-backed or
runtime-observed support, never treats op count as implementation effort, and
never treats logical payload as measured copy or latency.

## Quant Evidence Chains

The Quant workspace groups its 18 labs into four collapsible proof chains:
Lattice and Residual Contracts, Integer Arithmetic Safety, Numerical ABI and Build
Reproducibility, and Preprocessing Contract. The grouping changes navigation,
not evidence ownership; every lab keeps its schema, validator, digest, and
download contract.

Integer Arithmetic Safety adds an exact-channel convergence summary only when
the same `op_index` and `channel_index` are present in assessed accumulator,
kernel-witness, requantization, and vitality ledgers, and the vitality ledger
proves that channel constant under both pinned rounding paths. It never joins
different worst-channel rankings into one claim. Accumulator Reachability also
states why a kernel-local source can be promoted to a complete model-input
tensor witness only when its activation input is a declared model input;
intermediate activation sources remain explicitly upstream-unresolved.

## Quantization Lattice

For every eligible TFLite `ADD`, `SUB`, `MUL`, `MAXIMUM`, and `MINIMUM`, the
public Rust/WASM analyzer reads both input contracts and the output contract,
then exhaustively enumerates all `256 x 256` legal input-code pairs under the
operator's exact real-value semantics. `CONCATENATION` uses a separate branch
projection: every legal code of every input is requantized into the output
contract, for `input_count x 256` exact projections.

Each row retains the input/output real intervals, legal-result interval, continuous
interval coverage, 65,536-pair endpoint-escape and rounded-clamp counts,
in-range nearest-grid error, all-pair clamped projection error, exact 256-bin
output histogram, and exact 16x16 tile aggregates. Candidate operators outside
the v1.4 contract remain present with a `not_assessed` reason instead of zero-valued
metrics. The browser independently repeats the complete enumeration and rejects
tampered counts, histograms, tiles, errors, worst pairs, summaries, or rankings.

The Contract Design pass evaluates every legal output zero-point and adjusts the
analytical scale by one binary64 ULP until emitted endpoint arithmetic is both
containing and minimal. It retains the non-dominated `|zero-point delta|` versus
scale frontier, then reprojects the fixed-zero-point and globally finest
contracts across all 65,536 pairs to preserve clamp, code-utilization, and
mean/maximum-error ledgers.

The Quantization Lattice Lab exposes all six operator families as independent
tabs with artifact-specific assessed/candidate counts. The ADD family retains
domain escape, projection error, projected-output-code, and containment-design
views, links rows to Graph Explorer, exports validated JSON, and adds the
highest-ranked ADD map to the existing Visual PNG set. Engineering and
quantization evidence retain the same schema-bound ledger, while conformance
reconstructs it from tensor contracts.

This is uniform legal-code-domain geometry. It is not an observed activation
distribution, runtime saturation frequency, calibration result, accuracy loss,
or task-risk estimate. The ideal projection does not claim the executed
fixed-point multiplier, kernel rounding, fused activation, or hardware result.
Containment contracts are re-export counterfactuals, not calibration advice or
safe FlatBuffer edits; downstream contracts and task outputs require validation.

## Accumulator Headroom

The public Rust/WASM analyzer assesses every constant 8-bit TFLite
`CONV_2D`, `DEPTHWISE_CONV_2D`, and rank-2 `FULLY_CONNECTED` candidate. It
decodes stored weights and INT32 bias, expands per-tensor/per-axis zero-points,
and solves per-output-channel dot-product extrema over the complete independent
legal centered input-code interval by summing positive and negative centered
weights separately. The retained envelope includes zero, pre-bias extrema, and
post-bias extrema so bias cancellation cannot hide intermediate headroom.

All integer extrema are serialized as decimal strings to avoid JavaScript
precision loss. Each op retains every channel lower/upper envelope, signed-bit
requirement, histogram, overflow indices, worst-channel witnesses, a
metadata-only comparison, and a canonical SHA-256 channel ledger. The browser
independently re-decodes model bytes and reconstructs all 18,057 sample-channel
rows with `BigInt`, then verifies every digest before marking the viewer
independently verified. Report, static/quantization evidence, ML-BOM, JSON, and
Visual PNG exports preserve the same schema and source commit.

For the bundled quantized MobileNetV2 sample, 53/53 candidate ops and 18,057
channels are assessed; the maximum envelope is 4,482,645, maximum INT32 use is
0.208739%, maximum width is 24 signed bits, minimum headroom is 8 bits, and no
channel exceeds INT32. The top exact bound is 10.34653x tighter than its
metadata-only magnitude bound. These are exact legal-code-domain results under
the pinned TFLite reference algebra, not observed activation frequency,
accuracy, or an executed backend/microkernel claim.

## Requantization Fidelity

Every channel accepted by Accumulator Headroom is joined to the artifact's full
input, weight, and output float32 scale metadata. Public Rust/WASM reproduces
the pinned TensorFlow effective-scale calculation and `QuantizeMultiplier`
Q0.31 significand/exponent encoding for both default and
`TFLITE_SINGLE_ROUNDING` builds. The exact post-bias accumulator domain then
bounds the output-code drift caused only by multiplier encoding, while the full
accumulator envelope remains the intermediate INT32-safety domain. Separate source-derived
rounding envelopes cover the default double-rounding path and the single-round
path, and positive default pre-shifts are checked against signed INT32 before
the high multiply.

Each op retains all channel multipliers, shifts, represented scales, errors,
build-mode differences, and bounds. Canonical ledger rows encode every f64 as
its 16-digit IEEE-754 bit pattern before SHA-256. Browser JavaScript repeats the
bit decomposition, Q0.31 rounding, envelope propagation, aggregate ranking,
and digest verification independently.

The bundled quantized MobileNetV2 assesses 53/53 ops and 18,057 channels. Its
shift range is -11..-1, maximum relative multiplier error is
3.3838871e-10 (0.000338389 ppm), maximum encoding drift is 4.49658565e-6
output codes, default double-rounding bound is at most 0.75 code, and the
single-rounding bound is at most 0.500004497 code. No positive pre-shift hazard
or build-mode encoding divergence is present. The artifact does not reveal the
runtime compile flag or executed delegate, so the analysis never converts these
bounds into mismatch frequency, accuracy, or executed-kernel claims.

## Quantized Kernel Witness

Every constant 8-bit TFLite Conv/Depthwise/FC output channel accepted by the
Accumulator Headroom and Requantization Fidelity contracts receives two exact
synthetic full-valid receptive-field patterns. The canonical minimum assigns
input `qmin` to positive centered weights, `qmax` to negative weights, and the
input zero-point to zero weights; the maximum reverses `qmin` and `qmax`. The
resulting integer dot products and post-bias endpoints are checked against the
stored-weight accumulator ledger channel by channel.

Both endpoints execute the pinned TensorFlow default double-rounding and
`TFLITE_SINGLE_ROUNDING` equations through output zero-point addition and fused
activation clamping. Each op retains channel/term counts, exact assignment and
execution totals, mismatch and clamp counts, collapsed output spans, eight
ranked channel witnesses, an op ledger digest, and six source-file identities.
The browser independently decodes all constants and reconstructs every term,
endpoint, aggregate, ranking, op digest, and selected pattern digest before the
viewer is promoted to `independently verified`.

The bundled quantized MobileNetV2 produces 53 assessed ops, 18,057 channels,
6,942,080 canonical assignments, and 72,228 fixed-point endpoint executions.
The pinned default and single-rounding paths differ at 125 endpoints by at
most one output code. Op `#55` channel `767` provides the highest-ranked exact
example: maximum ideal/default/single output codes are `137/138/137` and its
pattern SHA-256 is
`35445b96d727a11cf2bdeab9dc0df29210496ffb5cac395e59bff80c97f7e31c`.

The viewer provides op and arbitrary numeric-channel selection, endpoint and
field segmented controls, term tooltips, fixed-point path comparison, ranking,
selected Witness JSON, report, static/quantization evidence, conditional
finding, ML-BOM properties, and Visual PNG. It is exact per-channel synthetic
evidence for one full-valid receptive field, not one simultaneous full-model
input, an activation distribution, a task-quality result, or observed runtime
execution. Runtime compile flags and observed target output remain separate.

## Quantized Channel Vitality

Every exact Kernel Witness post-bias interval is joined to the pinned default
and `TFLITE_SINGLE_ROUNDING` endpoint projections. Because the multiplier,
zero-point addition, and clamp paths are nondecreasing, equal endpoint output
codes prove one constant output code over the entire full-valid
receptive-field legal-code domain for that build path. Wider inclusive spans
are exact interval-hull cardinalities and upper bounds on distinct reachable
codes, not proofs that every interior code is reachable.

The ledger separates constant accumulators, lower/upper code clamps,
fixed-point projection collapse, and nonconstant channels. It also records
negative/positive post-bias sign lock, zero-containing intervals, default and
single-rounding compact code arrays, exact constant coordinates, ranked op
rows, and SHA-256 bindings to the source witness ledger. The browser
independently reconstructs all classifications and hashes before rendering.

The bundled quantized MobileNetV2 assesses 53 ops and 18,057 channels. Fourteen
channels are constant under both build paths: 11 constant accumulators and
three variable-accumulator channels at op `#1` (`3`, `12`, `16`) forced to code
zero by the lower clamp. Op `#1` channel `26` is mode-dependent: default is
`0..1`, while single-rounding is constant `0`. The report, static and
quantization evidence, High `EA-QNT-0112` finding, ML-BOM, conformance report,
selected Channel JSON, and Visual PNG retain the same coordinates and digests.
The proof is not full-model reachability, activation frequency, edge-padding,
delegate execution, calibration, or task-quality evidence.

## Fixed-Point Rounding Equivalence

Every Kernel Witness post-bias interval is partitioned into maximal contiguous
segments with one ordered `(default_output, single_output)` pair. Both pinned
paths are monotone integer step functions, so binary-searching each segment end
and advancing by one covers the complete closed interval exactly once. The
8-bit output range bounds the merged partition to 511 segments per channel.

For the bundled quantized MobileNetV2, all 53 ops and 18,057 channels cover
13,933,008,957 interval states. Exactly 2,874,544 states diverge by one output
code (`0.0206311789%` under uniform interval-state counting), 974 channels are
complete-interval equivalent, and 17,083 contain at least one divergence. The
portfolio contains 7,280,734 exact pair segments. The top coordinate is op
`#007` channel `37`: 191 of 2,041 states diverge, from accumulator `14`
(`3/2`) through the last difference at `1498` (`255/254`), with 447 segments
and ledger SHA-256
`6b42280ab896789a75ce996634eb5251c01c8fbd554216f1cdaddbf3ee62e9ab`.

The browser worker independently reconstructs all source witnesses, segments,
arrays, aggregates, rankings, histograms, selected traces, and per-op digests.
The result is an exact certificate over each integer interval hull. Interior
integers can still be unreachable due to discrete, correlated legal dot
products, so the ratio is not activation frequency, runtime mismatch rate,
accuracy, or evidence that either compile-time rounding path executed.

## Accumulator Reachability Lattice

Each Kernel Witness channel is reduced to a bounded nonnegative sum of stored
absolute centered-weight denominations. After GCD normalization, sorted
denomination groups extend an exact reachable prefix `[0, R]` when
`d <= R + 1`; the new bound is `R + d*c`, where `c` is the group's aggregate
legal-code capacity. The analyzer distinguishes complete integer, complete
modular, partial endpoint-band, and singleton proofs, then exactly intersects
the certified lattice with Rounding Equivalence divergence segments.

The bundled quantized MobileNetV2 has 13,320 complete integer channels, 34
complete modular channels, 4,692 partial endpoint-band channels, and 11
singletons. Its 13,933,008,957 interval states conserve as 13,755,523,449
certified reachable, 328,950 residue-excluded, and 177,156,558 unresolved.
The 2,874,544 divergent states conserve as 2,239,435 exact reachable numerical
ABI counterexamples, 3,585 residue-excluded states, and 631,524 unresolved
states. The browser independently reconstructs all proofs, selected aggregate
coefficient witnesses, arrays, rankings, and SHA-256 ledgers.

This is an exact full-valid kernel-local certificate under independently
variable legal quantized input codes. It is not full-model-input reachability,
activation frequency, padding freedom, runtime-path evidence, proof of a
declared-output change, or a task-accuracy measurement.

## Numerical ABI Propagation

Every build-mode-divergent rounding source and its Accumulator Reachability
partition are joined to the complete producer/tensor/consumer graph. The
analyzer retains exact source corridors, shortest declared-output paths,
residual reconvergence and single-branch merge classes, acyclic edge-sequence
route multiplicity, and predicted execution-domain edge payloads. The browser
independently reconstructs the graph and every source join, checks state
conservation, and verifies graph, rounding, reachability, and propagation
SHA-256 ledgers before rendering.

The bundled quantized MobileNetV2 has 52 divergent source ops. All 52 contain a
constructively reachable kernel-local counterexample and are structurally
output-reachable. Its exact state conservation is
`2,874,544 = 2,239,435 exact-local + 3,585 residue-excluded + 631,524 unresolved`;
9 sources have a residue facet and 32 have an unresolved facet, overlapping the
52 exact-local sources. The graph has 74 unique edges, 2,021 exact-qualified
source-corridor edge instances, 260 source/reconvergence instances, 29
source/single-branch merge instances, and a maximum of 1,024 exact output
routes. Its two unique predicted boundaries carry 64,000 B of logical payload.
The 3,264,000 B source-boundary sum repeats the same edges for every source
corridor and is explicitly an exposure inventory, not runtime traffic.

Exact-local qualification proves one full-valid kernel-local receptive-field
assignment under independently legal input codes. The downstream corridor is
structural potential only: it does not prove a full-model input realizes that
assignment, a declared output changes, a runtime copy occurs, or either pinned
rounding path executed. Paired deployed-build output tests remain authoritative.

## Model Input Tensor ABI Witness

Every exact-local divergent source is joined to its first activation input and
classified as tensor-ABI constructive, upstream-activation unresolved, or not
assessed. A direct quantized Conv/Depthwise source is promoted only after the
analyzer constructs a complete static rank-four model-input tensor: every
element begins at the quantized zero point, exact bounded-sum codes are placed
at full-valid NHWC receptive-field coordinates, and the full raw byte array is
hashed. The browser independently rebuilds the tensor, sparse override set,
patch, term products, dot-plus-bias accumulator, output-code divergence,
histogram, witness ledger, and portfolio ledger.

The bundled MobileNetV2 result conserves all 52 exact-local sources as one
constructive direct-input source and 51 unresolved intermediate-activation
sources. The representative first-layer certificate covers 18 channels and
2,918 exact divergent states. It constructs one 150,528-byte UINT8 input with
26 sparse overrides and proves `-13,115 + 13,159 = 44`, default output code 1,
and single-rounding output code 0. The raw tensor SHA-256 is
`89265147c9669c94eccbbdd5593623e04f1ba76190054786d88989aa6e5d3035`.

The certificate is exact at the model tensor ABI. Application preprocessing
realizability, deployed build selection, runtime frequency, declared-output
change, and task impact remain outside its proof boundary. The viewer exposes
the patch, 27-term arithmetic ledger, source portfolio, JSON certificate, and
raw tensor; reports, ML-BOM, findings, conformance, Visual PNG, Raw Data ZIP,
and the compact Engineering Bundle retain the same evidence.

## Pixel-to-Tensor Contract Lab

The Contract Lab evaluates the constructive input witness against eight
explicit RGB8 preprocessing counterfactuals instead of inferring an application
contract from model convention. For every tensor channel, public Rust/WASM
maps all 256 source pixel codes through an explicit direct-storage or exact
rational normalized transform. Normalized rows apply nearest-ties-away
quantization, zero point, and saturation. It inventories reachable
tensor codes, holes, collisions, inverse pixels, and the globally minimum
per-element approximation over the complete 150,528-element witness. The
browser independently regenerates all LUTs and an RGB fixture, encodes a
deterministic non-interlaced RGB8 PNG with stored DEFLATE blocks, decodes it,
and checks PNG CRC-32, zlib Adler-32, raw pixel SHA-256, candidate ledgers, and
the portfolio ledger before enabling downloads.

The bundled sample has four exact contracts and four non-exact contracts. Raw
RGB, raw BGR, artifact-affine RGB, and `(pixel-128)/128` RGB reproduce every
tensor code. `[-1,1]` has one code hole per channel and leaves 150,502 elements
one code away. `[0,1]` leaves only 18 elements unrealizable but their minimum
total absolute code error is 2,304. The exact/non-exact conservation is
`8 = 4 + 4`, and the portfolio SHA-256 is
`35a1ca877c09dd440ed75e7cfbe1789c6b3693ea6572da5069cf6a01aadd5bb5`.

These rows are counterfactuals, not observations of the production app. The
proof does not identify its decoder, resize/interpolation, color management,
channel order, normalization implementation, or quantizer. The exported PNG
becomes an authoritative replay fixture only after the production contract is
matched to the corresponding row and its produced tensor digest is checked.

## Preprocessing Consequence Atlas

The browser-local Consequence Atlas closes the gap between source-contract
realizability and model response without claiming that a production preprocessor
was observed. LiteRT.js 2.5.2 WASM compiles the model once and executes the
canonical model-input witness plus all eight Contract Lab tensors twice each.
Every repeat must be byte-identical. The browser independently reconstructs all
input tensors, complete output tensor-set hashes, exact integer differences,
equivalence classes, candidate ledgers, and the portfolio ledger before the
result is promoted to `MEASURED_SYNTHETIC`.

The bundled MobileNetV2 result is `8 contracts -> 4 input classes -> 4 output
classes`. The exact four contracts conserve the baseline output. The non-exact
contracts change `751`, `250`, `769`, and `769` of 1,001 output codes with
maximum absolute output-code differences `5`, `1`, `5`, and `5`; all retain raw
first-output top-1 index `535`. Both ImageNet channel-order rows collapse to one
input/output class for this witness. The runtime-bound portfolio SHA-256 is
`d51906882f2c5011c22b8883e3524b7e5c88d3c19e86a0a5171627d4558813e3`.

The panel exposes the contract-to-input-to-output fingerprint flow, selected
delta spectrum, exact certificate, full consequence matrix, equivalence classes,
JSON, selected candidate JSON, and selected raw output tensor. Engineering
Report, Raw and Engineering Evidence, findings, ML-BOM, conformance, Visual PNG,
Raw Data ZIP, and the compact Engineering Bundle retain the same result and the
same counterfactual boundary. Production decode/resize/normalization behavior,
device runtime, label semantics, representative-data frequency, task accuracy,
and user impact remain outside the evidence.

## Contract Migration Impact

For both exhaustive residual-containment candidates, public Rust/WASM traces
every exact direct tensor consumer and the complete reachable downstream graph.
Direct quantized Conv/Depthwise/FC consumers regenerate each channel's
effective multiplier, pinned TensorFlow Q0.31 encoding, shift, and INT32 bias
code. The bias calculation decodes the stored real bias and applies
`round_ties_away(bias_old * input_scale_old / input_scale_candidate)`, while
retaining representability, absolute error, error in both old and candidate
bias steps, complete channel arrays, and canonical SHA-256 ledgers.

Direct quantized ADD consumers reproduce the pinned TensorFlow general 8-bit
prepare path: left shift 20, twice the larger input scale, two input
multipliers, output multiplier, offsets, and all three Q0.31 encodings. The
browser independently re-parses the artifact and checks every consumer,
channel, parameter row, graph-depth row, aggregate, source binding, and digest
before rendering or export.

For the bundled quantized MobileNetV2 sample, 10 residuals produce 20 candidate
scenarios across 15 distinct direct consumers. All 30 consumer scenarios are
assessed: 9,504 kernel-channel scenarios, 9,504 changed multiplier encodings,
7,584 changed shifts, 9,500 changed bias codes, no candidate bias INT32
overflow, and 24 changed direct-ADD parameter encodings. The structural radius
contains 55 distinct downstream ops and reaches 28 edges, but it is kept
separate from the exact direct parameter-regeneration boundary.

This is a counterfactual re-export impact analysis. It is not a FlatBuffer
patch, runtime trace, calibration recommendation, task-quality result, or
proof that every reachable downstream op needs metadata regeneration. Runtime
build flags, delegated lowering, and executed microkernels remain runtime
evidence requirements.

## Residual Step Response

For every assessed residual ADD contract, public Rust/WASM exhaustively holds
one branch code fixed and increments the other branch by one legal 8-bit code.
Both sums are projected with the declared candidate output scale and zero-point,
round-ties-away-from-zero, and legal output clamping. The result separates
visible, silent, unclipped-silent, clamp-associated-silent, and multi-code-jump
transitions, retains output-delta histograms and reproduction error, and
classifies every 255x255 common interior pair as both/input-0/input-1/neither
visible.

The bundled quantized MobileNetV2 yields 10 residuals, 30 contracts, 3,916,800
branch transitions, and 1,950,750 joint cells. Its ten current contracts contain
404,848 silent transitions. Across both containment portfolios, 199,410 rounded
clamp-pair instances are removed at the cost of 511,902 additional silent
transitions relative to two current portfolios; the maximum per-ADD visibility
loss is 24.735%. Every contract carries a SHA-256 over ordered binary rows of
nine signed little-endian i64 fields. The browser worker independently rebuilds
all transitions, aggregate counts, and hashes before verification status is
promoted.

The field is exact for a uniform legal-code domain, not an activation sample or
probability distribution. It does not estimate entropy, mutual information,
calibration quality, task accuracy, branch activity, or executed runtime-kernel
behavior. Candidate containment remains a re-export experiment rather than an
in-place patch recommendation.

## Residual Contract Distortion

For each assessed residual ADD, public Rust/WASM compares the artifact output
contract against both containment candidates over every legal `256 x 256` input
code pair. It retains current and candidate raw/projected codes, represented
real values, clamp states, signed displacement in current-output steps, and
absolute ideal-projection error. Improved, worsened, and equal classifications
use `max(current_scale, candidate_scale) * 2^-40` as the explicit binary64
equality tolerance.

The bundled quantized MobileNetV2 yields 10 residuals, 20 candidate scenarios,
and 1,310,720 pair comparisons. Relative to two copies of the current-contract
portfolio, 199,410 clamp instances are rescued, 1,304,974 represented outputs
change, 532,893 ideal errors improve, 772,081 worsen, 5,746 are equal within
tolerance, and 3,764 change sign class. The maximum scenario RMS displacement
is 23.0624893 current-output steps and maximum p99 is 94.5209618 steps.

Every scenario retains exact 16x16 tile counts, displacement p50/p90/p99, mean
ideal-error delta, a worst-pair witness, and a SHA-256 binary pair ledger. The
browser worker independently reconstructs all 1,310,720 rows before the three
field views, histogram, report, evidence, ML-BOM, finding, JSON, and Visual PNG
are accepted. This is uniform legal-code-domain counterfactual evidence, not an
activation probability, calibration or task-quality recommendation, runtime
mismatch result, or executed-kernel observation.

## TFLite Runtime Evidence

The public Kernel & Runtime view accepts two official binary protobufs from
one declared `benchmark_model` capture:

- `ModelRuntimeDetails` supplies the execution plan, delegate names, delegate
  node IDs, and symmetric replaced-original-node maps;
- `BenchmarkProfilingData` supplies aggregate regular-run timing statistics;
- a user-supplied capture ID must match across the two imports;
- non-delegated execution nodes map by the formatter-emitted node-ID suffix and
  validated op type;
- delegate-node timing is retained once as a partition total and is never
  replicated onto replaced original ops;
- `Delegate/*` rows use delegate-internal IDs; primary-subgraph profiled events
  and nested delegate-section events remain separate, unassigned subtotals;
- primary execution-node run counts are derived only when the formatter reports
  one event per run; the graph total is withheld unless every execution-plan
  node has that same count;
- the official formatter computes `times_called` with integer division, so
  delegate-internal and other events never recover run count by inverting it.
  Their per-run sums use the common primary execution-node run count or remain
  withheld.

The parser independently checks `sum`, `count`, `avg`, min/max, first/last,
`times_called`, run order, node identity, and every emitted subtotal. Both raw
protobuf SHA-256 values and the pinned TensorFlow source-file digests are kept
in engineering evidence. Artifact binding is deterministic through exact
topology; runtime build, collection context, and capture identity remain
declared because the official protobufs do not embed them. Executed
microkernel symbols and tensor-copy materialization remain unobserved.

## Evidence Boundary

Reports and bundles use these evidence labels: `OBSERVED`, `DERIVED`,
`MEASURED_SYNTHETIC`, `ESTIMATED`, `PREDICTED`, `PROXY`, and
`NOT_ASSESSABLE`.

Default app exposes login-free watermarked Engineering and Regulatory Support
Reports plus profile-based Evidence Packages. Reusable raw derivatives, the full
Engineering Bundle, and controlled research evidence remain authorization-bound.
`/medical` presents medical-AI evidence organization, not market
authorization, certification, clinical validation, cybersecurity attestation,
or population-level generalizability.

## Canonical Artifact Evidence

Version 1.91 normalizes every supported adapter into
`deepbom.artifact_evidence_envelope.v1`. The envelope separates artifact
identity, interfaces, tensors/graph, required external files, provenance,
runtime requirements, findings, format extensions, and a complete capability
manifest. Evidence classes distinguish observed artifact facts, deterministic
derivations, heuristic derivations, imported declarations, measurements,
not-assessable results, and not-applicable analyses.

The CycloneDX 1.7 projector is canonical. The broad ML-BOM export reuses that
projector and retains its older flat properties only as a conformance
compatibility layer. Findings become `declarations.claims`; the hash-bound
envelope is declaration evidence; citations attribute the projection to the
analyzer; ONNX external data, Core ML package members, and SafeTensors shards
become file components connected to the model subject through `dependencies`.
Unknown training data or runtime versions remain provenance gaps rather than
invented components.

The format registry fails closed. TFLite uses the release-hardened Rust/WASM
analyzer. Every TFLite SubGraph now carries one shared-calculation,
one-invocation intrinsic ledger: shape-validated nominal dense MACs remain
separate from modeled-scenario and unassessed compute operators; tensor,
operator-I/O, graph-I/O, and constant payloads use one cached tensor assessment
with checked conservation. These rows are never summed across IF, WHILE,
CALL_ONCE, SignatureDef, or StableHLO computation references without observed
invocation counts. Every serialized subgraph is also passed through the same
target-aware op builder and independently receives quantization, source-pinned
XNNPACK candidate, boundary, liveness, pinned ArenaPlanner, movement,
weight-integrity, accumulator, requantization, and applicable fixed-point proof
ledgers. These local proofs are not promoted to a cross-control-flow execution,
arena, delegation, or latency total without runtime invocation evidence. ONNX
uses the bounded browser protobuf engine. GGUF and SafeTensors
decode metadata and tensor directories, conserve exact byte ranges, and stream
source-bound full-payload numerical-integrity scans where the stored encoding is
implemented. GGUF v3 little- and big-endian scalar/block payloads use source-field
endianness rather than whole-block reversal; byte-packed lanes remain byte ordered,
and Q8_K redundant block sums are checked against the serialized quants. GGUF
additionally checks 138 pinned llama.cpp architecture
registrations and nine build/backend prerequisite profiles without inventing an
execution graph. A SafeTensors bundle with `config.json` receives a bounded,
pinned Transformers architecture, canonical tensor-shape and parameter,
GQA/KV-state, sparse-MoE active/total expert, SSM recurrent-state, and bounded
compute-scenario contract for the registered Llama, Mistral, Qwen2, Qwen3,
Gemma, Gemma2, OLMo2, Granite, Phi-3, Cohere, Cohere2, Nemotron, Ministral,
SmolLM3, EXAONE 4, OLMo, Mixtral, and Mamba families. Configuration defaults and canonical state-dict module
  names are independently bound to pinned configuration and modeling file
  digests; split and fused projection layouts are not conflated. GGUF emits the same
KV-state and canonical decoder compute scenario when its registered architecture,
tokenizer cardinality, and required dimensions are serialized. A hash-bound
`deepbom.runtime.json` SafeTensors sidecar can separately conserve declared or
observed runtime identity, exclusive weight residency, complete layer placement,
and KV/SSM paging. Both LLM container paths now preserve a content-addressed
encoding inventory and tensor-to-encoding assignment signature. Static memory
rows add exact serialized tensor bytes to conditional logical KV/SSM state bytes
under an emitted simultaneous-residency, no-weight-paging assumption. Reference
capacity comparisons are valid only under that assumption and never promote an
at-or-above tier to a runtime fit. A bound runtime manifest separately
reports exclusive primary resident and allocated lower bounds while leaving
packing, replicas, workspaces, allocator, application, and OS reserve unbound.
Neither container is promoted to an executable DAG.

ONNX and TFLite additionally emit
`deepbom.serialized_llm_graph_evidence.v1`. The ledger preserves a deterministic
serialized graph signature, explicit Attention-family operators, bounded
MatMul/Softmax/normalization/Gather counts, and external state-name candidates
with exact logical bytes only when every dimension and dtype width is static.
An explicit Attention operator establishes only that serialized operator's
semantics; a decomposed motif remains `HEURISTIC`, and an interface name remains
a candidate. These rows do not infer architecture family, layer count, KV
layout, tokenizer, generation policy, task, or accuracy.

Accelerator placement now shares
`deepbom.backend_placement_projection.v1` across TFLite delegate and ONNX Runtime
provider adapters. Every original operator is identified by scope plus op index
and receives exactly one of `CONDITIONALLY_ELIGIBLE`, `DEFINITE_EXCLUSION`, or
`UNRESOLVED`. Segment, state, graph-edge, and logical-boundary-payload totals are
conserved. The associated workload envelope partitions only artifact MAC and
logical-byte ledgers; GPU peak compute, bandwidth, occupancy, generated
shader/kernel, transfer volume, and latency remain `NOT_ASSESSED` rather than
receiving a transplanted CPU roofline.

TensorRT ONNX preflight is split by execution path. The native collector calls
only `supportsModelV2` and the subgraph-query APIs, stages the model and explicit
external data in an isolated directory, verifies plugin hashes before loading,
and binds output to model/profile/binary/source-set/Git/TensorRT/CUDA/device
identity. It contains source guards against engine build, deserialization, and
execution APIs. ORT TensorRT EP requires a distinct identity-bound
`GetCapability` and profile-assignment capture. No native parser observation is
reused as ORT EP assignment.

An optional `deepbom.tensorrt_engine_inspector_evidence.v1` import separately
binds an existing serialized engine and TensorRT 10.x/11.x engine-information
JSON to the ONNX artifact, build profile, runtime/device identity, capture tool,
and invocation. It preserves optimized layer/tensor rows and selected tactic
identifiers without promoting source-name metadata to original-op assignment or
claiming tactic timing, kernel execution, physical transfer, allocation, latency,
or omitted next-generation optimizer subgraphs.

SafeTensors packages may include `tensorrt_llm_engine_config.json` and
`deepbom.tensorrt-llm.json`. The static TensorRT-LLM contract validates exact
world-size conservation across TP/PP/CP, explicit or quotient/remainder pipeline
layer partition, build limits, quantization declarations, plugin-config digest,
and conditional logical KV-state bytes. Its binding manifest identifies the
non-circular model-source digest computed only from shard-index,
architecture-config, and tensor-shard roles; adding the binding manifest changes
the complete bundle digest but not that subject digest. Per-rank weight bytes,
engine tactics, runtime allocation, workspace, occupancy, throughput, latency,
accuracy, and device feasibility are intentionally not inferred.

Core ML decodes the top-level description and FeatureTypes; legacy
NeuralNetwork and ML Program graphs; GLM, SVM, and TreeEnsemble numerical and
structural contracts; and named Pipeline stage composition where each nested
payload is supported. It scans supported embedded WeightParams, recurrent and
Conv3D parameters, classical FLOAT64 arrays, and complete package-bound ML
Program blobs; derives implemented shape/MAC and logical-liveness evidence; and
derives the necessary specification-to-OS floor. An identity/op-order-bound
MLComputePlan estimate is accepted only for decoded NeuralNetwork or ML Program
representations and must bind the selected function, compiled-model digest,
compute units, macOS version/build, hardware/device inventory, pinned plan source,
and collector source. No actual Apple-host capture is committed yet, and a plan is
never promoted to executed placement or timing. Legacy deterministic shape coverage includes convolution,
pooling, padding/resize/crop-resize, static reductions, tile/stack,
gather/scatter variants, static TopK and ArgMin/ArgMax, reshape families,
constant padding, embeddings, recurrent layers, and broadcast matrix products.
Dynamic shape-value inputs, unsupported model types, remaining operation-specific shape/cost rules,
custom-layer semantics, and runtime-only lowering remain explicit. Core ML
packages and SafeTensors shard sets use bounded strict JSON manifests, safe
relative paths, duplicate/case-collision rejection, streaming file hashes, and
complete manifest-to-file conservation. PyTorch pickle formats are rejected and
never deserialized.

## Code Map

- `src/`: TFLite Rust/WASM analyzer and preprocessing helpers.
- `web/app.js`: session orchestration, event wiring, workflow control.
- `web/lib/session-evidence.js`: deterministic Engineering Report HTML envelope,
  session privacy provenance, and complete runtime/weight evidence context
  assembly. Browser state is passed explicitly so report generation does not
  silently depend on hidden globals.
- `web/onnx.js`: ONNX protobuf parser and static-analysis path; it has no ONNX
  Runtime dependency.
- `web/lib/artifact-evidence-envelope.js`, `report-export-contracts.js`:
  canonical cross-format evidence model and CycloneDX 1.7 projection.
- `web/lib/metadata-model-adapters.js`, `coreml-metadata-adapter.js`,
  `artifact-bundle.js`: bounded GGUF/SafeTensors/Core ML metadata parsing,
  package/shard conservation, and streaming hash dependency inventories.
- `web/lib/onnx-benchmark.js`, `tflite-benchmark.js`: fail-closed runtime
  input/output binding, benchmark phase orchestration, output observation, and
  deterministic tensor/session cleanup.
- `web/lib/analysis.js`, `audit-ui.js`, `auth-ui.js`,
  `performance-visuals.js`,
  `graph-ui.js`, `benchmark-ui.js`, `preprocess.js`, `preprocess-ui.js`,
  `visual-export.js`, `runtime.js`: static interpretation, audit
  summary/dashboard/table UI, visual panes, graph row/detail UI, benchmark UI,
  preprocessing logic/UI, PNG exports, and shared runtime contracts.
- `src/quantization_lattice.rs`, `web/lib/quantization-lattice.js`: exhaustive
  residual ADD code-domain engine, independent verifier, and three-mode viewer.
- `web/lib/research.js`: advanced local evidence helpers.
- `web/lib/runtime-profile-adapter.js`, `kernel-inspector.js`: raw ORT profile
  and pinned native dual-profile identity mapping, canonical runtime evidence
  validation, production transformed-node ledgers, provider-transition
  comparison data, and Kernel & Runtime viewer rendering.
- `web/lib/bundle.js`: bundle specs, module envelopes, progress logs.
- `web/lib/dom.js`, `account-ui.js`: reusable browser-local DOM builders for
  cards, request rows, visual shells, protocol blocks, and canvas export
  helpers.
- `web/lib/elements.js`, `admin-elements.js`, `app-surface.js`: DOM bindings
  and `/medical` augmentation.
- `web/lib/report*.js`: reports, context, manifests, ML-BOM, findings,
  integrity/security posture, evidence sections, markdown/CSV/JSON/ZIP helpers.
- `web/lib/metric-coverage.js`, `decision-coverage.js`: metric ownership and
  field-routing manifest, plus independently conserved decision-domain status
  and front-page claim boundaries.
- `web/lib/telemetry.js`: opt-in structure fingerprint and timing payloads.
- `worker/`: Cloudflare Worker auth, authorization, admin, feedback, D1 routes.
- `scripts/`: local/deploy guardrails.
- `protected/deepbom_wasm/`: protected DEEPBOM WASM.
- `native/`: pinned capture contract, runtime trace sink, automatic collector, and deterministic contract probe.

## Quality Guards

Automated checks cover JS syntax/imports, dead/unused guards, DOM/workflow/CSS
contracts, advanced copy, export/report/bundle contracts, default vs
`/medical` separation, runtime warning copy, perturbation probe distinction,
hash/fingerprint helpers, git/dist privacy, private research exclusion,
Worker/D1/protected routes, service-worker coverage/versioning, deploy-skip
classification, low-height workspace CSS behavior, source/dist budgets, Rust
fmt/clippy/tests, private optional release-WASM build/load/runtime-call
integrity, private WASM registry schema integrity, and visualizer parity.
Release-dist checks also reject protected selector signatures in the public
analyzer WASM, require them in the authorization-scoped DEEPBOM WASM, and verify
the full Advanced result-to-Kernel-Inspector path at desktop and mobile widths.
The deploy assembler minifies every project JavaScript/CSS asset without source
maps, strips source-map references from third-party runtime files, emits a
`deepbom.deployment_hardening.v1` manifest, and rejects any `.map` artifact.
WASM checks reject custom/debug sections and internal source paths and pin both
the public analyzer and protected analysis export surfaces. These controls raise
copying cost and prevent accidental source disclosure; they do not claim that
browser-delivered code is cryptographically secret.
Formal release preflight additionally verifies 55 pinned XNNPACK configuration
source references against the immutable upstream commit; the networked check is
available independently as `npm run verify:xnnpack-source-pin`. The same
preflight fetches the pinned ONNX v1.21.0 generated schema changelog, verifies
its SHA-256, regenerates all 152-op/514-version formal contracts in memory, and
requires byte identity with the checked-in browser table via
`npm run verify:onnx-shape-schema`. It also re-fetches and SHA-256 verifies all
four pinned Sequence/Optional inference sources through
`npm run verify:onnx-container-source-pin`, and the six ONNX/ORT
TfIdfVectorizer schema, reference, kernel, and test files through
`npm run verify:onnx-tfidf-source-pin`.
Public-browser E2E checks independently cover raw ORT profile preview/import,
strict unnamed-node mapping, digest/event coverage, provider-only tracks,
observed transition edges, explicit mismatch non-applicability, no inferred
partitions or microkernels, and desktop/mobile overflow.
The pinned native ORT E2E additionally performs real dual-session captures for
both embedded and external-data models, verifies npm package and host-binary
identities, executes only an immutable verified snapshot, rejects envelope,
model, sidecar, file-ledger, and tensor-range tampering, requires 9/9 and 1/1
identity mapping, preserves the optimized production ledger without inferred
original mapping, verifies Engineering Report content-set disclosure, and
checks paired-output deltas in the desktop/mobile viewer.
The Deployment Frontier E2E uses the real quantized TFLite and ONNX samples,
recomputes bindings and arithmetic, rejects tampering, exercises all three
views, and checks desktop/mobile overflow for both format-specific matrices.
The TFLite E2E path additionally imports both protobufs against a real sample,
checks full execution-node timing, partition-total attribution,
delegate-internal isolation, latency hotspots, and desktop/mobile overflow.
The Residual Lattice E2E analyzes the real quantized MobileNetV2 sample,
reconstructs 655,360 legal code pairs, fixes the ten-ADD ranking and exact top
counts, rejects histogram/tile/arithmetic tampering, verifies the report and
quantization-evidence bindings, checks three distinct canvas pixel signatures,
tests the Visual PNG renderer, and enforces desktop/mobile overflow contracts.
The Accumulator Headroom E2E independently reconstructs all 53 ops and 18,057
channel envelopes, verifies every SHA-256 ledger, rejects channel/histogram/hash
tampering, checks exact overflow finding wiring, exercises all three canvas
modes and the Visual PNG renderer, and enforces desktop/mobile horizontal and
vertical scrolling contracts.
The Requantization Fidelity E2E independently reconstructs all 53 ops and
18,057 Q0.31 channel rows, verifies pinned source digests and every IEEE-754
ledger hash, rejects multiplier/bound/hash tampering, checks conditional action
wiring, exercises three canvas modes and the Visual PNG renderer, and enforces
desktop/mobile overflow and scrolling contracts.
The Quantized Kernel Witness E2E independently reconstructs all 53 ops, 18,057
channels, 6,942,080 canonical term assignments, and 72,228 pinned fixed-point
executions. It verifies every op and selected-pattern SHA-256, rejects endpoint,
count, op-ledger, and pattern-ledger tampering, checks report,
static/quantization evidence, conditional finding, ML-BOM, and conformance
bindings, exercises four endpoint/field canvas states, validates selected
Witness JSON and the Visual PNG, and enforces desktop/mobile overflow contracts.
The Quantized Channel Vitality E2E independently reconstructs all 18,057
channel classifications and both fixed-point output hulls, verifies op-ledger
SHA-256 values, rejects count/array/hash tampering, checks report,
static/quantization evidence, High finding, ML-BOM, and conformance bindings,
exercises build-mode plus code-span/cause/sign canvas states, validates selected
Channel JSON and the Visual PNG, and enforces desktop/mobile overflow contracts.
The Fixed-Point Rounding Equivalence E2E independently partitions all
13,933,008,957 interval states across 18,057 channels, conserves every segment
and divergent region, verifies per-op SHA-256 ledgers, rejects count/array/hash
tampering, checks report, evidence, Medium finding, ML-BOM, and conformance
bindings, exercises all three heatmaps and exact traces, validates selected
Certificate JSON and the Visual PNG, and enforces desktop/mobile overflow.
The Accumulator Reachability E2E independently reconstructs all 18,057 bounded
sum proofs, conserves all 13,933,008,957 interval states and 2,874,544 divergent
states across reachable/excluded/unresolved classes, verifies aggregate
coefficient witnesses and per-op SHA-256 ledgers, rejects arithmetic and digest
tampering, checks report, evidence, Medium finding, ML-BOM, and conformance
bindings, exercises four heatmaps and exact traces, validates selected Channel
JSON and the Visual PNG, and enforces desktop/mobile overflow.
The Numerical ABI Propagation E2E independently joins every rounding and
reachability source certificate to all 74 graph edges, conserves the
2,874,544-state exact/excluded/unresolved partition, verifies 52 exact-local
output-reachable sources, 2,021 exact-qualified corridor instances, 1,024-route
maximum multiplicity, and both unique predicted boundaries. It rejects state,
reachability-ledger, route, edge-payload, and propagation-ledger tampering,
checks report, findings, ML-BOM, and conformance bindings, exercises the four
overlapping source facets, validates both JSON contracts and the Visual PNG,
and enforces desktop/mobile overflow.
The Model Input Tensor ABI Witness E2E independently reconstructs all 150,528
input bytes from zero-point fill plus 26 sparse overrides, verifies the 27-term
NHWC arithmetic ledger, exact dot-plus-bias accumulator, divergent output
codes, complete tensor/witness/portfolio SHA-256 values, and all 52 source
classifications. It rejects term, override, digest, and classification
tampering; checks report, finding, ML-BOM, conformance, compact-bundle raw
binary, and Visual PNG bindings; validates JSON and raw-tensor downloads; and
enforces readable desktop/mobile layouts with zero page overflow.
The Pixel-to-Tensor Contract Lab E2E independently reconstructs eight complete
3x256 transfer LUTs, four exact and four non-exact candidate outcomes, every
witness inverse row, complete RGB fixtures, deterministic PNG bytes, CRC/Adler
checksums, fixture/candidate/portfolio SHA-256 values, and the exact minimum
error accounting. It rejects LUT, fixture, candidate-ledger, and portfolio
tampering; validates report, finding, ML-BOM, conformance, Raw Data, JSON, PNG,
RGB, and Visual PNG exports; exercises exact, `[-1,1]`, and `[0,1]` views; and
enforces desktop/mobile overflow contracts.
The Contract Migration E2E independently reconstructs both candidates for all
10 residual ADDs, all 15 direct-consumer identities, all 9,504 kernel-channel
scenarios, every direct-ADD prepare parameter, graph reachability and depth,
and every canonical ledger digest. It rejects channel, ADD-parameter, and hash
tampering, checks Engineering Report, evidence, ML-BOM, and conformance wiring,
exercises candidate switching and both canvas views, verifies its Visual PNG,
and enforces desktop/mobile overflow contracts.
The Residual Step Response E2E reconstructs all 3,916,800 adjacent-code branch
transitions and 1,950,750 joint-cell classifications, verifies every binary
ledger digest, rejects tile/branch/hash tampering, checks report, evidence,
ML-BOM, finding, and conformance bindings, exercises candidate switching and
both exact canvases, verifies its Visual PNG, and enforces desktop/mobile
overflow contracts.
The Residual Contract Distortion E2E reconstructs all 1,310,720 artifact versus
candidate pair rows, verifies every aggregate, percentile, tile, worst witness,
and SHA-256 ledger, rejects arithmetic and digest tampering, checks report,
static/quantization evidence, ML-BOM, finding, and conformance bindings,
exercises both candidates and all three exact fields, verifies its Visual PNG,
and enforces desktop/mobile overflow contracts.

Source budgets are reported as separate public runtime code, docs, private
optional runtime, private optional tests, and dev-check buckets. This keeps
shipped product code growth, documentation growth, local validation growth,
private verification growth, and unreleased private-module growth visible
without mixing their operational meanings.

Regression guardrails:

- SSE4 and WASM SIMD target profiles must not collapse into identical
  ridge/throughput/break-overhead behavior.
- Source-enumerated kernel rows must preserve every unresolved selector branch,
  artifact selector fact, no-match reason, source location, source-file SHA-256,
  candidate-specific padded/inactive channel arithmetic, and aggregate worst-op
  tie set; they must never claim an executed runtime microkernel.
- ONNX must never execute the TFLite XNNPACK selector rulepack; unsupported
  formats/profiles emit an explicit `not_available_for_profile` assessment.
- ONNX runtime provider rows must never be compared with TFLite static
  delegation predictions. Missing ORT partition IDs and microkernel symbols
  remain unassessed, while provider segments are labeled separately.
- Packing warnings distinguish general `>=10us` from small-FC/setup and
  low-reuse `>=5us` criteria.
- DEEPBOM UI uses JSON signal values as display source.
- Service worker app import coverage must include new browser modules.
- Residual-lattice pair counts, endpoint escape, rounded clamp, histograms,
  tiles, worst pair, and ranking must reproduce from tensor scale/zero-point
  contracts; uniform-domain ratios must never be presented as runtime rates.
- Accumulator envelopes, signed widths, histograms, rankings, and per-op SHA-256
  ledgers must reconstruct from the original model bytes; legal-code headroom
  must never be presented as an observed overflow frequency or executed-kernel
  guarantee.
- Requantization multipliers, shifts, encoding drift, dual rounding bounds,
  rankings, and SHA-256 ledgers must reproduce from artifact scales and exact
  post-bias accumulator domains. Full accumulator envelopes remain the source
  for intermediate INT32 path safety. The absent `TFLITE_SINGLE_ROUNDING` build flag must
  remain unresolved rather than being inferred from the model.
- Channel-vitality constant proofs, reason codes, sign classes, span
  histograms, rankings, compact endpoint arrays, and SHA-256 ledgers must
  reconstruct from the source witness endpoints. Span one is an exact
  full-valid receptive-field proof; wider interval hulls must never be
  presented as exact reachable-code counts, observed activation coverage, or
  full-model reachability.
- Rounding-equivalence interval-state counts, pair segments, divergence
  directions, regions, histograms, rankings, selected traces, and per-op
  SHA-256 ledgers must reconstruct from the pinned fixed-point paths and source
  witness intervals. Uniform interval-state ratios must never be presented as
  activation frequency, runtime mismatch rate, task accuracy, or proof that
  every interior accumulator is reachable.
- Numerical-ABI graph edges, source corridors, merge classes, shortest paths,
  exact acyclic route counts, predicted boundary payloads, rankings, and graph/
  source SHA-256 ledgers must reproduce independently. Repeated per-source
  boundary inventory must never be presented as physical runtime traffic, and
  structural reachability must never be presented as observed output change.
- Contract-migration candidate counts, direct-consumer identities, complete
  kernel-channel arrays, ADD prepare parameters, reachability depths, and
  SHA-256 ledgers must reproduce from the artifact and pinned source algebra.
  Every candidate/direct-consumer pair must be assessed or carry an explicit
  reason; direct parameter regeneration and reachable structural impact must
  never be conflated.
- Contract-migration results must remain labeled counterfactual re-export
  analysis. They must never be presented as an applied FlatBuffer patch,
  calibration suitability, task accuracy, runtime assignment, or executed
  microkernel evidence.

## Verification

Before functional pushes:

```bash
npm run check:deployable
npm run check
npm run check:rust
npm run check:source-budget
npm run check:private-wasm-build
npm run verify:local -- --target android_mid_a55 <representative.tflite> <second_model.tflite>
```

`npm run check:deployable` checks the current working tree against `HEAD` and
reports whether the change should trigger Cloudflare deploy work, Rust-only
quality checks, or check-only CI. Use it before committing when deciding whether
to bump the service worker and run live deployment verification.

After deployable pushes:

```bash
npm run wait:live-sw
```

Smoke:

- `https://deepbom.org/`
- `https://deepbom.org/medical`
- changed `/web/` module paths
- `https://deepbom.org/api/auth/config`

Keep generated `dist/`, `reports/`, Rust `target/`, private optional module
outputs, local model files, and local private notes out of git.

## Near-Term Plan

Priority 1: stabilization and maintainability.

- Keep the extracted graph/research workspace, report, target-profile, arena,
  and influence modules behind import, source-budget, and provenance-slice
  checks. `web/app.js` is below 300 KiB and `src/lib.rs` is below 400 KiB;
  further extraction must preserve those budgets rather than create new public
  surface by default.
- Preserve workflow order: Input, Audit, Graph, Runtime, advanced modules,
  Output.
- Keep Output as final report/export review, not execution surface.
- Keep model/session context stable while panes change.
- Continue improving small-screen and low-height workspace usability without
  changing the established workflow hierarchy.
- Keep source and dist budgets healthy before adding public modules.

Priority 2: report and evidence maturity.

- Keep scope/evidence boundary in every report and bundle summary.
- Maintain analyzer version, rulepack version, schema IDs, target profile,
  model SHA-256, and report-generation time.
- Maintain the exact TFLite byte-ownership ledger: referenced FlatBuffer
  ranges, metadata ZIP ranges, unowned trailing bytes, overlaps, collisions,
  and file-size conservation must remain identical across findings, viewer,
  Engineering Report, evidence envelope, and CycloneDX projection.
- Operate the official-release ES256 key lifecycle through the published
  `/.well-known/deepbom-signing-keys.json` trust anchor. Keep local browser keys
  distinct from official release keys and test validity, retirement,
  revocation, and DOI/domain binding on every key rotation.
- Use `Application-Observed Local Processing Record`, not formal no-egress
  proof.

Priority 3: local advanced evidence.

- Representative-dataset capture import now binds the audited artifact and
  complete external I/O contract, then reports exact bounded-integer endpoint
  saturation, supplied-reference output drift, and repeat-run nondeterminism
  with explicit sample/comparison/tensor/value denominators.
- Keep population representativeness, calibration quality, task accuracy,
  clinical validity, and release readiness outside that ledger unless a
  separately bound evaluation protocol establishes them.
- Add layer-wise robustness where metadata permits.
- Add safe deploy-artifact weight perturbation where mutable constant buffers
  are detectable.
- Keep runtime benchmark interpretation aligned with browser/runtime changes.

Priority 4: backend evidence closure.

- ONNX symbolic rank and dimension propagation plus explicit runtime I/O symbol
  binding are implemented. Across 48 unique-byte artifacts from 40 pinned
  repositories, all 41,462 observed nodes now have a source-bound local
  output-shape rule, including all 1,397 observed ORT contrib nodes. This is a
  measured-corpus result, not a claim of complete ONNX/ORT operator coverage.
  Slice inference preserves rank and unaffected dimensions under dynamic
  controls, recognizes only the dtype-specific INT32/INT64 extrema as open
  bounds, retains other unsafe INT64 values as exact finite bounds, and keeps
  clipping against symbolic extents as an explicit expression.
  Pad distinguishes absent attributes from runtime-bound input vectors,
  validates vector rank/cardinality, and preserves rank plus unaffected axes
  without inventing pad values. Expand emits the actual multidirectional
  broadcast result rather than assuming the requested shape is the result.
  Exact dimension-valued Range controls remain symbolic expressions; a delta
  that is not artifact-bound to a finite nonzero value remains unresolved.
  Bounded ConstantOfShape fills and dimension-only comparison/Where chains
  propagate only when every selected value is statically determined.
  ConvTranspose applies the source-defined output-size equation while retaining
  symbolic batch/spatial identities and checking channel and attribute
  conservation. Conv nominal MAC formulas use the output and weight contracts
  when the input rank is omitted, while still rejecting an explicitly declared
  incompatible scalar/rank or channel contract.
  The regenerated 48-artifact ledger contains 43,180 dense node outputs:
  40,785 have complete contracts, while 1,995 are unconditionally invalid and
  400 have at least one condition-bound invalid variant. The non-conflict
  static shape residual is zero. Six unconditional root conflicts and 543
  condition-bound invalid variants across nine artifacts invalidate those
  outputs; 1,901 downstream nodes preserve the upstream conflict instead of
  being relabeled as generic unknowns. Those artifact defects also block 213
  MAC-bearing rows. No total-MAC row remains blocked by an analyzer rule or an
  otherwise valid serialized contract in this measured population. Guarded
  integer expression IR v2 closes cropped ConvTranspose with the exact
  piecewise overlap sum and rank-omitted uncropped ConvTranspose with guarded
  inverse output-size arithmetic. Finite conditional Squeeze rank unions and
  nested Loop/If semantic failures are likewise preserved without selecting a
  runtime branch or computing through an invalid declaration. This is a
  measured-corpus closure result, not a claim that every possible ONNX custom
  domain or runtime-valued graph is statically decidable. Numeric arena and
  liveness values remain unbound wherever valid runtime dimensions are absent.
- Add `NPU Coverage` only with source-backed support tables and version labels.
- Selected-build inventory contracts now cover QNN, NNAPI, Core ML, WebGPU, and
  WebNN while keeping configured inclusion, capability acceptance, original-op
  assignment, and execution as four separate evidence layers. Convert further
  provider-specific support predicates into machine-evaluated artifact checks
  only where pinned source and build inputs make the condition deterministic.
- Extend selected-build evidence to DirectML and additional vendor backends only
  where a build system exposes a complete generated kernel/provider inventory.
- Add MediaTek NeuroPilot only with concrete public source and version caveats.
- Show NPU chain coverage, fallback boundaries, and unsupported op families
  without claiming runtime behavior.

Priority 5: model-family coverage expansion.

- The validation corpus now includes hash-bound SafeTensors dense-decoder,
  sparse-MoE, and recurrent-SSM packages; eight GGUF architecture/encoding
  anchors across four architecture classes and ten observed encodings; and
  five source-pinned Core ML MLProgram contracts covering static external
  blobs, enumerated shapes, bounded shape ranges, blockwise affine compression,
  and vector LUT palettization. These establish selected
  parser and contract behavior, not ecosystem prevalence or device placement.
- Core ML ML Program compression inspection now validates source-pinned iOS 18
  blockwise affine, LUT palettization, and sparse-mask shape/cardinality
  contracts. Block dimensions, scale count, index bit width, palette count,
  vector expansion, and logical output cardinality are deterministic; sparse
  mask population remains an explicit payload-decoding boundary. Runtime
  materialization, compressed-buffer residency, CPU/GPU/ANE placement, and
  latency are never inferred from these serialized contracts.
- ExecuTorch ET12/FT01 parsing now binds every KernelCall to a generated,
  content-digest-pinned v1.4.1 portable operator-signature registry. The
  registry covers 209 entries (200 operators plus nine functions), preserves
  source argument/return positions, validates EValue arity and input/output
  direction, and emits exact nominal MACs only for source-closed matrix and
  convolution signatures. Seven pinned backend IDs preserve processed payload
  byte ranges and hashes; public-schema payloads receive a bounded FlatBuffer
  root-envelope check. A strict `deepbom.executorch-build.json` can bind source,
  CMake options, backend/operator inventories, and runtime binary digests in
  both web and CLI. Delegate initialization, executed placement, kernels,
  allocation, physical transfer, correctness, and latency remain external.
- Continue expanding license-clear public populations before ranking ecosystem
  frequency or adding family-wide prevalence claims.
- A hash-bound upstream `ai.onnx.contrib` custom-op conformance artifact now
  proves external-domain identity preservation and fail-closed semantics. Add a
  repository-stratified public application cohort before implementing vendor
  semantics or making frequency claims; one conformance fixture is not an
  ecosystem denominator.
- Llama, Mistral, Qwen2, Qwen3, Gemma, Gemma2, OLMo, OLMo2, Granite, Phi-3,
  Cohere, Cohere2, Nemotron, Ministral, SmolLM3, EXAONE 4, Mixtral, and Mamba
  configuration plus modeling sources are content-digest pinned.
  The contract distinguishes split/fused QKV, gated/two-matrix MLP, parallel
  residual normalization, conditional Q/K normalization, and serialized bias
  layouts. Mixtral separates total expert parameters, active top-k compute, and
  router work; Mamba separates recurrent state from KV state and withholds
  selective-scan compute that is not source-closed. Configuration defaults are
  never treated as execution graphs.
- Jamba configuration and modeling sources are also content-digest pinned.
  Its periodic attention/Mamba and expert/dense schedules, canonical state-dict
  tensors, active MoE arithmetic, attention KV state, and recurrent Mamba state
  are independently conserved. Hybrid memory is exactly
  `(KV elements per token * context + recurrent elements per batch) * batch *
  bytes per element`; selective-scan arithmetic remains excluded because the
  implementation does not expose a source-closed primitive count.
- Expand further hybrid SSM/MoE layouts only after their state-dict modules and
  compute equations are separately source-bound. Keep runtime-observed KV
  or recurrent-state layout,
  paging, residency, and backend offload as a separate frame rather than
  forcing them into serialized-container or CNN assumptions.

## Public Communication

Publish browser-local static audit, XNNPACK chain prediction, quantization risk,
target-aware roofline/cache/packing, local benchmark path, ML-BOM, Engineering
Bundle, `/medical` evidence workspace, and Cloudflare deployment.

Do not publish unreleased private mechanics, claims requiring real-device
profiling or training artifacts, or claims that artifact-only analysis proves
clinical safety, regulatory compliance, or generalization. Prefer clear
assumptions and evidence boundaries.

# Changelog

All notable archival releases of DEEPBOM are documented here.

## Unreleased

## 1.96.13 - 2026-09-10

- Added a serverless Agent-native path for Codex, Claude Code, and compatible
  local agents. `deepbom integrate` previews, installs, verifies, and removes
  only hash-bound managed Skill files; user edits, symlinks, and path escapes
  fail closed.
- Added `deepbom.agent_capabilities.v1`, a public Agent Skill, clean-agent
  selection evaluations, and exact version-pinned `npx` commands. Every bounded
  human summary now carries an artifact-hash-bound reproduction command.
- Kept local stdio MCP as the persistent tool-call surface and verified its
  four tools with the official MCP Inspector. Added a version-matched Claude
  Desktop MCPB whose CLI and TFLite WASM bytes are identical to the npm channel.
- Added `/for-agents/`, `/agent-capabilities.json`, assistant-readable discovery
  metadata, and an explicit no-hosted-analysis boundary. Plain chat can
  recommend the pinned local command but cannot claim to have run an audit.
- Added Custom Domain-bound `medbom.org` redirects to the canonical `deepbom.org`,
  allowing Cloudflare to provision DNS and TLS without creating a second product,
  content surface, or origin.
- Removed standards-development material from the website, public product
  documentation, normal quality and deployment gates, and release artifacts.
  CycloneDX product output remains fixed to version 1.7.
- Unified explicit human-summary handling for audit, verify, diff, and explore,
  and expanded installed-channel checks across TFLite, ONNX, GGUF, paths with
  spaces, MCP, Agent Skill lifecycle, and the Cargo/Python/npm capability
  contract.
- Kept the three review verdict counts in one bounded mobile row so the audit
  workbench remains reachable in the first 390 x 844 review viewport.
- Raised only the source-budget categories consumed by the new Agent runtime,
  evaluation fixtures, MCPB builder, and release checks; per-file and generated
  rulepack limits remain unchanged.
- Aligned the build-content provenance manifest with the deployed file set so
  excluded standards-development paths cannot reappear through generated
  release metadata.

## 1.96.12 - 2026-09-09

- Closed the remaining reproducible external-review correctness boundaries with
  hash-bound fixtures: TFLite `TRANSPOSE_CONV` uses an explicit nominal dense
  scatter MAC convention, unresolved extents remain symbolic, and saturation
  and quantization-grid denominators are emitted instead of implied.
- Separated GGUF F16 floating-point storage from Q4/Q8 block quantization,
  accepted the source-backed Core ML attribute form of
  `constexpr_affine_dequantize`, and bound ML Program operands to exact package
  blob files and byte ranges. Invalid or contradictory ranges still fail closed.
- Added `deepbom.semantic_artifact_diff.v1` for same-format TFLite, ONNX,
  Core ML, GGUF, SafeTensors, and ExecuTorch comparisons. Quantization mapping,
  granularity, axis, scale, and zero-point changes are first-class; TFLite also
  retains its target-bound deployment delta.
- Completed the explanation registry for all 119 canonical `EA-*` findings and
  made new unclassified identifiers a CI failure. CLI, MCP, SARIF, and the web
  evidence drawer now consume the same trigger, evidence, boundary, and
  remediation records.
- Added explicit engineering and regulatory evidence-completeness gate
  profiles, six-format maturity and stable machine-contract declarations, and
  separate stable/prerelease publication behavior. Neither policy profile
  claims safety, clinical validity, legal compliance, or regulatory acceptance.
- Added a CycloneDX 1.7 root-component compatibility fixture and documentation:
  the analyzed model is `metadata.component`, while `components[]` contains
  additional inventory. The root remains valid without duplication.
- Bound the external-review golden boundary corpus to TFLite, GGUF, Core ML,
  and ONNX manifests and their format-tier verifiers. ExecuTorch native-runtime
  and TensorRT GPU observations remain deferred evidence gaps rather than
  statically inferred results.
- Classified the reproducibly generated finding catalog as generated runtime
  data instead of handwritten code. Raised only the measured 1.96.12 ceilings:
  handwritten runtime by 16 KiB for the semantic-diff and policy surfaces,
  generated runtime data by 192 KiB for the 119-rule catalog, development
  tooling by 32 KiB for source-pinned fixture generators, and corpus evidence
  by 16 KiB for the bound Core ML, GGUF, ONNX, and CycloneDX fixtures.
- Refreshed the public-source boundary with all 39 new public contracts,
  fixtures, generators, checks, and runtime modules. Public export now fails
  before copying when any tracked non-private path is absent from or stale in
  the reviewed exact-file allowlist. The manifest and source-bound build
  metadata generated only inside an export remain outside that source-coverage
  comparison.

## 1.96.11 - 2026-09-09

- Unified the bounded human audit summary behind one shared output contract
  consumed by the CLI, MCP server, and capability document. Explicit
  `--output-format summary` and the `--summary` compatibility alias now produce
  the same projection derived from `deepbom.review_summary.v1`.
- Corrected externally reproduced deployment-artifact evidence defects without
  broad severity escalation: non-finite learned tensors block the defect gate,
  all-zero tensors remain cautions, TFLite nominal MAC and dynamic-shape
  confidence contracts retain exact or symbolic totals, and quantization-risk
  summaries preserve the affected operation.
- Bound the external-review fixtures, expected values, lifecycle states, and
  fix/verification commits in a machine-readable ledger. The fastest public CI
  tiers now retain the non-finite defect gate and the format tier retains the
  TFLite correctness corpus.
- Rebased the documentation budget from 384 KiB to 416 KiB and the verification
  budget from 3328 KiB to 3392 KiB after adding the external-review triage,
  product-direction record, and bound regression fixtures. These are explicit
  non-runtime allowances and do not increase the runtime source budget.
- Known issue: a source-pinned Core ML iOS 16/17
  `constexpr_affine_dequantize` package using operation-attribute constant
  serialization may still be rejected during static analysis. This release
  does not claim complete support for compressed Core ML MLProgram artifacts;
  such a rejection must not be interpreted as proof that the artifact itself
  is malformed.

## 1.96.10 - 2026-09-08

- Hardened the local MCP stdio server with request cancellation, responsive
  control requests during analysis, bounded concurrency and queueing, tool
  timeouts, input-frame and response limits, and launch-root filesystem
  confinement. The server now negotiates MCP `2025-11-25`, returns unknown
  tools as JSON-RPC invalid-parameter errors, and preserves a cancelled
  request's no-response contract while terminating its child analysis.
- Made the bounded human audit summary the MCP default. Detailed evidence,
  section and pointer selection, large-model scan depth, offline acquisition,
  and download bounds are explicit tool inputs. JSON results expose matching
  `structuredContent`; a defect-gate exit keeps the first JSON block unchanged
  and reports policy status separately.
- Added read-only tool metadata and `deepbom_explain_rule`, corrected
  `deepbom_diff` to disclose its TFLite-only scope, and expanded installed npm
  MCP checks to real TFLite WASM, ONNX, and bounded GGUF audits. Official MCP
  Inspector strict discovery and transport checks cover the same contract.
- Registered `artifact_set` under the artifact identity metric family, fixing
  an Engineering Bundle conformance failure exposed by full browser download
  validation. The package selector now states both accepted directory forms
  using their literal `.mlpackage` and `.safetensors` extensions.

## 1.96.9 - 2026-09-07

- Added `deepbom mcp`, a stdio Model Context Protocol server exposing
  `deepbom_capabilities`, `deepbom_audit`, and `deepbom_diff`. It is a transport
  rather than an analysis command: it declares no output contract of its own,
  answers every call by re-entering this CLI in a child process so the JSON-RPC
  stream never shares stdout with an evidence document, and rejects undeclared
  tool arguments. Its tool descriptions carry the finding-kind distinction and
  the static-evidence boundary, and `scripts/check-mcp-server.mjs` rejects a
  version that drops either. Source and installed-channel checks execute a real
  ONNX audit through the transport, and runtime validation rejects undeclared or
  conflicting arguments rather than silently ignoring them.
  `deepbom.cli_capabilities.v1` now declares the transport under
  `automation.mcp_stdio_server`.
- Published `llms.txt` at the domain root and an agent-facing skill at
  `skills/deepbom/SKILL.md`, both stating that analysis is local and that no
  hosted analysis endpoint exists. The discovery-metadata check asserts the
  generated file keeps that boundary and the three finding kinds.
- Raised the measured source ceilings that the preceding web and agent work
  exceeded: handwritten runtime to 11776 KiB, verification to 3328 KiB,
  development tooling to 1408 KiB, and docs to 384 KiB.

## 1.96.8 - 2026-09-05

- Added the bounded ONNX installation probe to every standalone engine and
  Python wheel, and to the immutable Cargo engine matrix. Python and Cargo now
  validate its declared path, size, and SHA-256 before executing the packaged
  self-test; channel and published-engine smoke tests execute that probe.
- Made Cloudflare deployment fail closed when credentials are absent or
  Wrangler uploads a version without activating a production target. Generated
  deployment configuration now retains and validates the explicit
  `deepbom.org` production routes.
- Raised only the measured 1.96.7 source ceilings: handwritten runtime by
  64 KiB, verification by 18 KiB, and development tooling by 24 KiB. The same
  source-budget gate now runs as part of local release validation, preventing
  a clean local release record from diverging from the Cloudflare deployment
  preflight. The generated public-source provenance manifest is excluded from
  handwritten development-source accounting in both repository layouts. The
  web workflow now installs its pinned public WASM build toolchain whenever a
  deployable artifact is built; private-module smoke remains independently
  conditional.

## 1.96.7 - 2026-09-04

- Kept the Cloudflare deployment preflight browser-free by separating the
  conversion-receipt module, IR, CLI, and tamper checks from the browser
  binding exercised by Full Quality. This preserves the bounded deployment
  gate without downloading Chromium into the release path.
- Added the packaged CLI self-test fixture to the exact public member
  allowlist and bound its source bytes, packaged bytes, and release-manifest
  SHA-256. This closes the six-platform package-boundary failure without
  weakening the public/private distribution boundary.

## 1.96.6 - 2026-09-04

- Reworked the website's first-use path around one local artifact or verified
  example, then presents the shared Artifact Evidence IR as artifact defects,
  cautions, evidence gaps, coverage, and direct next actions. Browser-local
  Review HTML, Evidence JSON, and Evidence Package exports no longer require an
  account.
- Added concise regulatory, quality, and engineering evaluation briefs while
  keeping licensed standards text out of the distribution. Public product
  surfaces and generated BOMs use stable CycloneDX 1.7 only.
- Added real frontend code splitting and interaction-loaded format analyzers,
  report code, and TFLite WebAssembly. Corrected bundled Worker URLs so source
  and deployed builds retain the same isolated analysis paths.
- Changed the CLI default to a bounded human review summary and added explicit
  section and JSON Pointer selection, defect-only gating, installation
  self-test, rule explanations, and categorized deployment-delta impact.
- Fixed npm and npx TFLite runtime asset discovery by resolving package assets
  from the installed CLI module rather than the unresolved executable symlink.
- Separated learned, quantization, control, and unresolved constants so shape,
  axis, index, and Slice sentinel values cannot contaminate learned-weight
  magnitude or sparsity statistics. Constant-wide numerical integrity remains
  independently assessed.
- Extended browser, installed-channel, public-boundary, mobile, theme, and
  service-worker contracts for the new onboarding and output semantics. The
  bounded deployment preflight now executes 84 checks with argument-safe test
  dispatch.

## 1.96.5 - 2026-09-04

- Moved every heavyweight TFLite WebAssembly analysis call behind one isolated
  Worker RPC boundary, including target/frontier, redesign, influence,
  calibration, histogram, landscape, tomography, and Haar analysis paths.
- Moved GGUF, SafeTensors, Core ML, and package range analysis to the same
  file-scoped metadata Worker while preserving package-relative paths,
  progress events, inactivity termination, and clean retry behavior.
- Added a static main-thread WASM boundary gate, Worker RPC contract checks,
  and a reproducible JavaScript screening benchmark for the only prospective
  streaming WASM candidates: incremental hashing and large payload decoding.
  Independent JavaScript validators remain authoritative cross-checks.
- Fixed stale Artifact IR consumer views after protected selector or provider
  evidence is merged, and kept the mobile Kernel Inspector controls and
  selector decision ledger in the first visible panel region.
- Raised only the measured ceilings needed by the separately modularized
  Worker protocol, RPC facade, enforcement checks, benchmark, and execution-
  boundary documentation. Generated data, corpus, native tooling, private
  source, and individual runtime-file ceilings are unchanged.

## 1.96.4 - 2026-09-04

- Replaced mutable pre-release schema pins with hash-verified compatibility
  fixtures. Incompatible or unresolved combinations now emit a deterministic
  status record instead of a false BOM conformance claim; the stable product
  export remains unchanged.
- Added an RFC 9535 Perspective evaluator for Web and CLI that records exact
  match counts and JSON Pointers while distinguishing zero, empty, multiple,
  and type-mismatched results. Reference projection is explicit and
  `relevance: required` remains unclassified until its normative cardinality
  semantics are resolved.
- Added `deepbom.conversion_receipt.v1` for hash-only source checkpoint,
  converter invocation, environment manifest, and deployment output lineage.
  The output hash/format binding is observed while converter and source claims
  remain `DECLARED_UNVERIFIED`; executable `.pt`, `.pth`, and `.h5` inputs are
  never deserialized.
- Advanced Artifact IR method v2 to 2.2.0 so the same conversion lineage is
  conserved through Web, CLI, evidence envelope, graph export, and CycloneDX.
  Added tamper, cross-artifact, secret-bearing argument, and cross-surface
  regression checks.
- Added bounded recovery for transient Engineering, regulatory, and raw-report
  formatter module fetches, and corrected Evidence Package membership checks
  so non-BOM compatibility-status records are not mistaken for `.cdx.json`
  documents.

## 1.96.3 - 2026-09-03

- Fixed the Cargo launcher's verified runtime-asset installation contract by
  separating the versioned GitHub Release filename from the canonical local
  wasm-pack filename consumed by the analysis engine.
- Extended the post-publication Cargo smoke test to execute the hash-pinned
  MobileNet V2 TFLite artifact and assert its operator, tensor, and MAC totals,
  in addition to the existing ONNX execution check. This prevents a successful
  engine-only smoke from masking a missing TFLite WASM runtime asset.
- Advanced the service-worker cache generation to v548 for the 1.96.3 release
  identity so clients cannot retain the preceding application configuration.

## 1.96.2 - 2026-09-03

- Added a self-hashed ONNX serialized-contract conflict capsule that preserves
  declaration/semantic roots, condition-bound variants, downstream impact,
  blocked MAC rows, and canonical Artifact IR subjects without repairing or
  computing through contradictory tensor declarations.
- Published a reproducible nine-artifact ONNX conflict corpus with six
  unconditional roots, 543 condition-bound invalid variants, 1,901 downstream
  blocked nodes, and 213 withheld MAC rows. Official ONNX checker and strict
  shape-inference runs are isolated and retained as comparative outcomes rather
  than treated as an oracle.
- Separated 27 hash-identified public Core ML artifacts, five generated
  MLProgram conformance fixtures, and the still-unobserved compiled-plan
  population in the residual evidence ledger. Their denominators are no longer
  interchangeable and no ecosystem-prevalence or device-placement claim is
  inferred.
- Migrated 49 report/export, viewer, and session/diff modules away from direct
  native `analysis.ops/tensors` reads into one compatibility selector facade.
  An AST-based policy now permits zero direct surface readers and rejects any
  regression. Repeated capability rendering also retains live rows without
  detached nodes while filters and formats change.
- Added the conflict capsule and Artifact IR selector facade to offline cache
  coverage, and made application startup recover from bounded transient module
  fetch failures without masking code errors. The service-worker cache advances
  from v544 to v547.

## 1.96.1 - 2026-09-03

- Closed the remaining Artifact IR round-trip gaps with a nested-scope runtime
  fusion fixture, rendered `review.html` verification, runtime subject
  resolution, applicability-state preservation, and CycloneDX sibling digest
  checks across the shared export context.
- Replaced the Artifact IR import-boundary text scan with parser-derived ESM
  edges and exact per-module importer policy, keeping the unchecked builder
  reachable only through the canonical context facade.
- Added real browser audits for TFLite, ONNX, Core ML, GGUF, SafeTensors, and
  ExecuTorch across fixed evidence domains and lenses, including mobile state,
  focus, touch-target, stale-content, heap, and detached-DOM regressions.
- Fixed retained Explorer and residual-distortion DOM state, made Arena peak and
  pan/zoom behavior responsive, and aligned local single-file artifact-set
  identity so Web and installed CLI Artifact IR digests match for TFLite and
  ONNX fixtures. Added the shared artifact-set module to offline precaching and
  advanced the service-worker cache generation from v543 to v544.
- Pinned deterministic expected-output evidence to the clean release envelope
  while requiring dirty development builds to add exactly one reproducibility
  finding and otherwise preserve the same canonical evidence body.
- Updated the development-only Ajv URI dependency lock to patched `fast-uri`
  3.1.7; the release source now reports zero npm audit vulnerabilities.

## 1.96.0 - 2026-09-02

- Closed the Artifact IR stabilization contract across six export surfaces,
  embedded a non-executable machine-readable state in `review.html`, rejected
  stale artifact/runtime context injection, and required every imported runtime
  subject to resolve to a canonical operator or architecture node.
- Split Artifact IR construction into identity, graph, storage, architecture,
  quantization, overlay, and validation modules behind one context facade. A
  machine-readable import and consumer policy now rejects unclassified raw
  ledger readers and direct unchecked-builder imports.
- Kept all five evidence domains and three analysis lenses discoverable across
  six formats, including explicit not-applicable, not-assessable, and
  not-assessed-yet states. Added desktop/mobile and light/dark browser
  regression baselines for all ten evidence classes.
- Added a clean-source release-validation manifest that records each local
  quality, deployment, public-source, and channel-equivalence command with its
  duration, toolchain, commit, log, and output digests. The bounded deployment
  gate now contains 73 checks and measured 155.533 seconds on Windows x64.
- Added every split Artifact IR module to service-worker cache coverage and
  advanced the cache generation from v542 to v543.
- Ordered quality validation so the current clean source provenance is generated
  and verified before deterministic expected-output checks; stale ignored build
  metadata can no longer alter a release check according to local run history.
- Made `--output-format` the unambiguous CLI spelling while retaining
  `--format` as a backward-compatible alias; existing raw analysis output and
  automation contracts remain unchanged for the 1.96 stabilization cycle.
- Made the cached Artifact IR context the only public construction path and
  added a source-boundary gate that rejects direct unchecked-builder imports.
  UI, reports, deployment packs, CycloneDX, `review.html`, raw ZIP evidence,
  and CLI graph output now retain one IR identity, nested-scope count, and
  runtime subject mapping.
- Replaced format-driven tab disappearance with five fixed evidence domains
  and explicit applicability records. Four primary workflow stages remain
  separate from optional analysis tools, and graphless containers expose why
  graph arithmetic or placement is not applicable instead of rendering an
  unexplained empty surface.
- Added a canonical ten-class evidence visual contract with text, border,
  pattern, tooltip, and ARIA encodings; improved arena peak occupancy and
  pan/zoom controls, and added a complete assessed-MAC cumulative view.
- Raised the handwritten runtime ceiling by 32 KiB and verification ceiling by
  16 KiB for the separately modularized evidence-class, applicability,
  runtime-reconciliation, and cross-output conservation contracts; generated
  rulepack, corpus, per-file, and application-entry budgets are unchanged.
- Added the evidence-preserving `deepbom.artifact_ir.v2` ledger with separate
  serialized graph, storage topology, architecture projection, scoped
  quantization, static placement, and imported runtime layers. Existing
  `deepbom.graph_ir.v1` visualization output is now a deterministic
  compatibility projection from this ledger, and graphless containers no
  longer receive synthetic execution-order edges.
- Advanced the Artifact IR method to `2.1.0`: all exactly decoded TFLite
  subgraphs and ONNX nested/function scopes are materialized with conserved
  scope-local ports and values, ExecuTorch is covered by a serialized PTE
  regression fixture, and static placement remains primary-scope-only.
- Added artifact-bound runtime reconciliation for canonical subject references
  and primary native op indices, including explicit one-to-many fusion. Missing
  bindings remain unreconciled and runtime names are never similarity-matched.
- Routed shared Web, CLI, report, diff, graph-export, and CycloneDX consumers
  through one cached primary-scope IR view. Canonical operators and values stay
  fixed while session findings, deployment deltas, and runtime imports remain
  live, and runtime-evidence changes rebuild only the affected overlay context.
- Tightened the public Artifact IR JSON Schema and semantic validator around
  nested-scope ownership, cross-scope ports, storage/quantization totals,
  overlay identity, and runtime reconciliation cardinality.
- Raised only the verification-source budget by 48 KiB to retain the new
  multi-format scope, reconciliation, consumer, tamper, browser, and release
  regressions; runtime, generated-data, documentation, and corpus budgets are
  unchanged.
- Published a JSON Schema and fail-closed multi-format conservation suite for
  Artifact IR identity, references, byte/MAC totals, graph applicability, and
  tamper detection. Engineering evidence, Raw Data, Deployment Contract Pack,
  CycloneDX references, Web visualization, and CLI graph JSON now bind the
  same IR digest.
- Added a shared `deepbom.mac_coverage.v1` calculation path so TFLite,
  ONNX, Core ML, GGUF, and SafeTensors ML-BOM exports retain explicit MAC
  numerators, denominators, and non-assessment reasons without coercing an
  unknown Core ML execution precision to zero.
- Added `deepbom.placement_comparison.v1`, Web N-way profile selection, and the
  `deepbom placement` CLI command over one canonical graph/tensor ledger.
  Source-pinned TFLite Core ML and LiteRT Qualcomm QNN profiles, imported Edge
  TPU compiler reports, and imported Qualcomm compiler/dispatch evidence remain
  independent evidence portfolios rather than a fabricated joint assignment.
- Separated CPU cost profiles from source-backed accelerator eligibility in the
  audit controls. TFLite GPU and NNAPI remain visibly unavailable until their
  pinned source ledgers are loaded; accelerator selection never changes the CPU
  roofline or claims kernel selection, timing, or observed assignment.
- Added a hash-bound Evidence Cursor, exact hierarchical graph contraction,
  topology-and-tensor-contract artifact diff, strict external node/edge overlay
  import, and explicit source-to-runtime graph reconciliation.
- Added a uniform evidence explanation drawer and a CSP-restricted,
  self-contained `review.html` export with normalized findings, static graph,
  runtime reconciliation, and replayable review state but no model payload.
- Added mobile bottom-sheet graph inspection and Resource Map evidence linking
  while retaining keyboard navigation, light/dark contrast, exact conservation,
  and desktop/mobile overflow checks. Internal `unnamed group` labels are
  normalized only in the presentation layer without changing ledger identity.
- Fixed the release-tier runner to pass script arguments separately from the
  script path, so argument-bearing gates such as CLI documentation validation
  execute instead of being interpreted as nonexistent filenames.
- Made the JavaScript source gate public-boundary aware: `web` and `scripts`
  remain mandatory, while private-only roots such as `worker` are checked when
  present instead of making a clean public export fail before validation.
- Made clean public source exports generate their own public-distribution build
  metadata and record only roots that exist in that export, keeping runtime
  imports, bundle hashes, and provenance complete without private-only paths.
- Made web import smoke cover the protected WASM integration only when that
  private artifact exists, while retaining mandatory imports for every public
  web module, ONNX adapter, and TFLite WASM package.
- Split the hosted-worker assertion from the public web workflow contract so a
  clean public checkout validates all shipped UI, privacy, accessibility, and
  export behavior without requiring the separately hosted backend source.
- Scoped the hosted-worker configuration gate to private checkouts that contain
  the worker; public smoke now runs its complete 15-check surface while private
  smoke retains the additional worker deployment check.
- Raised the handwritten runtime ceiling from 11,048 KiB to 11,184 KiB for the
  separately modularized evidence cursor, hierarchy, diff, overlay, accelerator
  switcher, explanation, and read-only review surfaces; per-file, generated,
  private-source, and application-entry ceilings remain unchanged.
- Raised the handwritten runtime ceiling from 11,184 KiB to 11,248 KiB for the
  separately modularized CPU-target provenance, accelerator lifecycle binding,
  Edge TPU compiler evidence, and identity-bound review-policy contracts;
  per-file, generated, private-source, and application-entry ceilings remain
  unchanged.
- Raised the handwritten runtime ceiling from 11,248 KiB to 11,264 KiB for the
  separately modularized cross-format MAC coverage, N-way placement comparison,
  and source-pinned Core ML/Qualcomm adapters; per-file, generated, private-
  source, and application-entry ceilings remain unchanged.
- Raised the handwritten runtime ceiling from 11,264 KiB to 11,392 KiB for the
  separately modularized Artifact Evidence IR and its v1 compatibility
  projection; per-file, generated, private-source, and application-entry
  ceilings remain unchanged.
- Raised the documentation-source ceiling from 320 KiB to 352 KiB for the
  published Artifact IR JSON Schema and its CLI contract documentation;
  runtime and per-file ceilings remain unchanged.
- Raised the verification-source ceiling from 3,072 KiB to 3,136 KiB for the
  isolated parser coordinator and the accelerator, review-policy, expected-
  output, and Worker timeout regression contracts; runtime and per-file
  ceilings are unchanged.
- Preserved incomplete ONNX MAC ledgers as not assessed instead of exposing
  their assessed subtotal as a complete zero-valued model total; UI, graph
  exports, engineering/regulatory reports, conformance checks, and CLI output
  now share the same nullable-total contract and explicit coverage denominator.
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

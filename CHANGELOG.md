# Changelog

## 1.106.0 — 2026-09-22

- Add an interactive Web weight pruning workbench with linked tensor/channel views, target-sparsity controls, energy curves, original/candidate/difference heatmaps, zoom, pan and full-screen inspection.
- Export hash-bound pruning simulation JSON and SVG/PNG views; preserve exact removal counts, deterministic tie handling, existing zeros and explicit floating/dequantized value scope. This release does not export a modified model or infer task quality or runtime acceleration.
- Verify the simulation with 1,212 independent numerical cases, six format decoders and browser interaction/export checks. Publish the Web, CLI and local/hosted MCP channels at the same engine version.
- Reserve 4 KiB of runtime-source bookkeeping and 32 KiB of verification source for the interactive workbench and its recorded release qualification.

## 1.105.0 — 2026-09-22

- Add optional Weight IR derived analysis for six serialized formats: channel statistics, native kernel slices, complete signed similarity, bounded Jacobi SVD, structured sparsity, validated affine dequantization and aligned original/candidate comparison.
- Expose matching numerical results in Web, CLI and local MCP, with SVG/PNG views, hash-bound JSON, explicit per-feature budgets, native axis order and unassessed reasons.
- Preserve Weight IR v1 and Model IR identities; add separate weight_analysis.v1 and weight_comparison.v1 contracts. Numerical properties do not imply task quality or measured acceleration.


All notable archival releases of DEEPBOM are documented here.

## 1.104.0 — Optional numerical evidence

- Add optional Weight IR and Activation IR linked to the existing Model IR hash,
  storage, tensor, port and operation IDs. Preserve static defaults and the Model
  IR v1 schema. Account for every stored object, decoder limitation and missing
  capture without converting unknown values to zero.
- Reuse ONNX, TFLite, GGUF, SafeTensors, Core ML and ExecuTorch numeric decoders.
  Report exact decimal counts, shared fixed-bin histograms, stable floating
  moments, nonfinite counts and exact large-integer extrema. Withhold unsafe
  floating integer statistics and label stored codes versus dequantized values.
- Add local CLI/MCP opt-ins, web inspection/export controls, ChatGPT Weight IR
  downloads and an explicit local ONNX/LiteRT activation collector. Bind runtime,
  inputs, collector, instrumentation and configuration; do not infer attestation,
  production timing or per-node GPU placement.
- Qualify the new numerical layer with count, identity, reference, schema, numeric
  edge-case and capture-coverage checks. Budget changes account for these optional
  modules, the collector, public schema, guide and regression tests.

## Unreleased

- Add an eleven-chapter CLI Handbook and searchable command reference under
  Guides. Generate the hosted reference and downloadable help from the same
  executable as the Markdown reference, and reject drift during web builds.
  Include copyable examples, print/PDF layouts, mobile navigation, and explicit
  CLI-versus-Web SPDX availability. Reserve 64 KiB of runtime source and
  16 KiB of development tooling for the documentation and generation path.

- Add a public `/get-started/` hub for web, Windows/macOS/Linux Desktop,
  CLI installers, and AI connections. Pin verified download/setup versions,
  distinguish preview and review status, and expose the page in mobile and
  desktop navigation. Include keyboard-accessible OS tabs, copyable commands,
  and complete instructions without JavaScript. Reserve 40 KiB of handwritten
  runtime source for this separate page and its navigation; analyzer and tool
  contracts remain at their existing versions.

- Qualify the Claude MCP App in a simulated cross-origin browser host. Load
  TFLite's existing self-contained worker bundle through the deployment origin
  and clear stale results before a new analysis. Keep engine 1.103.0 and the
  existing tool contract; actual Claude account validation remains required.

## 1.103.1 - 2026-09-21

- Reconcile quantization counts and every `mlbom:model:` property against fresh
  artifact observations. Count every supplied property occurrence, including
  unknown namespaces and duplicate names; reject hidden conflicting hashes.
- Report `NO_CONTRADICTION` and explicit coverage instead of a blanket human
  `PASS`. Keep exit-code compatibility and distinguish missing, mismatching,
  and ambiguous subject hashes without inferring forgery from a mismatch.
- Support both CLI and browser CycloneDX profiles. Preserve unavailable
  quantization observations as unknown, rather than emitting false or zero.
- Keep SARIF artifact defects, cautions, and evidence gaps distinct while
  preserving original severity. Improve SafeTensors byte-range diagnostics.
- Surface BOM verification, batch, summary, and hash-binding options near the
  start of help; add automatic light/dark SVG rendering.
- Expand the exact-name property taxonomy and reserve 24 KiB of documentation
  budget for compatibility mappings and explicit reconciliation boundaries.
- Retain empty or unassessed GGUF/SafeTensors tensors in namespace counts
  without dangling payload references. Preserve unknown namespace byte totals,
  offsets and exact integers as null, not zero; this corrects some derived
  Artifact IR hashes without changing the artifact SHA-256. Regenerate older
  IR-bound BOMs when adopting the corrected analysis.
- Report known invalid GGUF block-row cardinalities, offsets and overlapping
  ranges as artifact defects even in structure-only scans. Keep unsupported
  encodings distinct and label incomplete human byte totals as known subtotals.

## 1.103.0 - 2026-09-17

- Publish the maintainer-requested channel usage update under the explicit
  stable-release interval exception. Agent contract 1.0.2 is a review candidate
  for the expanded website/privacy notice; evidence contract 1.0.0 is unchanged.
- Add consented Web analysis usage alongside ChatGPT, channel/date filters,
  weekly browser estimates, unfinished-run counts, and explicit unmeasured
  coverage for Claude, CLI and local MCP. Keep legacy ChatGPT history and
  channel-specific browser identities; never sum identities across channels.
- Separate cached public npm download trends from opt-in analysis evidence.
  Preserve missing dates and provider outages rather than displaying zeros.
  CLI/local MCP execution remains offline with no new telemetry.
- Raise runtime/verification budgets by 32 KiB, docs/development tooling by
  8 KiB and the app entry by 2 KiB for channel summaries, bounded registry reads,
  browser controls, regressions and recorded release qualification.

- Add Google sign-in directly to the administrator console, returning to the
  console after authentication and offering account switching to non-admins.

## 1.102.0 - 2026-09-17

- Publish the coordinated web, CLI, and MCP update at the maintainer's explicit
  request. This requested release uses the stable-release interval exception.
  Prepare Agent contract 1.0.1 for the optional usage/privacy change while
  retaining evidence contract 1.0.0. Source validation and server deployment
  do not assert OpenAI approval of this new contract candidate.

- Add optional ChatGPT usage statistics with explicit browser consent, random
  pseudonymous identifiers, UTC revisit estimates, test exclusion, withdrawal,
  and deletion. Retain only allowlisted events and remove expired records daily.
  Keep model files, names, hashes, and conversation text out of usage storage.
- Add administrator-only 7/30/90-day dashboards, format/export counts, daily MCP
  operational counts, and aggregate CSV/JSON evidence exports. Distinguish
  consenting browser estimates from verified people and downloads. Update the
  privacy notice and review preparation files; keep CLI contracts unchanged.
- Raise handwritten runtime, verification, and documentation ceilings by 48,
  32, and 16 KiB respectively for the collector, consent controls, administrator
  dashboard, SQLite/browser regressions, and usage/review documentation. Add
  8 KiB to development tooling for the associated checks, build metadata, and
  measured release qualification records.

- Add an explicit Save via ChatGPT action for prepared exports in hosted
  widget revision 20260916.6. Use the host's file upload, temporary HTTPS
  download URL, and external navigation APIs when iframe downloads are blocked.
  Offer Send link to chat and retain file IDs across URL failures. Local export
  preparation does not upload anything; the separate save action shares only
  the generated export, never the original model. Keep unsupported host APIs and
  upload failures visible instead of claiming a file was saved.
- Raise the verification source ceiling by 16 KiB for sandbox-blocked download,
  exact JSON file handoff, schema/hash, upload failure, and URL retry regressions.

- Make hosted widget revision 20260916.5 work with `worker-src blob:` by
  downloading the self-contained classic Worker bundle before starting it.
  Bind WASM asset resolution to the deployment URL at build time, without a
  remote module import, runtime eval, or main-thread parser fallback. Exercise
  the stricter CSP inside nested sandboxed iframes and report analyzer-loading
  failures separately from attachment failures. Clarify that opening the panel
  does not establish successful or ongoing analysis when the panel has failed.

- Fix hosted ChatGPT TFLite analysis in a different-origin sandbox by starting
  a local Blob Worker that imports the deployed, CORS-enabled worker bundle.
  Preserve deployment-relative WASM loading and verify cross-origin MobileNet
  analysis with the actual production asset paths. Hosted widget revision
  20260916.4 retains the 1.101.0 analyzer and CLI contracts.
- Raise the deployment ceiling by 2 MiB for the self-contained ChatGPT worker
  bundle; keep cross-origin access limited to the declared public runtime assets.

## 1.101.0 - 2026-09-16

- Publish this coordinated web, CLI, and MCP release immediately at the
  maintainer's request to restore accessible visualization file downloads.
  This explicitly requested release precedes the usual stable-release interval.
- Move ChatGPT visualization and file controls above long evidence tables and
  report component height changes. Keep SVG, 300-DPI PNG, and Word-ready ZIP
  Save links available after download requests. PNG handoff now requests an
  inline image and a real ChatGPT-issued download link when available.
- Add browser-local CycloneDX 1.7 and SPDX 2.3 JSON export buttons. Reuse the
  existing CycloneDX artifact-evidence projection; represent SPDX artifacts as
  FILE-purpose packages with measured SHA-256, explicit license unknowns, and
  evidence annotations. Validate against pinned official schemas and exercise
  actual file downloads, narrow-panel visibility, and optional URL failures.
- Raise the reviewed development-tooling source ceiling by 8 KiB for the
  pinned SPDX schema validator and this release's channel qualification records.

## 1.100.0 - 2026-09-16

- Use the documented stable-release interval exception for a P0 public-tool
  correctness failure: the advertised ChatGPT `Send PNG to chat` path could
  fail in the host sandbox after the static audit had succeeded. The release
  gate now exercises the complete render-to-upload handoff. The additive Model
  Summary ships in the same corrected engine rather than publishing a second
  immediate release.
- Add `deepbom.model_summary.v1`, one deterministic Common Model IR projection
  for operation, reversible block, storage, and identity-only views. The CLI,
  browser workbench, local MCP, ChatGPT component, and Claude MCP App reuse the
  same source-bound rows without inferring framework trainability or runtime
  order. Publish the JSON Schema and cross-format regression coverage for
  TFLite, ONNX, GGUF, SafeTensors, Core ML, ExecuTorch, GraphDef, SavedModel,
  Keras, PT2, HDF5, and safe PyTorch checkpoint envelopes.
- Fix ChatGPT visualization PNG export by rasterizing the validated render
  model directly to an opaque monochrome canvas. This removes the sandbox-
  dependent SVG Blob decode step while retaining canonical SVG as the source
  and preserving the PNG derivation manifest.
- Raise the reviewed handwritten-runtime ceiling by 32 KiB for the shared
  Model Summary projection and sandbox-safe rasterizer, the documentation
  ceiling by 16 KiB for its public schema and contracts, and the verification
  ceiling by 16 KiB for cross-format, MCP, and browser handoff regressions.
- Keep `Report in chat` explicitly user-initiated and reusable after the
  browser-local audit completes. ChatGPT can acknowledge a component follow-up
  request without rendering a new turn, so the widget no longer reports
  delivery as complete or disables the only recovery control. The bounded
  result prompt still identifies the Model IR SVG, PNG, and Word-ready exports.
- Raise the reviewed documentation source ceiling by 16 KiB for the official
  ChatGPT submission import, portal values, demo protocol, icons, and explicit
  plugin privacy disclosures. Raise the development-tooling ceiling by 24 KiB
  for the corresponding schema and asset checks; runtime and generated-data
  ceilings are unchanged.

## 1.99.2 - 2026-09-14

- Use the documented same-day stable-release interval exception for a P0
  release-contract correctness fix: engine-only updates were incorrectly
  coupled to the reviewed Agent surface. The bounded
  `check-agent-contract-release` mutation suite proves that engine-only drift
  preserves the reviewed fingerprint while OpenAI, Claude, and evidence
  contract drift are classified separately. This record is a release-policy
  exception, not evidence of platform approval.
- Separate the frequently changing analyzer engine version from the stable
  `deepbom.agent_contract.v1` and
  `deepbom.artifact_evidence_envelope.v1` contract versions. Public plugin
  manifests, host-evaluation cases, and Skill behavior no longer change merely
  because an engine patch is released.
- Make the public Skill resolve a local or explicitly downloaded engine by
  declared Agent/evidence compatibility, then bind the exact observed engine
  version with a self-test before analysis. Registry access remains opt-in.
- Add a deterministic reviewed-surface fingerprint for OpenAI MCP metadata,
  UI resource policy metadata, plugin and Skill contents, privacy/transfer
  boundaries, the Claude remote surface, and the evidence-contract identity.
  Ordinary engine releases now fail closed if any of those contracts drift.
- Add release classification tests proving that an engine-only version change
  does not request a new platform review, while OpenAI metadata, Claude remote
  metadata, and evidence-contract mutations produce their distinct required
  actions. Source checks continue to distinguish a prepared candidate from an
  account-observed approval or listing.
- Add the Agent contract gate to both the web deployment preflight and the
  installation-channel publication boundary, with a public operations guide
  for deliberate contract revisions.
- Keep the publication-repository gate executable without exporting the
  private remote-MCP worker implementations: private release validation
  captures live metadata, while the exact public export recomputes its public
  Skill, manifest, policy, endpoint, and listing surfaces against the pinned
  live-metadata hashes.
- Raise the reviewed documentation source ceiling by 8 KiB for the public
  contract-versioning policy; executable and generated-data ceilings are
  unchanged.
- Raise the verification source ceiling by 32 KiB for the contract snapshot,
  classification, mutation, and release-gate checks introduced in this release.
- Raise the development-tooling source ceiling by 24 KiB for the deterministic
  contract-snapshot builder; runtime and generated-data ceilings remain unchanged.

## 1.99.1 - 2026-09-14

- Pin npm 11.6.0 on Windows installation-channel runners before invoking the
  official MCP Inspector. This avoids the reproducible npm 11.6.2
  `ECOMPROMISED: Lock compromised` regression while leaving the tested
  Inspector version and DEEPBOM runtime contract unchanged.

## 1.99.0 - 2026-09-14

- Add a separate Claude remote MCP App candidate at
  `https://deepbom.org/mcp/claude`. It opens an explicit browser-sandbox file
  picker, never claims automatic Claude attachment access, and returns only a
  bounded static-evidence result to the conversation.
- Keep the remote Claude path distinct from the local Claude Desktop MCPB.
  Public documentation now states the different byte-transfer, result-transfer,
  host-support, and confidentiality boundaries without claiming Anthropic
  review or directory approval.
- Add runtime titles and read-only, non-destructive annotations to the local MCP
  tools, declare the MCPB privacy policy, limit its manifest to Windows and
  macOS, and exercise all four packaged tools plus outside-root rejection with
  the official MCP Inspector on both release runners.
- Add a bounded MCP App browser client, result and diagnostic validation,
  protocol/version negotiation, origin and frame limits, and executable bridge
  round-trip tests. Selected model bytes remain in the app sandbox; the service
  receives the bounded result rather than the artifact.
- Expand brandless host-evaluation material to five positive and three negative
  cases for both ChatGPT and the Claude remote path. These catalogs remain blank
  test plans until real host runs are recorded by the publisher.
- Pin the Cloudflare deployment CLI used by CI after validating the production
  configuration against its packaged schema and dry-run output.
- Raise the reviewed documentation ceiling by 8 KiB for the separate Claude
  remote qualification boundary and account-owned OpenAI/Anthropic submission
  runbooks; executable and generated-data ceilings are unchanged.

## 1.98.0 - 2026-09-14

- Add the bounded `deepbom.model_ir.v1` preview without replacing the stable
  `deepbom.artifact_ir.v2` contract. The new hash-bound projection separates
  serialized programs, partial dependency order, logical values, storage,
  parameter bindings, quantization, reversible hierarchy, static-runtime
  projections, observed-runtime overlays, applicability, completeness, and
  explicit loss records.
- Add format-neutral analysis passes and traceable Transformer, CNN,
  encoder-decoder, and MoE profile projections. Native operation identities
  remain preserved, model-name inference cannot create execution edges, and
  graphless containers cannot be promoted to serialized execution graphs.
- Project the existing ONNX, TFLite, Core ML, ExecuTorch, GGUF, and
  SafeTensors evidence into Model IR, and add bounded no-execution previews for
  TensorFlow GraphDef, first-MetaGraph SavedModel signatures, Keras declarative
  configuration, and PT2 declarative JSON. HDF5 and legacy PyTorch checkpoints
  intentionally remain safe-envelope inventories rather than full model
  analyzers.
- Add deterministic Model IR visualization bundles with canonical monochrome
  A4 SVG pages, 300-DPI PNG derivatives, captions, cross-page references,
  subject locators, conservation checks, and a Word insertion manifest. These
  are engineering-evidence projections and do not claim regulatory approval or
  standards conformance.
- Expose Model IR and visualization-manifest access through CLI, browser,
  Python, npm, Cargo, and MCP-backed channel contracts while retaining one
  analyzer implementation and exact artifact/IR digests.
- Pin and verify the official TensorFlow, HDF5, Keras, and PyTorch source files
  used to bound the new preview adapters. Verification can use reviewed local
  sparse clones or temporary immutable checkouts; upstream source bytes are
  not redistributed.
- Raise the reviewed handwritten-runtime source ceiling by 288 KiB,
  verification by 64 KiB, development tooling by 32 KiB, and documentation by
  16 KiB for the Common Model IR, bounded source adapters, visualization
  pipeline, source-pin verifier, and their executable contracts. Generated
  runtime-data and per-file ceilings are unchanged.

## 1.97.4 - 2026-09-13

- Add a portable Agent Plugins package and a compatibility manifest that bind
  the public Skill to the existing ChatGPT Streamable HTTP MCP endpoint.
- Make the bundled Skill verify exact local installations before offering an
  explicit npm download fallback, with structured resolution evidence.
- Add host-observed, brandless selection cases and blank recording templates
  that distinguish deterministic CLI checks from real ChatGPT, Claude, and
  Codex tool selection.
- Add a bounded Claude Desktop MCPB review package and descriptive listing copy
  without claiming submission, approval, or remote-connector compatibility.
- Publish problem-focused ONNX quantization, GGUF tensor encoding, and artifact
  comparison guides with reproducible commands and explicit static-evidence
  limits.
- Distinguish the optional browser-observed runtime benchmark from CLI and
  assistant-driven static artifact analysis in discovery metadata.
- Raise the reviewed documentation source ceiling from 480 KiB to 512 KiB for
  the agent review packages and three substantive problem guides; executable
  and generated-data ceilings are unchanged.

## 1.97.3 - 2026-09-13

- Made the published-release provenance job check out the exact tagged source
  before invoking the private-repository fallback verifier. A release-contract
  test now requires that checkout and the verifier to remain in the same job.
- Restricted registry publication to the public `JunHwan-Kwon/deepbom`
  repository, preventing a private-source workflow from publishing release
  assets that the public Cargo launcher cannot resolve.

## 1.97.2 - 2026-09-13

- Published the ChatGPT integration guide and MCP registry metadata at their
  canonical production URLs, and added deployment-asset checks so the guide,
  widget, icon, and version-bound server metadata cannot be omitted silently.
- Preserved GitHub identity-backed artifact attestations where the repository
  supports them. For user-owned private repositories, the release now records
  the unavailable-attestation boundary explicitly and verifies the bundled
  unsigned local provenance subject instead; npm and PyPI provenance remain
  separately required and verified after registry publication.

## 1.97.1 - 2026-09-13

- Added a ChatGPT plugin path for one authorized model attachment. The public
  MCP endpoint supplies a browser-sandbox analyzer and receives only a bounded
  result; model bytes are not fetched or retained by the DEEPBOM service.
  Local CLI and stdio MCP remain the documented path for confidential, large,
  sharded, or external-data artifacts.
- Added exact ChatGPT file-input metadata, public/private tool separation,
  protocol negotiation, bounded request/result contracts, domain-verification
  support, legal and support pages, evaluation prompts, and a real browser E2E
  covering range reads, hashing, analysis, result publication, and recovery.
- Registered cross-format MAC-summary fields with the serialized-container
  metric owners so normalized GGUF, SafeTensors, Core ML, and ExecuTorch
  results remain exportable through the Engineering Bundle.
- Raised the handwritten runtime ceiling by 128 KiB, documentation by 60 KiB,
  verification by 64 KiB, and development tooling by 64 KiB for the reviewed
  ChatGPT MCP control plane, browser analyzer, submission documentation, and
  their executable boundary tests. Per-file and generated-data ceilings are
  unchanged.

## 1.97.0 - 2026-09-13

- Added bounded CycloneDX 1.7 BOM-to-artifact reconciliation. Subject binding
  is exact by SHA-256 or explicit unique `bom-ref`; ambiguous identities fail
  closed, and each comparison distinguishes matches, contradictions, absent
  declarations, unsupported mappings, and facts not assessable from the
  serialized artifact.
- Added artifact-derived external-interface baseline capture through
  `deepbom contract capture`. Captures bind the source artifact and analyzer
  identities and explicitly remain unapproved baselines until a separate
  review process adopts them.
- Reworked standalone CycloneDX output to carry compact structured evidence in
  `declarations.evidence`, bind the omitted full envelope by digest, and avoid
  dangling relative references. Deployment bundles retain hash-bound sidecars;
  identity evidence and model-card I/O fields are no longer overloaded with
  unrelated tensor detail.
- Restored all four single-executable-backed MCP calls, added true MCP
  end-to-end and cross-channel equivalence gates, repaired text rendering for
  every public explanation ID, and made invocation failures preserve their
  exit status across output formats.
- Added tensor-aware diff and Markdown projections, shorthand tensor tables,
  bounded MCP result pages, structured recovery diagnostics, a compact agent
  capability view, option suggestions, and explicit terminology, numeric-type,
  and shape-order contracts.
- Made partial all-zero findings identify the affected tensor count and names,
  completed repository-side chat-template discovery, and retained conservative
  static-QDQ wording without inferring a runtime-fused execution path.
- Added independently computed GGUF, ONNX, and SafeTensors measurement-oracle
  gates, a documented DeepBOM property taxonomy, and generic local integration
  seams for Actions, pre-commit, containers, model stores, authoring pipelines,
  OCI attestations, Python pipelines, and bounded engineering-review records.
- Added a release self-SBOM, identity-backed release attestation and readback
  workflow, digest-pinned demonstration containers, PyPI discovery metadata,
  packaged Agent Skill content, and a public-adoption boundary test. `npm audit`
  reports zero known vulnerabilities for this release dependency graph.

## 1.96.15 - 2026-09-11

- Made the GGUF tensor-table assignment signature reproducible from the table
  alone by documenting the `encoding`-to-`dtype` projection, recording the
  GGUF `ne0`-first shape order, and retaining the existing signature identity.
- Completed the Python facade contract: `audit()` now exposes the existing
  engineering/regulatory policy profiles and defect gate, policy-blocked
  results retain their parsed evidence document, section selection defaults to
  analysis only when the caller omits an output kind, and tensor-table scalar
  values use Python-native `int` and `Decimal` types. The module entry point no
  longer imports through `deepbom.__main__`, removing the first-run warning.
- Corrected SafeTensors precision-category counts for reduced- and
  full-precision floating-point tensors and recomputed those counts over every
  member of a sharded repository. Mixed single-file and cross-shard fixtures
  enforce category conservation.
- Separated ONNX serialized quantization representation from runtime claims.
  Static QDQ, dynamic QDQ, weight-only QDQ, QOperator, mixed, and unresolved
  forms are identified explicitly; runtime fusion and the executed integer
  path remain unasserted, and compute coverage counts only serialized integer
  compute operators.
- Added `deepbom.tensor_encoding_inventory.v1`, a bounded format-neutral
  encoding projection for GGUF, SafeTensors, ONNX, and TFLite with reproducible
  inventory and tensor-assignment hashes and no duplicated decimal mirrors.
- Added a Windows-safe `deepbom.batch_manifest.v1` input path. Every member is
  separately hash-bound and audited into a non-clobbering result file; the
  batch result does not infer a combined model or lineage.
- Extended commit-pinned Hugging Face SafeTensors closure acquisition to
  standard optional configuration and tokenizer sidecars, including chat
  templates when present. Local directory selection binds the same sidecars,
  while a lone local tensor file deliberately ignores ambient neighbors.
- Narrowed the all-zero payload finding title to the tensors actually observed
  and advanced the service-worker cache identity for the deployed contract.
- Raised the handwritten runtime and verification source ceilings by 32 KiB
  each for the batch, neutral encoding-inventory, Python facade, and
  quantization-classification contracts, plus the development-tooling ceiling
  by 16 KiB for the measured delivery record. Per-file and generated-data
  ceilings are unchanged.

## 1.96.14 - 2026-09-10

- Added a structure-only `deepbom gguf --tensors` projection with a concise
  human table and `deepbom.tensor_table.v1` JSON, including absolute byte
  ranges, storage ratios, and existing assignment identities without nested
  numerical payload ledgers.
- Added GGUF-specific help and machine capability discovery for that bounded
  projection.
- Added an experimental Python facade (`audit`, `capabilities`, `tensors`, and
  `tensor_inventory`) over the same verified engine, with timeout/output bounds
  and typed exit-code exceptions.
- Raised only the documentation source ceiling by 4 KiB for the generated CLI
  reference and the new tensor/Python contract; runtime file ceilings are
  unchanged.
- Archived source-only, content-addressed CycloneDX review evidence for PRs
  #990, #1067, and #1075 plus issue #862. These research fixtures remain
  excluded from the website and installation artifacts.

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
- Bound the canonical `deepbom.org` apex and the `medbom.org` redirects as
  Cloudflare Custom Domains, allowing Cloudflare to provision DNS and TLS
  without creating a second product, content surface, or origin.
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

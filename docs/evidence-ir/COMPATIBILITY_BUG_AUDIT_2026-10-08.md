# Compatibility and correctness follow-up audit

Date: 2026-10-08. Scope: the current working tree, including the experimental
native model and training extension. This is an engineering verification record,
not a claim that all possible bugs, models or execution environments are covered.
No publication or deployment is part of this audit.

This record supplements, rather than replaces, the
[October 7 implementation qualification](NATIVE_IMPLEMENTATION_VALIDATION_2026-10-07.md).
The existing deployment-artifact contracts and the new trusted local framework
path are tested separately and together at their shared calculation boundary.

## Reproduced defects and corrections

| Area | Failure or omission | Correction and regression coverage |
| --- | --- | --- |
| PyTorch folding | A custom `Sequential.forward` or custom ancestor can reorder or bypass registered children. Folding from containment alone can change the output. | Automatic folding requires exact built-in Sequential ancestry from the root; unsupported call paths are not proposed and explicit requests are rejected. A reversed-order model and a bypassing parent reproduce the boundary. |
| Keras numeric precision | A folded replacement identity and an inverted block could acquire default float32 behavior for a float64 model. | Preserve the original dtype policy and the block input's compute dtype. Float64 forward comparison and exported/restored candidates are exercised. |
| Training recorder lifecycle | A rejected call after `finish()` could leave an update token behind, making the closed event stream inconsistent. | Check the running lifecycle and validate event inputs before updating token state; rejected terminal calls leave a readable run. |
| Sampling with manual updates | Out-of-order returns could use the latest global attempt counter rather than their own sampling decision. | Store the sampling decision with each optimizer-bound token. Invalid return coordinates do not consume the token. |
| JSON metadata | Converting keys to strings could silently merge distinct keys such as `1` and `"1"`. | Reject collisions before canonicalization. |
| Validation interpretation | A TensorFlow probe with no differentiable loss could report a successful backward check. | Report `not_assessed_no_differentiable_loss`; do not claim an optimizer smoke step ran. |
| Candidate package identity | Independently valid hashes did not fully establish agreement among manifest, snapshot and comparison identities; a required file could be omitted from the manifest. | Require the package inventory and exact evidence schemas, validate the evidence bundle before loading, and cross-check comparison baseline/candidate identities. Rehashed but inconsistent manifests are rejected. Package paths use portable POSIX separators. |
| TensorBoard exports | Exporting successive complete run heads into one event directory duplicated earlier measurements. | Export each immutable head under `deepbom/<training_ir_sha256>` and return that directory in projection v2 receipts. Re-exporting one head is idempotent. |
| TensorBoard numeric precision | Legacy float32 scalar summaries changed `1.0000000000000002` to `1.0` and finite `1e100` to infinity. | Emit float64 tensor summaries with the scalar plugin metadata. Both values survive event loading and import back into Training IR exactly. |
| Shared scalar calculations | The native engine introduced another scalar byte-width table. | Use the existing scalar registry through `scalarDtypeBytes`; retain only a supported-type set. A 13-type decoding oracle and the common-rule audit cover this boundary. |
| Verification coverage | The default check runner did not include new native/numerical, SDK and local MCP checks. | Add the missing checks to the full runner and appropriate format/release tiers. Optional real-framework checks remain separately executable. |
| Offline web delivery | Three new numerical-detail modules were missing from the service worker asset set, and the changed cached application had no cache-version advance. | Include their transitive imports and advance the cache from v600 to v601. Import coverage, worker lifecycle, offline controllers and cache invalidation are checked. |
| Existing fixture pin | The CycloneDX consumer ledger retained an older digest after the tracked fixture changed in commit `ee44e53`. | Update the exact source pin and record why; preserve the schema and consumer assertions. This was a stale verification input, not a new BOM calculation change. |
| Source budget | An ignored private experiment directory was counted as product source. | Exclude only untracked, Git-ignored root paths, preserving tracked roots and same-named directories elsewhere. Account explicitly for the new public implementation and immutable documentation archives. |
| Distribution budget | New schemas, numerical views and preserved compatibility archives exceeded the previous total size ceiling. | Measure the additions (604,459 bytes for those files before other bundle changes) and allocate 768 KiB explicitly. The 16 MiB per-file ceiling remains unchanged; the final distribution is 67.45 MiB against 67.625 MiB. |

TensorBoard projection v2 changes the output directory and receipt key, not the
Training IR schema. Prior flat event directories remain importable; they are not
rewritten. See the [usage guide](../NATIVE_MODEL_TRAINING_GUIDE.md) for the returned
`event_logdir` and its import example.

## Verification record

| Check | Executed result |
| --- | --- |
| Full repository runner | All 267 checks passed across the complete run and resumed checks after corrections. The completion ledger contains every index from 1 to 267. The final catalog and size-helper changes were also checked separately. |
| Source framework suite | All 9 groups passed: real PyTorch and TensorFlow training, candidates, precision, frozen state, collectors, logger integration and rejection boundaries. |
| Installed native package | All 9 groups also passed from a newly built and installed Linux wheel using its bundled engine, with both source-engine overrides removed. The same 165-document schema checks passed. |
| Channel equivalence | Installed npm across five formats and two package forms; local MCP; standalone/Python TFLite and ONNX parity; SDK, CLI policy/SARIF/verify/diff/explore; Cargo and packaged engine/WASM tampering checks passed. Static Python installation was also exercised on Python 3.13.13. |
| Generated documents | 165 actual documents passed the schemas: 23 snapshots, 23 Model IRs, 26 Weight IRs, 3 Activation IRs, 28 Training IRs, 41 chunks, 10 candidate manifests, 10 comparisons and 1 result manifest. |
| Existing CLI behavior | 19 cases had byte-identical stdout/stderr and identical exit codes under matching build metadata. |
| Existing contracts | 10 existing schemas and 3 historical compatibility archives remained byte-identical; existing Python signatures were unchanged and Node exports remained a superset. |
| Native common engine | Exact scalar/statistics/SVD, transport, malformed input, budgets, binding and training continuity passed on Node 20.20.2 and 24.12.0. |
| Browser reports | DeepBoard timeline, six tensor views, activation binding, SVG/PNG and narrow/dark layouts passed on source and installed-wheel reports. Candidate structure expansion, clipboard and exact code downloads passed. |
| Compatibility catalog | Version 0.11.0: 48 endpoints and 57 mappings; owner digests, schema/field closure and immutable archives are verified with the final source. |
| Web distribution | Build, 448 cacheable assets with no private artifacts, size limits and the minified Explorer/Redesign browser test passed. The initial shell remains 3 JavaScript requests / 2,475 gzip bytes. |
| Public-source export | Exact member lists, file hashes, licenses, executable imports and private-path exclusion passed for 1,365 files plus the manifest. |

Failures were investigated and corrected before resuming at the failed check;
the original logs remain available locally. They were not skipped or marked as
successful. Source and distribution budget changes are explicit feature-size
allocations; import-coverage checks and per-file limits remain active.

Local execution logs and the 267-index completion ledger are in
`.local-validation/compatibility-review-2026-10-07/` (the audit began on October 7).
The completion ledger is `whole-suite-summary.json`; source and installed native
results are `native-precision-final.log` and `installed-native.log`. Channel
equivalence is recorded in `channel-equivalence.log`. These local artifacts are
not included in the public source export. The source changes remain uncommitted
and have not been pushed or deployed as part of this audit.

## Subsequent user-workflow exercise

A fresh installed-wheel walkthrough on October 8 found an additional native CLI
input defect: `--rules` reused an object-only JSON reader even though its
documented input is an array. Both `[]` and explicit transformation lists were
rejected before optimization. Python API calls alone had not exercised this path.
The common strict JSON reader now accepts an explicitly requested top-level type;
engine manifests and evidence retain the object-only default and duplicate-key
rejection. The CLI requests an array for rule files. A real-framework regression
exercises empty and populated rule files, rejects object and duplicate-key inputs,
and restores the resulting packages. Existing Python SDK checks also pass.

The corrected Python wheel was rebuilt with the previously verified, unchanged
engine and installed for the walkthrough. Fifteen CLI/API user actions completed
with their expected exit codes, including intentional overwrite refusal. PyTorch
candidate refinement and four updates, Keras file import, three GradientTape
updates, two epochs of `fit`, local MLflow, TensorBoard and offline W&B were
executed. Local command logs, immutable outputs and browser captures are retained
under `.local-validation/user-journeys-2026-10-08/`; they are private validation
artifacts, not public distributions. This follow-up does not repeat or extend the
earlier claim of a 267-check repository run to the new patch.

## Compatibility boundary

The comparison baseline is tracked commit
`b161cd4c383e1133e97200a04c8459935f4c7d97`. Baseline JavaScript, Python and schema
inputs are isolated, with the same dependencies, WASM and generated build
metadata used for both sides. Different clean/dirty provenance metadata correctly
changes the provenance finding, so comparisons must use matching build context.

Artifact-based Evidence IR family v1 and existing schemas remain unchanged.
Native Model/Weight/Activation v2 and Training v1 are additive contracts under
family v2. Compatibility catalog 0.11.0 records updated implementation hashes;
older catalog snapshots are retained. The native adapters remain experimental.

The tested framework environment is Linux x86-64, Python 3.12, PyTorch 2.8.0+cpu,
TensorFlow 2.20.0, Keras 3.11.3 and NumPy 2.2.6. Logger checks use local MLflow
3.17.0, TensorBoard 2.20.0 and W&B 0.30.0 offline. Node 20 and 24 are checked.
GPU, AMP, compiled/distributed training, arbitrary custom Python behavior and
native execution on macOS/Windows are not qualified by these runs. The browser
checks do not constitute a live ChatGPT or Claude host requalification.

Successful static inspection, a gradient smoke check, or a smaller model does not
establish task accuracy, clinical validity, measured latency or optimization
benefit. Those require separate task and runtime evidence.

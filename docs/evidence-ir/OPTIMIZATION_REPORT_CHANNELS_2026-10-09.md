# Saved optimization reports across channels

This extends the [2026-10-08 review](OPTIMIZATION_REPORT_DIFF_REVIEW_2026-10-08.md).
The previous review's missing Web/native-report MCP connections describe its
historical state. The implementation now supplies a website viewer and **local**
stdio MCP access. Hosted ChatGPT/Claude contracts have not changed.

## Ownership and boundaries

```text
Local framework model → native evidence → optimization_report.v1 JSON
                                             ↓
                    common schema + digest + reference/quantity checks
                                             ↓
                 optimization_diff.v1 + report query/export projections
                        ↙                    ↓                  ↘
             Web exploration          npm CLI/API         local stdio MCP
                                      Python PDF          embedded files
```

The report, diff and access documents are output contracts, not additional IR
layers. Artifact v1 and native v2 remain distinct. The native report viewer is a
separate workspace so an imported report cannot replace a currently audited
artifact or silently enter the artifact-only renderer.

- The existing native-state total owner is reused by the producer and importer.
  Storage identities conserve shape/count/bytes under the common dtype registry.
- The importer reconstructs the report projection from its recorded probe using
  the existing cost owner. Inconsistent totals, coverage or deltas are rejected.
- Schema validation is compiled from the existing JSON Schemas. The standalone
  validator bundles its small runtime helper; it does not fetch validators from
  a CDN. The generator check detects stale output.
- The website validates on a worker. Import failure hides prior results, and
  switching files cannot display a stale worker response as the new result.
- CLI/MCP read a bounded regular file through one descriptor. MCP additionally
  restricts paths to allowed roots, requires the report hash for exports and
  applies the existing subprocess limits, cancellation and scheduling.
- An MCP export is an embedded resource, not an uploaded file or a persistent
  server record. Host clients may handle downloads differently.
- PDF rendering uses the local Python report renderer on saved JSON. It does
  not import model factories, deserialize checkpoints, run inference or train.

Schema-valid, self-consistent evidence is not an attestation of its producer or
an independent measurement of the original model. Training, activation and
quality observations remain supplied evidence. Do not infer task improvement
from parameter or nominal MAC reductions.

## Verification

`check-optimization-report-channels.mjs` exercises synthetic large-list and
adversarial cases and optionally accepts actual framework report JSON paths.
The local run included PyTorch and TensorFlow reports. Checks cover schema and
arithmetic consistency, CLI/common output parity, Web paging and hierarchy,
downloads, dark mode/mobile layout, no file upload, malformed/resealed reports,
stale hashes, root/symlink escape rejection and live MCP resources. With
`DEEPBOM_REPORT_PYTHON` set it also renders and verifies a PDF through live MCP.

The native evidence, SDK and existing local MCP regression suites cover shared
calculation and transport behavior. Simulation files and local build/test
artifacts remain under ignored `.local-validation/`. A local build or test is
not a deployment receipt.

Local verification completed on 2026-10-09: the channel checks (including live
MCP PDF export), native evidence, SDK, local MCP, ChatGPT/Claude remote MCP,
schema family, compatibility, module imports, HTML entrypoints and build budgets
passed. The final npm executable and installed Python wheel rendered the actual
PyTorch and TensorFlow reports to PDF (67,627 and 89,051 bytes respectively), with
report and file hashes checked. The installed wheel also passed the framework
report suite, including the 1,500-module diff fixture and deterministic output.
The browser print document and complete HTML export were checked separately.
This records tested cases, not a proof of absence of all bugs.

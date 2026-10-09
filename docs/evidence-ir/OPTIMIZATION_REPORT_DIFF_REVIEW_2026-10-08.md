# Optimization report and diff integration review

Scope: native optimization reports, HTML/PDF rendering, common comparison and
cost ownership, packaging and channel compatibility. This is an implementation
review, not certification or proof of absence of all bugs. The native framework
workflow remains an experimental local extension.

## Data flow and ownership

```text
Explicit local PyTorch / TensorFlow model
  → copied model state snapshot + Model IR v2 + Weight IR v2
  → baseline/candidate bundles + rule receipts + shape observations
  → common comparison and optimization_report.v1
  → common optimization_diff.v1 output projection
  → HTML unified/split view and PDF field-level Before/After tables
```

The report and diff are output projections, not new Evidence IR layers. They do
not replace Model IR, Weight IR, Activation IR, Training IR or Provenance IR.
Existing artifact v1 and native v2 contracts retain their identifiers. Stored
report JSON stays unchanged; the diff is derived separately. The compatibility
catalog advances to 0.12.0 without rewriting older snapshots.

`web/lib/native-evidence.js` supplies native-node comparison to both bundle
comparison and diff validation. Counts retain the snapshot comparison owner;
MACs reuse the qualified ONNX contraction mapping. `native-optimization-diff.js`
owns canonical equality, change classes, binding checks and changed-field rows.
Python adds typography, pagination and unified text formatting only.

## Findings and corrections

- PDF had a change index without the field diff. Section 02B now prints every
  changed field with Before/After values. `Absent` and present `null` are distinct.
- Python-local equality could disagree with JS numeric canonicalization. Field
  comparison now uses the common canonical JSON owner. JSON types remain
  distinct, pointer tokens are escaped, and arrays are atomic rather than
  matched by shifted tensor indices.
- Package comparison checks omitted tensor rows. The full canonical comparison
  is now checked against common-engine recomputation.
- Module-only navigation could hide unowned state changes. They now have a
  separate entry. Limitations and source-order changes are retained; an otherwise
  unprojected Model IR identity change is shown instead of reporting no change.
- Cost mapping selected the first matching kernel. It now requires exactly one
  binding; ambiguity is unassessed. Mapped depthwise dimensions must remain exact.
- A supplied transformation index could override its generated index. The report
  now sets the final sequential index, and the diff validates it.

## Channel boundaries

| Surface | Connection | Boundary |
| --- | --- | --- |
| Python CLI/API | `optimization report`, `optimization-report`, `Candidate.export_report` generate HTML/PDF/JSON | PDF needs the report extra; saved-report export does not execute models |
| Common Node bridge | `evidence-native` operations `optimization_report` and `optimization_diff` share JS owners | Structured evidence requests, not arbitrary Python execution |
| Browser HTML report | Self-contained searchable diff, rationale, appendices and explicit exports | Local viewing without network requests |
| Main deepbom.org workspace | Existing deployment-artifact analysis and visualization retained | Native report import is not integrated; Model IR v2 is not silently sent to the artifact v1 renderer |
| Local stdio MCP | Existing static audit/diff tools retained | No native report tool or Python execution tool added |
| Hosted ChatGPT / remote MCP | Existing attachment and bounded static-query contracts retained | No native training-model loading or optimization-report execution claimed |

A standalone HTML report does not mean the main website or hosted MCP has the
native optimization workflow. Adding those surfaces needs a separate read-only
import contract and must not cause implicit Python execution. Existing approved
remote tool schemas are not repurposed.

## Interpretation and checks

Exact native-ID matching does not infer renames. Module containment is not a
complete execution graph. A report digest detects changes relative to that
digest; it does not authenticate a producer or independently verify model bytes.
Parameters/MACs do not prove latency, memory, cache behavior or quality. A `.diff`
download is a review document, not an executable model patch.

Checks: `check-native-evidence.mjs`, `check-optimization-report.mjs`, the real
PyTorch/TensorFlow `check-optimization-report.py`, and
`check-report-diff-browser.mjs`. Existing SDK, local MCP, hosted ChatGPT, schema
and compatibility checks cover channel boundaries. Executed logs and simulation
outputs remain in `.local-validation/`; local builds are not deployment receipts.

## Executed verification

The following checks passed on the local working tree on 2026-10-08:

- Common report/diff schema and calculations: exact integer sizes, repeated
  invocations, missing shapes, ambiguous kernels, canonical numeric equality,
  absent versus null, stale verdicts, dangling references and modified digests.
- Packaged Python report tests with actual PyTorch and TensorFlow models:
  replacement, inference folding, saved-package CLI export without the original
  factory, deterministic PDF bytes and rejection of altered tensor comparisons.
- Browser tests: 1,500-module paging, search, change navigation, unified/split
  views, state-only and connection-only changes, complete diff downloads, print
  expansion, escaped identifiers and narrow-screen layout.
- `check-optimization-diff-parity.mjs`: the browser common module, Node output
  and the installed Python package's embedded HTML projection matched exactly.
- Existing SDK, local MCP, ChatGPT MCP and ChatGPT widget checks passed. The
  widget test covered analysis, follow-ups, visualization downloads and host
  file handoff; it did not test an unimplemented native optimization MCP tool.
- Native evidence and Evidence IR family checks, compatibility schema/source
  digests, web imports, source budget and the static web build passed.

Ten existing simulation packages (five stages in each framework) had their HTML
and PDF regenerated. Package contents and report JSON were preserved. The saved
simulation verifier and both standalone browser reports passed, including exact
module coverage and generated-code downloads. PDF pages containing the field
diff were also rendered and visually inspected. Complete nested configurations
can span several pages; they are not truncated to make the report shorter.

This review verifies the listed paths, fixtures and boundaries. It is not a
claim that every possible framework model, graph or deployment has been tested.
No package, website or MCP deployment was performed by this verification run.

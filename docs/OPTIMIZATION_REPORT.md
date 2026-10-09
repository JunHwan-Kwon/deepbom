# Official model optimization report

The native optimization workflow exports an official monochrome report from
the actual before and after model evidence. It is a product-generated deliverable,
not a separately authored experiment narrative. This workflow is part of the
native development extension; availability in a published package depends on
that release including the extension.

## CLI

Install the framework extra needed to create models, and the optional PDF extra:

```bash
pip install "deepbom[torch,report]"
# Or: pip install "deepbom[tensorflow,report]"
```

Create a candidate and all three official report files:

```bash
python -m deepbom.optimization optimize \
  --factory my_model:build_model \
  --objective structure \
  --inputs inputs.json \
  --output candidate \
  --report-format pdf
```

`inputs.json` describes the explicitly requested synthetic validation probe:

```json
{"args":[{"shape":[2,3,16,16],"dtype":"float32"}],"kwargs":{}}
```

The candidate directory contains `report.pdf`, `report.html`, and `report.json`.
HTML and JSON are generated even without the PDF extra. The normal model package
and its loadable state remain available. With explicit `--rules rules.json`, add
`--rationale "Preserve the first convolution; change only the second block"`
to record a human decision separately from the rule's technical rationale.

Re-export an already saved candidate, with no factory or framework execution:

```bash
deepbom optimization-report candidate --output deliverable.pdf
python -m deepbom.optimization report candidate --output deliverable.html
python -m deepbom.optimization report candidate --output deliverable.json
```

`deepbom optimization-report` is the Python package's entry point. Use
`python -m deepbom.optimization report` to avoid ambiguity when the npm command
is also installed.

The last filename selects the format, or use `--format pdf|html|json`. A different
existing file is never overwritten. Exporting identical bytes is idempotent.
For an older package without `report-source.json`, pass `--baseline` with the
original package. The snapshot must match; absent shape probes remain unassessed.
The command does not reconstruct a missing baseline by executing a factory.

## Python

```python
from deepbom.optimization import optimize, export_report

candidate = optimize(model, objective="structure", example_args=(example,),
                     output_dir="candidate")
candidate.export_report("deliverable.pdf")
export_report("candidate", "deliverable.html")
```

## Report content

1. Before/after snapshot identities, objective, target, selection source and
   optional user rationale.
2. Every applied rule, exact affected subjects, rule version, reason, state
   handling and equivalence/training limitations.
3. Monochrome vector diagrams and complete before/after native structure diff,
   including configuration summaries. Large hierarchies continue on further pages.
4. Complete recorded containment relationships, explicitly distinguished from
   execution edges.
5. Exact state and trainable element counts, state payload bytes, signed deltas,
   and mapped nominal MACs when shape observations exist.
6. Per-invocation input/output shapes, logical output bytes, cost coverage and
   reasons for unmapped arithmetic.
7. Complete weight/buffer correspondence and recorded validation results.

The renderer uses only black, white and neutral grays. PDF embeds its fonts and
uses automatic pagination, repeated table headings and a report digest in the
footer. Identifiers outside the embedded font coverage are explicitly escaped
in PDF; HTML/JSON retain their original Unicode. No missing glyph is silently
substituted. JSON contains full configurations even where printed cells show a
declared concise projection. Structure/state rows are not silently truncated.

The HTML additionally preserves expandable before/after configurations, Python
code copying, and exact `.py`/Markdown downloads from the hash-checked package.
Generated code is embedded as text, never executed by the report. These controls
are excluded from printing; the code still requires its associated model package.

### Model diff for large hierarchies

The standalone HTML opens with a Git-style, read-only evidence diff. It initially
shows changed subjects and selects a changed module when available. Search by
native path, operator or applied rule; filter added/removed/changed modules,
state changes or connection changes. Previous/next change and 50-entry pages
keep navigation bounded. Switch between unified `-`/`+` lines and complete
Before/After records. Rule rationale and trade-offs accompany affected subjects.

Unchanged modules remain available under **All modules**. The complete hierarchy,
connections, tensor contracts and state tables are collapsed in HTML, retained
in the JSON and PDF, and expanded for browser printing. PDF includes a change
index and field-level Before/After tables in section 02B before the full structure
appendix. Missing fields and present nulls are distinguished. No changed field
is silently truncated; long tables continue across pages with repeated headings.

Module verdicts come from the existing common native comparison. The view also
exposes differences in saved state records, incident relationships and observed
calls; a module with unchanged configuration may still have changed weights or
connections. Contract/capture changes and reordered shared source IDs appear as
a separate context entry. Source order is not runtime execution order. Exact
native-ID matching does not infer renames or functional equivalence.

**Download evidence diff** exports all changed entries with report and snapshot
identities, regardless of the active filter/page. It is a review document, not
an executable patch to model code. This view adds no new IR schema and does not
change the report JSON or its digest. It works locally without external scripts
or network requests.

## Evidence and computation ownership

`report-source.json` stores the original baseline evidence, selection context
and optional shape observations. It is covered by the existing candidate
manifest's file hashes. Candidate evidence continues to use the existing
Model IR / Weight IR / model-state-snapshot contracts unchanged.

`web/lib/native-optimization-report.js` is the common projection owner. It
validates native bundles and reuses `compareNativeEvidence` for counts and
correspondence, `estimateOnnxMacs` for supported contraction counts, and the
shared tensor-size registry for logical bytes. Python collects shape facts and
renders this same `deepbom.optimization_report.v1` document to HTML or PDF.
The report is an output projection, not a new analytical IR. Its contract is
[`deepbom-optimization-report-v1.schema.json`](schemas/deepbom-optimization-report-v1.schema.json).

`web/lib/native-optimization-diff.js` owns the separately versioned
[`deepbom.optimization_diff.v1`](schemas/deepbom-optimization-diff-v1.schema.json)
output projection. It checks report/comparison digests and bindings, reuses the
common native-node comparison, and supplies the same canonical changed-field
rows to HTML and PDF. Python formats unified text without recomputing verdicts.
Unowned state and otherwise unprojected identity changes are explicit entries.
The Node evidence bridge exposes the projection without model execution.

The website now has a dedicated [optimization report workspace](https://deepbom.org/reports/optimization/),
linked from the artifact workspace before an audit is required. Open a local
`report.json`; no original model or framework installation is needed for viewing.
Schema, digest, references, state totals and shape-derived cost projections are
checked locally using the common owners before displaying the report. This checks
internal consistency, not the truth of imported observations or producer identity.
The viewer does not import arbitrary HTML or execute supplied model code.

The page provides changed/all-subject search, paginated Before/After hierarchies,
field changes, rule reasons, JSON/diff JSON/HTML downloads and **Print / Save PDF**.
Browser PDF uses the browser's print dialog; the CLI/MCP PDF uses the deterministic
ReportLab monochrome renderer. Both use the same common report and changed-field
evidence. The layouts and file bytes are not asserted to be identical. Web HTML
downloads are complete static documents; the Python package's standalone HTML
also retains its interactive diff and package-code attachments.

### npm CLI and local MCP access

The npm CLI accepts the saved JSON, whereas the Python command above accepts a
candidate directory. They share the report schema; they do not load model code.

```bash
npx deepbom optimization-report candidate/report.json --section summary
npx deepbom optimization-report candidate/report.json --section changes --limit 25
npx deepbom optimization-report candidate/report.json --section subject --subject spatial
npx deepbom optimization-report candidate/report.json --format diff --output report.diff.json
npx deepbom optimization-report candidate/report.json --format pdf --output report.pdf
```

`--expected-sha256` refers to the canonical **report digest**, not the raw JSON
file digest or a model snapshot digest. Follow-ups should use the returned
`report_sha256`. Queries return counts and `next_offset`; exports always include
the complete report/diff, regardless of the query page. Input JSON is limited to
16 MiB. Existing output files are not overwritten.

PDF requires a matching Python `deepbom[report]` installation. Set
`DEEPBOM_REPORT_PYTHON` to that environment's Python executable if it is not
`python3`. Missing PDF support gives an explicit error; JSON/HTML access works
without Python. A report-only render never loads factories or frameworks.

Local stdio MCP exposes:

- `deepbom_optimization_report`: `path` to `report.json`, optional `section`,
  `subject`, `search`, `offset`, `limit` (1–100), and `expected_sha256`.
- `deepbom_export_optimization_report`: `path`, `expected_sha256` and `format`
  (`json`, `diff`, `html`, `pdf`). Returns an embedded MCP file resource with its
  MIME type, byte length, file hash and bound report hash. No user file is written.

Both tools enforce `DEEPBOM_MCP_ALLOWED_ROOTS`, run within the existing queue,
timeout/cancellation and output limits, and return errors rather than silently
truncate. File display or saving depends on the MCP client. For a file exceeding
the MCP response limit, export it with the CLI. Hosted ChatGPT/Claude MCP tools
are unchanged; these additions do not claim availability in the approved remote
plugin. See the [channel verification](evidence-ir/OPTIMIZATION_REPORT_CHANNELS_2026-10-09.md).

Structure rendering currently follows **Native Model IR v2 → optimization report
projection → report-local vector layout → HTML/PDF**. It does not pass through
the existing artifact Model IR visualization manifest, which targets artifact
Model IR v1. The diagram is a module hierarchy, not a complete executable graph.
Native functional operations, data-flow edges and unobserved branches are not
reconstructed by this renderer.

Every captured module must appear exactly once on each side across the diagram
pages; rendering fails if this conservation check fails. The coverage table
reports captured/drawn modules, all recorded relationships, drawn relationships,
and relationships available only in the complete relationship table. Cross-page
relationships are included in that table even when their lines cannot be drawn.
Full identifiers are retained in SVG subject attributes and the structure table.

**Before / After** names the model states in the presentation. **Input / Output**
is reserved for tensor contracts and shapes. The existing JSON keys `baseline`
and `candidate` and the report content identity are unchanged by these labels.

When examples are supplied to optimization, independent clones collect shapes.
Only qualified native Conv/Linear/Dense mappings are counted. The MAC figure is
a **mapped invocation subtotal on the supplied probe path**, not a complete
program count. Functional arithmetic in custom forward code and unobserved
branches remain outside its scope. Missing examples or unsupported mappings
produce explicit unassessed results, never fabricated zeros.

Report export reads and checks saved evidence without importing the factory,
loading executable checkpoints or executing inference. The HTML/PDF/JSON
projections are outside the model-state manifest; their shared report content
digest identifies the evidence projection, not its producer's authenticity.

Applied-rule reasons do not imply measured candidate ranking. Parameter/MAC
reductions do not prove latency, peak memory, cache efficiency, delegate placement
or task quality improvements. The report preserves these distinctions.

## Validation

`scripts/check-optimization-report.py` exercises real PyTorch and TensorFlow
transformations, shared-cost cardinalities, no-input/no-change behavior, exact
state preservation, deterministic PDF output, CLI export with an unavailable
factory, source tampering and older-package baseline checks.

The diff checks include unchanged modules with changed weights or relationships,
source-order context, native-ID additions/removals without inferred renames,
and a 1,500-module presentation fixture. Run
`node scripts/check-report-diff-browser.mjs <report-test-output-directory>` after
the Python report tests to check filtering, pagination, change navigation,
Before/After records, full diff downloads, escaping, printing and mobile layout.

Local validation on 2026-10-08 also exercised a built and installed Linux wheel,
long paginated hierarchies, Unicode identifiers and escaped markup. The browser
regression checks retained exact code copying/downloads and the narrow layout.
The native framework regression suite covered training collectors and MLflow,
TensorBoard and W&B exports. A local build is not evidence of registry publication
or of validation on every operating system.

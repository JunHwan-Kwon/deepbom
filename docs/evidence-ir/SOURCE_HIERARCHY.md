# Source hierarchy and exact tensor connections

Release line: 2.4.0. Website and package publication are verified separately; the engine version does not establish a ChatGPT directory approval.

The hierarchy is now the default **presentation**, not an extra opt-in panel: Explorer, the Web summary, the ChatGPT widget, CLI graph exports, human-readable CLI summaries, and the architecture/group-detail pages in document bundles share it. `model-views/hierarchy-view.json` records the exact grouping behind those document pages. SVG and PNG use the same group membership and serialized connections. Missing source annotations use labelled structural groups.

New Model IR method `1.1.3` uses structural partition rule `1.1.0`: connected data components are divided into bounded display-order windows (at most 50 operations), then repeated signatures and their remaining subranges are examined. New results do not emit catch-all `head`/`tail` groups. A declared source scope with more than 50 direct operations keeps its name and membership, with derived connected children for readable expansion. These are navigation units, not claims about native residual blocks. Older Model IR methods and their original hashes remain readable. Old, large tail groups are repartitioned for presentation without modifying the imported document.

Compatibility: the default machine-readable `model-summary --format json` and `audit --section model_summary` remain the operation/storage ledger `deepbom.model_summary.v1`. Use `--level source --format json` for the exact hierarchy view shown in human summaries. `graph` now defaults to `architecture`; `--view structure` explicitly retains the full serialized-operation view. This changes presentation defaults, not model computation or the hosted MCP tool schemas.

The source hierarchy is a **versioned projection owned by Model IR**, not another Evidence IR family member. Existing Artifact IR, Model IR, Weight IR, Activation IR, Training IR and Snapshot contracts retain their identities. A source path is an exporter declaration observed in a file, not proof of a native module object, runtime invocation, parameter sharing or numerical equivalence.

## One data path

```text
Artifact bytes → format adapter → Artifact IR → Model IR
                         │                        │
                         └─ bounded annotations ──┤
                                                  ↓
                         source hierarchy + exact port boundaries
                                                  ↓
                       common aggregation and selected projection
                             ├─ Web diagram and table
                             ├─ CLI / SDK / local MCP
                             └─ SVG / CSV / JSON / HTML
```

Only the adapter reads native ONNX metadata. Renderers, queries and summaries consume the common contracts. The import-policy check restricts `collectModelSourceAnnotations` to the context builder. Web PNG is a derivative of the same selected SVG; HTML can be printed to PDF by the browser. CLI source-view output supports SVG, HTML and JSON. The existing document visualization command still supplies its original SVG/PNG bundles.

Engineering Evidence includes the supplied projection at `/evidence/model_hierarchy`, and Raw Data includes `static/model_hierarchy.json`. Both validate and retain the common source identity. Older contexts without a hierarchy remain readable; report generation does not independently reconstruct source annotations.

Field-coverage discovery applies its bounded traversal separately to the raw analysis and the four explicitly recognized embedded documents (Artifact IR, Model IR, summary and hierarchy). It still visits every field and reports each document's visit count; an oversized document or an arbitrary unregistered object still fails the coverage gate. A complete hierarchy does not imply that numerical equivalence or runtime performance was assessed.

### Contracts

| Document | Purpose |
| --- | --- |
| `deepbom.model_hierarchy.v1` | Source declarations, group membership, exact input-port edges, model-output boundaries, other program relationships, aggregate metrics and coverage. |
| `deepbom.model_hierarchy_view.v1` | Selected/collapsed rows, bundled edge membership, omitted/internal edge ledger, upstream data reachability and exact source identity. |
| `deepbom.model_hierarchy_query.v1` | Paged group summaries, exact counts, `next_offset`, and JSON Pointers to complete group records. |
| `deepbom.model_summary.v1`, method `1.2.0` | Existing summary shape with corrected block boundaries. Earlier methods remain readable; source rederivation requires the matching method. |
| `deepbom.model_ir_visualization_manifest.v2` | Opt-in conservation semantics: an unrequested exhaustive rendering has `not_requested` status and null assessment counts. Legacy v1 remains the default CLI contract. |

[JSON Schema](../schemas/deepbom-model-hierarchy-v1.schema.json) checks document shape. `validateModelHierarchy(document, model)` additionally verifies hashes, source identity and complete rederivation. Imported annotations remain declarations; verifying that they came from the artifact requires re-analysis of the original bytes. A self-consistent hash is not authenticity.

## Supported evidence and limits

- ONNX primary-graph `pkg.torch.onnx.name_scopes` and `class_hierarchy` are parsed using a bounded string-list grammar. No Python evaluation or model code runs. Native metadata is retained; the projection does not expose stack-trace contents.
- Aligned `aten.*` and related call annotations are separated from the module scope. Several serialized operations can share a scope; their operation IDs and ports remain distinct. The annotation does not establish how many native calls occurred.
- Root functional operations are grouped by their actual connectivity, so unrelated pre/postprocessing is not folded into one apparent loop. Unclassified operations are not relabelled as input/output processing.
- Other formats, missing annotations and nested/function regions retain explicit structural fallback or unavailable annotations. No source architecture is invented from an artifact filename or release label.
- Producer/consumer **ports** determine data edges. `Add(x,x)` retains two input edges. Model outputs remain boundary edges even when internally consumed. Control, state and call relationships stay in a separate ledger and are not drawn as tensor edges.
- Group storage uses a union of storage identities. Do not sum parent and child groups or assume sibling storage totals are disjoint. Equal serialized values do not prove shared training parameters.
- MACs, storage elements, storage bytes and trainable parameter counts have distinct meanings. MAC assessment categories remain separate; unsupported values remain unknown. Native axis order is preserved; no inferred NCHW/NHWC conversion occurs.
- Upstream highlighting means serialized data reachability. It does not show measured influence, autograd, loss dependencies or clinical importance.

## Usage

```sh
# Default readable architecture and summary; no special source flag needed.
deepbom graph model.onnx --format svg -o architecture.svg
deepbom model-summary model.onnx

# Read the full source-bound document.
deepbom audit model.onnx --section model_hierarchy --compact -o hierarchy.json

# List groups, then page or search without returning every operation.
deepbom model-summary model.onnx --level source --source-query --format json \
  --source-limit 20 --source-offset 0 --source-search dec

# Use the exact returned group ID. It is scoped to this artifact/Model IR.
deepbom model-summary model.onnx --level source --source-group '<group-id>' --format markdown
deepbom graph model.onnx --view source --source-group '<group-id>' --format svg -o decoder.svg
deepbom graph model.onnx --view source --source-group '<group-id>' --format html -o decoder.html

# Query the exact details_pointer returned for a group.
deepbom audit model.onnx --pointer '/model_hierarchy/groups/1' --compact

# Overview-only is not an exhaustive preservation test.
deepbom visualize model.onnx --view architecture-overview --manifest-version 2 --format json
```

The existing Node SDK uses the same CLI engine:

```js
import { audit } from "deepbom";

const result = await audit("model.onnx", {
  sections: ["model_ir", "model_hierarchy"],
  expectedSha256: "<independently supplied artifact SHA-256>",
});
const hierarchy = result.sections.model_hierarchy;
```

Local MCP exposes `deepbom_model_hierarchy` with `path`, optional `expected_sha256`, `parent_ref`, `search`, `offset` and `limit` (1–100). It returns `details_pointer` for a subsequent `deepbom_audit` pointer query. Follow-up requests should retain the returned artifact hash. The tool is local and does not fetch remote artifacts. Oversized responses fail explicitly; they are not silently truncated.

Hosted ChatGPT keeps its existing tool names and input schemas. Operator queries add source information inside the existing bounded `details` object and can search source paths. The local hierarchy tool is not automatically a hosted tool, and these changes are not a claim of a new published app review.

## Web interaction and export

Explorer adds a source-hierarchy panel: select a group, expand it, focus its subtree, search paths, or select an operation and trace upstream data. The table, detail records and exports use the same projection. External edges remain in a selected view even if the opposite endpoint is outside it.

Redesign exposes the **original source hierarchy as read-only evidence**. Selecting an original operation locates it in the editable graph. Collapsing a source group does not merge operations, rewrite the model or promote original evidence into a changed candidate.

The SVG has readable fixed-size text in a scrolling viewport. Large visible views require narrowing the group instead of shrinking thousands of nodes to unreadable text. Light, dark and monochrome styles are scoped to this SVG. PNG has explicit pixel/dimension limits. Exported HTML preserves the selected scope and exact detail records; choose a small module before printing to PDF. Automatic publication pagination and Parquet are not implemented in this projection.

## Verification and further work

The regression suite includes an independently constructed three-operation ONNX with repeated source scope, repeated input ports, multiple outputs and an internally consumed output. It checks malformed metadata, invalid Unicode, injection strings, rehashed tampering, schema validation, grouping conservation, CLI/SDK parity and graphless fallback. Browser checks exercise actual file analysis, diagram/table selection, independent Redesign operations, theme isolation, PNG/SVG export and mobile text size. The real stdio MCP test checks pagination, detail pointers, source mismatch rejection and response bounds.

A large private ONNX was also checked locally against independently counted source nodes and port edges. The private model, paths, detailed figures and measurements remain outside Git. This did not rerun training, clinical evaluation, inference equivalence or device profiling.

The [full development plan](SOURCE_HIERARCHY_VISUALIZATION_PLAN_2026-10-12.md) includes further stages. Native module/call capture, many-to-many transformation receipts including confirmed folding, runtime profiler joins, loss/gradient graphs, residual probes, ROI attribution, controlled ablation and CKA are **not implemented by this change**. Existing native and training evidence contracts remain available; this projection does not manufacture these observations from an ONNX graph. Supplemental research-pipeline cache/resume logic is also outside this repository change.

Public checks:

```sh
node scripts/generate-model-hierarchy-schema.mjs --check
node scripts/check-model-hierarchy.mjs
node scripts/check-model-hierarchy-browser.mjs
node scripts/check-mcp-server.mjs
```

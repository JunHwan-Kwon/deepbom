import assert from "node:assert/strict";
import { readFile, mkdir, writeFile } from "node:fs/promises";
import { spawnSync } from "node:child_process";
import { hierarchyFixture } from "./fixtures/source-hierarchy.mjs";
import { analyzeOnnxModel } from "../web/onnx.js";
import { getArtifactIrContext } from "../web/lib/artifact-ir-context.js";
import { projectModelHierarchy, validateModelHierarchy, queryModelHierarchy, resolveModelHierarchyPresentation } from "../web/lib/model-hierarchy.js";
import { buildModelSummary } from "../web/lib/model-summary.js";
import { parseSourceStringList } from "../web/lib/model-source-metadata.js";
import { renderModelHierarchySvg, renderModelHierarchyTable, renderModelHierarchyHtml } from "../web/lib/model-hierarchy-view.js";
import { buildModelIrVisualizationBundle, modelIrVisualizationFiles } from "../web/lib/model-ir-visualization.js";
import { exportGraphVisualization } from "../web/lib/graph-export.js";
import { exportGraphPng } from "../bin/graph-png-export.mjs";
import { canonicalJson } from "../web/lib/report-utils.js";
import { sha256BytesHex, sha256TextHex } from "../web/lib/sha256-sync.js";
import Ajv2020 from "ajv/dist/2020.js";
import { buildHierarchicalGraphProjection } from "../web/lib/graph-hierarchy.js";
import { audit as sdkAudit } from "../sdk/index.mjs";
import { buildEngineeringEvidenceDocument, buildRawDataArtifactFiles } from "../web/lib/report-evidence.js";
import { buildMlBomDocument } from "../web/lib/report-mlbom.js";

const context = options => {
  const bytes = hierarchyFixture(options), analysis = analyzeOnnxModel(bytes, "source.onnx");
  return getArtifactIrContext(analysis, { sha256: sha256BytesHex(bytes), filename: "source.onnx", format: "onnx", size: bytes.length });
};
const c = context(), h = c.model_hierarchy, model = c.model_ir;
assert.equal(h.coverage.annotated_operation_count, 3);
assert.equal(h.coverage.operator_edge_count, 3, "Repeated Add input ports remain distinct edges");
assert.equal(h.coverage.boundary_edge_count, 4, "One input and three output boundaries");
assert.equal(c.graph_ir.edges.filter(row => row.relation === "model_output").length, 3, "Internally consumed graph output remains visible");
const enc = h.groups.find(row => row.path === "enc.0"), dec = h.groups.find(row => row.path === "dec.0");
assert.equal(enc.member_refs.length, 2, "Repeated source scope groups calls without merging their operations");
assert.deepEqual(enc.aggregation.outputs.map(row => row.name).sort(), ["a", "c"]);
assert.equal(dec.aggregation.boundary.incoming_edge_refs.length, 2, "One value, two actual input ports");
assert.equal(enc.aggregation.trainable_parameter_count, null);
const view = projectModelHierarchy(model, h);
assert.equal(view.conservation.represented_operations, 3);
assert.equal(view.rows.length, 2);
const presentation = resolveModelHierarchyPresentation(model, h);
assert.deepEqual(presentation.view, view, "Default presentation uses the exact interactive projection");
const overview = buildModelIrVisualizationBundle(model, { hierarchy: h, views: ["architecture-overview", "block-detail"] });
assert.equal(overview.manifest.architecture_source.view_sha256, view.view_sha256);
assert.deepEqual(overview.pages.filter(page => page.view === "architecture-overview").flatMap(page => page.render_model.rows.map(row => row.subject_ref)), view.rows.map(row => row.id));
assert.deepEqual(overview.pages.filter(page => page.view === "block-detail").flatMap(page => page.render_model.rows.filter(row => row.kind.startsWith("Operation ")).map(row => row.subject_ref)).sort(), model.program.operations.map(op => op.id).sort());
const visualFiles = modelIrVisualizationFiles(model, { hierarchy: h, views: ["architecture-overview"] });
assert.deepEqual(JSON.parse(visualFiles.files.find(file => file.name === "model-views/hierarchy-view.json").data), view);
for (const format of ["svg", "html", "json", "mermaid", "dot"]) {
  const result = exportGraphVisualization(c.graph_ir, { view: "architecture", format, modelIr: model, hierarchy: h });
  assert.equal(result.manifest.hierarchy_view_sha256, view.view_sha256, `${format}: identical default source membership`);
  if (format === "json") assert.deepEqual(JSON.parse(result.text).hierarchy_view, view);
}
assert.equal(exportGraphPng(c.graph_ir, { view: "architecture", modelIr: model, hierarchy: h }).manifest.hierarchy_view_sha256, view.view_sha256);
assert.equal(view.edges.flatMap(row => row.edge_refs).length + view.conservation.internal_edge_refs.length + view.conservation.omitted_edge_refs.length, 7);
const expanded = projectModelHierarchy(model, h, { expanded: [enc.id, dec.id] });
assert.equal(expanded.rows.length, 3);
assert.equal(expanded.edges.flatMap(row => row.edge_refs).length, 7);
const selection = projectModelHierarchy(model, h, { selected: dec.id, expanded: [dec.id] });
assert.equal(selection.conservation.represented_operations, 1);
assert.equal(selection.edges.filter(row => row.from_ref === null).flatMap(row => row.edge_refs).length, 2);
assert.equal(projectModelHierarchy(model, h, { search: "enc.0" }).conservation.represented_operations, 2);
assert.deepEqual(validateModelHierarchy(JSON.parse(JSON.stringify(h)), model), h);
const tampered = structuredClone(h); tampered.groups[0].member_refs.pop(); delete tampered.hierarchy_sha256;
tampered.hierarchy_sha256 = sha256TextHex(canonicalJson(tampered));
assert.throws(() => validateModelHierarchy(tampered, model), /differs/);
const forgedModel = structuredClone(model); forgedModel.program.operations[0].name = "tampered";
assert.throws(() => projectModelHierarchy(forgedModel, h), /SHA-256/);
assert.throws(() => projectModelHierarchy(model, h, { selected: "wrong" }), /Unknown/);
assert.throws(() => projectModelHierarchy(model, h, { expanded: [enc.id, enc.id] }), /duplicated/);
assert.throws(() => queryModelHierarchy(model, h, { parent_ref: "missing" }), /Unknown/);
const one = queryModelHierarchy(model, h, { limit: 1 });
const two = queryModelHierarchy(model, h, { limit: 1, offset: one.next_offset });
assert.notEqual(one.rows[0].id, two.rows[0].id);
assert.equal(one.returned, 1);
assert.equal(queryModelHierarchy(model, h, { search: "enc.0" }).matched, 1);
const validateSchema = new Ajv2020({ strict: true, allErrors: true }).compile(JSON.parse(await readFile("docs/schemas/deepbom-model-hierarchy-v1.schema.json", "utf8")));
for (const doc of [h, view, expanded, selection, one]) assert(validateSchema(doc), JSON.stringify(validateSchema.errors));
const contracted = buildHierarchicalGraphProjection(c.graph_ir, { chunkSize: 8 });
assert.equal(contracted.edge_ledger.length, c.graph_ir.edges.length);
assert.deepEqual(contracted.edge_ledger.map(row => row.source_edge), c.graph_ir.edges, "Legacy contraction keeps exact internal and external edge records");
const fallback = context({ annotate: false });
assert.equal(fallback.model_hierarchy.status, "structural_fallback");
const partialSource = {
  format: "onnx", input_tensor_indices: [0], output_tensor_indices: [321],
  ops: Array.from({ length: 321 }, (_, index) => ({ index, name: "Identity", domain: "ai.onnx",
    inputs: [index], outputs: [index + 1], macs: 0,
    ...(index === 0 ? { onnx_metadata_properties: c.primary_view.ops[0].onnx_metadata_properties } : {}) })),
  tensors: Array.from({ length: 322 }, (_, index) => ({ index, name: `v${index}`, dtype: "FLOAT32", shape: [1, 8] })),
  mac_assessment: { status: "assessed", total_assessed_macs: 0, total_assessed_macs_decimal: "0", compute_ops: 0, assessed_compute_ops: 0, not_assessed_compute_ops: 0 },
};
const partial = getArtifactIrContext(partialSource, { filename: "partial.onnx", format: "onnx", sha256: "a".repeat(64), size: 1024 });
assert.equal(partial.model_hierarchy.coverage.annotated_operation_count, 1);
assert.equal(partial.model_hierarchy.coverage.unannotated_operation_count, 320);
assert(partial.model_hierarchy.groups.filter(group => group.kind === "unclassified").every(group => group.member_refs.length <= 50));
assert.equal(resolveModelHierarchyPresentation(partial.model_ir, partial.model_hierarchy).view.conservation.represented_operations, 321);
assert.equal(partial.model_hierarchy.groups.some(group => group.kind === "root_functional"), false, "Unmapped operations are not relabelled as input/output processing");
const shallowSource = structuredClone(partialSource);
for (const op of shallowSource.ops) op.onnx_metadata_properties = c.primary_view.ops[0].onnx_metadata_properties;
const shallow = getArtifactIrContext(shallowSource, { filename: "shallow.onnx", format: "onnx", sha256: "b".repeat(64), size: 1024 });
const scope = shallow.model_hierarchy.groups.find(group => group.path === "enc.0");
assert.equal(scope.member_refs.length, 321, "Source scope identity and membership remain intact");
assert.equal(scope.direct_member_refs.length, 0);
const expandedScope = projectModelHierarchy(shallow.model_ir, shallow.model_hierarchy, { expanded: [scope.id] });
assert.equal(expandedScope.conservation.represented_operations, 321);
assert(expandedScope.rows.every(row => row.member_refs.length <= 50), "Large shallow source scopes expand into readable derived children");
for (const distinctWeights of [false, true]) {
  const weighted = context({ weights: true, distinctWeights });
  assert.equal(weighted.model_hierarchy.groups.find(row => row.parent_ref === null).aggregation.serialized_element_count.decimal, distinctWeights ? "4" : "2", "Storage union counts identities, not content equality or sibling usages");
  assert.equal(weighted.model_hierarchy.groups.find(row => row.parent_ref === null).aggregation.trainable_parameter_count, null);
}
const unknown = context({ unknown: true });
assert.equal(unknown.model_hierarchy.groups.find(row => row.path === "dec.0").aggregation.macs.decimal, null, "Unsupported MACs must not become zero");
assert.equal(unknown.model_hierarchy.groups.find(row => row.parent_ref === null).aggregation.macs.status, "partial");
for (const options of [{ duplicate: true }, { malformed: true }]) {
  const bad = context(options);
  assert.equal(bad.model_hierarchy.coverage.invalid_annotation_count, 3);
  assert.equal(bad.model_ir.program.operations.length, 3, "Invalid annotation does not destroy valid topology");
}
assert.deepEqual(parseSourceStringList("['', 'a.b', 'a\\\'s']"), ["", "a.b", "a's"]);
for (const text of ["[foo()]", "[];alert(1)", "['unterminated]", "[['nested']]", "[1]", "['\\q']", "['\\uD800']"]) assert.throws(() => parseSourceStringList(text));
const malicious = context({ name: '<script>alert(1)</script>' });
const maliciousView = projectModelHierarchy(malicious.model_ir, malicious.model_hierarchy);
assert(!renderModelHierarchySvg(maliciousView).includes("<script>"));
assert(!renderModelHierarchyHtml(maliciousView).includes("<script>alert"));
assert(renderModelHierarchyTable(view, "csv").includes('"Source / operation"'));
const blockSummary = buildModelSummary(model, { level: "block" });
assert.deepEqual(blockSummary.rows.flatMap(row => row.outputs.map(v => v.name)).sort(), ["a", "b", "c"], "Block outputs include every graph boundary");
assert.equal(buildModelIrVisualizationBundle(model, { views: ["architecture-overview"], manifestVersion: 1 }).manifest.conservation.status, "failed", "Explicit old manifest contract preserved");
assert.equal(buildModelIrVisualizationBundle(model, { views: ["architecture-overview"], manifestVersion: 2 }).manifest.conservation.status, "not_requested");
assert.equal(buildModelIrVisualizationBundle(model, { views: ["architecture-overview"], manifestVersion: 2 }).manifest.conservation.omitted_count, null, "Unrequested checks must not report missing source nodes");
assert.equal(buildModelIrVisualizationBundle(model, { views: ["exhaustive"], manifestVersion: 2 }).manifest.conservation.status, "conserved");

// ExportedProgram signatures contain newlines and Markdown delimiters. Escaped
// report cells must still pass evidence preservation, with original JSON intact.
const metadataRows = [["signature`key|part", "\n# inputs\nx: USER_INPUT | tensor\r\n# outputs\ny: USER_OUTPUT"], ["empty", ""]];
const meta = context({ modelMetadata: metadataRows });
const identity = { filename: "source.onnx", format: "onnx", sha256: meta.model_ir.artifact.sha256, byte_length: meta.model_ir.artifact.byte_length.number };
const reportContext = { identity, artifactIrContext: meta, runtimeEvidence: {} };
const bom = buildMlBomDocument(meta.primary_view, { hash: identity.sha256, fileSizeBytes: identity.byte_length, artifactIr: meta.artifact_ir, timestamp: "2026-10-12T00:00:00.000Z" });
const evidence = buildEngineeringEvidenceDocument(meta.primary_view, { reportContext, rawEvidenceContext: reportContext, mlBomDocument: bom });
assert.deepEqual(evidence.evidence.static_analysis.metadata_presence.metadata_properties.map(({ key, value }) => [key, value]), metadataRows);
assert.equal(evidence.evidence.model_hierarchy.hierarchy_sha256, meta.model_hierarchy.hierarchy_sha256);
const rawFiles = buildRawDataArtifactFiles(meta.primary_view, { rawEvidenceContext: reportContext, mlBomDocument: bom });
const reportViewFile = rawFiles.find(file => file.name === "model-views/hierarchy-view.json");
assert(reportViewFile, "Report ZIP includes the exact hierarchy view behind its overview");
const reportView = JSON.parse(typeof reportViewFile.data === "string" ? reportViewFile.data : new TextDecoder().decode(reportViewFile.data));
assert.deepEqual(reportView, resolveModelHierarchyPresentation(meta.model_ir, meta.model_hierarchy).view);

await mkdir(".local-validation/model-hierarchy", { recursive: true });
await writeFile(".local-validation/model-hierarchy/source.onnx", hierarchyFixture());
await writeFile(".local-validation/model-hierarchy/context.json", JSON.stringify({ model_ir: model, model_hierarchy: h }));
const run = args => { const r = spawnSync(process.execPath, ["bin/deepbom.mjs", ...args], { encoding: "utf8", maxBuffer: 32 * 1024 * 1024 }); assert.equal(r.status, 0, r.stderr); return r.stdout; };
const cli = JSON.parse(run(["audit", ".local-validation/model-hierarchy/source.onnx", "--section", "model_hierarchy", "--compact"]));
assert.equal(cli.sections.model_hierarchy.coverage.operator_edge_count, 3);
const sdk = await sdkAudit(".local-validation/model-hierarchy/source.onnx", { sections: ["model_hierarchy"] });
assert.deepEqual(sdk, cli, "SDK and CLI share the complete hierarchy contract");
const cliView = JSON.parse(run(["model-summary", ".local-validation/model-hierarchy/source.onnx", "--level", "source", "--format", "json"]));
assert.deepEqual(cliView.rows.map(row => row.member_refs.length), [2, 1]);
assert.match(run(["graph", ".local-validation/model-hierarchy/source.onnx", "--view", "source", "--format", "svg"]), /data-view-sha256/);
const cliQuery = JSON.parse(run(["model-summary", ".local-validation/model-hierarchy/source.onnx", "--level", "source", "--source-query", "--source-limit", "1", "--source-offset", "1", "--format", "json"]));
const cliContext = JSON.parse(run(["graph", ".local-validation/model-hierarchy/source.onnx", "--view", "source", "--format", "json"]));
const defaultGraph = JSON.parse(run(["graph", ".local-validation/model-hierarchy/source.onnx", "--format", "json"]));
assert.deepEqual(defaultGraph.hierarchy_view, cliContext.view, "Default graph selects source hierarchy without an extra flag");
assert.equal(run(["model-summary", ".local-validation/model-hierarchy/source.onnx"]), renderModelHierarchyTable(cliContext.view, "table"));
assert.equal(JSON.parse(run(["model-summary", ".local-validation/model-hierarchy/source.onnx", "--format", "json"])).schema, "deepbom.model_summary.v1", "Default machine-readable summary remains compatible");
const defaultVisual = JSON.parse(run(["visualize", ".local-validation/model-hierarchy/source.onnx", "--view", "architecture-overview", "--format", "json"]));
assert.equal(defaultVisual.architecture_source.view_sha256, cliContext.view.view_sha256, "CLI report overview and default graph agree");
assert.deepEqual(cliQuery, queryModelHierarchy(cliContext.model_ir, cliContext.model_hierarchy, { offset: 1, limit: 1 }));
assert.deepEqual(cliQuery.rows, queryModelHierarchy(model, h, { offset: 1, limit: 1 }).rows, "Native parser and CLI projection have identical metrics and membership");
const cliPointer = JSON.parse(run(["audit", ".local-validation/model-hierarchy/source.onnx", "--pointer", cliQuery.rows[0].details_pointer, "--compact"]));
assert.equal(cliPointer.value.id, cliQuery.rows[0].id);
const storage = JSON.parse(run(["audit", "web/samples/nanofable-1m-fp16.safetensors", "--section", "model_hierarchy", "--compact"]));
assert.equal(storage.sections.model_hierarchy.status, "not_applicable_no_program");
assert.equal(storage.sections.model_hierarchy.edges.length, 0);
console.log("Model hierarchy: source metadata, repeated calls, port multiset/boundaries, selected exports, schema identity, rederivation, hostile metadata, fallback, CLI and graphless checks passed.");

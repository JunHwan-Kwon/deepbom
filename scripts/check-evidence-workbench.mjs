import assert from "node:assert/strict";

import { buildGraphDiffSnapshot, compareGraphDiffSnapshots } from "../web/lib/artifact-diff.js";
import { createEvidenceCursor, validateEvidenceCursor } from "../web/lib/evidence-cursor.js";
import { normalizeEvidenceExplanation } from "../web/lib/evidence-why-drawer.js";
import { buildHierarchicalGraphProjection } from "../web/lib/graph-hierarchy.js";
import {
  buildNodeEdgeEvidenceOverlayTemplate,
  validateNodeEdgeEvidenceOverlay,
} from "../web/lib/node-edge-evidence-overlay.js";
import { buildReviewState, buildSelfContainedReviewHtml, compareReviewSessions } from "../web/lib/review-export.js";

const SHA_A = "a".repeat(64);
const SHA_B = "b".repeat(64);
const base = analysis(SHA_A);
base.cpu_cost_target_binding = {
  schema: "deepbom.cpu_cost_target_binding.v1",
  profile_id: "android_mid_a55",
  profile_sha256: "c".repeat(64),
  binding_source: "explicit_id",
  host_observed: false,
  source_input: null,
};

const cursor = createEvidenceCursor({ schema: "deepbom.evidence_cursor.v1", artifact_sha256: SHA_A });
const cursorEvents = [];
cursor.subscribe((next) => cursorEvents.push(next));
cursor.select({ op_index: 11, tensor_index: 3, report_anchor: "finding:EA-1" }, { source: "test" });
assert.equal(cursor.get().op_index, 11);
assert.equal(cursorEvents.length, 1);
assert.equal(validateEvidenceCursor(cursor.get()).valid, true);
assert.equal(validateEvidenceCursor({ ...cursor.get(), artifact_sha256: "bad" }).valid, false);
assert.equal(validateEvidenceCursor({ ...cursor.get(), revision: -1 }).valid, false);
assert.equal(validateEvidenceCursor({ ...cursor.get(), revision: null }).valid, false);
cursor.select({ finding_id: "EA-2" }, { source: "test-finding" });
assert.equal(cursor.get().op_index, null, "A new evidence selection must not inherit an unrelated operator coordinate.");
assert.equal(cursor.get().finding_id, "EA-2");
assert.equal(normalizeEvidenceExplanation({ evidence_class: "ESTIMATED_TARGET_PROFILE" }).evidence_class, "ESTIMATED");

const hierarchy = buildHierarchicalGraphProjection({
  nodes: [
    node("op:10", 10, "stem"), node("op:11", 11, "stem"),
    node("op:20", 20, "head"), node("op:21", 21, "head"),
  ],
  edges: [
    edge("op:10", "op:11", 1, 8),
    edge("op:11", "op:20", 2, 16),
    edge("op:10", "op:20", 3, null),
    edge("op:20", "op:21", 4, 4),
  ],
});
assert.equal(hierarchy.nodes.length, 2);
assert.equal(hierarchy.edges.length, 1, "Only a serialized inter-group relationship may be rendered.");
assert.equal(hierarchy.edges[0].edge_count, 2);
assert.equal(hierarchy.edges[0].byte_length, null, "A partial payload aggregate must not be promoted to a complete zero or sum.");
assert.deepEqual(hierarchy.conservation, {
  original_node_count: 4,
  projected_group_count: 2,
  covered_node_count: 4,
  original_edge_count: 4,
  internal_group_edge_count: 2,
  inter_group_edge_count: 2,
  external_edge_count: 0,
});

const changed = analysis(SHA_B);
changed.ops[1].quantization_state = "none";
changed.ops[1].output_shapes = [[1, 8]];
changed.tensors.find((tensor) => tensor.index === 3).shape = [1, 8];
const diff = compareGraphDiffSnapshots(buildGraphDiffSnapshot(base), buildGraphDiffSnapshot(changed));
assert.equal(diff.matched_count, 3);
assert.ok(diff.changed_match_count >= 1);
assert.match(diff.interpretation_boundary, /Name or operator index alone is never treated as identity/);

const ambiguousLeft = buildGraphDiffSnapshot(duplicateAnalysis(SHA_A, [0, 1]));
const ambiguousRight = buildGraphDiffSnapshot(duplicateAnalysis(SHA_B, [5, 6]));
const ambiguous = compareGraphDiffSnapshots(ambiguousLeft, ambiguousRight);
assert.equal(ambiguous.matched_count, 0, "Duplicate name/contract candidates must remain unresolved.");
assert.ok(ambiguous.ambiguous.length > 0);

const template = buildNodeEdgeEvidenceOverlayTemplate(base);
const overlay = validateNodeEdgeEvidenceOverlay(template, base);
assert.equal(overlay.nodes[0].op_index, 10);
assert.equal(overlay.edges[0].tensor_index, 2);
assert.throws(() => validateNodeEdgeEvidenceOverlay({ ...template, artifact_sha256: SHA_B }, base), /different artifact/);
assert.throws(() => validateNodeEdgeEvidenceOverlay({ ...template, nodes: [{ op_index: 999, metrics: template.nodes[0].metrics }] }, base), /unknown op_index/);
assert.throws(() => validateNodeEdgeEvidenceOverlay({ ...template, edges: [{ ...template.edges[0], tensor_index: 999 }] }, base), /does not identify/);
assert.throws(() => validateNodeEdgeEvidenceOverlay({ ...template, nodes: [{ ...template.nodes[0], metrics: [{ ...template.nodes[0].metrics[0], value: Number.NaN }] }] }, base), /finite/);
assert.throws(() => validateNodeEdgeEvidenceOverlay({ ...template, nodes: [{ ...template.nodes[0], metrics: [{ ...template.nodes[0].metrics[0], value: null, evidence_class: "MEASURED" }] }] }, base), /NOT_ASSESSABLE/);
assert.throws(() => validateNodeEdgeEvidenceOverlay({ ...template, source: { label: "runtime", collected_at: "yesterday" } }, base), /RFC 3339/);

const reviewState = buildReviewState({ analysis: base, cursor: cursor.get(), graphView: { overlay: "structure", viewport: { scale: 1 } }, workspace: "graph", coverageSummary: { assessed: ["graph", "interfaces"] } });
assert.equal(reviewState.schema, "deepbom.review_session.v1");
assert.equal(reviewState.artifact_identity.sha256, SHA_A);
assert.equal(reviewState.selected_subject.finding_id, "EA-2");
assert.deepEqual(reviewState.viewport, { scale: 1 });
assert.equal(reviewState.cpu_cost_target_binding.binding_source, "explicit_id");
assert.equal(reviewState.cpu_cost_target_binding.host_observed, false);
const changedSession = buildReviewState({ analysis: changed, coverageSummary: { assessed: ["graph"] } });
const sessionComparison = compareReviewSessions(reviewState, changedSession);
assert.equal(sessionComparison.axes.artifact_changed, true);
assert.equal(sessionComparison.axes.cpu_target_changed, true);
assert.equal(sessionComparison.coverage.status, "coverage_regression");
assert.deepEqual(sessionComparison.coverage.regressions, ["interfaces"]);
const html = buildSelfContainedReviewHtml({
  analysis: { ...base, model_bytes: "DO_NOT_EXPORT_THIS_PAYLOAD", findings: [{ id: "EA-1", title: "Review", severity: "medium", category: "deployment", confidence: "static", impact: "Check the condition.", evidence: [{ source: "/ops/1", text: "Derived evidence" }] }] },
  graphSvg: '<svg xmlns="http://www.w3.org/2000/svg"><text>graph</text></svg>',
  reviewState,
});
assert.match(html, /DEEPBOM READ-ONLY REVIEW/);
assert.match(html, /Content-Security-Policy/);
assert.match(html, /data:image\/svg\+xml/);
assert.doesNotMatch(html, /DO_NOT_EXPORT_THIS_PAYLOAD/);
assert.match(html, /not a task-accuracy, clinical-validity, or release-readiness certificate/);

console.log(JSON.stringify({ status: "pass", cursor_events: cursorEvents.length, hierarchy_groups: hierarchy.nodes.length, diff_matches: diff.matched_count, ambiguous_groups: ambiguous.ambiguous.length }));

function analysis(sha) {
  return {
    format: "onnx", filename: "fixture.onnx", model_sha256: sha,
    input_tensor_indices: [0], output_tensor_indices: [4],
    tensors: [
      { index: 0, name: "input", shape: [1, 4], dtype: "FLOAT32" },
      { index: 2, name: "hidden", shape: [1, 4], dtype: "FLOAT32" },
      { index: 3, name: "branch", shape: [1, 4], dtype: "FLOAT32" },
      { index: 4, name: "output", shape: [1, 4], dtype: "FLOAT32" },
    ],
    ops: [
      { index: 10, name: "MatMul", domain: "ai.onnx", inputs: [0], outputs: [2], output_shapes: [[1, 4]], output_dtypes: ["FLOAT32"], macs: 16, graph_node_name: "encoder/a" },
      { index: 11, name: "Relu", domain: "ai.onnx", inputs: [2], outputs: [3], output_shapes: [[1, 4]], output_dtypes: ["FLOAT32"], macs: 4, graph_node_name: "encoder/b", quantization_state: "float" },
      { index: 20, name: "Add", domain: "ai.onnx", inputs: [3], outputs: [4], output_shapes: [[1, 4]], output_dtypes: ["FLOAT32"], macs: 4, graph_node_name: "head" },
    ],
  };
}

function duplicateAnalysis(sha, indices) {
  return {
    format: "onnx", filename: "duplicates.onnx", model_sha256: sha,
    tensors: indices.flatMap((index, offset) => [
      { index: offset * 2, shape: [1, 4], dtype: "FLOAT32" },
      { index: offset * 2 + 1, shape: [1, 4], dtype: "FLOAT32" },
    ]),
    ops: indices.map((index, offset) => ({ index, name: "Relu", domain: "ai.onnx", inputs: [offset * 2], outputs: [offset * 2 + 1], output_shapes: [[1, 4]], output_dtypes: ["FLOAT32"], macs: 4 })),
  };
}

function node(id, index, stage) { return { id, index, label: `op${index}`, stage, domain: "x", topo_depth: index, macs: { decimal: "1", number: 1 }, estimated_bytes: { decimal: "4", number: 4 }, quantization: { state: "none" }, placement: { status: "UNASSESSED" } }; }
function edge(from, to, tensorIndex, bytes) { return { id: `e${tensorIndex}`, from, to, tensor_index: tensorIndex, tensor_name: `t${tensorIndex}`, byte_length: bytes == null ? null : { decimal: String(bytes), number: bytes } }; }

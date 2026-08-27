import assert from "node:assert/strict";
import {
  BACKEND_PLACEMENT_STATES as S,
  buildBackendPlacementProjection,
  ortProviderProjectionRows,
  tfliteDelegateProjectionRows,
  validateBackendPlacementProjection,
} from "../web/lib/backend-placement-projection.js";

const sha = "a".repeat(64);
const analysis = {
  format: "tflite",
  model_sha256: sha,
  ops: [
    { index: 10, name: "CONV_2D", inputs: [0], outputs: [1], macs: 100, macs_status: "assessed", estimated_bytes: 50, estimated_bytes_status: "assessed" },
    { index: 11, name: "CONV_2D", inputs: [1], outputs: [2], macs: 200, macs_status: "assessed", estimated_bytes: 100, estimated_bytes_status: "assessed" },
    { index: 12, name: "ADD", inputs: [1, 2], outputs: [3], macs: 30, macs_status: "assessed", estimated_bytes: 60, estimated_bytes_status: "assessed" },
    { index: 13, name: "ADD", inputs: [3], outputs: [4], macs: null, macs_status: "not_assessed", estimated_bytes: 20, estimated_bytes_status: "assessed" },
  ],
  tensors: [
    { index: 0, name: "input", dtype: "UINT8", shape: [1, 4] },
    { index: 1, name: "branch", dtype: "INT4", shape: [3] },
    { index: 2, name: "dynamic", dtype: "FLOAT16", shape: [1, 4], shape_signature: [-1, 4] },
    { index: 3, name: "merge", dtype: "FLOAT32", shape: [1, 2] },
    { index: 4, name: "output", dtype: "FLOAT32", shape: [1, 2] },
  ],
};

const projection = buildBackendPlacementProjection({
  analysis,
  profileId: "test_backend",
  label: "Test backend",
  evidenceClass: "SOURCE_PINNED_ARTIFACT_PRECHECK",
  rows: [
    { op_index: 10, state: S.CONDITIONALLY_ELIGIBLE },
    { op_index: 11, state: S.CONDITIONALLY_ELIGIBLE },
    { op_index: 12, state: S.DEFINITE_EXCLUSION, reason_codes: ["unsupported_add"] },
    { op_index: 13, state: S.UNRESOLVED, unresolved_predicates: ["build_flag"] },
  ],
});

assert.deepEqual(projection.state_counts, {
  CONDITIONALLY_ELIGIBLE: 2,
  DEFINITE_EXCLUSION: 1,
  UNRESOLVED: 1,
});
assert.equal(projection.segment_count, 3);
assert.deepEqual(projection.segments.map((row) => row.op_count), [2, 1, 1]);
assert.equal(projection.graph_edge_count, 4, "Branch tensor must create one edge per distinct consumer.");
assert.equal(projection.boundary_edge_count, 3);
assert.equal(projection.boundary_edges.filter((row) => row.tensor_index === 1).length, 1,
  "The branch edge to the excluded ADD must remain independently visible.");
assert.equal(projection.graph_edges.find((row) => row.tensor_index === 1)?.logical_payload_bytes, 2,
  "Three INT4 elements require ceil(12/8) = 2 logical bytes.");
assert.equal(projection.graph_edges.find((row) => row.tensor_index === 2)?.logical_payload_bytes, null,
  "A dynamic shape signature must not reuse the serialized shape as an exact payload.");
assert.equal(projection.boundary_payload.assessed_edge_count, 2);
assert.equal(projection.boundary_payload.unassessed_edge_count, 1);
assert.equal(projection.boundary_payload.assessed_edge_payload_bytes, 10);
assert.equal(projection.boundary_payload.summed_edge_payload_bytes, null);
assert.equal(projection.workload_envelope.by_state.CONDITIONALLY_ELIGIBLE.complete_macs_decimal, "300");
assert.equal(projection.workload_envelope.by_state.CONDITIONALLY_ELIGIBLE.complete_logical_bytes_decimal, "150");
assert.equal(projection.workload_envelope.by_state.CONDITIONALLY_ELIGIBLE.mac_equivalent_ops_per_logical_byte, 4);
assert.equal(projection.workload_envelope.total.complete_macs_decimal, null);
assert.equal(projection.workload_envelope.total.assessed_macs_decimal, "330");
assert.equal(projection.workload_envelope.conditionally_eligible_mac_share, null,
  "Incomplete total MAC coverage must suppress the candidate MAC share.");
assert.equal(projection.workload_envelope.conditionally_eligible_logical_byte_share_decimal, "0.652173913043");
assert.equal(projection.workload_envelope.backend_cost_model.latency, null);
assert.equal(validateBackendPlacementProjection(projection, { analysis }), true);

const tfliteRows = tfliteDelegateProjectionRows({ rows: [
  { op_index: 0, artifact_precheck_status: "source_candidate_partial", unresolved_predicates: [{ id: "gpu_build" }] },
  { op_index: 1, artifact_precheck_status: "definite_exclusion", definite_exclusion_reasons: [{ id: "version" }] },
] });
assert.deepEqual(tfliteRows.map((row) => row.state), [S.CONDITIONALLY_ELIGIBLE, S.DEFINITE_EXCLUSION]);
assert.deepEqual(tfliteRows[0].unresolved_predicates, ["gpu_build"]);

const ortRows = ortProviderProjectionRows({ ops: [
  { op_index: 0, status: "SOURCE_KERNEL_VERSION_MATCH", source_candidate_after_artifact_precheck: true, artifact_precheck_status: "ARTIFACT_PRECHECK_PASS" },
  { op_index: 1, status: "SOURCE_KERNEL_VERSION_MATCH", source_candidate_after_artifact_precheck: true, artifact_precheck_status: "ARTIFACT_PRECHECK_UNRESOLVED" },
  { op_index: 2, status: "SOURCE_RULE_NOT_FOUND", source_candidate_after_artifact_precheck: false, artifact_precheck_status: "NOT_APPLICABLE_SOURCE_VERSION_GAP" },
  { op_index: 3, status: "EXTERNAL_CUSTOM_REGISTRY_REQUIRED", source_candidate_after_artifact_precheck: false, artifact_precheck_status: "NOT_APPLICABLE_SOURCE_VERSION_GAP" },
] });
assert.deepEqual(ortRows.map((row) => row.state), [
  S.CONDITIONALLY_ELIGIBLE, S.UNRESOLVED, S.DEFINITE_EXCLUSION, S.UNRESOLVED,
]);

assert.throws(() => buildBackendPlacementProjection({
  analysis,
  profileId: "missing",
  label: "Missing row",
  evidenceClass: "TEST",
  rows: projection.rows.slice(0, 3),
}), /cover every operator/);
assert.throws(() => buildBackendPlacementProjection({
  analysis: { ...analysis, ops: [...analysis.ops, { ...analysis.ops[0] }] },
  profileId: "duplicate",
  label: "Duplicate op",
  evidenceClass: "TEST",
  rows: [],
}), /duplicates op/);
assert.throws(() => buildBackendPlacementProjection({
  analysis: {
    ...analysis,
    ops: analysis.ops.map((op, index) => index === 2 ? { ...op, outputs: [1] } : op),
  },
  profileId: "producer",
  label: "Duplicate producer",
  evidenceClass: "TEST",
  rows: projection.rows,
}), /multiple producers/);

const tampered = structuredClone(projection);
tampered.boundary_payload.assessed_edge_payload_bytes += 1;
assert.throws(() => validateBackendPlacementProjection(tampered, { analysis }), /payload does not reproduce/);

console.log("Backend placement projection passed (identity, three-state conservation, branch edges, sub-byte bytes, dynamic-shape non-assessment, adapters, and fail-closed mutations).");

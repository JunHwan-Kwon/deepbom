import assert from "node:assert/strict";

import { buildExecutionPlacementEvidence } from "../web/lib/execution-placement-evidence.js";
import { buildPlacementComparison } from "../web/lib/placement-comparison.js";
import { parseLiteRtQualcommEvidence } from "../web/lib/litert-qualcomm-evidence.js";
import { tfliteAcceleratorSourceManifest } from "../web/lib/tflite-accelerator-source-profiles.js";

const SHA = "a".repeat(64);
const analysis = {
  format: "tflite",
  model_sha256: SHA,
  ops: [
    op(0, "CONV_2D", [0], [1], true),
    op(1, "RELU6", [1], [2], true),
    op(2, "QUANTIZE", [2], [3], false),
    op(3, "CUSTOM", [3], [4], false),
  ],
  tensors: [
    tensor(0, "FLOAT32"), tensor(1, "FLOAT32"), tensor(2, "FLOAT32"),
    tensor(3, "INT8"), tensor(4, "INT8"),
  ],
};

const sourceEvidence = buildExecutionPlacementEvidence(analysis);
const profileIds = sourceEvidence.static_profiles.map((row) => row.profile_id);
assert.deepEqual(profileIds, ["xnnpack_cpu", "tflite_coreml_delegate", "litert_qualcomm_qnn"]);
assert.equal(sourceEvidence.placement_comparison.schema, "deepbom.placement_comparison.v1");
assert.equal(sourceEvidence.placement_comparison.rows.length, 3);
assert.equal(sourceEvidence.placement_comparison.conservation.all_rows_cover_graph, true);

const coreMl = sourceEvidence.static_profiles.find((row) => row.profile_id === "tflite_coreml_delegate");
assert.equal(coreMl.state_counts.CONDITIONALLY_ELIGIBLE, 2);
assert.equal(coreMl.state_counts.DEFINITE_EXCLUSION, 2);
assert.ok(coreMl.rows[0].unresolved_predicates.includes("selected_apple_os_coreml_version_and_delegate_build"));

const fp16Full = coreMlProfile({
  ops: [
    op(0, "DEQUANTIZE", [1], [2], false),
    op(1, "CONV_2D", [0, 2], [3], false),
  ],
  tensors: [
    tensor(0, "FLOAT32"),
    { ...tensor(1, "FLOAT16"), constant_buffer: true },
    tensor(2, "FLOAT32"),
    tensor(3, "FLOAT32"),
  ],
});
assert.equal(fp16Full.rows[0].state, "CONDITIONALLY_ELIGIBLE");
assert.ok(fp16Full.rows[0].unresolved_predicates.includes("all_non_dequantize_nodes_must_be_accepted_by_the_selected_coreml_delegate_build"));

const fp16Partial = coreMlProfile({
  ops: [
    op(0, "DEQUANTIZE", [1], [2], false),
    op(1, "CONV_2D", [0, 2], [3], false),
    op(2, "CUSTOM", [3], [4], false),
  ],
  tensors: [
    tensor(0, "FLOAT32"),
    { ...tensor(1, "FLOAT16"), constant_buffer: true },
    tensor(2, "FLOAT32"),
    tensor(3, "FLOAT32"),
    tensor(4, "FLOAT32"),
  ],
});
assert.equal(fp16Partial.rows[0].state, "DEFINITE_EXCLUSION");
assert.deepEqual(fp16Partial.rows[0].reason_codes, ["fp16_constant_dequantize_kept_on_cpu_during_partial_coreml_delegation"]);

const qualcomm = sourceEvidence.static_profiles.find((row) => row.profile_id === "litert_qualcomm_qnn");
assert.equal(qualcomm.state_counts.CONDITIONALLY_ELIGIBLE, 3);
assert.equal(qualcomm.state_counts.DEFINITE_EXCLUSION, 1);

const sourceProfile = tfliteAcceleratorSourceManifest().profiles.find((row) => row.id === "litert_qualcomm_qnn");
analysis.litert_qualcomm_evidence = parseLiteRtQualcommEvidence({
  schema: "deepbom.litert_qualcomm_compiler_dispatch_evidence.v1",
  artifact_sha256: SHA,
  source: { litert_commit: sourceProfile.source.commit, rulepack_sha256: sourceProfile.rulepack_sha256 },
  compiler: { name: "litert-qualcomm-compiler", version: "fixture", binary_sha256: "b".repeat(64) },
  invocation: { options: ["--soc_model=fixture"] },
  compiled_plan_sha256: "c".repeat(64),
  source_file_sha256: "d".repeat(64),
  operations: analysis.ops.map((row) => ({
    op_index: row.index,
    op_name: row.name,
    compile_status: row.index < 3 ? "compiled" : "not_compiled",
    reason: row.index < 3 ? null : "compiler_rejected_custom_op",
  })),
  dispatch: { status: "not_observed" },
  summary: { operation_count: 4, compiled_operation_count: 3, not_compiled_operation_count: 1, dispatch_status: "not_observed" },
}, analysis);

const compiledEvidence = buildExecutionPlacementEvidence(analysis);
const compiled = compiledEvidence.static_profiles.find((row) => row.profile_id === "litert_qualcomm_qnn_compiled_plan");
assert.equal(compiled.state_counts.CONDITIONALLY_ELIGIBLE, 3);
assert.equal(compiled.state_counts.DEFINITE_EXCLUSION, 1);
assert.equal(compiled.source.dispatch_status, "not_observed");

const selected = buildPlacementComparison(analysis, compiledEvidence.static_profiles, {
  selectedProfileIds: ["xnnpack_cpu", "tflite_coreml_delegate", "litert_qualcomm_qnn_compiled_plan"],
});
assert.deepEqual(selected.selected_profile_ids, ["xnnpack_cpu", "tflite_coreml_delegate", "litert_qualcomm_qnn_compiled_plan"]);
assert.equal(selected.rows.length, 3);
assert.throws(() => buildPlacementComparison(analysis, compiledEvidence.static_profiles, { selectedProfileIds: ["missing"] }), /unavailable/);

const tampered = structuredClone(analysis.litert_qualcomm_evidence);
tampered.operations[0].op_name = "ADD";
assert.throws(() => buildExecutionPlacementEvidence({ ...analysis, litert_qualcomm_evidence: tampered }), /name mismatch|SHA-256/);

console.log("Placement comparison checks passed: N-way source profiles, compiler-observed Qualcomm projection, graph conservation, and fail-closed identity binding.");

function coreMlProfile({ ops, tensors }) {
  return buildExecutionPlacementEvidence({ format: "tflite", model_sha256: SHA, ops, tensors })
    .static_profiles.find((row) => row.profile_id === "tflite_coreml_delegate");
}

function op(index, name, inputs, outputs, xnnpack) {
  return { index, name, version: 1, inputs, outputs, macs: index === 0 ? 64 : 0, xnnpack_chain_id: xnnpack ? 0 : -1, xnnpack_supported: xnnpack };
}

function tensor(index, dtype) {
  return { index, name: `t${index}`, dtype, shape: [1, 4], shape_signature: [1, 4] };
}

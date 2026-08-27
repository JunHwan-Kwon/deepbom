import { createHash } from "node:crypto";
import { parseRuntimeAssignmentDocument } from "../web/lib/kernel-inspector.js";
import { runtimeEnvironmentMarkdown } from "../web/lib/report-sections.js";
import {
  buildTfliteRuntimeAssignmentDocument,
  parseTfliteRuntimeInfoSource,
  previewTfliteRuntimeInfoMapping,
  TFLITE_RUNTIME_INFO_SOURCE,
} from "../web/lib/tflite-runtime-info-adapter.js";
import { createCheck } from "./check-assert.mjs";

const { done, expect, expectEqual, expectThrows } = createCheck("TFLite runtime-info adapter check");

const sha = "a".repeat(64);
const profileSha = "b".repeat(64);
const analysis = {
  format: "tflite",
  model_sha256: sha,
  target_profile: { id: "android_mid_a55", profile_sha256: profileSha },
  ops: [
    { index: 0, name: "CONV_2D", inputs: [0, 1], outputs: [2], xnnpack_chain_id: 0, macs: 100, estimated_bytes: 16 },
    { index: 1, name: "RELU", inputs: [2], outputs: [3], xnnpack_chain_id: 0, macs: 0, estimated_bytes: 16 },
    { index: 2, name: "SQUEEZE", inputs: [3], outputs: [4], xnnpack_chain_id: -1, macs: 0, estimated_bytes: 16 },
    { index: 3, name: "SOFTMAX", inputs: [4], outputs: [5], xnnpack_chain_id: -1, macs: 20, estimated_bytes: 16 },
  ],
  tensors: [
    { index: 0, name: "input", bytes: 16 },
    { index: 1, name: "weight", bytes: 16 },
    { index: 2, name: "conv", bytes: 16 },
    { index: 3, name: "relu", bytes: 16 },
    { index: 4, name: "squeeze", bytes: 16 },
    { index: 5, name: "output", bytes: 16 },
  ],
};

const validBytes = modelRuntimeDetails();
const profile = parseTfliteRuntimeInfoSource(validBytes, analysis);
expectEqual(profile.schema, "deepbom.tflite_runtime_info_adapter.v1", "TFLite runtime-info adapter schema should be explicit.");
expectEqual(profile.original_node_count, 4, "All original graph ops should bind by exact topology.");
expectEqual(profile.delegate_node_count, 1, "Delegate node inventory should come from ModelRuntimeDetails.");
expectEqual(profile.execution_plan_node_count, 3, "Execution-plan nodes should preserve observed plan cardinality.");
expectEqual(profile.delegated_op_count, 2, "Symmetrically replaced original ops should be observed as delegated.");
expectEqual(profile.partitions[0].delegate_name, "TfLiteXNNPackDelegate", "Delegate provider name should come from delegate metadata.");
expectEqual(profile.partitions[0].replaced_op_ids.join(","), "0,1", "Delegate partition should retain exact replaced original node IDs.");
expectEqual(profile.assignments[0].partition_id, "subgraph:0/delegate_node:4", "Observed delegate node ID should define an explicit partition ID.");
expectEqual(profile.assignments[2].provider, "TFLite non-delegated kernel", "Execution-plan original ops should remain explicitly non-delegated.");
expectEqual(profile.topology_binding.source_artifact_sha256_embedded, false, "The adapter must not imply the source proto embeds artifact SHA-256.");

const preview = previewTfliteRuntimeInfoMapping(profile, analysis);
expectEqual(preview.assignment_count, 4, "Preview should expose full assignment coverage.");
expectEqual(preview.delegate_partition_count, 1, "Preview should expose exact delegate partition count.");

const profileSha256 = createHash("sha256").update(validBytes).digest("hex");
const document = buildTfliteRuntimeAssignmentDocument(profile, analysis, {
  runtimeVersion: "tensorflow/tensorflow@runtime-build",
  runtimeBuild: "benchmark_model release; XNNPACK enabled",
  collectedAt: "2026-07-16T00:00:00.000Z",
  profileSha256,
  captureId: "fixture-capture-001",
});
const parsed = parseRuntimeAssignmentDocument(JSON.stringify(document), analysis);
expectEqual(parsed.schema, "deepbom.runtime_assignment.v1.9", "TFLite runtime plan should normalize to runtime-assignment v1.9.");
expectEqual(parsed.evidence_class, "OBSERVED_RUNTIME", "Imported execution-plan placement should be runtime-observed evidence.");
expectEqual(parsed.source.duration_semantics, "not_collected", "ModelRuntimeDetails must not imply timing evidence.");
expectEqual(parsed.source.adapter.artifact_binding, "active_artifact_exact_original_op_topology", "Artifact binding method should remain explicit.");
expectEqual(parsed.source.adapter.source_artifact_sha256_embedded, false, "Normalized evidence should disclose absent source artifact digest.");
expectEqual(parsed.comparison.placement_assessment.match_count, 4, "Observed assignment should compare deterministically with static prediction.");
expectEqual(parsed.comparison.observed_partitions.explicit_partition_count, 1, "Observed delegate node ID should create one explicit runtime partition.");
expectEqual(parsed.comparison.duration_comparison.total_duration_us, null, "ModelRuntimeDetails evidence must never emit a timing total.");
expect(parsed.assignments.every((row) => row.kernel == null), "ModelRuntimeDetails evidence must never claim an executed microkernel.");
expectEqual(parsed.source.adapter.source_sha256, TFLITE_RUNTIME_INFO_SOURCE.source_sha256, "Generator source digest should survive strict normalization.");
const reportSection = runtimeEnvironmentMarkdown({ runtimeAssignmentEvidence: parsed });
expect(reportSection.includes(`runtime-plan protobuf SHA-256 ${profileSha256}`), "Engineering runtime section should bind the imported protobuf digest.");
expect(reportSection.includes("exact topology 4/4 original op(s)"), "Engineering runtime section should disclose exact topology coverage.");
expect(reportSection.includes("source artifact SHA-256 embedded false"), "Engineering runtime section should disclose that the source proto has no artifact digest.");
expect(reportSection.includes("executed microkernel symbols"), "Engineering runtime section should preserve the executed-microkernel evidence limit.");
expect(!reportSection.includes("undefined"), "TFLite adapter report prose must not leak ORT-only undefined counters.");

expectThrows(() => parseTfliteRuntimeInfoSource(modelRuntimeDetails({ op1DelegatedTo: null }), analysis), "symmetric", "Importer should reject a missing original-to-delegate reverse mapping.");
expectThrows(() => parseTfliteRuntimeInfoSource(modelRuntimeDetails({ executionPlan: [4, 1, 2, 3] }), analysis), "must not remain", "Importer should reject replaced original ops left in the execution plan.");
expectThrows(() => parseTfliteRuntimeInfoSource(modelRuntimeDetails({ executionPlan: [4, 2, 99] }), analysis), "unknown node 99", "Importer should reject unknown execution-plan node IDs.");
expectThrows(() => parseTfliteRuntimeInfoSource(modelRuntimeDetails({ op1Name: "RELU6" }), analysis), "op name mismatch", "Importer should reject original-op name drift.");
expectThrows(() => parseTfliteRuntimeInfoSource(modelRuntimeDetails({ op1Inputs: [99] }), analysis), "input tensor IDs", "Importer should reject original-op tensor topology drift.");
expectThrows(() => parseTfliteRuntimeInfoSource(modelRuntimeDetails({ duplicateNodeIdField: true }), analysis), "duplicate singular field node.id", "Importer should reject ambiguous duplicate singular fields.");
const optionalInputAnalysis = { ...analysis, ops: analysis.ops.map((op) => op.index === 1 ? { ...op, inputs: [-1, 2] } : op) };
const optionalInputProfile = parseTfliteRuntimeInfoSource(modelRuntimeDetails({ op1Inputs: [-1, 2] }), optionalInputAnalysis);
expectEqual(optionalInputProfile.assignments[1].op_index, 1, "Importer should accept the TFLite kTfLiteOptionalTensor -1 sentinel in original-op tensor topology.");
expectThrows(() => parseTfliteRuntimeInfoSource(Uint8Array.of(0x12, 0x05, 0x08), analysis), "boundary", "Importer should reject truncated length-delimited fields.");
expectThrows(() => parseTfliteRuntimeInfoSource(validBytes, { ...analysis, format: "onnx" }), "requires an active TFLite", "Importer should reject a non-TFLite active artifact.");

done("TFLite ModelRuntimeDetails adapter contract passed (strict protobuf, exact topology, symmetric delegation, and evidence limits). ");

function modelRuntimeDetails(options = {}) {
  const op1DelegatedTo = Object.hasOwn(options, "op1DelegatedTo") ? options.op1DelegatedTo : 4;
  const originalNodes = [
    node({ id: 0, name: "CONV_2D", type: "3", inputs: [0, 1], outputs: [2], delegatedTo: 4 }),
    node({ id: 1, name: options.op1Name || "RELU", type: "19", inputs: options.op1Inputs || [2], outputs: [3], delegatedTo: op1DelegatedTo, duplicateId: options.duplicateNodeIdField }),
    node({ id: 2, name: "SQUEEZE", type: "43", inputs: [3], outputs: [4] }),
    node({ id: 3, name: "SOFTMAX", type: "25", inputs: [4], outputs: [5] }),
  ];
  const delegate = node({
    id: 4,
    name: "TfLiteXNNPackDelegate",
    type: "Delegate/TfLiteXNNPackDelegate",
    inputs: [0, 1],
    outputs: [3],
    delegateName: "TfLiteXNNPackDelegate",
    replacedIds: [0, 1],
  });
  const subgraph = message([
    intField(1, 0),
    ...[...originalNodes, delegate].map((value) => bytesField(3, value)),
    packedIntField(4, options.executionPlan || [4, 2, 3]),
    intField(5, 1),
    stringField(6, "main"),
  ]);
  return message([stringField(1, "fixture"), bytesField(2, subgraph)]);
}

function node({ id, name, type, inputs, outputs, delegatedTo = null, delegateName = null, replacedIds = [], duplicateId = false }) {
  const fields = [intField(1, id)];
  if (duplicateId) fields.push(intField(1, id));
  fields.push(stringField(2, name), stringField(3, type), packedIntField(4, inputs), packedIntField(5, outputs));
  if (delegateName != null) fields.push(bytesField(8, message([stringField(1, delegateName), packedIntField(2, replacedIds)])));
  else if (delegatedTo != null) fields.push(intField(9, delegatedTo));
  return message(fields);
}

function intField(field, value) { return concat(varint((field << 3) | 0), varint(value)); }
function bytesField(field, value) { return concat(varint((field << 3) | 2), varint(value.length), value); }
function stringField(field, value) { return bytesField(field, new TextEncoder().encode(value)); }
function packedIntField(field, values) { return bytesField(field, concat(...values.map(varint))); }
function message(fields) { return concat(...fields); }

function varint(value) {
  let remaining = BigInt(value);
  if (remaining < 0) remaining = BigInt.asUintN(64, remaining);
  const bytes = [];
  do {
    let byte = Number(remaining & 0x7fn);
    remaining >>= 7n;
    if (remaining) byte |= 0x80;
    bytes.push(byte);
  } while (remaining);
  return Uint8Array.from(bytes);
}

function concat(...arrays) {
  const output = new Uint8Array(arrays.reduce((sum, value) => sum + value.length, 0));
  let offset = 0;
  for (const value of arrays) {
    output.set(value, offset);
    offset += value.length;
  }
  return output;
}

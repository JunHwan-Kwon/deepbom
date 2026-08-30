import { createHash } from "node:crypto";
import { createCheck } from "./check-assert.mjs";
import { parseRuntimeAssignmentDocument } from "../web/lib/kernel-inspector.js";
import { buildTfliteRuntimeAssignmentDocument, parseTfliteRuntimeInfoSource } from "../web/lib/tflite-runtime-info-adapter.js";
import { buildTfliteProfiledAssignmentDocument, parseTfliteBenchmarkProfileSource, previewTfliteBenchmarkProfileMapping } from "../web/lib/tflite-profile-info-adapter.js";
import { buildTfliteRuntimeTimingCsv } from "../web/lib/report-evidence.js";
import { runtimeEnvironmentMarkdown } from "../web/lib/report-sections.js";

const { done, expect, expectEqual, expectThrows } = createCheck("TFLite BenchmarkProfilingData adapter check");
const analysis = {
  format: "tflite",
  model_sha256: "a".repeat(64),
  target_profile: { id: "android_mid_a55", profile_sha256: "b".repeat(64) },
  tensors: [
    { index: 0, name: "input", dtype: "FLOAT32", shape: [1, 4] },
    { index: 1, name: "weights", dtype: "FLOAT32", shape: [4, 4] },
    { index: 2, name: "conv", dtype: "FLOAT32", shape: [1, 4] },
    { index: 3, name: "relu", dtype: "FLOAT32", shape: [1, 4] },
    { index: 4, name: "squeezed", dtype: "FLOAT32", shape: [4] },
    { index: 5, name: "prob", dtype: "FLOAT32", shape: [4] },
  ],
  ops: [
    { index: 0, name: "CONV_2D", inputs: [0, 1], outputs: [2], xnnpack_chain_id: 0, macs: 100, estimated_bytes: 64 },
    { index: 1, name: "RELU", inputs: [2], outputs: [3], xnnpack_chain_id: 0, macs: 4, estimated_bytes: 32 },
    { index: 2, name: "SQUEEZE", inputs: [3], outputs: [4], xnnpack_chain_id: -1, macs: 0, estimated_bytes: 32 },
    { index: 3, name: "SOFTMAX", inputs: [4], outputs: [5], xnnpack_chain_id: -1, macs: 20, estimated_bytes: 32 },
  ],
};

const captureId = "pixel8-run-001";
const runtimeInfoBytes = modelRuntimeDetails();
const runtimeProfile = parseTfliteRuntimeInfoSource(runtimeInfoBytes, analysis);
const runtimeDocument = buildTfliteRuntimeAssignmentDocument(runtimeProfile, analysis, {
  runtimeVersion: "tensorflow/tensorflow@runtime-build",
  runtimeBuild: "benchmark_model release; XNNPACK per-op profiling",
  collectedAt: "2026-07-16T01:00:00.000Z",
  profileSha256: sha(runtimeInfoBytes),
  captureId,
});
const runtimeEvidence = parseRuntimeAssignmentDocument(JSON.stringify(runtimeDocument), analysis);
expectEqual(runtimeEvidence.source.capture_id, captureId, "Runtime-plan evidence should retain the declared capture ID.");

const completeBytes = benchmarkProfile({ includePartition: true, includeInternal: true });
const completeProfile = parseTfliteBenchmarkProfileSource(completeBytes);
const preview = previewTfliteBenchmarkProfileMapping(completeProfile, runtimeEvidence, analysis);
expectEqual(preview.mapped_execution_node_count, 3, "All three execution-plan nodes should map.");
expectEqual(preview.delegate_partition_timing_count, 1, "The delegate node row should map to one partition total.");
expectEqual(preview.delegate_internal_event_count, 1, "Delegate-internal timing should remain separately inventoried.");
expectEqual(preview.graph_total_us, 125, "Complete common-run execution-node timing should produce an exact graph total.");

const combinedDocument = buildTfliteProfiledAssignmentDocument(completeProfile, runtimeEvidence, analysis, {
  profileSha256: sha(completeBytes),
  captureId,
  collectedAt: "2026-07-16T01:00:02.000Z",
});
const combined = parseRuntimeAssignmentDocument(JSON.stringify(combinedDocument), analysis);
expectEqual(combined.schema, "deepbom.runtime_assignment.v1.10", "Combined evidence should use runtime-assignment v1.10.");
expectEqual(combined.source.adapter.schema, "deepbom.tflite_runtime_info_adapter.v2", "Combined evidence should use the timing-capable adapter schema.");
expectEqual(combined.assignments[0].duration_us, null, "A partition total must not be copied to delegated original op #0.");
expectEqual(combined.assignments[1].duration_us, null, "A partition total must not be copied to delegated original op #1.");
expectEqual(combined.assignments[2].duration_us, 10, "A non-delegated original execution node should retain its measured mean per run.");
expectEqual(combined.assignments[3].duration_us, 15, "A second non-delegated execution node should retain its measured mean per run.");
expectEqual(combined.source.adapter.timing_profile.execution_node_total_us, 125, "Execution total should sum partition and CPU nodes once.");
expectEqual(combined.source.adapter.timing_profile.cpu_execution_node_subtotal_us, 25, "CPU subtotal should be reconstructed from original execution nodes.");
expectEqual(combined.source.adapter.timing_profile.delegate_partition_subtotal_us, 100, "Partition subtotal should use the delegate-node row once.");
expectEqual(combined.source.adapter.timing_profile.delegate_internal_profiled_subtotal_us, 60, "Delegate-internal subtotal should remain separately reported.");
expectEqual(combined.source.adapter.timing_profile.primary_delegate_profiled_subtotal_us, null, "A delegate-section event must not be relabeled as a primary delegate-profiled event.");
expectEqual(combined.source.adapter.timing_profile.delegate_internal_section_subtotal_us, 60, "Nested delegate-section timing should be identified explicitly.");
expectEqual(combined.comparison.duration_comparison.total_duration_us, 125, "Runtime comparison should expose the validated execution-plan total.");
expectEqual(combined.comparison.observed_partitions.partitions[0].duration_us, 100, "Observed partition inventory should expose the partition total without original-op duplication.");

const timingCsv = buildTfliteRuntimeTimingCsv(combined);
expect(timingCsv.includes("delegate_partition") && timingCsv.includes("delegate_internal"), "Timing CSV should preserve separate partition and delegate-internal evidence groups.");
expect(timingCsv.includes(sha(completeBytes)), "Timing CSV should bind the raw profiling protobuf digest.");
const report = runtimeEnvironmentMarkdown({ runtimeAssignmentEvidence: combined });
expect(report.includes(`timing protobuf SHA-256 ${sha(completeBytes)}`), "Engineering report should bind the profiling protobuf digest.");
expect(report.includes("Primary delegate-profiled subtotal") && report.includes("Nested delegate-internal subtotal") && report.includes("not added to partition totals"), "Engineering report should separate unassigned primary delegate events from nested delegate timing.");

const internalOnlyBytes = benchmarkProfile({ includePartition: false, includeInternal: true, internalInPrimary: true });
const internalOnly = parseTfliteBenchmarkProfileSource(internalOnlyBytes);
const partialPreview = previewTfliteBenchmarkProfileMapping(internalOnly, runtimeEvidence, analysis);
expectEqual(partialPreview.mapped_execution_node_count, 2, "CPU execution nodes should still map when XNNPACK emits only delegate-profiled rows.");
expectEqual(partialPreview.graph_total_available, false, "Missing delegate-node timing must withhold the graph total.");
expectEqual(partialPreview.delegate_internal_profiled_subtotal_us, 60, "XNNPACK internal timing should remain a separately labeled subtotal.");
expectEqual(partialPreview.primary_delegate_profiled_subtotal_us, 60, "Primary delegate-profiled events should have their own subtotal.");
expectEqual(partialPreview.delegate_internal_section_subtotal_us, null, "Primary delegate-profiled events must not appear as nested delegate-section timing.");
const variableInternal = previewTfliteBenchmarkProfileMapping(
  parseTfliteBenchmarkProfileSource(benchmarkProfile({ includeInternal: true, internalEventCount: 3, internalEventDuration: 40 })),
  runtimeEvidence,
  analysis,
);
expectEqual(variableInternal.delegate_internal_profiled_subtotal_us, 60, "Delegate-internal per-run timing must use the common primary run count, not invert integer-truncated times_called.");
const conditionalProfile = parseTfliteBenchmarkProfileSource(benchmarkProfile({ zeroTimesCalled: true }));
const conditionalPreview = previewTfliteBenchmarkProfileMapping(conditionalProfile, runtimeEvidence, analysis);
expectEqual(conditionalPreview.mapped_execution_node_count, 3, "A valid row with non-derivable run count should remain mapped.");
expectEqual(conditionalPreview.graph_total_available, false, "A zero integer times_called value should withhold per-run graph totals.");
const conditionalDocument = buildTfliteProfiledAssignmentDocument(conditionalProfile, runtimeEvidence, analysis, {
  profileSha256: sha(benchmarkProfile({ zeroTimesCalled: true })), captureId, collectedAt: "2026-07-16T01:00:03.000Z",
});
const conditionalEvidence = parseRuntimeAssignmentDocument(JSON.stringify(conditionalDocument), analysis);
expectEqual(conditionalEvidence.assignments[2].duration_us, null, "A non-derivable run count must not produce an original-op per-run duration.");
expectEqual(conditionalEvidence.comparison.duration_comparison.total_duration_us, null, "A non-derivable run count must survive normalization with the graph total withheld.");

expectThrows(() => buildTfliteProfiledAssignmentDocument(completeProfile, runtimeEvidence, analysis, {
  profileSha256: sha(completeBytes), captureId: "different-capture", collectedAt: "2026-07-16T01:00:02.000Z",
}), "capture ID does not match", "Timing attachment should reject a different declared capture ID.");
expectThrows(() => previewTfliteBenchmarkProfileMapping(parseTfliteBenchmarkProfileSource(benchmarkProfile({ partitionNodeType: "CONV_2D" })), runtimeEvidence, analysis), "does not match runtime node 4", "Execution-node timing should reject a mismatched node type.");
expectThrows(() => parseTfliteBenchmarkProfileSource(benchmarkProfile({ inconsistentAverage: true })), "arithmetically inconsistent", "Profiling parser should reject inconsistent aggregate statistics.");
expectThrows(() => parseTfliteBenchmarkProfileSource(message([stringField(1, "fixture")])), "no runtime_profile", "Profiling parser should reject a proto without runtime timing.");

done("TFLite profiling adapter contract passed (strict stats, capture binding, execution-node coverage, partition totals, and delegate-internal isolation). ");

function modelRuntimeDetails() {
  const nodes = [
    runtimeNode({ id: 0, name: "CONV_2D", type: "3", inputs: [0, 1], outputs: [2], delegatedTo: 4 }),
    runtimeNode({ id: 1, name: "RELU", type: "19", inputs: [2], outputs: [3], delegatedTo: 4 }),
    runtimeNode({ id: 2, name: "SQUEEZE", type: "43", inputs: [3], outputs: [4] }),
    runtimeNode({ id: 3, name: "SOFTMAX", type: "25", inputs: [4], outputs: [5] }),
    runtimeNode({ id: 4, name: "TfLiteXNNPackDelegate", type: "Delegate/TfLiteXNNPackDelegate", inputs: [0, 1], outputs: [3], delegateName: "TfLiteXNNPackDelegate", replacedIds: [0, 1] }),
  ];
  const subgraph = message([intField(1, 0), ...nodes.map((value) => bytesField(3, value)), packedIntField(4, [4, 2, 3]), intField(5, 1), stringField(6, "main")]);
  return message([stringField(1, "fixture"), bytesField(2, subgraph)]);
}

function runtimeNode({ id, name, type, inputs, outputs, delegatedTo = null, delegateName = null, replacedIds = [] }) {
  const fields = [intField(1, id), stringField(2, name), stringField(3, type), packedIntField(4, inputs), packedIntField(5, outputs)];
  if (delegateName != null) fields.push(bytesField(8, message([stringField(1, delegateName), packedIntField(2, replacedIds)])));
  else if (delegatedTo != null) fields.push(intField(9, delegatedTo));
  return message(fields);
}

function benchmarkProfile({ includePartition = true, includeInternal = false, internalInPrimary = false, partitionNodeType = "TfLiteXNNPackDelegate", inconsistentAverage = false, zeroTimesCalled = false, internalEventCount = 2, internalEventDuration = 60 } = {}) {
  const profiles = [
    opProfile("SQUEEZE", "[squeezed]:2", 1, 10, inconsistentAverage ? 9 : 10, zeroTimesCalled ? 0 : 1),
    opProfile("SOFTMAX", "[prob]:3", 2, 15),
  ];
  if (includePartition) profiles.unshift(opProfile(partitionNodeType, "[relu]:4", 0, 100));
  if (includeInternal && internalInPrimary) profiles.unshift(opProfile("xnn_f32_gemm", "Delegate/xnn_f32_gemm:0", 0, internalEventDuration, internalEventDuration, 1, internalEventCount));
  const subgraph = message([stringField(1, "main"), intField(2, 0), ...profiles.map((value) => bytesField(3, value))]);
  const modelFields = [bytesField(1, subgraph)];
  if (includeInternal && !internalInPrimary) {
    const delegate = message([stringField(1, "TfLiteXNNPackDelegate"), bytesField(2, opProfile("xnn_f32_gemm", "Delegate/xnn_f32_gemm:0", 0, internalEventDuration, internalEventDuration, 1, internalEventCount))]);
    modelFields.push(bytesField(2, delegate));
  }
  return message([stringField(1, "fixture"), bytesField(3, message(modelFields))]);
}

function opProfile(nodeType, name, runOrder, duration, average = duration, timesCalled = 1, eventCount = 2) {
  const stat = message([
    intField(1, duration), intField(2, duration), intField(3, average), floatField(4, 0), floatField(5, 0),
    intField(6, duration), intField(7, duration), intField(8, duration * eventCount), intField(9, eventCount),
  ]);
  return message([stringField(1, nodeType), bytesField(2, stat), intField(4, timesCalled), stringField(5, name), intField(6, runOrder)]);
}

function sha(bytes) { return createHash("sha256").update(bytes).digest("hex"); }
function intField(field, value) { return concat(varint((field << 3) | 0), varint(value)); }
function bytesField(field, value) { return concat(varint((field << 3) | 2), varint(value.length), value); }
function stringField(field, value) { return bytesField(field, new TextEncoder().encode(value)); }
function packedIntField(field, values) { return bytesField(field, concat(...values.map(varint))); }
function floatField(field, value) { const bytes = new Uint8Array(4); new DataView(bytes.buffer).setFloat32(0, value, true); return concat(varint((field << 3) | 5), bytes); }
function message(fields) { return concat(...fields); }
function varint(value) { let remaining = BigInt(value); if (remaining < 0) remaining = BigInt.asUintN(64, remaining); const bytes = []; do { let byte = Number(remaining & 0x7fn); remaining >>= 7n; if (remaining) byte |= 0x80; bytes.push(byte); } while (remaining); return Uint8Array.from(bytes); }
function concat(...arrays) { const output = new Uint8Array(arrays.reduce((total, item) => total + item.length, 0)); let offset = 0; for (const item of arrays) { output.set(item, offset); offset += item.length; } return output; }

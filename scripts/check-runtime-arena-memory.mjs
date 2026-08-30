import { readFileSync } from "node:fs";
import { initSync, analyze_tflite_for_target } from "../pkg/tflite_wasm_audit.js";
import { parseRuntimeAssignmentDocument } from "../web/lib/kernel-inspector.js";
import {
  buildEngineeringEvidenceDocument,
  buildEngineeringReport,
  buildMlBomDocument,
  buildRawDataArtifactFiles,
  buildRuntimeArenaReconciliationCsv,
} from "../web/lib/report.js";
import { sha256TextHex } from "../web/lib/sha256-sync.js";
import { createCheck } from "./check-assert.mjs";
import { assertCompactMlBomProjection } from "./compact-mlbom-assert.mjs";

const { done, expect, expectEqual, expectThrows } = createCheck("Runtime arena memory contract check");
const modelBytes = new Uint8Array(readFileSync("web/samples/mobilenet_v2_1.0_224_quant.tflite"));
initSync({ module: readFileSync("pkg/tflite_wasm_audit_bg.wasm") });
const analysis = analyze_tflite_for_target(modelBytes, "mobilenet_v2_1.0_224_quant.tflite", "android_mid_a55");
analysis.model_sha256 = "a".repeat(64);
const plan = analysis.tensor_arena_plan;
expectEqual(plan?.status, "assessed", "The sample must expose a complete declared-shape ArenaPlanner projection.");

const owningAllocations = (plan.allocations || []).filter((item) => item.allocation_status === "allocated").map((item) => ({
  tensor_index: item.tensor_index,
  arena: item.arena,
  offset_bytes: item.offset_bytes,
  size_bytes: item.size_bytes,
  first_node: item.first_node,
  last_node: item.last_node ?? null,
}));
const runtimeTemporary = {
  tensor_index: analysis.tensors.length,
  arena: "kTfLiteArenaRwPersistent",
  offset_bytes: plan.persistent_arena_bytes,
  size_bytes: 64,
  first_node: 0,
  last_node: 0,
};
const allocations = [...owningAllocations, runtimeTemporary].sort((left, right) => left.tensor_index - right.tensor_index);
const aliases = (plan.aliases || []).map((item) => ({
  tensor_index: item.tensor_index,
  shared_with_tensor_index: item.shared_with_tensor_index,
})).sort((left, right) => left.tensor_index - right.tensor_index);
const snapshots = [{
  memory_snapshot_id: 0,
  non_persistent_arena_bytes: plan.non_persistent_arena_bytes,
  persistent_arena_bytes: plan.persistent_arena_bytes + runtimeTemporary.size_bytes,
  combined_arena_bytes: plan.combined_arena_bytes + runtimeTemporary.size_bytes,
  tensor_count: analysis.tensors.length + 1,
  execution_node_count: analysis.ops.length,
  allocation_count: allocations.length,
  alias_count: aliases.length,
  allocated_interval_bytes: allocations.reduce((sum, item) => sum + item.size_bytes, 0),
  allocations,
  aliases,
}];
const canonicalSnapshots = snapshots.map((snapshot) => ({
  memory_snapshot_id: snapshot.memory_snapshot_id,
  non_persistent_arena_bytes: snapshot.non_persistent_arena_bytes,
  persistent_arena_bytes: snapshot.persistent_arena_bytes,
  combined_arena_bytes: snapshot.combined_arena_bytes,
  tensor_count: snapshot.tensor_count,
  execution_node_count: snapshot.execution_node_count,
  allocation_count: snapshot.allocation_count,
  alias_count: snapshot.alias_count,
  allocated_interval_bytes: snapshot.allocated_interval_bytes,
  allocations: snapshot.allocations.map((item) => ({
    tensor_index: item.tensor_index,
    arena: item.arena,
    offset_bytes: item.offset_bytes,
    size_bytes: item.size_bytes,
    first_node: item.first_node,
    last_node: item.last_node,
  })),
  aliases: snapshot.aliases.map((item) => ({
    tensor_index: item.tensor_index,
    shared_with_tensor_index: item.shared_with_tensor_index,
  })),
}));
const runtimeMemory = {
  schema: "deepbom.runtime_memory.v1",
  status: "assessed",
  evidence_class: "OBSERVED_RUNTIME",
  tensorflow_source_commit: "87bbf65b8d23d3f06912b1b2183587e1884bc45c",
  snapshot_count: 1,
  peak_non_persistent_arena_bytes: snapshots[0].non_persistent_arena_bytes,
  peak_persistent_arena_bytes: snapshots[0].persistent_arena_bytes,
  peak_combined_arena_bytes: snapshots[0].combined_arena_bytes,
  final_non_persistent_arena_bytes: snapshots[0].non_persistent_arena_bytes,
  final_persistent_arena_bytes: snapshots[0].persistent_arena_bytes,
  final_combined_arena_bytes: snapshots[0].combined_arena_bytes,
  allocation_ledger_sha256: sha256TextHex(JSON.stringify(canonicalSnapshots)),
  snapshots,
  method: "Instrumented post-commit ArenaPlanner fixture.",
  interpretation_boundary: "TFLite arenas only; delegate buffers, scratch, and process RSS are excluded.",
};
const xnnpackCommit = "23a67314f7afdbb76191589ae090d82bf55afbfa";
const buildIdentifierSha = "b".repeat(64);
const capture = {
  schema: "deepbom.runtime_assignment.v1.10",
  artifact_sha256: analysis.model_sha256,
  target_profile_id: analysis.target_profile.id,
  target_profile_sha256: analysis.target_profile.profile_sha256,
  runtime: { name: "LiteRT", version: "2.20.0", backend: "XNNPACK", build: "instrumented release", binary_sha256: "c".repeat(64) },
  source: {
    kind: "deepbom_native_runtime_capture",
    collected_at: "2026-07-23T00:00:00.000Z",
    capture_id: "runtime-arena-fixture",
    assignment_semantics: "original_graph_op_assignment",
    partition_semantics: "partition_id_identifies_runtime_partition_when_present",
    duration_semantics: "not_collected",
    dispatch_sample_semantics: "unique_context_function_selection_per_process",
    collector: {
      schema: "deepbom.native_runtime_collector.v1.1",
      name: "deepbom-runtime-collector",
      version: "0.1.0",
      source_commit: `deepbom@${"d".repeat(40)}`,
      binary_sha256: "e".repeat(64),
      attestation_status: "not_attested",
      instrumentation: { lowering_ids: false, microkernel_ids: false, arena_allocations: true },
    },
  },
  selector_context: {
    schema: "deepbom.runtime_selector_context.v1.1",
    backend_library: "XNNPACK",
    device: { architecture: "aarch64", identity: "fixture-host", cpu_feature_source: "native_os_api", cpu_features: ["asimd", "fp", "neon"] },
    build: {
      runtime_binary_sha256: "c".repeat(64),
      xnnpack_source_commit: xnnpackCommit,
      microkernel_build_identifier_sha256: buildIdentifierSha,
      build_manifest_sha256: "f".repeat(64),
      compile_definitions: [
        { name: "XNN_BUILD_ALL_MICROKERNELS", value: "OFF" },
        { name: "XNN_ENABLE_ASSEMBLY", value: "ON" },
      ],
    },
    invocation: { inputs: analysis.inputs.map((input) => ({ tensor_index: input.index, name: input.name, shape: input.shape })), thread_count: 1, runtime_options_sha256: "1".repeat(64) },
  },
  runtime_memory: runtimeMemory,
  assignments: analysis.ops.map((op) => ({
    op_index: op.index,
    op_name: op.name,
    provider: op.xnnpack_chain_id >= 0 ? "XNNPACK" : "TFLite non-delegated kernel",
    delegated: op.xnnpack_chain_id >= 0,
    partition_id: op.xnnpack_chain_id >= 0 ? `xnn-${op.xnnpack_chain_id}` : null,
    mapping_method: "native_runtime_original_op_instrumentation",
    lowering_id: null,
    kernel_id: null,
    kernel: null,
    kernel_source_ref: null,
    kernel_build_identifier_sha256: null,
    duration_us: null,
    duration_sum_us: null,
    sample_count: null,
    lowerings: [],
    dispatches: [],
  })),
};

const assignment = parseRuntimeAssignmentDocument(JSON.stringify(capture), analysis, { fileSha256: "2".repeat(64) });
expectEqual(assignment.arena_reconciliation.peak_delta_bytes, 64, "Observed runtime arena peak must retain the exact added persistent temporary bytes.");
expectEqual(assignment.arena_reconciliation.runtime_temporary_allocation_count, 1, "The runtime-only tensor must be classified as a Prepare-time temporary.");
expectEqual(assignment.arena_reconciliation.runtime_temporary_interval_bytes, 64, "Runtime temporary interval bytes must conserve the observed allocation row.");
expectEqual(assignment.arena_reconciliation.missing_observed_allocation_count, 0, "Every projected owning allocation must remain observed in the fixture.");

const identity = {
  filename: analysis.filename,
  format: "tflite",
  sha256: analysis.model_sha256,
  target_id: analysis.target_profile.id,
  target_label: analysis.target_profile.label,
  target_profile_sha256: analysis.target_profile.profile_sha256,
  operator_count: analysis.operator_count,
  tensor_count: analysis.tensor_count,
  total_macs: analysis.total_macs,
};
const runtimeEvidence = { runtimeAssignmentEvidence: assignment };
const mlBomDocument = buildMlBomDocument(analysis, {
  hash: analysis.model_sha256,
  fileSizeBytes: modelBytes.byteLength,
  target: analysis.target_profile,
  targetId: analysis.target_profile.id,
  runtimeAssignmentEvidence: assignment,
});
const reportContext = { identity, runtimeEvidence };
const evidence = buildEngineeringEvidenceDocument(analysis, {
  reportContext,
  rawEvidenceContext: { identity, runtimeEvidence },
  mlBomDocument,
});
expectEqual(evidence.evidence.conformance_report.status, "pass", "Runtime arena report, finding, ML-BOM, and reconciliation must pass independent conformance.");
expect(evidence.evidence.findings_register.findings.some((item) => item.finding_id === "EA-MEM-0002" && item.technical_priority === "Medium"), "A positive observed peak delta must enter the authoritative finding queue without being called a defect.");
expectEqual(evidence.evidence.metric_coverage_manifest.entries.find((item) => item.metric_id === "runtime.arena_memory")?.status, "assessed", "Runtime arena memory must be a first-class assessed metric family.");
expectEqual(evidence.evidence.metric_coverage_manifest.decision_coverage.rows.find((item) => item.domain_id === "runtime_observation")?.status, "partial", "Arena observation alone must make runtime observation partial rather than imply complete execution validation.");
assertCompactMlBomProjection(mlBomDocument, {
  expect,
  expectEqual,
  omittedProperties: ["deepbom:runtime:arenaPeakDeltaBytes"],
  label: "Runtime-arena compact ML-BOM",
});
const report = buildEngineeringReport(analysis, reportContext);
expect(report.includes("### Static Projection Vs Observed TFLite Arena") && report.includes(runtimeMemory.allocation_ledger_sha256), "Engineering Report must render the runtime arena comparison and ledger identity.");
expect(report.includes("Observed TFLite arena allocation differs from the declared-shape projection"), "Engineering Report action queue must retain the runtime arena difference finding.");
const csv = buildRuntimeArenaReconciliationCsv(assignment);
expect(csv.includes("row_type,tensor_index") && csv.includes(`allocation,${runtimeTemporary.tensor_index},runtime_temporary_${runtimeTemporary.tensor_index}`), "Runtime arena CSV must retain the complete runtime-only allocation row.");
const rawFiles = buildRawDataArtifactFiles(analysis, { rawEvidenceContext: { identity, runtimeEvidence }, mlBomDocument });
expect(rawFiles.some((file) => file.name === "runtime/arena_reconciliation.csv"), "Raw Data must add one consolidated runtime arena reconciliation CSV.");

const tamperedAssignment = structuredClone(assignment);
tamperedAssignment.arena_reconciliation.runtime_temporary_interval_bytes += 1;
expectThrows(() => buildEngineeringEvidenceDocument(analysis, {
  reportContext: { identity, runtimeEvidence: { runtimeAssignmentEvidence: tamperedAssignment } },
  rawEvidenceContext: { identity, runtimeEvidence: { runtimeAssignmentEvidence: tamperedAssignment } },
  mlBomDocument,
}), "CF-RUNTIME-018", "A self-consistent-looking reconciliation aggregate tamper must fail independent conformance.");

done("Runtime arena memory contract passed (collector ledger, reconciliation, report, ML-BOM, CSV, and tamper rejection).");

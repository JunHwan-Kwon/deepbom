import { readFileSync } from "node:fs";

import {
  initSync,
  analyze_tflite_for_target,
  compute_delegation_repair,
} from "../pkg/tflite_wasm_audit.js";
import { validateDelegationRepair } from "../web/lib/delegation-repair.js";
import { buildEngineeringReport } from "../web/lib/report-engineering.js";
import { buildConformanceReport } from "../web/lib/report-conformance.js";
import { buildStaticAnalysisExport } from "../web/lib/report-evidence.js";
import { buildFindingsRegister } from "../web/lib/report-findings.js";
import { ANALYZER_METADATA } from "../web/lib/report-metadata.js";
import { buildPublicShareAnalysis } from "../web/lib/public-export.js";
import { createCheck } from "./check-assert.mjs";

const { done, expect, expectEqual, expectThrows } = createCheck("Delegation repair check");
const targetId = "android_mid_a55";
const filename = "mobilenet_v2_1.0_224_quant.tflite";

initSync({ module: readFileSync("pkg/tflite_wasm_audit_bg.wasm") });
const bytes = new Uint8Array(readFileSync(`web/samples/${filename}`));
const analysis = analyze_tflite_for_target(bytes, filename, targetId);
const repair = compute_delegation_repair(bytes, filename, targetId);
analysis.model_sha256 = repair.artifact_sha256;
analysis.delegation_repair = repair;

expect(validateDelegationRepair(repair, analysis), "Delegation repair should satisfy the independent browser reconstruction.");
expectEqual(repair.schema, "deepbom.delegation_repair.v1.3", "Delegation-repair schema should be stable.");
expectEqual(repair.operator_count, 65, "Actual quantized MobileNetV2 should expose 65 toggle rows.");
expectEqual(repair.graph_edge_count, 74, "Actual quantized MobileNetV2 graph-edge count should remain fixed.");
expectEqual(repair.baseline.delegate_segment_count, 2, "Baseline should contain two predicted delegate segments.");
expectEqual(repair.baseline.cpu_segment_count, 1, "Baseline should contain one CPU segment.");
expectEqual(repair.baseline.boundary_edge_count, 2, "Baseline should contain two producer-consumer partition edges.");
expectEqual(repair.baseline.summed_edge_payload_bytes, 64_000, "Baseline logical partition-edge payload should be exactly 64,000 B.");
expectEqual(repair.repair_opportunity_count, 1, "Exactly one single-op support extension should reduce fragmentation.");
expectEqual(repair.fragmentation_risk_count, 62, "Every currently delegated op should expose its support-loss fragmentation effect.");
expectEqual(repair.toggles.filter((row) => !row.repair_opportunity && !row.fragmentation_risk).length, 2, "Two non-ranked support-loss payload effects should remain explicit.");
expectEqual(repair.cpu_island_count, 1, "Actual quantized MobileNetV2 should contain one contiguous predicted CPU island.");
expectEqual(repair.full_segment_repair_count, 1, "The complete CPU-island intervention should remove fragmentation.");
expectEqual(repair.group_only_repair_count, 0, "The one-op CPU island should not be mislabeled as a group-only repair.");

const topRepair = repair.toggles.find((row) => row.op_index === repair.repair_ranking_op_indices[0]);
expectEqual(topRepair.op_index, 62, "AVERAGE_POOL_2D should be the highest-ranked repair coordinate.");
expectEqual(topRepair.op_name, "AVERAGE_POOL_2D", "Repair coordinate identity should remain bound to the parsed op.");
expectEqual(topRepair.outcome_class, "bridge_merges_delegate_segments", "The support-extension toggle should merge both neighboring delegate segments.");
expectEqual(topRepair.signed_delegate_segment_count, -1, "Repair should remove one delegate segment split.");
expectEqual(topRepair.signed_boundary_edge_count, -2, "Repair should remove exactly two graph boundary edges.");
expectEqual(topRepair.boundary_payload_reduction_bytes, 64_000, "Repair should remove exactly 64,000 B of logical boundary payload.");
expectEqual(topRepair.edge_changes.length, 2, "Repair should expose both removed boundary edges.");
expectEqual(topRepair.edge_changes.map((edge) => edge.payload_bytes).sort((left, right) => right - left).join(","), "62720,1280", "Repair edge ledger should preserve exact tensor payloads.");
expect(topRepair.edge_changes.every((edge) => edge.transition === "removed"), "Repair edge ledger should classify both changes as removed.");

const topFragility = repair.toggles.find((row) => row.op_index === repair.fragility_ranking_op_indices[0]);
expectEqual(topFragility.op_index, 4, "The deterministic fragility ranking should start at op #4.");
expectEqual(topFragility.op_name, "DEPTHWISE_CONV_2D", "Top fragmentation identity should remain stable.");
expectEqual(topFragility.signed_delegate_segment_count, 1, "Top support-loss row should split one delegate segment.");
expectEqual(topFragility.signed_boundary_edge_count, 2, "Top support-loss row should add two boundary edges.");
expectEqual(topFragility.signed_boundary_payload_bytes, 1_505_280, "Top support-loss row should expose the exact logical payload increase.");

const island = repair.cpu_islands[0];
expectEqual(island.island_index, 1, "CPU-island identity should be stable.");
expectEqual(island.op_indices.join(","), "62", "The actual CPU island should contain op #62 only.");
expectEqual(island.baseline_incident_boundary_edge_count, 2, "CPU island should bind both incident partition edges.");
expectEqual(island.baseline_incident_boundary_payload_bytes, 64_000, "CPU island should bind exact incident logical payload.");
expectEqual(island.boundary_edge_reduction_count, 2, "Complete island support should remove both boundaries.");
expectEqual(island.boundary_payload_reduction_bytes, 64_000, "Complete island support should remove the exact logical payload.");
expectEqual(island.best_single_op_index, 62, "Single-member island should bind its member as the best single coordinate.");
expectEqual(island.additional_edge_reduction_over_best_single, 0, "Single-member island should have no group-only edge gain.");
expectEqual(island.additional_payload_reduction_over_best_single, 0, "Single-member island should have no group-only payload gain.");
expectEqual(island.outcome_class, "eliminates_cpu_island_and_merges_delegate_segments", "Full-island outcome should preserve the exact segment merge.");

const repeated = compute_delegation_repair(bytes, filename, targetId);
expectEqual(JSON.stringify(repeated), JSON.stringify(repair), "Delegation repair should be byte-for-byte deterministic as JSON.");
const report = buildEngineeringReport(analysis, {
  identity: {
    filename,
    format: "tflite",
    sha256: analysis.model_sha256,
    target_label: analysis.target_profile.label,
    operator_count: analysis.operator_count,
    tensor_count: analysis.tensor_count,
    total_macs: analysis.total_macs,
  },
});
expect(report.includes("## Delegation Repair Lab (PREDICTED/DERIVED COUNTERFACTUAL)"), "Engineering report should render the evidence-qualified repair lab.");
expect(report.includes("#062 AVERAGE_POOL_2D") && report.includes("bridge_merges_delegate_segments"), "Engineering report should render the highest-ranked repair identity and outcome.");
expect(report.includes("61.3 KiB"), "Engineering report should render the larger removed edge payload.");
expect(report.includes("### Predicted CPU-Island Assignment Portfolio")
  && report.includes("island 1: #062-#062")
  && report.includes("eliminates_cpu_island_and_merges_delegate_segments"), "Engineering report should render the complete CPU-island portfolio and exact outcome.");
expect(report.includes("65 = 1 repair + 62 fragility + 2 unranked other effect(s)")
  && report.includes("### Unranked Toggle Outcomes")
  && report.includes("#061 CONV_2D")
  && report.includes("#063 CONV_2D"), "Engineering report should conserve and expose every toggle class instead of dropping two non-ranked effects.");
const runtimeBuildFinding = buildFindingsRegister(analysis).find((finding) => finding.finding_id === "EA-DEL-0004");
expectEqual(runtimeBuildFinding?.technical_priority, "High", "A missing QU8 build flag that collapses predicted coverage must enter the High action queue.");
expect(runtimeBuildFinding?.observation.includes("from 64 to 0 op(s)")
  && runtimeBuildFinding?.observation.includes("100% of modeled MACs")
  && runtimeBuildFinding?.observation.includes("tflite_with_xnnpack_qu8"), "The QU8 runtime-build finding should preserve the exact conditional coverage collapse and required flag.");
const conformance = buildConformanceReport({
  analysis,
  staticAnalysis: buildStaticAnalysisExport(analysis),
  quantization: { schema: ANALYZER_METADATA.schemas.quantizationEvidence },
  findingsRegister: { authoritative_action_source: "findings", raw_analyzer_signals: [], findings: [] },
  runtimeResults: {},
  securityPosture: { execution_integrity: {} },
  mlBomDocument: {},
  engineeringReport: report,
});
const repairConformance = conformance.checks.filter((check) => check.id.startsWith("CF-REPAIR-"));
expectEqual(repairConformance.length, 5, "Conformance should expose five delegation-repair cross-output checks.");
expect(repairConformance.every((check) => check.status === "pass"), "All delegation-repair conformance checks should pass for the actual sample.");
const publicAnalysis = buildPublicShareAnalysis(analysis);
expectEqual(publicAnalysis.delegation_repair.artifact_sha256, "", "Public repair export should redact artifact SHA-256.");
expectEqual(publicAnalysis.delegation_repair.target_id, "PUBLIC-TARGET", "Public repair export should redact planning-target identity.");

const tamperedBaseline = structuredClone(repair);
tamperedBaseline.baseline.boundary_edge_count += 1;
expectThrows(() => validateDelegationRepair(tamperedBaseline, analysis), "baseline", "Validation should reject a tampered baseline summary.");
const tamperedEdge = structuredClone(repair);
tamperedEdge.toggles[62].edge_changes[0].tensor_index += 1;
expectThrows(() => validateDelegationRepair(tamperedEdge, analysis), "edge", "Validation should reject a tampered edge identity.");
const tamperedRanking = structuredClone(repair);
tamperedRanking.fragility_ranking_op_indices.reverse();
expectThrows(() => validateDelegationRepair(tamperedRanking, analysis), "ranking", "Validation should reject a tampered fragility ranking.");
const tamperedIsland = structuredClone(repair);
tamperedIsland.cpu_islands[0].additional_edge_reduction_over_best_single += 1;
expectThrows(() => validateDelegationRepair(tamperedIsland, analysis), "additional edge", "Validation should reject tampered CPU-island synergy arithmetic.");
const tamperedIslandEdge = structuredClone(repair);
tamperedIslandEdge.cpu_islands[0].edge_changes[0].tensor_index += 1;
const tamperedAnalysis = structuredClone(analysis);
tamperedAnalysis.delegation_repair = tamperedIslandEdge;
const tamperedConformance = buildConformanceReport({
  analysis: tamperedAnalysis,
  staticAnalysis: buildStaticAnalysisExport(tamperedAnalysis),
  quantization: { schema: ANALYZER_METADATA.schemas.quantizationEvidence },
  findingsRegister: { authoritative_action_source: "findings", raw_analyzer_signals: [], findings: [] },
  runtimeResults: {},
  securityPosture: { execution_integrity: {} },
  mlBomDocument: {},
  engineeringReport: report,
});
expectEqual(tamperedConformance.checks.find((check) => check.id === "CF-REPAIR-005")?.status, "fail", "Independent conformance should reject a tampered CPU-island changed-edge ledger.");

done(`65 single-op toggles plus 1 complete CPU-island intervention; #62 removes 2 boundaries/64,000 B; #4 adds 2 boundaries/1,505,280 B; deterministic JSON and tamper rejection passed.`);

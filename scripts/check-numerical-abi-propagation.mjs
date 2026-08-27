import { readFileSync } from "node:fs";

import { analyze_tflite_for_target, initSync } from "../pkg/tflite_wasm_audit.js";
import {
  reconstructNumericalAbiPropagation,
  validateNumericalAbiPropagationAgainstReconstruction,
  validateNumericalAbiPropagationAnalysis,
  validateNumericalAbiPropagationDigestsAgainstReconstruction,
} from "../web/lib/numerical-abi-propagation.js";
import { buildConformanceReport } from "../web/lib/report-conformance.js";
import { buildEngineeringReport } from "../web/lib/report-engineering.js";
import { buildQuantizationEvidence, buildStaticAnalysisExport } from "../web/lib/report-evidence.js";
import { buildFindingsRegister } from "../web/lib/report-findings.js";
import { buildMlBomDocument } from "../web/lib/report-mlbom.js";
import { assertCompactMlBomProjection } from "./compact-mlbom-assert.mjs";
import { ANALYZER_METADATA } from "../web/lib/report-metadata.js";
import { buildQuantizationContractChecks } from "../web/lib/report-quantization-contracts.js";
import { visualPngSpecs } from "../web/lib/visual-export.js";
import { createCheck } from "./check-assert.mjs";

const { done, expect, expectEqual, expectThrows } = createCheck("Numerical ABI propagation check");
const filename = "mobilenet_v2_1.0_224_quant.tflite";
initSync({ module: readFileSync("pkg/tflite_wasm_audit_bg.wasm") });
const bytes = new Uint8Array(readFileSync(`web/samples/${filename}`));
const analysis = analyze_tflite_for_target(bytes, filename, "android_mid_a55");
const propagation = analysis.numerical_abi_propagation;
const reconstructed = await validateNumericalAbiPropagationAnalysis(analysis);

expectEqual(propagation.schema, "deepbom.numerical_abi_propagation.v1.1", "Propagation schema should retain reachability-qualified evidence.");
expectEqual(propagation.method_version, "2026-07-28.1", "Propagation method should retain the serialized-concrete-shape payload rule.");
expectEqual(propagation.status, "assessed", "The complete sample graph should be assessed.");
expectEqual(propagation.source_evidence_schema, "deepbom.rounding_equivalence.v1 + deepbom.accumulator_reachability.v1", "Propagation should bind both independent numerical evidence schemas.");
expectEqual(`${propagation.candidate_source_op_count}:${propagation.divergent_source_op_count}:${propagation.equivalent_source_op_count}:${propagation.unassessed_source_op_count}`, "53:52:1:0", "Source classification should remain exact.");
expectEqual(`${propagation.output_reachable_source_op_count}:${propagation.output_isolated_source_op_count}`, "52:0", "Every divergent source should structurally reach the declared output.");
expectEqual(`${propagation.exact_local_counterexample_source_op_count}:${propagation.residue_excluded_divergence_source_op_count}:${propagation.unresolved_divergence_source_op_count}:${propagation.local_reachability_unassessed_source_op_count}`, "52:9:32:0", "Reachability-qualified source facets should remain exact.");
expectEqual(propagation.exact_output_reachable_source_op_count, 52, "Every exact-local source should retain a structural path to the declared output.");
expectEqual(`${propagation.interval_divergent_state_count_decimal}:${propagation.exact_local_divergent_state_count_decimal}:${propagation.residue_excluded_divergent_state_count_decimal}:${propagation.unresolved_divergent_state_count_decimal}`, "2874544:2239435:3585:631524", "Global reachability-state partition should remain exact.");
expectEqual(
  BigInt(propagation.interval_divergent_state_count_decimal),
  BigInt(propagation.exact_local_divergent_state_count_decimal)
    + BigInt(propagation.residue_excluded_divergent_state_count_decimal)
    + BigInt(propagation.unresolved_divergent_state_count_decimal),
  "Global interval divergence should conserve exact, residue-excluded, and unresolved states.",
);
expectEqual(propagation.graph_edge_count, 74, "Producer/tensor/consumer graph edge count should remain exact.");
expectEqual(propagation.graph_ledger_sha256, "32f61ca0603b0b1da6749600ecde950293feffc2c74b56ededbb3f62d07e93aa", "Graph ledger digest should remain stable.");
expectEqual(`${propagation.source_corridor_edge_instance_count}:${propagation.source_boundary_edge_instance_count}:${propagation.assessed_source_boundary_edge_instance_payload_bytes_decimal}`, "2021:102:3264000", "Repeated source-corridor and boundary inventory should remain exact.");
expectEqual(`${propagation.unique_reachable_op_count}:${propagation.unique_reachable_tensor_count}:${propagation.unique_model_output_tensor_count}`, "64:65:1", "Union corridor inventory should remain exact.");
expectEqual(`${propagation.unique_predicted_boundary_edge_count}:${propagation.unique_predicted_boundary_logical_payload_bytes}`, "2:64000", "Unique predicted boundary inventory should remain exact.");
expectEqual(propagation.unique_predicted_boundary_edge_indices.join(","), "71,72", "Unique predicted boundary coordinates should remain stable.");
expectEqual(`${propagation.exact_source_corridor_edge_instance_count}:${propagation.exact_source_boundary_edge_instance_count}`, "2021:102", "Exact-qualified source corridors and boundary instances should remain exact.");
expectEqual(`${propagation.exact_unique_reachable_op_count}:${propagation.exact_unique_reachable_tensor_count}:${propagation.exact_unique_predicted_boundary_edge_count}:${propagation.exact_unique_predicted_boundary_logical_payload_bytes}`, "64:65:2:64000", "Exact-qualified union exposure should remain exact.");
expectEqual(propagation.exact_unique_predicted_boundary_edge_indices.join(","), "71,72", "Exact-qualified boundary coordinates should remain stable.");
expectEqual(`${propagation.reconvergence_source_op_instance_count}:${propagation.single_branch_merge_source_op_instance_count}`, "260:29", "Merge classification inventory should remain exact.");
expectEqual(`${propagation.maximum_model_output_op_hops}:${propagation.maximum_model_output_graph_route_count_decimal}:${propagation.graph_cycle_status}`, "34:1024:acyclic", "Acyclic route and path maxima should remain exact.");
expectEqual(propagation.propagation_ranking_op_indices.slice(0, 12).join(","), "0,1,2,3,4,5,6,7,8,10,11,12", "Structural exposure ranking should remain deterministic.");

const sourceZero = source(0);
expectEqual(`${sourceZero.reachable_op_count}:${sourceZero.corridor_edge_count}:${sourceZero.reconvergence_op_count}:${sourceZero.single_branch_merge_op_count}`, "64:74:10:0", "Root source corridor should cover the exact graph and reconvergence set.");
expectEqual(`${sourceZero.predicted_boundary_edge_count}:${sourceZero.assessed_boundary_logical_payload_bytes}:${sourceZero.minimum_model_output_op_hops}:${sourceZero.exact_model_output_graph_route_count_decimal}`, "2:64000:34:1024", "Root source route and boundary certificate should remain exact.");
expectEqual(`${sourceZero.local_reachability_status}:${sourceZero.exact_reachable_divergent_channel_count}:${sourceZero.unresolved_divergent_channel_count}`, "exact_local_counterexample:18:6", "Root source should preserve its constructive and unresolved channel facets.");
expectEqual(`${sourceZero.exact_reachable_divergent_state_count_decimal}:${sourceZero.provably_unreachable_divergent_state_count_decimal}:${sourceZero.unresolved_divergent_state_count_decimal}`, "2918:0:1062", "Root source reachability-state partition should remain exact.");
expectEqual(
  BigInt(sourceZero.divergent_state_count_decimal),
  BigInt(sourceZero.exact_reachable_divergent_state_count_decimal)
    + BigInt(sourceZero.provably_unreachable_divergent_state_count_decimal)
    + BigInt(sourceZero.unresolved_divergent_state_count_decimal),
  "Root source divergent states should conserve the reachability partition.",
);
expectEqual(sourceZero.source_equivalence_ledger_sha256, "6f46095f91c39665dd9943c46bb6271cbcf03a32a4ad9b8a890c4a0fa3782e0b", "Root source should bind the rounding-equivalence certificate.");
expectEqual(sourceZero.source_reachability_ledger_sha256, "67743dd329cabb5d6019924a46020b1ec59b6b69bb68f0136178184dfd1c4505", "Root source should bind the accumulator-reachability certificate.");
expectEqual(sourceZero.propagation_ledger_sha256, "0318b9ba433a62b8543aa3286130052cccea90769e0fe23d43106c54e2365229", "Root propagation digest should bind both numerical ledgers.");
expectEqual(sourceZero.model_output_paths[0].shortest_path_op_indices.join(","), "0,1,2,3,4,5,9,10,11,12,16,20,21,22,23,27,31,35,36,37,38,42,46,47,48,49,53,57,58,59,60,61,62,63,64", "Deterministic shortest output path should remain stable.");

const sourceSeven = source(7);
expectEqual(`${sourceSeven.reachable_op_count}:${sourceSeven.corridor_edge_count}:${sourceSeven.reconvergence_op_count}:${sourceSeven.single_branch_merge_op_count}`, "57:66:9:1", "Mid-graph source should distinguish reconvergence from a single-branch merge.");
expectEqual(`${sourceSeven.minimum_model_output_op_hops}:${sourceSeven.exact_model_output_graph_route_count_decimal}`, "30:512", "Mid-graph path multiplicity should remain exact.");
expectEqual(sourceSeven.merge_points[0].merge_class, "single_branch_merge", "The first residual ADD after source #7 should not be mislabeled as reconvergence.");
expectEqual(`${sourceSeven.exact_reachable_divergent_state_count_decimal}:${sourceSeven.provably_unreachable_divergent_state_count_decimal}:${sourceSeven.unresolved_divergent_state_count_decimal}`, "23756:0:2865", "Mid-graph source should preserve its exact reachability partition.");
expectEqual(sourceSeven.source_reachability_ledger_sha256, "e5b686df8947dadb90d4eccb5c59c09bca0714b9bcb76b1a5df06b91608e653f", "Mid-graph source should bind its reachability ledger.");
expectEqual(sourceSeven.propagation_ledger_sha256, "0ec4bf0801ccf9bcaa4ffff919c281bf76785e6e52e5bb27d3b98cbd7888e4cf", "Mid-graph propagation digest should remain stable.");

const equivalentSource = source(51);
expectEqual(`${equivalentSource.assessment_status}:${equivalentSource.reachable_op_count}:${equivalentSource.reachable_tensor_count}:${equivalentSource.route_count_status}`, "complete_interval_equivalent:0:1:not_applicable", "Complete-interval equivalence should retain its source tensor without claiming propagation.");
expectEqual(equivalentSource.local_reachability_status, "complete_interval_equivalent", "Equivalent source should remain equivalent after the reachability join.");
expectEqual(equivalentSource.source_reachability_ledger_sha256, "81ce1ccf2bfffab656ed6a1abb79de7108a120b5ca2fd9434d7bd5cd90b0a291", "Equivalent source should still bind its reachability assessment.");
expectEqual(equivalentSource.propagation_ledger_sha256, "", "Equivalent source should not emit a propagation certificate.");
const classifierSource = source(63);
expectEqual(`${classifierSource.reachable_op_count}:${classifierSource.corridor_edge_count}:${classifierSource.exact_model_output_graph_route_count_decimal}:${classifierSource.minimum_model_output_op_hops}`, "1:1:1:1", "Classifier source should expose its one-edge output path exactly.");
expectEqual(`${classifierSource.predicted_boundary_edge_count}:${classifierSource.reconvergence_op_count}:${classifierSource.single_branch_merge_op_count}`, "0:0:0", "Classifier source should not inherit unrelated boundaries or merges.");
expectEqual(`${classifierSource.local_reachability_status}:${classifierSource.exact_reachable_divergent_state_count_decimal}:${classifierSource.provably_unreachable_divergent_state_count_decimal}:${classifierSource.unresolved_divergent_state_count_decimal}`, "exact_local_counterexample:155085:0:0", "Classifier source should be fully exact within the analyzed divergent interval states.");
expectEqual(classifierSource.propagation_ledger_sha256, "1c43fbdab268c2106a52483ab925f3d7f3f28da55fa5d34e92536de65ccacd06", "Classifier propagation digest should remain stable.");

const boundaryEdges = propagation.unique_predicted_boundary_edge_indices.map((index) => propagation.graph_edges[index]);
expectEqual(boundaryEdges.map((edge) => `${edge.edge_index}:${edge.predicted_boundary_direction}:${edge.logical_payload_bytes}`).join(","), "71:delegate_to_cpu:62720,72:cpu_to_delegate:1280", "Predicted boundary direction and logical payload should remain exact.");
expectEqual(boundaryEdges.reduce((total, edge) => total + edge.logical_payload_bytes, 0), 64_000, "Unique boundary payload should conserve the two edge payloads.");
expectEqual(sourceZero.model_output_paths[0].exact_graph_route_count_decimal, "1024", "Per-output route count should conserve the source total.");

const identity = {
  filename,
  format: "tflite",
  sha256: analysis.model_sha256,
  target_label: analysis.target_profile.label,
  operator_count: analysis.operator_count,
  tensor_count: analysis.tensor_count,
  total_macs: analysis.total_macs,
};
const report = buildEngineeringReport(analysis, { identity });
expect(report.includes("## Numerical ABI Propagation Atlas (DERIVED EXACT LOCAL SOURCE + STRUCTURAL CORRIDOR)"), "Engineering report should render the qualified propagation atlas.");
expect(report.includes("2,874,544 = 2,239,435 + 3,585 + 631,524") && report.includes("2,021") && report.includes("1,024") && report.includes("3,264,000 B") && report.includes("62.5 KiB (64,000 B)"), "Engineering report should preserve state conservation, corridor, route, repeated-inventory, and unique-boundary values.");
expect(report.includes(propagation.graph_ledger_sha256) && report.includes(sourceZero.source_reachability_ledger_sha256) && report.includes(sourceZero.propagation_ledger_sha256), "Engineering report should preserve graph, reachability, and propagation digests.");
expect(report.includes("repeated source exposure inventory, not physical runtime traffic") && report.includes("Exact-local qualification proves") && report.includes("Downstream corridors remain tensor-level structural potential"), "Engineering report should preserve the exact-local and structural evidence boundary.");

const staticAnalysis = buildStaticAnalysisExport(analysis);
const quantization = buildQuantizationEvidence(analysis, identity);
expectEqual(staticAnalysis.numerical_abi_propagation.schema, ANALYZER_METADATA.schemas.numericalAbiPropagation, "Static evidence should retain the propagation schema.");
expectEqual(JSON.stringify(quantization.numerical_abi_propagation), JSON.stringify(staticAnalysis.numerical_abi_propagation), "Static and quantization exports should retain one propagation ledger.");
const contract = buildQuantizationContractChecks(analysis).numerical_abi_propagation;
expectEqual(`${contract.status}:${contract.divergent_sources}:${contract.output_reachable_sources}:${contract.maximum_model_output_graph_route_count_decimal}`, "review:52:52:1024", "Quantization contract check should expose the exact propagation portfolio.");
expectEqual(`${contract.exact_local_counterexample_sources}:${contract.residue_excluded_divergence_sources}:${contract.unresolved_divergence_sources}:${contract.exact_output_reachable_sources}`, "52:9:32:52", "Quantization contract should preserve reachability-qualified source facets.");
expectEqual(`${contract.interval_divergent_state_count_decimal}:${contract.exact_local_divergent_state_count_decimal}:${contract.residue_excluded_divergent_state_count_decimal}:${contract.unresolved_divergent_state_count_decimal}`, "2874544:2239435:3585:631524", "Quantization contract should preserve the exact state partition.");
expectEqual(`${contract.exact_source_corridor_edge_instances}:${contract.exact_unique_reachable_ops}:${contract.exact_unique_reachable_tensors}:${contract.exact_unique_predicted_boundary_edges}:${contract.exact_unique_predicted_boundary_logical_payload_bytes}`, "2021:64:65:2:64000", "Quantization contract should preserve exact-qualified exposure unions.");
expectEqual(contract.graph_ledger_sha256, propagation.graph_ledger_sha256, "Contract check should bind the graph digest.");

const findings = buildFindingsRegister(analysis);
const finding = findings.find((item) => item.finding_id === "EA-QNT-0114");
expect(finding, "Output-reachable divergence should enter the authoritative action queue.");
expectEqual(finding?.technical_priority, "Informational", "Structural propagation should remain supporting evidence under the model-input rounding ABI action.");
expect(finding?.title.includes("Exact kernel-local fixed-point counterexample") && finding?.observation.includes("2,239,435 exact reachable") && finding?.observation.includes("3,585 residue-excluded") && finding?.observation.includes("631,524 unresolved"), "Finding should lead with the qualified numerical evidence.");
expect(finding?.observation.includes("1,024") && finding?.observation.includes("62.5 KiB") && finding?.observation.includes(sourceZero.source_reachability_ledger_sha256) && finding?.observation.includes(sourceZero.propagation_ledger_sha256), "Finding should preserve exact routes, unique payload, and both source certificate digests.");
expect(finding?.interpretation.includes("downstream corridor is structural only") && finding?.recommendation.includes("paired full-model output comparisons"), "Finding should preserve the structural evidence boundary and runtime validation action.");

const mlBom = buildMlBomDocument(analysis, { hash: analysis.model_sha256, fileSizeBytes: bytes.byteLength, target: analysis.target_profile });
assertCompactMlBomProjection(mlBom, {
  expect,
  expectEqual,
  omittedProperties: [
    "deepbom:model:numericalAbiPropagationSchema",
    "deepbom:model:numericalAbiPropagationDivergentSources",
    "deepbom:model:numericalAbiPropagationExactLocalSources",
    "deepbom:model:numericalAbiPropagationResidueFacetSources",
    "deepbom:model:numericalAbiPropagationUnresolvedFacetSources",
    "deepbom:model:numericalAbiPropagationExactLocalDivergentStates",
    "deepbom:model:numericalAbiPropagationResidueExcludedDivergentStates",
    "deepbom:model:numericalAbiPropagationUnresolvedDivergentStates",
    "deepbom:model:numericalAbiPropagationCorridorEdgeInstances",
    "deepbom:model:numericalAbiPropagationExactCorridorEdgeInstances",
    "deepbom:model:numericalAbiPropagationExactUniqueBoundaryPayloadBytes",
    "deepbom:model:numericalAbiPropagationMaximumRoutes",
    "deepbom:model:numericalAbiPropagationGraphLedgerSha256",
  ],
  label: "Numerical-ABI compact ML-BOM",
});

const conformance = buildConformanceReport({
  analysis,
  staticAnalysis,
  quantization,
  findingsRegister: { authoritative_action_source: "findings", raw_analyzer_signals: [], findings },
  runtimeResults: {},
  securityPosture: { execution_integrity: {} },
  mlBomDocument: mlBom,
  engineeringReport: report,
});
const abiChecks = conformance.checks.filter((check) => check.id.startsWith("CF-ABI-"));
expectEqual(abiChecks.length, 4, "Conformance should expose four cross-output propagation checks.");
expect(abiChecks.every((check) => check.status === "pass"), `Every propagation conformance check should pass: ${JSON.stringify(abiChecks)}`);
expect(visualPngSpecs({ analysis, filename }).some(([path]) => path === "visuals/numerical_abi_propagation.png"), "Engineering bundle should include the propagation PNG.");

const repeated = analyze_tflite_for_target(bytes, filename, "android_mid_a55").numerical_abi_propagation;
expectEqual(JSON.stringify(repeated), JSON.stringify(propagation), "Propagation JSON should be byte-for-byte deterministic.");
const independent = reconstructNumericalAbiPropagation(analysis);
const batchDynamicAnalysis = structuredClone(analysis);
for (const tensor of batchDynamicAnalysis.tensors || []) {
  if (Array.isArray(tensor.shape) && tensor.shape.length > 0) {
    tensor.shape_signature = [-1, ...tensor.shape.slice(1)];
  }
}
const batchDynamicReconstruction = reconstructNumericalAbiPropagation(batchDynamicAnalysis);
expectEqual(
  batchDynamicReconstruction.graph_edges.map((edge) => edge.logical_payload_bytes).join(","),
  independent.graph_edges.map((edge) => edge.logical_payload_bytes).join(","),
  "A dynamic batch signature must not suppress payloads derived from a fully bound serialized concrete shape.",
);
for (const row of independent.sources) {
  for (const path of row.model_output_paths || []) {
    expectEqual(path.shortest_path_op_indices.length, path.shortest_op_hops + 1, "Shortest-path op inventory should include the source while hop count measures graph edges.");
    expectEqual(path.shortest_path_edge_indices.length, path.shortest_op_hops, "Shortest-path edge inventory should equal the edge-hop count.");
  }
}
const tamperedEdge = structuredClone(propagation);
tamperedEdge.graph_edges[71].logical_payload_bytes += 1;
expectThrows(() => validateNumericalAbiPropagationAgainstReconstruction(tamperedEdge, independent), "differs from independent graph reconstruction", "Independent validator should reject a tampered edge payload.");
const tamperedRoute = structuredClone(propagation);
tamperedRoute.sources.find((row) => row.op_index === 0).exact_model_output_graph_route_count_decimal = "1023";
expectThrows(() => validateNumericalAbiPropagationAgainstReconstruction(tamperedRoute, independent), "differs from independent graph reconstruction", "Independent validator should reject a tampered route count.");
const tamperedExactState = structuredClone(propagation);
tamperedExactState.sources.find((row) => row.op_index === 0).exact_reachable_divergent_state_count_decimal = "2917";
expectThrows(() => validateNumericalAbiPropagationAgainstReconstruction(tamperedExactState, independent), "does not conserve divergent states", "Independent validator should reject a tampered exact-local state count before comparison.");
const tamperedReachabilityLedger = structuredClone(propagation);
tamperedReachabilityLedger.sources.find((row) => row.op_index === 0).source_reachability_ledger_sha256 = "0".repeat(64);
expectThrows(() => validateNumericalAbiPropagationAgainstReconstruction(tamperedReachabilityLedger, independent), "differs from independent graph reconstruction", "Independent validator should reject a tampered reachability certificate.");
const tamperedLedger = structuredClone(propagation);
tamperedLedger.sources.find((row) => row.op_index === 0).propagation_ledger_sha256 = "0".repeat(64);
expectThrows(() => validateNumericalAbiPropagationDigestsAgainstReconstruction(tamperedLedger, reconstructed), "source ledger SHA-256 mismatch", "Digest validator should reject a tampered source certificate.");

done("Numerical ABI propagation passed (2,239,435 exact-local, 3,585 residue-excluded, 631,524 unresolved states; 52 exact output-reachable sources; independent reconstruction, reports, digests, and tamper rejection).");

function source(opIndex) {
  const row = propagation.sources.find((candidate) => candidate.op_index === opIndex);
  expect(Boolean(row), `Propagation source #${opIndex} should exist.`);
  return row;
}

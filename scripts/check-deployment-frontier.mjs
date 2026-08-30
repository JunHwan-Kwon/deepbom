import { readFileSync } from "node:fs";

import {
  initSync,
  analyze_tflite_for_target,
  compute_deployment_frontier,
} from "../pkg/tflite_wasm_audit.js";
import {
  initSync as initDeepBom,
  analyze_deepbom,
} from "../web/protected/deepbom/pkg/deepbom_wasm.js";
import {
  validateDeploymentFrontier,
  validateOrtEpPortabilityFrontier,
} from "../web/lib/deployment-frontier.js";
import { applyProtectedOrtCompatibilityEvidence } from "../web/lib/ort-compatibility-evidence.js";
import {
  deploymentFrontierMatchesTargetIds,
  deploymentFrontierTargetIds,
} from "../web/lib/app-config.js";
import { buildEngineeringReport } from "../web/lib/report-engineering.js";
import { buildModelAtGlance } from "../web/lib/model-glance.js";
import {
  buildGraphIndex,
  summarizeFrontierTargetComparison,
} from "../web/lib/analysis.js";
import { buildInputQuantizationConventionCheck } from "../web/lib/quantization-contract-summary.js";
import { collectFullGraph, layoutFullGraph } from "../web/lib/graph-layout.js";
import { analyzeOnnxModel } from "../web/onnx.js";
import { createCheck } from "./check-assert.mjs";

const { done, expect, expectEqual, expectThrows } = createCheck("Deployment frontier check");
const targets = ["android_mid_a55", "rpi4_a72", "x86_avx2", "wasm_simd"];

initSync({ module: readFileSync("pkg/tflite_wasm_audit_bg.wasm") });
const tfliteBytes = new Uint8Array(readFileSync("web/samples/mobilenet_v2_1.0_224_quant.tflite"));
const tflite = analyze_tflite_for_target(tfliteBytes, "mobilenet_v2_1.0_224_quant.tflite", targets[0]);
const frontier = compute_deployment_frontier(tfliteBytes, tflite.filename, JSON.stringify(targets));
tflite.model_sha256 = frontier.artifact_sha256;
tflite.deployment_frontier = frontier;

expect(validateDeploymentFrontier(frontier, tflite), "TFLite deployment frontier should satisfy browser invariants.");
expectEqual(frontier.schema, "deepbom.deployment_frontier.v1.6", "TFLite frontier schema should be stable.");
expect(/^[a-f0-9]{64}$/.test(frontier.artifact_sha256), "TFLite frontier should hash and bind the artifact inside WASM.");
expectEqual(frontier.target_count, 4, "TFLite frontier should cover the four public comparison targets.");
expectEqual(frontier.targets.length, 4, "TFLite target ledger should be complete.");
for (const target of frontier.targets) {
  const row = summarizeFrontierTargetComparison(target);
  expect(Math.abs(row.coldStartUs - target.total_us) <= 1e-12, `${target.target_id} Target Comparison cold total should bind the Frontier cold ledger.`);
  expect(Math.abs(row.totalUs - (target.total_us - target.components.packing_us - target.components.boundary_us)) <= 1e-12, `${target.target_id} Target Comparison steady total should exclude one-time packing and partition-planning setup exactly once.`);
  expect(row.coldStartUs >= row.totalUs && row.coldStartUs > 0, `${target.target_id} Target Comparison totals should be nonzero and ordered.`);
  expect(target.steady_state_low_us <= target.steady_state_us && target.steady_state_us <= target.steady_state_high_us, `${target.target_id} steady point should lie inside the disclosed modeled range.`);
}
expectEqual(frontier.cache_watch_ratio, 0.9, "Cache watch threshold should be explicit and deterministic.");
const a55Cache = frontier.targets.find((target) => target.target_id === "android_mid_a55");
expectEqual(a55Cache.l1_data_bytes, 32 * 1024, "Default A55 profile should use the conservative 32 KiB L1D reference.");
expectEqual(a55Cache.l2_bytes, 128 * 1024, "Default A55 profile should retain the 128 KiB private-L2 planning reference.");
expectEqual(a55Cache.l1_watch_count, tflite.ops.filter((op) => op.row_working_set_bytes / a55Cache.l1_data_bytes >= frontier.cache_watch_ratio).length, "A55 L1 watch count should reconstruct from exact row working sets and the selected denominator.");
expect(Math.abs(a55Cache.max_l1_ratio - Math.max(...tflite.ops.map((op) => op.row_working_set_bytes / a55Cache.l1_data_bytes))) <= 1e-12, "A55 maximum L1 ratio should conserve the op ledger.");
expect(a55Cache.max_l1_ratio >= 0.9 && a55Cache.l1_watch_count > 0, "The quantized MobileNet sample should expose the near-capacity A55 L1 row-working-set condition.");
const glance = buildModelAtGlance(tflite);
expectEqual(glance.schema, "deepbom.model_at_a_glance.v1.3", "The shared model-at-a-glance schema should expose direct-convolution PAD candidates without claiming runtime fusion.");
expectEqual(glance.memory.cache.l1.bytes, 32 * 1024, "The shared overview should use the bound A55 L1D denominator.");
expectEqual(glance.memory.cache.l1.watchCount, a55Cache.l1_watch_count, "The report/viewer summary should reuse the exact frontier cache watch policy.");
expectEqual(glance.latency.conservationStatus, "conserved", "Viewer latency components should conserve the WASM total.");
expect(Math.abs(glance.latency.totals.totalUs - tflite.ops.reduce((sum, op) => sum + Number(op.bottleneck_total_us || 0), 0)) <= 1e-9, "Viewer latency total should reconstruct from every op.");
expect(Math.abs(
  glance.latency.totals.coldStartUs
    - glance.latency.totals.steadyStateUs
    - glance.latency.totals.packingUs
    - glance.latency.totals.boundaryUs,
) <= 1e-9, "Cold-start total should equal steady-state plus one-time packing and partition-planning setup.");
expectEqual(glance.memory.artifactPlusArenaBytes, glance.artifact.fileSizeBytes + glance.memory.arenaBytes, "Artifact-plus-arena deployment footprint should conserve its two disclosed terms.");
expectEqual(tflite.block_inventory.stage_count, tflite.block_inventory.stages.length, "Blocks stage header should use the exact consolidated stage inventory.");
expect(glance.latency.totals.coldStartUs >= glance.latency.totals.steadyStateUs, "Cold-start total must not be lower than steady-state.");
expectEqual(glance.score.conservationStatus, "conserved", "Viewer score should conserve base minus every visible penalty.");
expectEqual(glance.score.base - glance.score.penaltyTotal, glance.score.final, "Viewer score arithmetic should remain independently reconstructible.");
const graphOutputContractOps = tflite.ops.filter((op) => op.channel_alignment_status === "graph-output-contract");
expect(graphOutputContractOps.some((op) => op.index === 63 && op.output_channels === 1001), "The classifier head should be traced through RESHAPE and protected as a graph-output semantic axis.");
expect(!tflite.recommendations.some((item) => item.op_index === 63 && /channel tail/i.test(item.title)), "A graph-output class axis must not receive generic SIMD padding advice.");
expect(tflite.recommendations.some((item) => item.title === `Weight packing watchlist: ${tflite.insights.packing_warn_ops} ops`), "Packing action count should use the same all-warning-op denominator as Top Signals.");
const packingAction = tflite.recommendations.find((item) => item.title.startsWith("Weight packing watchlist:"));
expect(packingAction?.detail.includes("at or above the single 10.0 us watch threshold")
  && packingAction.detail.includes("all-op one-time packing ledger")
  && packingAction.detail.includes("including sub-threshold rows"), "Packing action should separate the warning subset from the complete cold-start ledger.");
const unmatchedInputConvention = buildInputQuantizationConventionCheck({
  inputs: [{ index: 0, dtype: "INT8", scale_sample: [0.0316], zero_point_sample: [-92] }],
});
expectEqual(unmatchedInputConvention.status, "review", "An asymmetric [-1.14,6.92]-class input range should not be presented as a common preprocessing convention.");
expectEqual(unmatchedInputConvention.details[0].status, "no_common_full_domain_match", "Input convention result should retain the exact unmatched classification.");
const normalizedInputConvention = buildInputQuantizationConventionCheck({
  inputs: [{ index: 0, dtype: "INT8", scale_sample: [1 / 255], zero_point_sample: [-128] }],
});
expect(normalizedInputConvention.details[0].matched_convention_ids.includes("normalized_0_1"), "A full-code INT8 [0,1] mapping should match the normalized reference within one quantization step.");
const costCacheGlance = buildModelAtGlance(tflite, { l1DataBytes: 16 * 1024 });
expect(costCacheGlance.memory.cache.l1.watchCount >= glance.memory.cache.l1.watchCount, "A smaller viewer-only L1D denominator must not reduce the exact watch count.");
const fullGraph = collectFullGraph(tflite, buildGraphIndex(tflite));
const fullGraphLayout = layoutFullGraph(fullGraph.nodes, fullGraph.edges);
expectEqual(fullGraphLayout.positions.size, tflite.ops.length, "Top-down full-graph layout should retain every op exactly once.");
expectEqual(fullGraphLayout.orientation, "top-down", "Full-graph layout should declare its top-down orientation.");
expectEqual(fullGraphLayout.isDag, true, "The sample graph should complete a stable DAG traversal.");
expect(fullGraph.edges.every((edge) => fullGraphLayout.positions.has(edge.from) && fullGraphLayout.positions.has(edge.to)), "Top-down full-graph layout should retain both endpoints of every graph edge.");
expect(fullGraph.edges.every((edge) => fullGraphLayout.positions.get(edge.to).y > fullGraphLayout.positions.get(edge.from).y), "Every sample tensor edge should advance downward by at least one DAG rank.");
const multiInterfaceLayout = layoutFullGraph(fullGraph.nodes, fullGraph.edges, { minimumColumns: 4 });
expect(multiInterfaceLayout.bounds.width >= 48 * 2 + 238 * 4, "Multiple model interfaces should reserve non-overlapping top-down terminal columns.");
expectEqual(frontier.ops.length, tflite.ops.length, "TFLite frontier should cover every op.");
expect(frontier.robust_coverage.minimum_union_coverage >= 0.8, "Robust hotspot union should cover at least 80% on every target.");
expectEqual(frontier.target_divergence.pair_count, 6, "Four targets should produce six exact pairs.");
expect(frontier.target_divergence.pairs.every((pair) => pair.normalized_jensen_shannon_divergence >= 0 && pair.normalized_jensen_shannon_divergence <= 1), "Normalized target divergence should remain in [0,1].");
expect(frontier.target_divergence.pairs.every((pair) => pair.drivers.length === frontier.op_count && Math.abs(pair.attribution_sum - pair.normalized_jensen_shannon_divergence) <= 1e-12), "Every target pair should conserve normalized JSD across a complete per-op attribution ledger.");
expect(frontier.target_divergence.pairs.every((pair) => pair.attribution_prefix_coverage >= 0.8 && pair.attribution_prefix_op_count > 0), "Every nonzero target divergence should expose a deterministic 80% explanation prefix.");
const maximumPair = [...frontier.target_divergence.pairs].sort((left, right) => right.normalized_jensen_shannon_divergence - left.normalized_jensen_shannon_divergence)[0];
expect(maximumPair.drivers[0].attribution_share > 0 && maximumPair.drivers[0].component_contribution_delta.largest_absolute_component, "Maximum-divergence pair should identify its top op and largest modeled component shift.");
expect(frontier.interventions.every((item) => item.evidence_class === "ESTIMATED_COUNTERFACTUAL_UPPER_BOUND"), "Every counterfactual should retain its upper-bound evidence class.");
expect(frontier.targets.every((target) => Number(target.weight_packing_bandwidth_gbps) > 0
  && Number(target.effective_memory_bandwidth_gbps) > 0
  && Number(target.effective_peak_gops) > 0), "Every target should expose the profile constants needed to reproduce its counterfactual denominator.");
expectEqual(JSON.stringify(compute_deployment_frontier(tfliteBytes, tflite.filename, JSON.stringify(targets))), JSON.stringify(frontier), "TFLite deployment frontier should be byte-for-byte deterministic as JSON.");
const tfliteReport = buildEngineeringReport(tflite, { identity: { filename: tflite.filename, format: "tflite", sha256: tflite.model_sha256, target_label: tflite.target_profile.label, operator_count: tflite.operator_count, tensor_count: tflite.tensor_count, total_macs: tflite.total_macs } });
expect(tfliteReport.includes("## Deployment Frontier (DERIVED/ESTIMATED)"), "Engineering report should render the TFLite frontier.");
expect(tfliteReport.indexOf("## Model At A Glance") > tfliteReport.indexOf("## Read First") && tfliteReport.indexOf("## Model At A Glance") < tfliteReport.indexOf("## Static Audit Conclusion"), "Model At A Glance should immediately front-load practical metrics before detailed conclusions.");
expect(tfliteReport.includes("32 KiB / 128 KiB") && tfliteReport.includes("watch >=0.90x"), "Engineering report should expose the selected cache denominators and near-capacity policy.");
expect(tfliteReport.includes("### Counterfactual Denominator Ledger")
  && tfliteReport.includes("Packing and predicted-boundary setup shares use each target's cold-start point total")
  && tfliteReport.includes("Fallback shares use the steady-state point total")
  && tfliteReport.includes("GB/s planning profile"), "Engineering report should expose target-specific counterfactual numerators, denominators, and packing bandwidths.");
expect(tfliteReport.includes("36.8 KiB row working set / 32 KiB L1D = 1.15x; L2: 36.8 KiB / 128 KiB = 0.29x."), "The cache-pressure finding should expose reconstructible logical-payload L1D and L2 arithmetic for the limiting row working set.");

const tampered = structuredClone(frontier);
tampered.targets[0].total_us += 1;
expectThrows(() => validateDeploymentFrontier(tampered, tflite), "invariant failed", "Browser validation should reject a tampered target total.");
const tamperedAttribution = structuredClone(frontier);
tamperedAttribution.target_divergence.pairs[0].drivers[0].normalized_js_contribution += 0.001;
expectThrows(() => validateDeploymentFrontier(tamperedAttribution, tflite), "JSD contribution", "Browser validation should reject a tampered per-op divergence attribution.");
expectThrows(() => compute_deployment_frontier(tfliteBytes, tflite.filename, JSON.stringify([targets[0]])), "at least two", "WASM frontier should reject a single target.");

const availableProfiles = [
  ...targets,
  "android_flagship_x3_a715",
  "x86_sse4",
  "android_mid_a55_l1_16k",
  "android_mid_a55_l1_64k",
  "zynq_ultrascale_plus_a53",
].map((id) => ({ id }));
const flagshipTargetIds = deploymentFrontierTargetIds(availableProfiles, "android_flagship_x3_a715");
expectEqual(flagshipTargetIds.join(","), `${targets.join(",")},android_flagship_x3_a715`, "A selected non-canonical target should be appended to the stable comparison set.");
expect(!deploymentFrontierMatchesTargetIds(frontier, flagshipTargetIds), "A cached canonical frontier must not satisfy a request containing the selected flagship target.");
const flagship = analyze_tflite_for_target(tfliteBytes, tflite.filename, "android_flagship_x3_a715");
const flagshipFrontier = compute_deployment_frontier(tfliteBytes, tflite.filename, JSON.stringify(flagshipTargetIds));
flagship.model_sha256 = flagshipFrontier.artifact_sha256;
flagship.deployment_frontier = flagshipFrontier;
expect(validateDeploymentFrontier(flagshipFrontier, flagship), "A selected non-canonical target should remain bound to its five-target frontier.");
expect(deploymentFrontierMatchesTargetIds(flagshipFrontier, flagshipTargetIds), "Frontier cache identity should include the exact ordered target list.");
expectEqual(deploymentFrontierTargetIds(availableProfiles, "x86_sse4").at(-1), "x86_sse4", "Every available non-canonical selected target should be included deterministically.");
expectEqual(deploymentFrontierTargetIds(availableProfiles, "android_mid_a55_l1_16k").at(-1), "android_mid_a55_l1_16k", "A selected cost-chip cache variant should be included in the frontier.");
expectEqual(deploymentFrontierTargetIds(availableProfiles, "zynq_ultrascale_plus_a53").at(-1), "zynq_ultrascale_plus_a53", "The source-bound Zynq Cortex-A53 profile should be included deterministically.");
const zynq = analyze_tflite_for_target(tfliteBytes, tflite.filename, "zynq_ultrascale_plus_a53");
expectEqual(zynq.target_profile.hardware_spec.evidence_class, "SOURCE_BACKED_PRODUCT", "Zynq hardware facts should retain their product-source evidence class.");
expectEqual(`${zynq.target_profile.l1_data_bytes}:${zynq.target_profile.l2_bytes}`, `${32 * 1024}:${1024 * 1024}`, "Zynq cache denominators should match DS891.");
expectEqual(`${zynq.target_profile.hardware_spec.l1_instruction_ways}:${zynq.target_profile.hardware_spec.l1_data_ways}:${zynq.target_profile.hardware_spec.l2_ways}`, "2:4:16", "Zynq cache associativity should match DS891 plus the Cortex-A53 TRM.");
expectEqual(`${zynq.target_profile.hardware_spec.l1_line_bytes}:${zynq.target_profile.hardware_spec.l2_line_bytes}`, "64:64", "Zynq cache lines should match the Cortex-A53 TRM.");
expectEqual(zynq.target_profile.performance_model_evidence_class, "HEURISTIC", "Zynq runtime-performance constants must remain separate from source-backed hardware facts.");
const zynqReport = buildEngineeringReport(zynq, { identity: { filename: zynq.filename, format: "tflite", sha256: zynq.model_sha256, target_label: zynq.target_profile.label, operator_count: zynq.operator_count, tensor_count: zynq.tensor_count, total_macs: zynq.total_macs } });
for (const text of ["SOURCE_BACKED_PRODUCT", "DS891 v1.11.1", "1badf7142690c573987f3eacd788620ff8a8392425f13124f928aaed152265e9", "L1 data cache | 32 KiB; 4-way; 64 B line", "L2 cache | 1 MiB; 16-way; 64 B line", "FP16 register-element capacity must not be read as native FP16 arithmetic support"]) {
  expect(zynqReport.includes(text), `Zynq engineering report should include source-bound hardware evidence: ${text}`);
}

initDeepBom({ module: readFileSync("web/protected/deepbom/pkg/deepbom_wasm_bg.wasm") });
const protectedAnalysis = (analysis, bytes = Uint8Array.of(0)) => analyze_deepbom(bytes, JSON.stringify(analysis));
const costA55 = analyze_tflite_for_target(tfliteBytes, tflite.filename, "android_mid_a55_l1_16k");
const costA55Selector = protectedAnalysis(costA55, tfliteBytes).xnnpack_selector_evidence;
expectEqual(costA55Selector.assessment_status, "complete", "The protected selector should accept the A55 16 KiB cache variant without changing its source-backed kernel rules.");
expectEqual(costA55Selector.target_profile_id, "android_mid_a55_l1_16k", "Protected selector evidence should remain bound to the selected cache-variant profile.");
const onnxBytes = new Uint8Array(readFileSync("web/samples/sample_cnn_float.onnx"));
const onnx = analyzeOnnxModel(onnxBytes, "sample_cnn_float.onnx");
const ortEvidence = protectedAnalysis(onnx, onnxBytes).ort_compatibility_evidence;
applyProtectedOrtCompatibilityEvidence(onnx, ortEvidence);
const ortFrontier = onnx.ort_ep_portability_frontier;
expect(validateOrtEpPortabilityFrontier(ortFrontier, onnx), "ONNX EP portability frontier should satisfy browser invariants.");
expectEqual(ortFrontier.schema, "deepbom.ort_ep_portability_frontier.v2", "ONNX frontier schema should be stable.");
expectEqual(ortFrontier.execution_provider_count, 9, "ONNX frontier should cover CPU, CUDA, WebGPU, WebNN, QNN, DirectML, CoreML, NNAPI, and XNNPACK source profiles.");
expectEqual(ortFrontier.ops.length, onnx.ops.length, "ONNX portability frontier should cover every graph op.");
expect(ortFrontier.providers.every((provider) => provider.source_match_op_count + provider.top_gap_ops.length >= provider.source_match_op_count), "ONNX provider summaries should retain source gaps without support claims.");
expect(ortFrontier.providers.every((provider) => provider.artifact_precheck_candidate_op_count <= provider.source_match_op_count), "Artifact-precheck candidates must remain a subset of source-version matches.");
expect(ortFrontier.ops.every((op) => op.artifact_precheck_candidate_eps.every((ep) => op.source_match_eps.includes(ep))), "Every narrowed per-op EP candidate must retain its source-version match.");
expect(buildEngineeringReport(onnx, { identity: { filename: onnx.filename, format: "onnx", sha256: onnx.model_sha256, target_label: "ONNX", operator_count: onnx.operator_count, tensor_count: onnx.tensor_count, total_macs: onnx.total_macs } }).includes("## ONNX EP Portability Frontier (DERIVED_FROM_PINNED_SOURCE_AND_ARTIFACT_VISIBLE_DEFINITE_EXCLUSIONS)"), "Engineering report should render the ONNX EP frontier.");

done(`TFLite ${frontier.target_count} targets/${frontier.robust_coverage.selected_op_count} robust ops; max pair ${maximumPair.left_target_id}/${maximumPair.right_target_id} JSD ${maximumPair.normalized_jensen_shannon_divergence.toFixed(6)} explained 80% by ${maximumPair.attribution_prefix_op_count} ops, top #${maximumPair.top_driver_op_index} ${maximumPair.top_driver_op_name} ${(maximumPair.top_driver_attribution_share * 100).toFixed(1)}%; ONNX ${ortFrontier.execution_provider_count} EP source portfolios.`);

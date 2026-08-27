import { readFileSync } from "node:fs";

import { analyze_tflite_for_target, initSync } from "../pkg/tflite_wasm_audit.js";
import {
  reconstructAccumulatorReachabilityChannel,
  validateAccumulatorReachabilityAgainstReconstruction,
  validateAccumulatorReachabilityDigests,
} from "../web/lib/accumulator-reachability.js";
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

const { done, expect, expectEqual, expectThrows } = createCheck("Accumulator reachability check");
const filename = "mobilenet_v2_1.0_224_quant.tflite";
initSync({ module: readFileSync("pkg/tflite_wasm_audit_bg.wasm") });
const bytes = new Uint8Array(readFileSync(`web/samples/${filename}`));
const analysis = analyze_tflite_for_target(bytes, filename, "android_mid_a55");
const reachability = analysis.accumulator_reachability;

expectEqual(reachability.schema, "deepbom.accumulator_reachability.v1", "Reachability schema should remain stable.");
expectEqual(reachability.method_version, "2026-07-18.1", "Reachability method should remain stable.");
expectEqual(reachability.status, "assessed", "Every sample candidate should be assessed.");
expectEqual(`${reachability.candidate_op_count}:${reachability.assessed_op_count}:${reachability.unassessed_op_count}`, "53:53:0", "Every quantized kernel op should be represented exactly once.");
expectEqual(reachability.assessed_channel_count, 18_057, "Every fixed-point channel should receive a bounded-sum proof.");
expectEqual(`${reachability.complete_integer_interval_channel_count}:${reachability.complete_modular_lattice_channel_count}:${reachability.partial_band_channel_count}:${reachability.singleton_channel_count}`, "13320:34:4692:11", "Channel proof classes should remain exact.");
expectEqual(`${reachability.exact_reachable_divergent_channel_count}:${reachability.unresolved_divergent_channel_count}:${reachability.interval_only_divergent_channel_count}`, "13189:3978:3894", "Divergent channel classes should remain exact.");
expectEqual(`${reachability.interval_state_count_decimal}:${reachability.lattice_compatible_state_count_decimal}`, "13933008957:13932680007", "Interval and congruent lattice portfolios should remain exact.");
expectEqual(`${reachability.certified_reachable_state_count_decimal}:${reachability.provably_unreachable_state_count_decimal}:${reachability.unresolved_state_count_decimal}`, "13755523449:328950:177156558", "Reachability state partition should remain exact.");
expectEqual(`${reachability.interval_divergent_state_count_decimal}:${reachability.exact_reachable_divergent_state_count_decimal}:${reachability.provably_unreachable_divergent_state_count_decimal}:${reachability.unresolved_divergent_state_count_decimal}`, "2874544:2239435:3585:631524", "Divergent state partition should remain exact.");
expect(Math.abs(reachability.exact_reachable_divergent_ratio - 0.7790574783339549) < 1e-15, "Exact reachable divergent ratio should remain stable.");
expectEqual(reachability.maximum_lattice_gcd, 8, "Maximum observed lattice gcd should remain exact.");
expectEqual(reachability.reachability_ranking_op_indices.slice(0, 16).join(","), "58,61,54,50,63,39,47,43,44,40,28,32,24,55,36,60", "Reachability ranking should remain deterministic.");

const topOp = reachability.ops.find((row) => row.op_index === 58);
const topChannel = topOp.top_channels[0];
expectEqual(`${topOp.assessed_channel_count}:${topOp.complete_integer_interval_channel_count}:${topOp.exact_reachable_divergent_channel_count}`, "960:960:960", "Top op should have complete integer coverage and exact divergence in every channel.");
expectEqual(`${topOp.interval_divergent_state_count_decimal}:${topOp.exact_reachable_divergent_state_count_decimal}:${topOp.unresolved_divergent_state_count_decimal}`, "209280:209280:0", "Top op divergence should be completely reachable.");
expectEqual(topOp.reachability_ledger_sha256, "ed953ba2fc4fec5d1666cb2959ee629bc67a4bde3105965cac982a3ea0d55665", "Top op reachability ledger should remain stable.");
expectEqual(`${topChannel.channel_index}:${topChannel.proof_status}:${topChannel.lattice_gcd}`, "0:complete_integer_interval:1", "Top channel proof identity should remain stable.");
expectEqual(`${topChannel.exact_reachable_divergent_state_count_decimal}:${topChannel.first_exact_reachable_divergent_accumulator_decimal}:${topChannel.first_default_output_code}:${topChannel.first_single_output_code}`, "218:27:1:0", "Top channel exact counterexample should remain stable.");
expectEqual(`${topChannel.denomination_group_count}:${topChannel.first_exact_reachable_witness_group_count}`, "23:13", "Top channel compact witness cardinalities should remain stable.");

const selected = reconstructAccumulatorReachabilityChannel(analysis, bytes, 58, 0);
expectEqual(`${selected.post_bias_minimum_decimal}:${selected.post_bias_maximum_decimal}`, "-157985:152095", "Selected exact interval should reproduce stored weights.");
expectEqual(`${selected.denomination_coverage_steps.length}:${selected.first_exact_reachable_aggregate_coefficient_witness.length}`, "23:13", "Selected detail should reconstruct denomination and coefficient witnesses on demand.");
const witnessOffset = selected.first_exact_reachable_aggregate_coefficient_witness.reduce((sum, row) => sum + BigInt(row.normalized_denomination) * BigInt(row.aggregate_input_code_delta_decimal), 0n);
expectEqual((BigInt(selected.post_bias_minimum_decimal) + BigInt(selected.lattice_gcd) * witnessOffset).toString(), selected.first_exact_reachable_divergent_accumulator_decimal, "Aggregate coefficient witness should reproduce the first exact divergent accumulator.");

const modular = reconstructAccumulatorReachabilityChannel(analysis, bytes, 1, 5);
expectEqual(`${modular.proof_status}:${modular.lattice_gcd}:${modular.exact_reachable_divergent_state_count_decimal}:${modular.provably_unreachable_divergent_state_count_decimal}:${modular.unresolved_divergent_state_count_decimal}`, "complete_modular_lattice:8:23:163:0", "Modular proof should distinguish exact congruent and excluded residue states.");
const partial = reconstructAccumulatorReachabilityChannel(analysis, bytes, 0, 0);
expectEqual(`${partial.proof_status}:${partial.exact_reachable_divergent_state_count_decimal}:${partial.provably_unreachable_divergent_state_count_decimal}:${partial.unresolved_divergent_state_count_decimal}`, "partial_endpoint_bands:0:0:177", "Partial proof should retain an unresolved compatible middle without overclaiming reachability.");

const reconstructed = await validateAccumulatorReachabilityDigests(analysis, bytes);
expectEqual(reconstructed.exact_reachable_divergent_state_count_decimal, reachability.exact_reachable_divergent_state_count_decimal, "Independent JavaScript reconstruction should reproduce the Rust portfolio.");

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
expect(report.includes("## Accumulator Reachability Lattice (DERIVED EXACT KERNEL-LOCAL CERTIFICATE)"), "Engineering report should render the reachability section.");
expect(report.includes("2,239,435") && report.includes("3,585") && report.includes("631,524") && report.includes("#058 CONV_2D / ch 0"), "Engineering report should preserve the exact divergent partition and ranked coordinate.");
expect(report.includes("Divergent-channel sets (non-exclusive)") && report.includes("Exclusive assessed-channel partition"), "Engineering report should distinguish overlapping channel sets from the conserved exclusive partition.");
expect(report.includes(topOp.reachability_ledger_sha256) && report.includes("full-model-input reachability") && report.includes(reachability.source_commit), "Engineering report should preserve the ledger, source, and proof boundary.");

const staticAnalysis = buildStaticAnalysisExport(analysis);
const quantization = buildQuantizationEvidence(analysis, identity);
expectEqual(staticAnalysis.accumulator_reachability.schema, ANALYZER_METADATA.schemas.accumulatorReachability, "Static evidence should retain the reachability schema.");
expectEqual(JSON.stringify(quantization.accumulator_reachability), JSON.stringify(staticAnalysis.accumulator_reachability), "Static and quantization evidence should retain one reachability ledger.");
const contract = buildQuantizationContractChecks(analysis).accumulator_reachability;
expectEqual(contract.status, "review", "Exact reachable build-mode divergence should enter design review.");
expectEqual(`${contract.exact_reachable_divergent_state_count_decimal}:${contract.provably_unreachable_divergent_state_count_decimal}:${contract.unresolved_divergent_state_count_decimal}`, "2239435:3585:631524", "Contract summary should expose the exact divergent partition.");

const findings = buildFindingsRegister(analysis);
const finding = findings.find((item) => item.finding_id === "EA-QNT-0115");
expect(finding, "Exact kernel-local build-mode counterexamples should enter the authoritative action queue.");
expectEqual(finding?.technical_priority, "Informational", "Kernel-local reachability should remain supporting evidence under the model-input rounding ABI action.");
expect(finding?.observation.includes("2,239,435") && finding?.observation.includes("channel 0") && finding?.observation.includes(topOp.reachability_ledger_sha256), "Finding should preserve exact states, coordinate, and digest.");
expect(finding?.interpretation.includes("not proof that upstream model activations") && finding?.recommendation.includes("aggregate coefficient witness"), "Finding should preserve the reachability boundary and actionable replay path.");

const mlBom = buildMlBomDocument(analysis, { hash: analysis.model_sha256, fileSizeBytes: bytes.byteLength, target: analysis.target_profile });
assertCompactMlBomProjection(mlBom, {
  expect,
  expectEqual,
  omittedProperties: [
    "deepbom:model:accumulatorReachabilitySchema",
    "deepbom:model:accumulatorReachabilityExactDivergentStates",
    "deepbom:model:accumulatorReachabilityExcludedDivergentStates",
    "deepbom:model:accumulatorReachabilityUnresolvedDivergentStates",
  ],
  label: "Accumulator-reachability compact ML-BOM",
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
const reachabilityChecks = conformance.checks.filter((check) => check.id.startsWith("CF-REACH-"));
expectEqual(reachabilityChecks.length, 4, "Conformance should expose four cross-output reachability checks.");
expect(reachabilityChecks.every((check) => check.status === "pass"), "Every accumulator-reachability conformance check should pass for the sample.");
expect(visualPngSpecs({ analysis, filename }).some(([path]) => path === "visuals/accumulator_reachability.png"), "Engineering bundle should include the reachability PNG.");

const repeated = analyze_tflite_for_target(bytes, filename, "android_mid_a55").accumulator_reachability;
expectEqual(JSON.stringify(repeated), JSON.stringify(reachability), "Accumulator-reachability JSON should be byte-for-byte deterministic.");
const tampered = structuredClone(reachability);
tampered.exact_reachable_divergent_state_count_decimal = "2239436";
expectThrows(() => validateAccumulatorReachabilityAgainstReconstruction(tampered, reconstructed), "Global exact divergent states mismatch", "Validator should reject a tampered exact divergent total.");

done("Accumulator reachability passed (53 ops, 18,057 channels, exact bounded-sum classes, 2,239,435 reachable numerical ABI differences, reports, digests, and tamper rejection).");

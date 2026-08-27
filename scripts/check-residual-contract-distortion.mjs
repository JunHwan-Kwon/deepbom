import { readFileSync } from "node:fs";

import { analyze_tflite_for_target, initSync } from "../pkg/tflite_wasm_audit.js";
import {
  validateResidualContractDistortion,
  validateResidualContractDistortionDigests,
} from "../web/lib/residual-contract-distortion.js";
import { buildConformanceReport } from "../web/lib/report-conformance.js";
import { buildEngineeringReport } from "../web/lib/report-engineering.js";
import { buildQuantizationEvidence, buildStaticAnalysisExport } from "../web/lib/report-evidence.js";
import { buildFindingsRegister } from "../web/lib/report-findings.js";
import { buildMlBomDocument } from "../web/lib/report-mlbom.js";
import { assertCompactMlBomProjection } from "./compact-mlbom-assert.mjs";
import { ANALYZER_METADATA } from "../web/lib/report-metadata.js";
import { buildQuantizationContractChecks } from "../web/lib/report-quantization-contracts.js";
import { createCheck } from "./check-assert.mjs";

const { done, expect, expectEqual, expectThrows } = createCheck("Residual contract distortion check");
const filename = "mobilenet_v2_1.0_224_quant.tflite";
initSync({ module: readFileSync("pkg/tflite_wasm_audit_bg.wasm") });
const bytes = new Uint8Array(readFileSync(`web/samples/${filename}`));
const analysis = analyze_tflite_for_target(bytes, filename, "android_mid_a55");
const result = analysis.residual_contract_distortion;

expect(validateResidualContractDistortion(result, analysis), "Browser arithmetic should reconstruct every distortion scenario.");
await validateResidualContractDistortionDigests(result, analysis);
expectEqual(result.schema, "deepbom.residual_contract_distortion.v1.1", "Distortion schema should remain stable.");
expectEqual(result.method_version, "2026-07-29.1", "Distortion aggregate-count semantics should remain stable.");
expectEqual(result.status, "assessed", "Every sample residual should be assessed.");
expectEqual(result.assessed_add_count, 10, "All sample residual ADDs should be assessed.");
expectEqual(result.scenario_count, 20, "Both containment candidates should be compared per residual.");
expectEqual(result.total_enumerated_pair_count, 1_310_720, "Every candidate should cover the full legal pair domain.");
expectEqual(result.current_clamped_pair_instance_count, 99_705, "Current clamp pairs should be counted once per artifact ADD contract.");
expectEqual(result.scenario_current_clamped_pair_instance_count, 199_410, "Scenario-current clamp instances should retain candidate-comparison multiplicity.");
expectEqual(result.candidate_clamped_pair_count, 0, "Containment candidates should remain clamp-free.");
expectEqual(result.rescued_current_clamp_pair_instance_count, 199_410, "Every current clamp instance should be rescued.");
expectEqual(result.changed_represented_value_pair_count, 1_304_974, "Changed represented-value pairs should remain exact.");
expectEqual(result.ideal_error_improved_pair_count, 532_893, "Improved ideal-error pairs should remain exact.");
expectEqual(result.ideal_error_worsened_pair_count, 772_081, "Worsened ideal-error pairs should remain exact.");
expectEqual(result.ideal_error_equal_within_tolerance_pair_count, 5_746, "Equal-within-tolerance pairs should remain exact.");
expectEqual(result.sign_class_changed_pair_count, 3_764, "Sign-class changes should remain exact.");
expectEqual(result.distortion_ranking_op_indices[0], 27, "ADD #27 should remain the largest RMS distortion scenario.");
expectEqual(result.maximum_rms_contract_delta_current_steps, 23.062489300248952, "Maximum RMS displacement should remain exact.");
expectEqual(result.maximum_p99_contract_delta_current_steps, 94.52096182839742, "Maximum p99 displacement should remain exact.");

const top = result.residual_adds.find((row) => row.op_index === 27);
const fixed = top.scenarios.find((scenario) => scenario.design === "fixed_zero_point_minimum_containment");
const global = top.scenarios.find((scenario) => scenario.design === "globally_finest_minimum_containment");
expectEqual(fixed.rescued_current_clamp_pair_count, 14_792, "ADD #27 fixed candidate should rescue every current clamp.");
expectEqual(fixed.p50_absolute_contract_delta_current_steps, 0.6439215559420043, "ADD #27 fixed p50 should remain exact.");
expectEqual(fixed.p99_absolute_contract_delta_current_steps, 93.55157427173506, "ADD #27 fixed p99 should remain exact.");
expectEqual(fixed.pair_ledger_sha256, "cecf5fd321329802b95eb0fe520fd98eb099c2de38ed1ae5505868d5e19950b6", "ADD #27 fixed ledger should remain stable.");
expectEqual(global.ideal_error_improved_pair_count, 28_006, "ADD #27 global improved-pair count should remain exact.");
expectEqual(global.ideal_error_worsened_pair_count, 37_283, "ADD #27 global worsened-pair count should remain exact.");
expectEqual(global.p99_absolute_contract_delta_current_steps, 94.52096182839742, "ADD #27 global p99 should remain exact.");
expectEqual(global.pair_ledger_sha256, "0b5c3023ebe8ab105ff7451935b57f8c00ccaf201d4cb901d9ab81d5e356ba47", "ADD #27 global ledger should remain stable.");

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
expect(report.includes("## Residual Contract Distortion Atlas (DERIVED EXHAUSTIVE COUNTERFACTUAL DOMAIN)"), "Engineering report should render the distortion atlas section.");
expect(report.includes("deepbom.residual_contract_distortion.v1.1") && report.includes("1,310,720")
  && report.includes("532,893 / 772,081 / 5,746") && report.includes("#027 ADD"), "Engineering report should preserve schema, aggregate pair classes, and top identity.");
expect(report.includes("nine signed i64") && report.includes("six IEEE-754 binary64")
  && report.includes("not an observed activation distribution") && report.includes("complete re-export"), "Engineering report should preserve pair-ledger and interpretation boundaries.");
const staticAnalysis = buildStaticAnalysisExport(analysis);
const quantization = buildQuantizationEvidence(analysis, identity);
expectEqual(staticAnalysis.residual_contract_distortion.schema, ANALYZER_METADATA.schemas.residualContractDistortion, "Static evidence should retain the distortion schema.");
expectEqual(quantization.residual_contract_distortion.total_enumerated_pair_count, 1_310_720, "Quantization evidence should retain exact pair coverage.");
const contracts = buildQuantizationContractChecks(analysis).residual_contract_distortion;
expectEqual(contracts.checked_candidate_scenarios, 20, "Quantization checks should bind every distortion scenario.");
expectEqual(contracts.ideal_error_worsened_pairs, 772_081, "Quantization checks should retain worsened-pair count.");
const findings = buildFindingsRegister(analysis);
const finding = findings.find((item) => item.finding_id === "EA-QNT-0110");
expect(finding?.observation.includes("532,893") && finding?.observation.includes("772,081")
  && finding?.interpretation.includes("uniform-domain geometry") && finding?.recommendation.includes("reported worst pair witnesses"), "Finding should preserve exact trade-off counts and action boundary.");
const mlBom = buildMlBomDocument(analysis, { hash: analysis.model_sha256, fileSizeBytes: bytes.byteLength, target: analysis.target_profile });
assertCompactMlBomProjection(mlBom, {
  expect,
  expectEqual,
  omittedProperties: [
    "deepbom:model:residualContractDistortionPairs",
    "deepbom:model:residualContractDistortionRescuedClamps",
  ],
  label: "Residual-distortion compact ML-BOM",
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
const distortionChecks = conformance.checks.filter((check) => check.id.startsWith("CF-DISTORTION-"));
expectEqual(distortionChecks.length, 4, "Conformance should expose four distortion cross-output checks.");
expect(distortionChecks.every((check) => check.status === "pass"), "Every distortion conformance check should pass for the sample.");

const tamperedTile = structuredClone(result);
tamperedTile.residual_adds[0].scenarios[0].tile_ideal_error_worsened_pair_counts[0] += 1;
expectThrows(() => validateResidualContractDistortion(tamperedTile, analysis), "tile_ideal_error_worsened", "Validator should reject a tampered distortion tile.");
const tamperedWitness = structuredClone(result);
tamperedWitness.residual_adds[0].scenarios[0].worst_absolute_contract_delta_pair.candidate_projected_code += 1;
expectThrows(() => validateResidualContractDistortion(tamperedWitness, analysis), "worst witness", "Validator should reject a tampered worst witness.");
const tamperedDigest = structuredClone(result);
tamperedDigest.residual_adds[0].scenarios[0].pair_ledger_sha256 = "0".repeat(64);
await expectRejects(() => validateResidualContractDistortionDigests(tamperedDigest, analysis), "digest", "Validator should reject a tampered pair ledger.");
expect(result.interpretation_boundary.includes("not an observed activation distribution")
  && result.interpretation_boundary.includes("complete re-export"), "Interpretation boundary should reject distribution and in-place-edit overclaims.");

done(`10 residuals, 20 candidates, ${result.total_enumerated_pair_count.toLocaleString()} exact pair comparisons, full digests, and tamper rejection passed.`);

async function expectRejects(action, fragment, message) {
  try {
    await action();
    expect(false, message);
  } catch (error) {
    expect(String(error?.message || error).toLowerCase().includes(fragment.toLowerCase()), message);
  }
}

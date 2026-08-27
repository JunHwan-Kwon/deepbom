import { readFileSync } from "node:fs";

import { analyze_tflite_for_target, initSync } from "../pkg/tflite_wasm_audit.js";
import {
  reconstructResidualLatticeRow,
  validateQuantizationLattice,
} from "../web/lib/quantization-lattice.js";
import { buildConformanceReport } from "../web/lib/report-conformance.js";
import { buildEngineeringReport } from "../web/lib/report-engineering.js";
import { buildQuantizationEvidence, buildStaticAnalysisExport } from "../web/lib/report-evidence.js";
import { buildFindingsRegister } from "../web/lib/report-findings.js";
import { buildMlBomDocument } from "../web/lib/report-mlbom.js";
import { ANALYZER_METADATA } from "../web/lib/report-metadata.js";
import { buildQuantizationContractChecks } from "../web/lib/report-quantization-contracts.js";
import { createCheck } from "./check-assert.mjs";

const { done, expect, expectEqual, expectThrows } = createCheck("Quantization lattice check");
const filename = "mobilenet_v2_1.0_224_quant.tflite";
initSync({ module: readFileSync("pkg/tflite_wasm_audit_bg.wasm") });
const bytes = new Uint8Array(readFileSync(`web/samples/${filename}`));
const analysis = analyze_tflite_for_target(bytes, filename, "android_mid_a55");
const lattice = analysis.quantization_lattice;

expect(validateQuantizationLattice(lattice, analysis), "Browser reconstruction should validate the WASM lattice ledger.");
expectEqual(lattice.schema, "deepbom.quantization_lattice.v1.4", "Lattice schema should be stable.");
expectEqual(lattice.method_version, "2026-07-17.3", "Lattice method should be stable.");
expectEqual(lattice.status, "assessed", "All sample residual ADDs should be assessed.");
expectEqual(lattice.candidate_add_count, 10, "MobileNetV2 should expose ten residual ADDs.");
expectEqual(lattice.assessed_add_count, 10, "All ten residual ADDs should have complete 8-bit contracts.");
expectEqual(lattice.unassessed_add_count, 0, "No residual ADD should be silently omitted.");
expectEqual(lattice.candidate_operator_count, 10, "Overall candidate count should include every lattice family.");
expectEqual(lattice.assessed_operator_count, 10, "Overall assessed count should include every lattice family.");
expectEqual(lattice.residual_add_status, "assessed", "Residual ADD family status should be explicit.");

// Widened coverage: the enumerable binary operators beyond ADD and the
// per-input CONCATENATION projection are reported separately from the ADD
// fields, whose consumers are bound to residual-ADD semantics.
expectEqual(lattice.candidate_binary_count, 0, "MobileNetV2 exposes no SUB/MUL/MAXIMUM/MINIMUM.");
expectEqual(lattice.candidate_concatenation_count, 0, "MobileNetV2 exposes no CONCATENATION.");
expect(Array.isArray(lattice.binary_contracts) && Array.isArray(lattice.concatenation_contracts)
  && Array.isArray(lattice.binary_operator_coverage), "Widened lattice arrays should always be present.");
expectEqual(lattice.total_enumerated_code_pairs, 655_360, "Each ADD should enumerate the complete 256x256 code domain.");
expectEqual(lattice.residual_add_enumerated_code_pairs, 655_360, "Residual pair subtotal should remain explicit.");
expectEqual(lattice.total_enumerated_concatenation_codes, 0, "Absent CONCATENATION should contribute no codes.");
expectEqual(lattice.range_escape_add_count, 10, "Every sample ADD has legal-code sums outside its output endpoint interval.");
expectEqual(lattice.complete_domain_containment_add_count, 0, "No sample ADD contains the complete Cartesian sum domain.");
expectEqual(lattice.containment_design_add_count, 10, "Every assessed ADD should receive a containment design space.");
expectEqual(lattice.fixed_zero_point_containment_add_count, 10, "Every sample output zero-point should admit a containment scale.");
expectEqual(lattice.fixed_zero_point_scale_expansion_add_count, 10, "Every sample fixed-zero-point design should require scale expansion.");
expectEqual(lattice.global_zero_point_shift_add_count, 10, "Every globally finest sample design should require a zero-point shift.");
expectEqual(lattice.maximum_fixed_zero_point_scale_ratio, 1.9530308126873712, "Maximum fixed-zero-point scale ratio should remain deterministic.");
expectEqual(lattice.maximum_global_finest_scale_ratio, 1.8960080152366452, "Maximum global-finest scale ratio should remain deterministic.");
expectEqual(lattice.domain_escape_ranking_op_indices.join(","), "27,16,31,42,35,9,20,53,46,57", "Domain-escape ranking should remain deterministic.");
expectEqual(lattice.maximum_range_escape_pair_ratio, 0.2277069091796875, "Maximum endpoint-escape ratio should remain exact.");

const top = lattice.residual_adds.find((row) => row.op_index === 27);
expectEqual(top.domain_escape_rank, 1, "Op #27 should rank first.");
expectEqual(top.enumerated_code_pair_count, 65_536, "Top ADD pair count should be exhaustive.");
expectEqual(top.range_escape_pair_count, 14_923, "Top ADD endpoint-escape count should remain exact.");
expectEqual(top.rounded_projection_clamp_pair_count, 14_792, "Top ADD rounded-clamp count should remain separate and exact.");
expectEqual(top.output_code_histogram.reduce((sum, value) => sum + value, 0), 65_536, "Output histogram should conserve all code pairs.");
expectEqual(top.tile_range_escape_pair_counts.reduce((sum, value) => sum + value, 0), 14_923, "Tile ledger should conserve endpoint-escape pairs.");
expectEqual(top.tile_range_escape_pair_counts.length, 256, "The exact aggregate map should be 16x16.");
expect(Math.abs(top.mean_in_range_rounding_error_steps - 0.250030491019485) < 1e-12, "In-range rounding mean should remain reproducible.");
expect(Math.abs(top.mean_clamped_projection_error_steps - 9.123866253333475) < 1e-12, "All-pair projection mean should remain reproducible.");
expectEqual(top.worst_projection_pair.input_0_code, 255, "Worst pair q0 should remain stable.");
expectEqual(top.worst_projection_pair.input_1_code, 255, "Worst pair q1 should remain stable.");
expectEqual(top.worst_projection_pair.projected_output_code, 255, "Worst pair should clamp to the output endpoint.");
expectEqual(top.containment_candidate_count, 254, "Signed legal-sum containment should reject only the two UINT8 endpoint zero-points.");
expectEqual(top.containment_frontier.length, 5, "Top ADD should retain five non-dominated scale/zero-point-shift designs.");
expectEqual(top.fixed_zero_point_containment.output_zero_point, 122, "Fixed-zero-point design should preserve the artifact zero-point.");
expectEqual(top.fixed_zero_point_containment.output_scale, 0.3693381729430722, "Fixed-zero-point minimum binary64 scale should remain exact.");
expectEqual(top.fixed_zero_point_containment.rounded_projection_clamp_pair_count, 0, "Fixed-zero-point containment should eliminate ideal projection clamps.");
expectEqual(top.globally_finest_containment.output_zero_point, 118, "Global finest design zero-point should remain deterministic.");
expectEqual(top.globally_finest_containment.output_scale, 0.3585545766527635, "Global finest minimum binary64 scale should remain exact.");
expectEqual(top.globally_finest_containment.signed_zero_point_delta, -4, "Global finest design should expose the exact zero-point migration.");
expectEqual(top.globally_finest_containment.rounded_projection_clamp_pair_count, 0, "Global finest containment should eliminate ideal projection clamps.");
expectEqual(top.globally_finest_containment.distinct_projected_output_code_count, 256, "Global finest design should use the complete output codebook under uniform legal pairs.");

const repeated = analyze_tflite_for_target(bytes, filename, "android_mid_a55").quantization_lattice;
expectEqual(JSON.stringify(repeated), JSON.stringify(lattice), "Lattice JSON should be byte-for-byte deterministic.");

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
expect(report.includes("## Quantization Lattice Lab (DERIVED EXHAUSTIVE DOMAIN)"), "Engineering report should render the exhaustive lattice section.");
expect(report.includes("deepbom.quantization_lattice.v1.4") && report.includes("#027 ADD") && report.includes("14,923 / 65,536 (22.8%)"), "Engineering report should preserve schema, top identity, and exact pair ledger.");
expect(report.includes("No SUB, MUL, MAXIMUM, or MINIMUM candidate was present.")
  && report.includes("No CONCATENATION candidate was present."), "Engineering report should disclose widened-family applicability.");
expect(report.includes("1.9530x scale / zp +0 / 0 clamps")
  && report.includes("1.8960x scale / zp -4 / 0 clamps")
  && report.includes("min_binary64"), "Engineering report should preserve the top containment design and formula.");
expect(report.includes("not over observed activation values or their probability distribution")
  && report.includes("counterfactual output contracts"), "Engineering report should preserve the interpretation boundary.");

const quantization = buildQuantizationEvidence(analysis, identity);
const conformance = buildConformanceReport({
  analysis,
  staticAnalysis: buildStaticAnalysisExport(analysis),
  quantization,
  findingsRegister: { authoritative_action_source: "findings", raw_analyzer_signals: [], findings: [] },
  runtimeResults: {},
  securityPosture: { execution_integrity: {} },
  mlBomDocument: {},
  engineeringReport: report,
});
const latticeChecks = conformance.checks.filter((check) => check.id.startsWith("CF-LATTICE-"));
expectEqual(latticeChecks.length, 3, "Conformance should expose three lattice cross-output checks.");
expect(latticeChecks.every((check) => check.status === "pass"), "All lattice conformance checks should pass for the sample.");
expectEqual(quantization.residual_quantization_lattice.schema, ANALYZER_METADATA.schemas.quantizationLattice, "Quantization evidence should retain the lattice schema.");
const quantizationContracts = buildQuantizationContractChecks(analysis);
const topResidualContract = quantizationContracts.residual_add.details.find((row) => row.op_index === 27);
expectEqual(topResidualContract.exhaustive_legal_code_pair_count, 65_536, "Residual contract summary should bind the exhaustive lattice domain.");
expectEqual(topResidualContract.globally_finest_containment_scale_ratio, 1.8960080152366452, "Residual contract summary should retain the global containment ratio.");
expectEqual(topResidualContract.globally_finest_containment_zero_point_delta, -4, "Residual contract summary should retain the zero-point migration.");
const residualFinding = buildFindingsRegister(analysis).find((finding) => finding.finding_id === "EA-QNT-0108");
expect(residualFinding?.observation.includes("globally finest full-containment contract")
  && residualFinding?.observation.includes("direct parameter consumer(s)")
  && residualFinding?.observation.includes("structural behavior radius")
  && residualFinding?.recommendation.includes("Regenerate the listed direct-consumer Q0.31 parameters and INT32 bias constants")
  && residualFinding?.recommendation.includes("not a claim that every one needs metadata replacement"), "Residual action should preserve exact direct-consumer migration evidence and keep the reachable structural radius distinct.");
const mlBom = buildMlBomDocument(analysis, { hash: analysis.model_sha256, fileSizeBytes: bytes.byteLength, target: analysis.target_profile });
const mlBomProperties = new Map((mlBom.metadata.component.properties || []).map((item) => [item.name, item.value]));
expectEqual(mlBomProperties.get("deepbom:compatibility:detailLocation"), "engineering_evidence.json#/evidence/static_analysis", "Compact ML-BOM should bind omitted lattice detail to the canonical evidence ledger.");
expect(!mlBomProperties.has("deepbom:model:quantizationLatticeSchema")
  && !mlBomProperties.has("deepbom:model:residualContainmentDesignAdds"), "Compact ML-BOM should not duplicate the validated lattice ledger as compatibility properties.");

const genericTensors = [
  quantTensor(0, 0.25, 128),
  quantTensor(1, 0.5, 120),
  quantTensor(2, 0.125, 127),
];
const binaryRanges = new Map([
  ["ADD", [-92, 99.25]],
  ["SUB", [-99.5, 91.75]],
  ["MUL", [-2_160, 2_143.125]],
  ["MAXIMUM", [-32, 67.5]],
  ["MINIMUM", [-60, 31.75]],
]);
for (const [offset, [opName, expectedRange]] of [...binaryRanges].entries()) {
  const reconstructed = reconstructResidualLatticeRow({
    index: 7 + offset,
    name: opName,
    inputs: [0, 1],
    outputs: [2],
    fused_activation: "NONE",
  }, genericTensors);
  expectEqual(reconstructed.status, "assessed", `${opName} should be independently reconstructable.`);
  expectEqual(reconstructed.pairs, 65_536, `${opName} should enumerate every legal input-code pair.`);
  expectEqual(reconstructed.histogram.reduce((sum, count) => sum + count, 0), 65_536, `${opName} histogram should conserve its complete domain.`);
  expectEqual(reconstructed.sumRange[0], expectedRange[0], `${opName} legal range minimum should follow its operator semantics.`);
  expectEqual(reconstructed.sumRange[1], expectedRange[1], `${opName} legal range maximum should follow its operator semantics.`);
}

const tamperedHistogram = structuredClone(lattice);
tamperedHistogram.residual_adds.find((row) => row.op_index === 27).output_code_histogram[0] += 1;
expectThrows(() => validateQuantizationLattice(tamperedHistogram, analysis), "histogram", "Validator should reject a tampered output histogram.");
const tamperedTile = structuredClone(lattice);
tamperedTile.residual_adds.find((row) => row.op_index === 27).tile_range_escape_pair_counts[0] += 1;
expectThrows(() => validateQuantizationLattice(tamperedTile, analysis), "tile ledger", "Validator should reject a tampered exact tile ledger.");
const tamperedFrontier = structuredClone(lattice);
tamperedFrontier.residual_adds.find((row) => row.op_index === 27).containment_frontier[0].minimum_output_scale += 0.01;
expectThrows(() => validateQuantizationLattice(tamperedFrontier, analysis), "frontier scale", "Validator should reject a tampered containment frontier.");
const tamperedCandidate = structuredClone(lattice);
tamperedCandidate.residual_adds.find((row) => row.op_index === 27).globally_finest_containment.mean_absolute_projection_error += 0.01;
expectThrows(() => validateQuantizationLattice(tamperedCandidate, analysis), "mean_absolute_projection_error", "Validator should reject a tampered candidate projection ledger.");
const tamperedAnalysis = structuredClone(analysis);
tamperedAnalysis.quantization_lattice.residual_adds.find((row) => row.op_index === 27).range_escape_pair_count += 1;
const tamperedConformance = buildConformanceReport({
  analysis: tamperedAnalysis,
  staticAnalysis: buildStaticAnalysisExport(tamperedAnalysis),
  quantization: buildQuantizationEvidence(tamperedAnalysis, identity),
  findingsRegister: { authoritative_action_source: "findings", raw_analyzer_signals: [], findings: [] },
  runtimeResults: {},
  securityPosture: { execution_integrity: {} },
  mlBomDocument: {},
  engineeringReport: report,
});
expectEqual(tamperedConformance.checks.find((check) => check.id === "CF-LATTICE-002")?.status, "fail", "Conformance should reject tampered exhaustive arithmetic.");

done("Quantization lattice passed (10/10 model ADDs plus independent ADD/SUB/MUL/MAXIMUM/MINIMUM reconstruction, exact binary/CONCAT aggregation, containment reprojection, report binding, and tamper rejection).");

function quantTensor(index, scale, zeroPoint) {
  return {
    index,
    name: `T${index}`,
    dtype: "UINT8",
    shape: [1],
    quant_scales: 1,
    quant_zero_points: 1,
    scale_sample: [scale],
    zero_point_sample: [zeroPoint],
    quantized_dimension: 0,
  };
}

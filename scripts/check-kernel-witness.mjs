import { readFileSync } from "node:fs";

import { analyze_tflite_for_target, initSync } from "../pkg/tflite_wasm_audit.js";
import {
  reconstructKernelChannel,
  validateKernelWitnessAnalysis,
  validateKernelWitnessDigests,
} from "../web/lib/kernel-witness.js";
import { buildConformanceReport } from "../web/lib/report-conformance.js";
import { buildEngineeringReport } from "../web/lib/report-engineering.js";
import { buildQuantizationEvidence, buildStaticAnalysisExport } from "../web/lib/report-evidence.js";
import { buildFindingsRegister } from "../web/lib/report-findings.js";
import { buildMlBomDocument } from "../web/lib/report-mlbom.js";
import { ANALYZER_METADATA } from "../web/lib/report-metadata.js";
import { buildQuantizationContractChecks } from "../web/lib/report-quantization-contracts.js";
import { createCheck } from "./check-assert.mjs";
import { assertCompactMlBomProjection } from "./compact-mlbom-assert.mjs";

const { done, expect, expectEqual, expectThrows } = createCheck("Quantized kernel witness check");
const filename = "mobilenet_v2_1.0_224_quant.tflite";
initSync({ module: readFileSync("pkg/tflite_wasm_audit_bg.wasm") });
const bytes = new Uint8Array(readFileSync(`web/samples/${filename}`));
const analysis = analyze_tflite_for_target(bytes, filename, "android_mid_a55");
const witness = analysis.kernel_extremum_witness;

const reconstructed = validateKernelWitnessAnalysis(analysis, bytes);
await validateKernelWitnessDigests(analysis, bytes);
expectEqual(witness.schema, "deepbom.kernel_extremum_witness.v1", "Kernel-witness schema should remain stable.");
expectEqual(witness.method_version, "2026-07-17.1", "Kernel-witness method should remain stable.");
expectEqual(witness.status, "assessed", "Every candidate sample op should be assessed.");
expectEqual(witness.candidate_op_count, 53, "Every convolution-family op should enter witness analysis.");
expectEqual(witness.assessed_op_count, 53, "Every sample witness op should be assessed.");
expectEqual(witness.unassessed_op_count, 0, "No candidate op should be silently omitted.");
expectEqual(witness.assessed_channel_count, 18_057, "Every output channel should receive a witness.");
expectEqual(witness.fixed_point_assessed_channel_count, 18_057, "Every witness channel should receive both pinned fixed-point projections.");
expectEqual(witness.witness_assignment_count, 6_942_080, "Canonical min/max term assignments should remain exact.");
expectEqual(witness.fixed_point_endpoint_evaluation_count, 72_228, "Two endpoints and two build modes should execute for every channel.");
expectEqual(witness.default_ideal_mismatch_endpoint_count, 125, "Default double-rounding mismatch count should remain exact.");
expectEqual(witness.single_ideal_mismatch_endpoint_count, 0, "Single-rounding should match direct ideal projection for this sample.");
expectEqual(witness.build_mode_divergent_endpoint_count, 125, "Pinned build-mode output-code differences should remain exact.");
expectEqual(witness.default_activation_clamped_endpoint_count, 33_935, "Default activation-clamped endpoints should remain exact.");
expectEqual(witness.single_activation_clamped_endpoint_count, 33_928, "Single-rounding activation-clamped endpoints should remain exact.");
expectEqual(witness.default_collapsed_extrema_channel_count, 14, "Default collapsed extremum spans should remain exact.");
expectEqual(witness.single_collapsed_extrema_channel_count, 15, "Single-rounding collapsed extremum spans should remain exact.");
expectEqual(witness.maximum_default_ideal_delta_codes, 1, "Default maximum ideal delta should remain one code.");
expectEqual(witness.maximum_single_ideal_delta_codes, 0, "Single-rounding maximum ideal delta should remain zero codes.");
expectEqual(witness.witness_ranking_op_indices.slice(0, 15).join(","), "55,59,40,44,29,25,48,14,7,11,1,61,63,50,54", "Witness ranking should remain deterministic.");

const top = witness.ops.find((row) => row.op_index === 55);
const channel = top.worst_channel;
expectEqual(top.op_name, "DEPTHWISE_CONV_2D", "Top witness op identity should remain stable.");
expectEqual(top.assessed_channel_count, 960, "Top op should retain every output channel.");
expectEqual(top.accumulation_terms_per_channel, 9, "Top depthwise receptive field should contain nine terms.");
expectEqual(top.witness_assignment_count, 17_280, "Top-op witness assignment count should conserve channel terms and endpoints.");
expectEqual(top.fixed_point_endpoint_evaluation_count, 3_840, "Top-op fixed-point execution count should conserve channels, endpoints, and modes.");
expectEqual(top.default_ideal_mismatch_endpoint_count, 10, "Top-op default/ideal mismatch count should remain exact.");
expectEqual(top.build_mode_divergent_endpoint_count, 10, "Top-op build-mode difference count should remain exact.");
expectEqual(top.default_activation_clamped_endpoint_count, 1_638, "Top-op default clamp count should remain exact.");
expectEqual(top.single_activation_clamped_endpoint_count, 1_638, "Top-op single clamp count should remain exact.");
expectEqual(top.witness_ledger_sha256, "50337b6c43a23e27a392b4fbcd5675461a77548584c3078c06aa97583f099d97", "Top-op witness ledger should remain stable.");
expectEqual(channel.channel_index, 767, "Worst top-op witness channel should remain stable.");
expectEqual(channel.term_count, 9, "Selected witness should retain every depthwise term.");
expectEqual(channel.positive_centered_weight_count, 0, "Selected witness should have no positive centered weights.");
expectEqual(channel.negative_centered_weight_count, 9, "Selected witness should have nine negative centered weights.");
expectEqual(channel.minimum.dot_product_decimal, "-25500", "Minimum witness dot product should remain exact.");
expectEqual(channel.minimum.bias_decimal, "3211", "Selected witness bias should remain exact.");
expectEqual(channel.minimum.post_bias_accumulator_decimal, "-22289", "Minimum post-bias accumulator should remain exact.");
expectEqual(channel.minimum.ideal_output_code, 0, "Minimum direct ideal output should be activation-clamped to zero.");
expectEqual(channel.minimum.default_output_code, 0, "Minimum default output should be activation-clamped to zero.");
expectEqual(channel.minimum.single_output_code, 0, "Minimum single-rounding output should be activation-clamped to zero.");
expectEqual(channel.maximum.dot_product_decimal, "0", "Maximum witness dot product should remain exact.");
expectEqual(channel.maximum.post_bias_accumulator_decimal, "3211", "Maximum post-bias accumulator should remain exact.");
expectEqual(channel.maximum.ideal_output_code, 137, "Maximum direct ideal output should remain exact.");
expectEqual(channel.maximum.default_output_code, 138, "Maximum default output should expose the one-code difference.");
expectEqual(channel.maximum.single_output_code, 137, "Maximum single-rounding output should match direct ideal.");
expectEqual(channel.maximum.build_mode_output_delta_codes, 1, "Selected endpoint should prove a one-code build-mode delta.");
expectEqual(channel.witness_pattern_sha256, "35445b96d727a11cf2bdeab9dc0df29210496ffb5cac395e59bff80c97f7e31c", "Selected witness pattern digest should remain stable.");
expect((witness.source_references || []).length === 6 && witness.source_references.every((source) => /^[a-f0-9]{64}$/.test(source.sha256)
  && source.url.includes(witness.source_commit)), "Every pinned source reference should carry immutable commit and digest identity.");

const selected = reconstructKernelChannel(analysis, bytes, 55, 767);
expectEqual(selected.pattern_bytes.byteLength, 113, "Selected binary pattern ledger should retain its exact layout.");
expectEqual(Array.from(selected.term_rows.filter((_, index) => index % 3 === 1)).join(","), Array(9).fill(255).join(","), "Minimum witness should assign input qmax to every negative term.");
expectEqual(Array.from(selected.term_rows.filter((_, index) => index % 3 === 2)).join(","), Array(9).fill(0).join(","), "Maximum witness should assign input qmin to every negative term.");
expectEqual(reconstructed.witness_assignment_count, witness.witness_assignment_count, "Independent reconstruction should preserve all assignments.");
const repeated = analyze_tflite_for_target(bytes, filename, "android_mid_a55").kernel_extremum_witness;
expectEqual(JSON.stringify(repeated), JSON.stringify(witness), "Kernel-witness JSON should be byte-for-byte deterministic.");

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
expect(report.includes("## Quantized Kernel Witness Lab (DERIVED EXACT SYNTHETIC RECEPTIVE-FIELD DOMAIN)"), "Engineering report should render the kernel-witness section.");
expect(report.includes("6,942,080") && report.includes("72,228") && report.includes("#055 DEPTHWISE_CONV_2D")
  && report.includes(channel.witness_pattern_sha256) && report.includes(top.witness_ledger_sha256), "Engineering report should retain exact portfolio, coordinate, and digest evidence.");
expect(report.includes("not a full-model input") && report.includes("TFLITE_SINGLE_ROUNDING not embedded"), "Engineering report should preserve synthetic-domain and build-flag boundaries.");

const staticAnalysis = buildStaticAnalysisExport(analysis);
const quantization = buildQuantizationEvidence(analysis, identity);
expectEqual(staticAnalysis.kernel_extremum_witness.schema, ANALYZER_METADATA.schemas.kernelExtremumWitness, "Static evidence should retain the kernel-witness schema.");
expectEqual(JSON.stringify(quantization.kernel_extremum_witness), JSON.stringify(staticAnalysis.kernel_extremum_witness), "Static and quantization evidence should retain the same witness ledger.");
const contract = buildQuantizationContractChecks(analysis).kernel_extremum_witness;
expectEqual(contract.status, "review", "Build-mode divergence should enter quantization design review.");
expectEqual(contract.checked_channels, 18_057, "Contract summary should expose every checked witness channel.");
expectEqual(contract.canonical_witness_assignments, 6_942_080, "Contract summary should expose exact witness assignments.");
expectEqual(contract.fixed_point_endpoint_evaluations, 72_228, "Contract summary should expose exact endpoint executions.");

const findings = buildFindingsRegister(analysis);
const finding = findings.find((item) => item.finding_id === "EA-QNT-0111");
expect(finding, "Pinned build-mode divergence should enter the authoritative action queue.");
expectEqual(finding?.technical_priority, "Informational", "Kernel endpoint evidence should remain a supporting record under the grouped rounding ABI action.");
expect(finding?.observation.includes("6,942,080") && finding?.observation.includes("137 / 138 / 137")
  && finding?.observation.includes(channel.witness_pattern_sha256), "Finding should preserve exact execution count and reproducible endpoint witness.");
expect(finding?.interpretation.includes("not an observed runtime mismatch") && finding?.recommendation.includes("TFLITE_SINGLE_ROUNDING"), "Finding should preserve runtime-observation and action boundaries.");

const mlBom = buildMlBomDocument(analysis, { hash: analysis.model_sha256, fileSizeBytes: bytes.byteLength, target: analysis.target_profile });
assertCompactMlBomProjection(mlBom, {
  expect,
  expectEqual,
  omittedProperties: [
    "deepbom:model:kernelExtremumWitnessSchema",
    "deepbom:model:kernelWitnessAssignments",
    "deepbom:model:kernelWitnessEndpointExecutions",
    "deepbom:model:kernelWitnessBuildModeDifferences",
    "deepbom:model:kernelWitnessSourceCommit",
  ],
  label: "Kernel-witness compact ML-BOM",
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
const witnessChecks = conformance.checks.filter((check) => check.id.startsWith("CF-WITNESS-"));
expectEqual(witnessChecks.length, 4, "Conformance should expose four kernel-witness cross-output checks.");
expect(witnessChecks.every((check) => check.status === "pass"), "Every kernel-witness conformance check should pass for the sample.");

const tamperedEndpoint = structuredClone(analysis);
tamperedEndpoint.kernel_extremum_witness.ops.find((row) => row.op_index === 55).worst_channel.maximum.default_output_code = 137;
expectThrows(() => validateKernelWitnessAnalysis(tamperedEndpoint, bytes), "default_output_code", "Validator should reject a tampered fixed-point endpoint.");
const tamperedCount = structuredClone(analysis);
tamperedCount.kernel_extremum_witness.witness_assignment_count += 1;
expectThrows(() => validateKernelWitnessAnalysis(tamperedCount, bytes), "witness_assignment_count", "Validator should reject a tampered assignment total.");
const tamperedLedger = structuredClone(analysis);
tamperedLedger.kernel_extremum_witness.ops.find((row) => row.op_index === 55).witness_ledger_sha256 = "0".repeat(64);
await expectRejects(() => validateKernelWitnessDigests(tamperedLedger, bytes), "ledger SHA-256 mismatch", "Digest validator should reject a tampered op ledger.");
const tamperedPattern = structuredClone(analysis);
tamperedPattern.kernel_extremum_witness.ops.find((row) => row.op_index === 55).top_channels[0].witness_pattern_sha256 = "0".repeat(64);
await expectRejects(() => validateKernelWitnessDigests(tamperedPattern, bytes), "pattern SHA-256 mismatch", "Digest validator should reject a tampered channel pattern.");

done("Kernel witness passed (53 ops, 18,057 channels, 6,942,080 canonical assignments, 72,228 pinned executions, reports, digests, and tamper rejection).");

async function expectRejects(action, fragment, message) {
  try {
    await action();
    expect(false, message);
  } catch (error) {
    expect(String(error?.message || error).includes(fragment), `${message} Expected ${JSON.stringify(fragment)}, got ${JSON.stringify(error?.message || String(error))}.`);
  }
}

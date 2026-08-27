import { readFileSync } from "node:fs";

import { analyze_tflite_for_target, initSync } from "../pkg/tflite_wasm_audit.js";
import {
  validateChannelVitalityAnalysis,
  validateChannelVitalityDigests,
} from "../web/lib/channel-vitality.js";
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

const { done, expect, expectEqual, expectThrows } = createCheck("Channel vitality check");
const filename = "mobilenet_v2_1.0_224_quant.tflite";
initSync({ module: readFileSync("pkg/tflite_wasm_audit_bg.wasm") });
const bytes = new Uint8Array(readFileSync(`web/samples/${filename}`));
const analysis = analyze_tflite_for_target(bytes, filename, "android_mid_a55");
const vitality = analysis.channel_vitality;

const reconstructed = validateChannelVitalityAnalysis(analysis, bytes);
await validateChannelVitalityDigests(analysis, bytes);
expectEqual(vitality.schema, "deepbom.channel_vitality.v1", "Channel-vitality schema should remain stable.");
expectEqual(vitality.method_version, "2026-07-17.1", "Channel-vitality method should remain stable.");
expectEqual(vitality.status, "assessed", "Every sample candidate op should be assessed.");
expectEqual(vitality.candidate_op_count, 53, "Every convolution-family op should enter vitality analysis.");
expectEqual(vitality.assessed_op_count, 53, "Every sample vitality op should be assessed.");
expectEqual(vitality.unassessed_op_count, 0, "No candidate should be silently omitted.");
expectEqual(vitality.assessed_channel_count, 18_057, "Every output channel should enter vitality analysis.");
expectEqual(vitality.fixed_point_assessed_channel_count, 18_057, "Every output channel should have four pinned endpoint outputs.");
expectEqual(vitality.constant_accumulator_channel_count, 11, "Constant accumulator channels should remain exact.");
expectEqual(vitality.post_bias_negative_locked_channel_count, 10, "Negative sign-locked channels should remain exact.");
expectEqual(vitality.post_bias_positive_locked_channel_count, 532, "Positive sign-locked channels should remain exact.");
expectEqual(vitality.post_bias_zero_containing_channel_count, 17_515, "Zero-containing channels should remain exact.");
expectEqual(vitality.default_constant_output_channel_count, 14, "Default constant-output channels should remain exact.");
expectEqual(vitality.single_constant_output_channel_count, 15, "Single-rounding constant-output channels should remain exact.");
expectEqual(vitality.dual_mode_constant_output_channel_count, 14, "Dual-mode constant-output channels should remain exact.");
expectEqual(vitality.nonconstant_accumulator_dual_mode_constant_channel_count, 3, "Variable-accumulator constant channels should remain exact.");
expectEqual(vitality.mode_dependent_constant_output_channel_count, 1, "Build-mode-dependent constant channels should remain exact.");
expectEqual(vitality.default_severely_constrained_channel_count, 17, "Default spans up to fifteen should remain exact.");
expectEqual(vitality.single_severely_constrained_channel_count, 17, "Single-rounding spans up to fifteen should remain exact.");
expectEqual(vitality.default_full_activation_span_channel_count, 16_074, "Default full-span channels should remain exact.");
expectEqual(vitality.single_full_activation_span_channel_count, 16_077, "Single-rounding full-span channels should remain exact.");
expectEqual(vitality.vitality_ranking_op_indices.slice(0, 15).join(","), "1,0,6,39,55,51,44,40,14,29,25,4,7,18,33", "Vitality ranking should remain deterministic.");
expectEqual(vitality.span_histogram.map((bin) => `${bin.label}:${bin.default_channel_count}/${bin.single_rounding_channel_count}`).join(","), "1:14/15,2-3:1/0,4-15:2/2,16-63:318/318,64-127:595/595,128-255:1053/1050,256:16074/16077", "Inclusive-span histograms should remain exact.");

const opOne = vitality.ops.find((row) => row.op_index === 1);
expectEqual(opOne.op_name, "DEPTHWISE_CONV_2D", "Highest-ranked vitality op identity should remain stable.");
expectEqual(opOne.constant_accumulator_channel_count, 1, "Highest-ranked op constant accumulator count should remain exact.");
expectEqual(opOne.nonconstant_accumulator_dual_mode_constant_channel_count, 3, "Highest-ranked op variable collapse count should remain exact.");
expectEqual(opOne.default_constant_channel_indices.join(","), "3,12,16,20", "Default constant coordinates should remain exact.");
expectEqual(opOne.single_constant_channel_indices.join(","), "3,12,16,20,26", "Single-rounding constant coordinates should remain exact.");
expectEqual(opOne.mode_dependent_constant_channel_indices.join(","), "26", "Mode-dependent coordinate should remain exact.");
expectEqual(opOne.vitality_ledger_sha256, "d4b09fabf80bb0b6ac30e0686097e220720d918f59e1bec1cf6e84a5c2e3498d", "Highest-ranked op vitality digest should remain stable.");
expectEqual(vitality.ops.find((row) => row.op_index === 0).vitality_ledger_sha256, "192e1778c4b6da56068e374f184c0e19cf3247474bd051e1bdc2dcbdcf66bf7b", "First Conv vitality digest should remain stable.");
expectEqual(vitality.ops.find((row) => row.op_index === 55).vitality_ledger_sha256, "0506a5c305d8c5c8aa2aa555114f85cdd4402027743fca88deeedfa8c75f8164", "Narrow nonconstant depthwise digest should remain stable.");

for (const channelIndex of [3, 12, 16]) {
  const channel = opOne.top_channels.find((item) => item.channel_index === channelIndex);
  expect(channel && channel.accumulator_span_decimal !== "0", `Channel ${channelIndex} should preserve a variable accumulator interval.`);
  expectEqual(channel?.default_constant_reason, "lower_code_clamp", `Channel ${channelIndex} default collapse cause should be exact.`);
  expectEqual(channel?.single_constant_reason, "lower_code_clamp", `Channel ${channelIndex} single collapse cause should be exact.`);
  expectEqual(`${channel?.default_minimum_output_code}:${channel?.default_maximum_output_code}:${channel?.single_minimum_output_code}:${channel?.single_maximum_output_code}`, "0:0:0:0", `Channel ${channelIndex} should remain pinned to output code zero.`);
}
const modeChannel = opOne.top_channels.find((item) => item.channel_index === 26);
expectEqual(modeChannel.post_bias_minimum_decimal, "-254", "Mode-dependent channel minimum accumulator should remain exact.");
expectEqual(modeChannel.post_bias_maximum_decimal, "1", "Mode-dependent channel maximum accumulator should remain exact.");
expectEqual(`${modeChannel.default_minimum_preclamp_code}:${modeChannel.default_maximum_preclamp_code}:${modeChannel.default_minimum_output_code}:${modeChannel.default_maximum_output_code}`, "-88:1:0:1", "Default mode-dependent path should remain exact.");
expectEqual(`${modeChannel.single_minimum_preclamp_code}:${modeChannel.single_maximum_preclamp_code}:${modeChannel.single_minimum_output_code}:${modeChannel.single_maximum_output_code}`, "-87:0:0:0", "Single-rounding mode-dependent path should remain exact.");
expectEqual(reconstructed.nonconstant_accumulator_dual_mode_constant_channel_count, vitality.nonconstant_accumulator_dual_mode_constant_channel_count, "Independent reconstruction should preserve variable collapse count.");

const repeated = analyze_tflite_for_target(bytes, filename, "android_mid_a55").channel_vitality;
expectEqual(JSON.stringify(repeated), JSON.stringify(vitality), "Channel-vitality JSON should be byte-for-byte deterministic.");

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
expect(report.includes("## Quantized Channel Vitality Atlas (DERIVED EXACT MONOTONE ENDPOINT PROOF)"), "Engineering report should render the vitality section.");
expect(report.includes("18,057") && report.includes("Variable accumulator but constant under both paths") && report.includes("#001 DEPTHWISE_CONV_2D / ch 26"), "Engineering report should preserve exact counts and coordinates.");
expect(report.includes(opOne.vitality_ledger_sha256) && report.includes("not an exact reachable-code count"), "Engineering report should preserve digest and reachability boundary.");

const staticAnalysis = buildStaticAnalysisExport(analysis);
const quantization = buildQuantizationEvidence(analysis, identity);
expectEqual(staticAnalysis.channel_vitality.schema, ANALYZER_METADATA.schemas.channelVitality, "Static evidence should retain the vitality schema.");
expectEqual(JSON.stringify(quantization.channel_vitality), JSON.stringify(staticAnalysis.channel_vitality), "Static and quantization evidence should retain one vitality ledger.");
const contract = buildQuantizationContractChecks(analysis).channel_vitality;
expectEqual(contract.status, "review", "Variable or mode-dependent collapse should enter design review.");
expectEqual(contract.checked_channels, 18_057, "Contract summary should expose every checked channel.");
expectEqual(contract.nonconstant_accumulator_dual_mode_constant_channels, 3, "Contract summary should expose variable collapse count.");

const findings = buildFindingsRegister(analysis);
const finding = findings.find((item) => item.finding_id === "EA-QNT-0112");
expect(finding, "Variable-accumulator constant channels should enter the authoritative action queue.");
expectEqual(finding?.technical_priority, "High", "Channel-vitality collapse should be High priority.");
expect(finding?.observation.includes("ch 3") && finding?.observation.includes("ch 12") && finding?.observation.includes("ch 16") && finding?.observation.includes("ch 26"), "Finding should preserve exact collapse coordinates.");
expect(finding?.observation.includes(opOne.vitality_ledger_sha256) && finding?.interpretation.includes("not observed runtime activation"), "Finding should preserve digest and runtime boundary.");

const mlBom = buildMlBomDocument(analysis, { hash: analysis.model_sha256, fileSizeBytes: bytes.byteLength, target: analysis.target_profile });
assertCompactMlBomProjection(mlBom, {
  expect,
  expectEqual,
  omittedProperties: [
    "deepbom:model:channelVitalitySchema",
    "deepbom:model:channelVitalityAssessedChannels",
    "deepbom:model:channelVitalityVariableAccumulatorConstantChannels",
    "deepbom:model:channelVitalityModeDependentConstantChannels",
    "deepbom:model:channelVitalitySourceCommit",
  ],
  label: "Channel-vitality compact ML-BOM",
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
const vitalityChecks = conformance.checks.filter((check) => check.id.startsWith("CF-VITALITY-"));
expectEqual(vitalityChecks.length, 4, "Conformance should expose four channel-vitality cross-output checks.");
expect(vitalityChecks.every((check) => check.status === "pass"), "Every channel-vitality conformance check should pass for the sample.");
expect(visualPngSpecs({ analysis, filename }).some(([path]) => path === "visuals/channel_vitality.png"), "Engineering bundle should include the vitality PNG.");

const tamperedCount = structuredClone(analysis);
tamperedCount.channel_vitality.nonconstant_accumulator_dual_mode_constant_channel_count += 1;
expectThrows(() => validateChannelVitalityAnalysis(tamperedCount, bytes), "nonconstant_accumulator_dual_mode_constant_channel_count", "Validator should reject a tampered collapse total.");
const tamperedArray = structuredClone(analysis);
tamperedArray.channel_vitality.ops.find((row) => row.op_index === 1).single_maximum_output_codes[26] = 1;
expectThrows(() => validateChannelVitalityAnalysis(tamperedArray, bytes), "single_maximum_output_codes", "Validator should reject a tampered endpoint array.");
const tamperedLedger = structuredClone(analysis);
tamperedLedger.channel_vitality.ops.find((row) => row.op_index === 1).vitality_ledger_sha256 = "0".repeat(64);
await expectRejects(() => validateChannelVitalityDigests(tamperedLedger, bytes), "ledger SHA-256 mismatch", "Digest validator should reject a tampered vitality ledger.");

done("Channel vitality passed (53 ops, 18,057 channels, 14 dual-mode constants, 3 variable-accumulator constants, 1 build-mode-dependent collapse, reports, digests, and tamper rejection).");

async function expectRejects(action, fragment, message) {
  try {
    await action();
    expect(false, message);
  } catch (error) {
    expect(String(error?.message || error).includes(fragment), `${message} Expected ${JSON.stringify(fragment)}, got ${JSON.stringify(error?.message || String(error))}.`);
  }
}

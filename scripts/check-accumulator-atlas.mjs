import { readFileSync } from "node:fs";

import { analyze_tflite_for_target, initSync } from "../pkg/tflite_wasm_audit.js";
import {
  validateAccumulatorAtlas,
  validateAccumulatorAtlasDigests,
} from "../web/lib/accumulator-atlas.js";
import { buildConformanceReport } from "../web/lib/report-conformance.js";
import { buildEngineeringReport } from "../web/lib/report-engineering.js";
import { buildQuantizationEvidence, buildStaticAnalysisExport } from "../web/lib/report-evidence.js";
import { buildFindingsRegister } from "../web/lib/report-findings.js";
import { buildMlBomDocument } from "../web/lib/report-mlbom.js";
import { ANALYZER_METADATA } from "../web/lib/report-metadata.js";
import { buildQuantizationContractChecks } from "../web/lib/report-quantization-contracts.js";
import { createCheck } from "./check-assert.mjs";
import { assertCompactMlBomProjection } from "./compact-mlbom-assert.mjs";

const { done, expect, expectEqual, expectThrows } = createCheck("Accumulator atlas check");
const filename = "mobilenet_v2_1.0_224_quant.tflite";
initSync({ module: readFileSync("pkg/tflite_wasm_audit_bg.wasm") });
const bytes = new Uint8Array(readFileSync(`web/samples/${filename}`));
const analysis = analyze_tflite_for_target(bytes, filename, "android_mid_a55");
const atlas = analysis.accumulator_atlas;

expect(validateAccumulatorAtlas(analysis, bytes), "Independent BigInt reconstruction should validate the complete accumulator atlas.");
await expectAsyncSuccess(() => validateAccumulatorAtlasDigests(analysis, bytes), "Every per-op channel ledger digest should validate.");
expectEqual(atlas.schema, "deepbom.accumulator_atlas.v1.3", "Accumulator schema should be stable.");
expectEqual(atlas.method_version, "2026-07-30.4", "Accumulator method should be stable.");
expectEqual(atlas.status, "assessed", "All sample accumulator candidates should be assessed.");
expectEqual(atlas.candidate_op_count, 53, "MobileNetV2 should expose 53 Conv/Depthwise/FC accumulator candidates.");
expectEqual(atlas.assessed_op_count, 53, "All candidate accumulators should be exactly assessed.");
expectEqual(atlas.unassessed_op_count, 0, "No candidate should be silently omitted.");
expectEqual(atlas.assessed_channel_count, 18_057, "Every output-channel envelope should be retained.");
expectEqual(atlas.int32_safe_channel_count, 18_057, "All sample channels should fit INT32.");
expectEqual(atlas.int32_overflow_channel_count, 0, "The sample should not claim an INT32 overflow.");
expectEqual(atlas.bias_half_range_exceedance_channel_count, 0, "The sample should not cross the source bias half-range reference.");
expectEqual(atlas.bias_half_range_guard_adjacent_channel_count, 0, "The sample should not need the float32 guard-adjacent class.");
expectEqual(atlas.bias_half_range_material_exceedance_channel_count, 0, "The sample should not materially exceed the bias half-range reference.");
expectEqual(atlas.maximum_absolute_accumulator_decimal, "4482645", "The exact global accumulator maximum should remain stable.");
expectEqual(atlas.maximum_int32_ratio, 0.0020873942422156198, "The exact INT32 utilization should remain stable.");
expectEqual(atlas.maximum_required_signed_bits, 24, "The sample should require at most 24 signed bits.");
expectEqual(atlas.minimum_int32_headroom_bits, 8, "The sample should retain eight INT32 headroom bits.");
expectEqual(atlas.headroom_ranking_op_indices.slice(0, 10).join(","), "63,52,60,41,56,49,30,26,45,38", "Headroom ranking should remain deterministic.");
expectEqual(atlas.required_signed_bits_histogram.reduce((total, count) => total + count, 0), 18_057, "Global signed-bit histogram should conserve every assessed channel.");

const top = atlas.ops.find((row) => row.op_index === 63);
expectEqual(top.assessed_channel_count, 1_001, "Top op should retain every classifier output channel.");
expectEqual(top.accumulation_terms_per_channel, 1_280, "Top op fan-in should remain exact.");
expectEqual(top.metadata_only_magnitude_bound_decimal, "46379821", "Metadata-only comparison bound should remain reproducible.");
expect(Math.abs(top.exact_tightening_factor - 10.346530006279774) < 1e-12, "Stored-weight solving should tighten the metadata-only bound by 10.34653x.");
expectEqual(top.worst_channel.channel_index, 900, "Worst classifier channel should remain deterministic.");
expectEqual(top.worst_channel.positive_centered_weight_sum_decimal, "17579", "Worst-channel positive weight sum should remain exact.");
expectEqual(top.worst_channel.negative_centered_weight_sum_decimal, "-15672", "Worst-channel negative weight sum should remain exact.");
expectEqual(top.worst_channel.bias_decimal, "-2947", "Worst-channel stored bias should remain exact.");
expectEqual(top.worst_channel.accumulator_envelope_min_decimal, "-3999307", "Worst-channel lower envelope should remain exact.");
expectEqual(top.worst_channel.accumulator_envelope_max_decimal, "4482645", "Worst-channel upper envelope should remain exact.");
expectEqual(top.worst_channel.post_bias_min_decimal, "-3999307", "Worst-channel post-bias minimum should remain exact.");
expectEqual(top.worst_channel.post_bias_max_decimal, "4479698", "Worst-channel post-bias maximum should remain exact.");
expectEqual(top.worst_channel.required_signed_bits, 24, "Worst-channel bit width should remain exact.");
expectEqual(top.channel_ledger_sha256, "cdfa45b4d4051fe7e705eb5b0e35ff03809d95c9c0e8b693ea36ef78d6010ba6", "Top-op channel ledger digest should remain stable.");

const repeated = analyze_tflite_for_target(bytes, filename, "android_mid_a55").accumulator_atlas;
expectEqual(JSON.stringify(repeated), JSON.stringify(atlas), "Accumulator JSON should be byte-for-byte deterministic.");

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
expect(report.includes("## Accumulator Headroom Lab (DERIVED EXACT INTEGER DOMAIN)"), "Engineering report should render the exact accumulator section.");
expect(report.includes("deepbom.accumulator_atlas.v1.3 / 2026-07-30.4")
  && report.includes("#063 CONV_2D")
  && report.includes("4,482,645")
  && report.includes("24 signed bits / 8 bits")
  && report.includes("Bias half-range strict / float32-adjacent / material")
  && report.includes("10.35x")
  && report.includes(top.channel_ledger_sha256), "Engineering report should retain schema, exact arithmetic, ranking, and digest evidence.");
expect(report.includes("not an observed activation distribution") && report.includes(atlas.source_commit), "Engineering report should preserve source provenance and interpretation limits.");
expect(report.includes("### Interval Definition Cross-Check")
  && report.includes("INT32 accumulation envelope")
  && report.includes("Post-bias requantization interval")
  && report.includes("An envelope maximum of 0 and a post-bias maximum of -19"), "Engineering report should explicitly separate the headroom envelope from the signed requantization interval.");

const quantization = buildQuantizationEvidence(analysis, identity);
expectEqual(quantization.accumulator_headroom_atlas.schema, ANALYZER_METADATA.schemas.accumulatorAtlas, "Quantization evidence should retain the accumulator schema.");
expectEqual(JSON.stringify(quantization.accumulator_headroom_atlas), JSON.stringify(buildStaticAnalysisExport(analysis).accumulator_atlas), "Static and quantization evidence should retain the same atlas ledger.");
const contracts = buildQuantizationContractChecks(analysis);
expectEqual(contracts.accumulator_bound.bound_class, "exact_stored_weight_channel_integer_domain", "TFLite accumulator contract should use the exact stored-weight atlas.");
expectEqual(contracts.accumulator_bound.checked_channels, 18_057, "Accumulator contract should expose every checked channel.");
expectEqual(contracts.accumulator_bound.maximum_int32_ratio, atlas.maximum_int32_ratio, "Accumulator contract should not fall back to the coarse metadata ratio.");
const mlBom = buildMlBomDocument(analysis, { hash: analysis.model_sha256, fileSizeBytes: bytes.byteLength, target: analysis.target_profile });
assertCompactMlBomProjection(mlBom, {
  expect,
  expectEqual,
  omittedProperties: [
    "deepbom:model:accumulatorAtlasSchema",
    "deepbom:model:accumulatorAssessedChannels",
    "deepbom:model:accumulatorMaximumAbsoluteDecimal",
    "deepbom:model:accumulatorMinimumInt32HeadroomBits",
  ],
  label: "Accumulator compact ML-BOM",
});

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
const accumulatorChecks = conformance.checks.filter((check) => check.id.startsWith("CF-ACCUMULATOR-"));
expectEqual(accumulatorChecks.length, 3, "Conformance should expose three accumulator cross-output checks.");
expect(accumulatorChecks.every((check) => check.status === "pass"), "All accumulator conformance checks should pass for the sample.");
expect(!buildFindingsRegister(analysis).some((finding) => finding.finding_id === "EA-QNT-0106"), "A safe exact envelope should not create an overflow action.");

const tamperedMin = structuredClone(analysis);
tamperedMin.accumulator_atlas.ops.find((row) => row.op_index === 63).channel_accumulator_envelope_min_decimals[900] = "-3999306";
expectThrows(() => validateAccumulatorAtlas(tamperedMin, bytes), "channel_accumulator_envelope_min_decimals", "Validator should reject a tampered channel minimum.");
const tamperedPostBias = structuredClone(analysis);
tamperedPostBias.accumulator_atlas.ops.find((row) => row.op_index === 63).channel_post_bias_max_decimals[900] = "4479699";
expectThrows(() => validateAccumulatorAtlas(tamperedPostBias, bytes), "channel_post_bias_max_decimals", "Validator should reject a tampered post-bias accumulator bound.");
const tamperedBits = structuredClone(analysis);
tamperedBits.accumulator_atlas.ops.find((row) => row.op_index === 63).channel_required_signed_bits[900] = 25;
expectThrows(() => validateAccumulatorAtlas(tamperedBits, bytes), "channel_required_signed_bits", "Validator should reject a tampered channel bit width.");
const tamperedHistogram = structuredClone(analysis);
tamperedHistogram.accumulator_atlas.required_signed_bits_histogram[24] += 1;
expectThrows(() => validateAccumulatorAtlas(tamperedHistogram, bytes), "global bit histogram", "Validator should reject a tampered aggregate histogram.");
const tamperedDigest = structuredClone(analysis);
tamperedDigest.accumulator_atlas.ops.find((row) => row.op_index === 63).channel_ledger_sha256 = "0".repeat(64);
await expectAsyncFailure(() => validateAccumulatorAtlasDigests(tamperedDigest, bytes), "ledger SHA-256 mismatch", "Digest validator should reject a tampered channel ledger hash.");

const tamperedConformanceAnalysis = structuredClone(analysis);
tamperedConformanceAnalysis.accumulator_atlas.assessed_channel_count += 1;
const tamperedConformance = buildConformanceReport({
  analysis: tamperedConformanceAnalysis,
  staticAnalysis: buildStaticAnalysisExport(tamperedConformanceAnalysis),
  quantization: buildQuantizationEvidence(tamperedConformanceAnalysis, identity),
  findingsRegister: { authoritative_action_source: "findings", raw_analyzer_signals: [], findings: [] },
  runtimeResults: {},
  securityPosture: { execution_integrity: {} },
  mlBomDocument: {},
  engineeringReport: report,
});
expectEqual(tamperedConformance.checks.find((check) => check.id === "CF-ACCUMULATOR-002")?.status, "fail", "Conformance should reject an inconsistent channel aggregate.");

const overflowAnalysis = structuredClone(analysis);
const overflowAtlas = overflowAnalysis.accumulator_atlas;
const overflowRow = overflowAtlas.ops.find((row) => row.op_index === 63);
overflowAtlas.int32_safe_channel_count -= 1;
overflowAtlas.int32_overflow_channel_count = 1;
overflowAtlas.overflow_op_count = 1;
overflowAtlas.maximum_absolute_accumulator_decimal = "2147483648";
overflowAtlas.maximum_int32_ratio = 2147483648 / 2147483647;
overflowAtlas.maximum_required_signed_bits = 33;
overflowAtlas.minimum_int32_headroom_bits = -1;
overflowRow.int32_safe_channel_count -= 1;
overflowRow.int32_overflow_channel_count = 1;
overflowRow.overflow_channel_indices = [900];
overflowRow.maximum_absolute_accumulator_decimal = "2147483648";
overflowRow.maximum_int32_ratio = 2147483648 / 2147483647;
overflowRow.maximum_required_signed_bits = 33;
overflowRow.minimum_int32_headroom_bits = -1;
const overflowFinding = buildFindingsRegister(overflowAnalysis).find((finding) => finding.finding_id === "EA-QNT-0106");
expect(overflowFinding, "An exact INT32 overflow should enter the authoritative action queue.");
expectEqual(overflowFinding?.technical_priority, "High", "Accumulator overflow should be High priority.");
expect(overflowFinding?.evidence_json_pointers.includes("/evidence/static_analysis/accumulator_atlas"), "Overflow finding should point to the exact atlas evidence.");
expect(overflowFinding?.observation.includes("1 output channel") && overflowFinding?.observation.includes("33 signed bits")
  && overflowFinding?.observation.includes("900"), "Overflow action should carry exact channel and width evidence.");

done("Accumulator atlas passed (53 ops, 18,057 exact channel envelopes, digest/report binding, overflow action, and tamper rejection).");

async function expectAsyncSuccess(run, label) {
  try {
    await run();
    expect(true, label);
  } catch (error) {
    expect(false, `${label} ${error.message}`);
  }
}

async function expectAsyncFailure(run, messagePart, label) {
  try {
    await run();
    expect(false, `${label} Expected an error containing ${JSON.stringify(messagePart)}.`);
  } catch (error) {
    expect(String(error?.message || error).includes(messagePart), `${label} Expected ${JSON.stringify(messagePart)}, got ${JSON.stringify(error?.message || String(error))}.`);
  }
}

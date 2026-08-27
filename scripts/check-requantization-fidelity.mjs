import { readFileSync } from "node:fs";

import { analyze_tflite_for_target, initSync } from "../pkg/tflite_wasm_audit.js";
import {
  validateRequantizationFidelity,
  validateRequantizationFidelityDigests,
} from "../web/lib/requantization-fidelity.js";
import { buildConformanceReport } from "../web/lib/report-conformance.js";
import { buildEngineeringReport } from "../web/lib/report-engineering.js";
import { buildQuantizationEvidence, buildStaticAnalysisExport } from "../web/lib/report-evidence.js";
import { buildFindingsRegister } from "../web/lib/report-findings.js";
import { buildMlBomDocument } from "../web/lib/report-mlbom.js";
import { ANALYZER_METADATA } from "../web/lib/report-metadata.js";
import { buildQuantizationContractChecks } from "../web/lib/report-quantization-contracts.js";
import { createCheck } from "./check-assert.mjs";
import { assertCompactMlBomProjection } from "./compact-mlbom-assert.mjs";

const { done, expect, expectEqual, expectThrows } = createCheck("Requantization fidelity check");
const filename = "mobilenet_v2_1.0_224_quant.tflite";
initSync({ module: readFileSync("pkg/tflite_wasm_audit_bg.wasm") });
const bytes = new Uint8Array(readFileSync(`web/samples/${filename}`));
const analysis = analyze_tflite_for_target(bytes, filename, "android_mid_a55");
const fidelity = analysis.requantization_fidelity;

expect(validateRequantizationFidelity(analysis, bytes), "Independent browser arithmetic should reproduce every requantization channel.");
await expectAsyncSuccess(() => validateRequantizationFidelityDigests(analysis, bytes), "Every requantization ledger SHA-256 should validate.");
expectEqual(fidelity.schema, "deepbom.requantization_fidelity.v1", "Requantization schema should be stable.");
expectEqual(fidelity.method_version, "2026-07-17.2", "Requantization method should be stable.");
expectEqual(fidelity.status, "assessed", "The real quantized sample should be fully assessed.");
expectEqual(fidelity.candidate_op_count, 53, "Every accumulator candidate should enter requantization analysis.");
expectEqual(fidelity.assessed_op_count, 53, "Every sample requantization op should be assessed.");
expectEqual(fidelity.unassessed_op_count, 0, "No sample requantization op should be silently omitted.");
expectEqual(fidelity.assessed_channel_count, 18_057, "Every output channel should retain a multiplier ledger.");
expectEqual(fidelity.fixed_point_bound_channel_count, 18_057, "Every channel should receive a conservative fixed-point bound.");
expectEqual(fidelity.default_pre_shift_overflow_channel_count, 0, "No sample channel should violate default positive pre-shift INT32 safety.");
expectEqual(fidelity.single_rounding_encoding_divergence_channel_count, 0, "The sample should not hit the single-rounding shift saturation branch.");
expectEqual(fidelity.half_code_encoding_drift_channel_count, 0, "Encoding-only drift should stay below half a code.");
expectEqual(fidelity.one_code_encoding_drift_channel_count, 0, "Encoding-only drift should stay below one code.");
expectEqual(fidelity.minimum_shift, -11, "Minimum Q0.31 exponent should be stable.");
expectEqual(fidelity.maximum_shift, -1, "Maximum Q0.31 exponent should be stable.");
expectEqual(fidelity.shift_histogram.reduce((total, bin) => total + bin.channel_count, 0), 18_057, "Shift histogram should conserve every channel.");
expectEqual(fidelity.maximum_relative_multiplier_error, 3.38388710137892e-10, "Maximum relative multiplier error should remain deterministic.");
expectEqual(fidelity.maximum_multiplier_error_ppm, 0.00033838871013789197, "Maximum multiplier ppm should remain deterministic.");
expectEqual(fidelity.maximum_encoding_drift_bound_codes, 0.0000044965856487687306, "Maximum encoding drift should remain deterministic.");
expectEqual(fidelity.maximum_default_double_rounding_bound_codes, 0.75, "Default double-rounding bound should remain source-derived.");
expectEqual(fidelity.maximum_single_rounding_bound_codes, 0.5000044965856487, "Single-rounding bound should remain source-derived.");
expectEqual(fidelity.fidelity_ranking_op_indices.slice(0, 10).join(","), "61,3,58,39,56,50,6,10,26,52", "Fidelity ranking should remain deterministic.");

const top = fidelity.ops.find((row) => row.op_index === 61);
expectEqual(top.assessed_channel_count, 1_280, "Top op should retain all output channels.");
expectEqual(top.minimum_shift, -5, "Top op minimum shift should be stable.");
expectEqual(top.maximum_shift, -5, "Top op maximum shift should be stable.");
expectEqual(top.worst_channel.channel_index, 1_001, "Worst encoding witness channel should be stable.");
expectEqual(top.worst_channel.post_bias_accumulator_min_decimal, "-916457", "Worst encoding witness should retain the post-bias minimum.");
expectEqual(top.worst_channel.post_bias_accumulator_max_decimal, "900163", "Worst encoding witness should retain the post-bias maximum.");
expectEqual(top.worst_channel.maximum_absolute_post_bias_accumulator_decimal, "916457", "Worst encoding witness should bind the exact post-bias domain.");
expectEqual(top.worst_channel.quantized_multiplier, 1_764_866_200, "Worst encoding witness Q0.31 multiplier should be stable.");
expectEqual(top.worst_channel.shift, -5, "Worst encoding witness shift should be stable.");
expectEqual(top.worst_channel.real_multiplier, 0.025682183336716426, "Worst encoding witness real multiplier should be stable.");
expectEqual(top.worst_channel.represented_multiplier, 0.025682183331809938, "Worst encoding witness represented multiplier should be stable.");
expectEqual(top.worst_channel.encoding_drift_bound_codes, fidelity.maximum_encoding_drift_bound_codes, "Worst encoding witness should explain the global drift maximum.");
expectEqual(top.channel_ledger_sha256, "a98c633aa422010196ec28e09605311550b48aaa94bf81f1f3c79defbe1b3778", "Top-op requantization ledger digest should remain stable.");
expect(fidelity.source_references.every((source) => /^[a-f0-9]{64}$/.test(source.sha256)
  && source.url.includes(fidelity.source_commit)), "Every source reference should carry pinned commit and digest identity.");

const repeated = analyze_tflite_for_target(bytes, filename, "android_mid_a55").requantization_fidelity;
expectEqual(JSON.stringify(repeated), JSON.stringify(fidelity), "Requantization JSON should be byte-for-byte deterministic.");

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
expect(report.includes("## Requantization Fidelity Lab (DERIVED PINNED Q0.31 DOMAIN)"), "Engineering report should render the requantization section.");
expect(report.includes("deepbom.requantization_fidelity.v1 / 2026-07-17.2")
  && report.includes("#061 CONV_2D")
  && report.includes("4.496586e-6")
  && report.includes("0.750000000 / 0.500004497")
  && report.includes(top.channel_ledger_sha256), "Engineering report should retain schema, ranking, bounds, and digest evidence.");
expect(report.includes("TFLITE_SINGLE_ROUNDING compile flag") && report.includes(fidelity.source_commit), "Engineering report should preserve build/runtime boundaries and pinned source identity.");
expect(report.includes("Shift extrema coordinates")
  && report.includes("min -11 at #063 CONV_2D/ch0")
  && report.includes("max -1 at #001 DEPTHWISE_CONV_2D/ch0"), "Engineering report should bind both shift extrema to reproducible op/channel coordinates.");

const staticAnalysis = buildStaticAnalysisExport(analysis);
const quantization = buildQuantizationEvidence(analysis, identity);
expectEqual(staticAnalysis.requantization_fidelity.schema, ANALYZER_METADATA.schemas.requantizationFidelity, "Static evidence should retain the requantization schema.");
expectEqual(JSON.stringify(quantization.requantization_fidelity), JSON.stringify(staticAnalysis.requantization_fidelity), "Static and quantization evidence should retain the same ledger.");
const contracts = buildQuantizationContractChecks(analysis);
expectEqual(contracts.requantization_fidelity.status, "pass", "Sample requantization contract should pass without build-dependent risk.");
expectEqual(contracts.requantization_fidelity.checked_channels, 18_057, "Contract summary should expose every checked channel.");
expectEqual(contracts.requantization_fidelity.maximum_encoding_drift_bound_codes, fidelity.maximum_encoding_drift_bound_codes, "Contract summary should preserve the exact encoding drift.");

const mlBom = buildMlBomDocument(analysis, { hash: analysis.model_sha256, fileSizeBytes: bytes.byteLength, target: analysis.target_profile });
assertCompactMlBomProjection(mlBom, {
  expect,
  expectEqual,
  omittedProperties: [
    "deepbom:model:requantizationFidelitySchema",
    "deepbom:model:requantizationAssessedChannels",
    "deepbom:model:requantizationMaximumEncodingDriftCodes",
  ],
  label: "Requantization compact ML-BOM",
});

const conformance = buildConformanceReport({
  analysis,
  staticAnalysis,
  quantization,
  findingsRegister: { authoritative_action_source: "findings", raw_analyzer_signals: [], findings: [] },
  runtimeResults: {},
  securityPosture: { execution_integrity: {} },
  mlBomDocument: {},
  engineeringReport: report,
});
const requantizationChecks = conformance.checks.filter((check) => check.id.startsWith("CF-REQUANT-"));
expectEqual(requantizationChecks.length, 3, "Conformance should expose three requantization cross-output checks.");
expect(requantizationChecks.every((check) => check.status === "pass"), "All requantization conformance checks should pass.");
expect(!buildFindingsRegister(analysis).some((finding) => finding.finding_id === "EA-QNT-0107"), "A build-mode-invariant, pre-shift-safe artifact should not create a requantization action.");

const tamperedMultiplier = structuredClone(analysis);
tamperedMultiplier.requantization_fidelity.ops.find((row) => row.op_index === 61).channel_quantized_multipliers[1_001] += 1;
expectThrows(() => validateRequantizationFidelity(tamperedMultiplier, bytes), "channel_quantized_multipliers", "Validator should reject a tampered Q0.31 multiplier.");
const tamperedBound = structuredClone(analysis);
tamperedBound.requantization_fidelity.ops.find((row) => row.op_index === 61).channel_default_double_rounding_bound_codes[1_001] += 0.01;
expectThrows(() => validateRequantizationFidelity(tamperedBound, bytes), "channel_default_double_rounding_bound_codes", "Validator should reject a tampered execution bound.");
const tamperedDigest = structuredClone(analysis);
tamperedDigest.requantization_fidelity.ops.find((row) => row.op_index === 61).channel_ledger_sha256 = "0".repeat(64);
await expectAsyncFailure(() => validateRequantizationFidelityDigests(tamperedDigest, bytes), "ledger SHA-256 mismatch", "Digest validator should reject a tampered ledger hash.");

const riskyAnalysis = structuredClone(analysis);
riskyAnalysis.requantization_fidelity.default_pre_shift_overflow_channel_count = 1;
riskyAnalysis.requantization_fidelity.fixed_point_bound_channel_count -= 1;
const riskyRow = riskyAnalysis.requantization_fidelity.ops.find((row) => row.op_index === 61);
riskyRow.default_pre_shift_overflow_channel_count = 1;
riskyRow.fixed_point_bound_channel_count -= 1;
const riskyFinding = buildFindingsRegister(riskyAnalysis).find((finding) => finding.finding_id === "EA-QNT-0107");
expect(riskyFinding, "A default pre-shift overflow should enter the authoritative action queue.");
expectEqual(riskyFinding?.technical_priority, "High", "Default pre-shift overflow should be High priority.");
expect(riskyFinding?.observation.includes("1 channel") && riskyFinding?.evidence_json_pointers.includes("/evidence/static_analysis/requantization_fidelity"), "Requantization action should carry exact count and evidence pointer.");

done("Requantization fidelity passed (53 ops, 18,057 independently reconstructed channels, pinned Q0.31 bounds, reports, and tamper rejection).");

async function expectAsyncSuccess(run, label) {
  try { await run(); expect(true, label); } catch (error) { expect(false, `${label} ${error.message}`); }
}

async function expectAsyncFailure(run, messagePart, label) {
  try {
    await run();
    expect(false, `${label} Expected an error containing ${JSON.stringify(messagePart)}.`);
  } catch (error) {
    expect(String(error?.message || error).includes(messagePart), `${label} Expected ${JSON.stringify(messagePart)}, got ${JSON.stringify(error?.message || String(error))}.`);
  }
}

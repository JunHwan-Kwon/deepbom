import { readFileSync } from "node:fs";

import { analyze_tflite_for_target, initSync } from "../pkg/tflite_wasm_audit.js";
import {
  reconstructRoundingEquivalenceChannel,
  validateRoundingEquivalenceAgainstReconstruction,
  validateRoundingEquivalenceDigests,
  validateRoundingEquivalenceDigestsAgainstReconstruction,
} from "../web/lib/rounding-equivalence.js";
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

const { done, expect, expectEqual, expectThrows } = createCheck("Rounding equivalence check");
const filename = "mobilenet_v2_1.0_224_quant.tflite";
initSync({ module: readFileSync("pkg/tflite_wasm_audit_bg.wasm") });
const bytes = new Uint8Array(readFileSync(`web/samples/${filename}`));
const analysis = analyze_tflite_for_target(bytes, filename, "android_mid_a55");
const equivalence = analysis.rounding_equivalence;

const reconstructed = await validateRoundingEquivalenceDigests(analysis, bytes);
expectEqual(equivalence.schema, "deepbom.rounding_equivalence.v1", "Rounding-equivalence schema should remain stable.");
expectEqual(equivalence.method_version, "2026-07-17.1", "Rounding-equivalence method should remain stable.");
expectEqual(equivalence.status, "assessed", "Every sample candidate op should be assessed.");
expectEqual(equivalence.candidate_op_count, 53, "Every convolution-family op should enter equivalence analysis.");
expectEqual(equivalence.assessed_op_count, 53, "Every sample equivalence op should be assessed.");
expectEqual(equivalence.unassessed_op_count, 0, "No candidate should be silently omitted.");
expectEqual(equivalence.assessed_channel_count, 18_057, "Every fixed-point channel should receive an interval certificate.");
expectEqual(equivalence.equivalent_channel_count, 974, "Complete interval-equivalent channel count should remain exact.");
expectEqual(equivalence.divergent_channel_count, 17_083, "Build-mode-divergent channel count should remain exact.");
expectEqual(equivalence.divergent_op_count, 52, "Build-mode-divergent op count should remain exact.");
expectEqual(equivalence.interval_state_count_decimal, "13933008957", "Complete interval-state portfolio should remain exact.");
expectEqual(equivalence.divergent_state_count_decimal, "2874544", "Divergent interval-state portfolio should remain exact.");
expectEqual(equivalence.default_lower_state_count_decimal, "162765", "Default-lower state count should remain exact.");
expectEqual(equivalence.default_higher_state_count_decimal, "2711779", "Default-higher state count should remain exact.");
expectEqual(equivalence.maximum_absolute_output_delta, 1, "Maximum build-mode output delta should remain one code.");
expectEqual(equivalence.pair_segment_count, 7_280_734, "Exact ordered-pair segment count should remain stable.");
expectEqual(equivalence.divergent_region_count, 2_874_544, "Exact divergent-region count should remain stable.");
expect(Math.abs(equivalence.divergent_state_ratio - 0.00020631178870776634) < 1e-18, "Global divergent-state ratio should remain exact within binary64 serialization.");
expectEqual(equivalence.equivalence_ranking_op_indices.slice(0, 16).join(","), "7,44,40,14,59,18,55,33,3,25,29,1,37,11,4,22", "Exposure ranking should remain deterministic.");
expectEqual(equivalence.divergence_histogram.map((bin) => `${bin.label}:${bin.channel_count}/${bin.interval_state_count_decimal}/${bin.divergent_state_count_decimal}`).join(","), "0%:974/25891124/0,(0,0.01%]:2804/10472636309/438290,(0.01%,0.1%]:6483/3113112528/1123946,(0.1%,1%]:4789/287269939/744859,(1%,10%]:3002/34091657/566561,(10%,50%]:5/7400/888,(50%,100%]:0/0/0", "Divergence histogram should conserve exact channel and state counts.");

const opSeven = equivalence.ops.find((row) => row.op_index === 7);
const channel = opSeven.top_channels[0];
expectEqual(opSeven.op_name, "DEPTHWISE_CONV_2D", "Highest-ranked op identity should remain stable.");
expectEqual(`${opSeven.equivalent_channel_count}:${opSeven.divergent_channel_count}`, "0:144", "Highest-ranked op channel classification should remain exact.");
expectEqual(`${opSeven.interval_state_count_decimal}:${opSeven.divergent_state_count_decimal}`, "1112199:26621", "Highest-ranked op state ledger should remain exact.");
expectEqual(opSeven.maximum_pair_segment_count, 447, "Highest-ranked op segment bound should remain exact.");
expectEqual(opSeven.maximum_divergent_region_count, 191, "Highest-ranked op region count should remain exact.");
expectEqual(opSeven.equivalence_ledger_sha256, "6b42280ab896789a75ce996634eb5251c01c8fbd554216f1cdaddbf3ee62e9ab", "Highest-ranked certificate digest should remain stable.");
expectEqual(channel.channel_index, 37, "Highest-ranked channel coordinate should remain stable.");
expectEqual(`${channel.post_bias_minimum_decimal}:${channel.post_bias_maximum_decimal}:${channel.interval_state_count_decimal}`, "-417:1623:2041", "Selected accumulator interval should remain exact.");
expectEqual(`${channel.divergent_state_count_decimal}:${channel.pair_segment_count}:${channel.divergent_region_count}`, "191:447:191", "Selected exact partition should remain stable.");
expectEqual(`${channel.first_divergent_accumulator_decimal}:${channel.first_default_output_code}:${channel.first_single_output_code}`, "14:3:2", "First counterexample should remain exact.");
expectEqual(`${channel.last_divergent_accumulator_decimal}:${channel.last_default_output_code}:${channel.last_single_output_code}`, "1498:255:254", "Last counterexample should remain exact.");
expectEqual(`${channel.default_quantized_multiplier}:${channel.default_shift}:${channel.single_quantized_multiplier}:${channel.single_shift}`, "1458735232:-2:1458735232:-2", "Pinned encodings should remain exact.");

const equivalentOp = equivalence.ops.find((row) => row.op_index === 51);
expectEqual(`${equivalentOp.equivalent_channel_count}:${equivalentOp.divergent_channel_count}:${equivalentOp.divergent_state_count_decimal}`, "960:0:0", "Op #51 should retain a complete interval-equivalence certificate.");
expectEqual(equivalentOp.equivalence_ledger_sha256, "c680ff362b120b5a461db98a302ab29c8e486528e3cb2c3b94656ec215c12eac", "Equivalent-op certificate digest should remain stable.");
const signedOp = equivalence.ops.find((row) => row.op_index === 63);
expectEqual(`${signedOp.default_lower_state_count_decimal}:${signedOp.default_higher_state_count_decimal}`, "35035:120050", "Classifier output should preserve both build-mode delta directions.");
expectEqual(signedOp.equivalence_ledger_sha256, "ba7b72ac4e9ae8b6e0bbc895e723242cfc3e0fedadf57dc083b51377d0a97eef", "Classifier certificate digest should remain stable.");

const selected = reconstructRoundingEquivalenceChannel(analysis, bytes, 7, 37, true);
expectEqual(selected.segments.length, 447, "Selected reconstruction should expose every exact pair segment.");
expectEqual(selected.segments.reduce((sum, segment) => sum + BigInt(segment.state_count_decimal), 0n).toString(), "2041", "Selected segments should conserve the complete interval.");
expectEqual(selected.segments.filter((segment) => segment.divergent).reduce((sum, segment) => sum + BigInt(segment.state_count_decimal), 0n).toString(), "191", "Selected divergent segments should conserve exact exposure.");
expect(selected.segments.every((segment, index) => index === 0 || BigInt(selected.segments[index - 1].accumulator_maximum_decimal) + 1n === BigInt(segment.accumulator_minimum_decimal)), "Selected segments should be contiguous without overlap or gaps.");

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
expect(report.includes("## Fixed-Point Rounding Equivalence Lab (DERIVED EXACT CLOSED-INTERVAL CERTIFICATE)"), "Engineering report should render the equivalence certificate section.");
expect(report.includes("13,933,008,957") && report.includes("2,874,544") && report.includes("#007 DEPTHWISE_CONV_2D / ch 37"), "Engineering report should preserve exact state counts and ranked coordinate.");
expect(report.includes(opSeven.equivalence_ledger_sha256) && report.includes("interior integers can be unreachable") && report.includes(equivalence.source_commit), "Engineering report should preserve certificate digest, source, and interval-hull boundary.");
expect(report.includes("94.3% default-higher / 5.7% default-lower")
  && report.includes("double-versus-single difference is sign-dependent")
  && report.includes("not an activation-weighted runtime bias"), "Engineering report should expose the exact directional skew without promoting it to a runtime-distribution claim.");

const staticAnalysis = buildStaticAnalysisExport(analysis);
const quantization = buildQuantizationEvidence(analysis, identity);
expectEqual(staticAnalysis.rounding_equivalence.schema, ANALYZER_METADATA.schemas.roundingEquivalence, "Static evidence should retain the equivalence schema.");
expectEqual(JSON.stringify(quantization.rounding_equivalence), JSON.stringify(staticAnalysis.rounding_equivalence), "Static and quantization evidence should retain one equivalence ledger.");
const contract = buildQuantizationContractChecks(analysis).rounding_equivalence;
expectEqual(contract.status, "review", "Build-mode divergence should enter design review without failing artifact integrity.");
expectEqual(contract.checked_channels, 18_057, "Contract summary should expose every certified channel.");
expectEqual(contract.divergent_channels, 17_083, "Contract summary should expose every divergent channel.");
expectEqual(contract.divergent_state_count_decimal, "2874544", "Contract summary should expose exact divergent states.");

const findings = buildFindingsRegister(analysis);
const finding = findings.find((item) => item.finding_id === "EA-QNT-0113");
expect(finding, "Build-mode interval divergence should enter the authoritative action queue.");
expectEqual(finding?.technical_priority, "Informational", "Interval exposure should remain a supporting record under the grouped rounding ABI action.");
expect(finding?.observation.includes("13,933,008,957") && finding?.observation.includes("2,874,544") && finding?.observation.includes("channel 37") && finding?.observation.includes(opSeven.equivalence_ledger_sha256), "Finding should preserve exact portfolio, coordinate, and digest.");
expect(finding?.interpretation.includes("not an activation probability") && finding?.recommendation.includes("TFLITE_SINGLE_ROUNDING"), "Finding should preserve the interval-hull boundary and actionable build provenance.");

const mlBom = buildMlBomDocument(analysis, { hash: analysis.model_sha256, fileSizeBytes: bytes.byteLength, target: analysis.target_profile });
assertCompactMlBomProjection(mlBom, {
  expect,
  expectEqual,
  omittedProperties: [
    "deepbom:model:roundingEquivalenceSchema",
    "deepbom:model:roundingEquivalenceAssessedChannels",
    "deepbom:model:roundingEquivalenceDivergentChannels",
    "deepbom:model:roundingEquivalenceIntervalStates",
    "deepbom:model:roundingEquivalenceDivergentStates",
    "deepbom:model:roundingEquivalenceSourceCommit",
  ],
  label: "Rounding-equivalence compact ML-BOM",
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
const equivalenceChecks = conformance.checks.filter((check) => check.id.startsWith("CF-ROUND-"));
expectEqual(equivalenceChecks.length, 4, "Conformance should expose four cross-output equivalence checks.");
expect(equivalenceChecks.every((check) => check.status === "pass"), "Every rounding-equivalence conformance check should pass for the sample.");
expect(visualPngSpecs({ analysis, filename }).some(([path]) => path === "visuals/rounding_equivalence.png"), "Engineering bundle should include the equivalence PNG.");

const repeated = analyze_tflite_for_target(bytes, filename, "android_mid_a55").rounding_equivalence;
expectEqual(JSON.stringify(repeated), JSON.stringify(equivalence), "Rounding-equivalence JSON should be byte-for-byte deterministic.");
const tamperedCount = structuredClone(equivalence);
tamperedCount.divergent_state_count_decimal = "2874545";
expectThrows(() => validateRoundingEquivalenceAgainstReconstruction(tamperedCount, reconstructed), "divergent_state_count_decimal", "Validator should reject a tampered state total.");
const tamperedArray = structuredClone(equivalence);
tamperedArray.ops.find((row) => row.op_index === 7).channel_divergent_state_counts_decimal[37] = "190";
expectThrows(() => validateRoundingEquivalenceAgainstReconstruction(tamperedArray, reconstructed), "channel_divergent_state_counts_decimal", "Validator should reject a tampered channel array.");
const tamperedLedger = structuredClone(equivalence);
tamperedLedger.ops.find((row) => row.op_index === 7).equivalence_ledger_sha256 = "0".repeat(64);
await expectRejects(() => validateRoundingEquivalenceDigestsAgainstReconstruction(tamperedLedger, reconstructed), "SHA-256 mismatch", "Digest validator should reject a tampered certificate ledger.");

done("Rounding equivalence passed (53 ops, 18,057 channels, 13,933,008,957 interval states, 2,874,544 exact one-code divergences, reports, digests, and tamper rejection).");

async function expectRejects(action, fragment, message) {
  try {
    await action();
    expect(false, message);
  } catch (error) {
    expect(String(error?.message || error).includes(fragment), `${message} Expected ${JSON.stringify(fragment)}, got ${JSON.stringify(error?.message || String(error))}.`);
  }
}

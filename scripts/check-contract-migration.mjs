import { readFileSync } from "node:fs";

import { analyze_tflite_for_target, initSync } from "../pkg/tflite_wasm_audit.js";
import {
  validateContractMigration,
  validateContractMigrationDigests,
} from "../web/lib/contract-migration.js";
import { buildConformanceReport } from "../web/lib/report-conformance.js";
import { buildEngineeringReport } from "../web/lib/report-engineering.js";
import { buildQuantizationEvidence, buildStaticAnalysisExport } from "../web/lib/report-evidence.js";
import { buildFindingsRegister } from "../web/lib/report-findings.js";
import { buildMlBomDocument } from "../web/lib/report-mlbom.js";
import { assertCompactMlBomProjection } from "./compact-mlbom-assert.mjs";
import { ANALYZER_METADATA } from "../web/lib/report-metadata.js";
import { buildQuantizationContractChecks } from "../web/lib/report-quantization-contracts.js";
import { createCheck } from "./check-assert.mjs";

const { done, expect, expectEqual, expectThrows } = createCheck("Contract migration check");
const filename = "mobilenet_v2_1.0_224_quant.tflite";
initSync({ module: readFileSync("pkg/tflite_wasm_audit_bg.wasm") });
const bytes = new Uint8Array(readFileSync(`web/samples/${filename}`));
const analysis = analyze_tflite_for_target(bytes, filename, "android_mid_a55");
const migration = analysis.contract_migration;

expect(validateContractMigration(analysis, bytes), "Browser arithmetic should reconstruct every migration scenario.");
await validateContractMigrationDigests(analysis, bytes);
expectEqual(migration.schema, "deepbom.contract_migration.v1", "Migration schema should be stable.");
expectEqual(migration.method_version, "2026-07-17.1", "Migration method should be stable.");
expectEqual(migration.status, "assessed", "Every direct sample consumer should be assessed.");
expectEqual(migration.residual_contract_count, 10, "All residual containment designs should enter migration analysis.");
expectEqual(migration.candidate_scenario_count, 20, "Both containment candidates should be evaluated for every residual.");
expectEqual(migration.direct_consumer_count, 15, "The graph should expose ten kernel and five ADD consumers.");
expectEqual(migration.direct_consumer_edge_count, 15, "Every direct sample consumer should use one input edge.");
expectEqual(migration.kernel_consumer_count, 10, "Every residual should feed one expansion CONV_2D.");
expectEqual(migration.add_consumer_count, 5, "Five residual outputs should feed a later residual ADD.");
expectEqual(migration.other_consumer_count, 0, "No sample direct consumer should be silently unmodeled.");
expectEqual(migration.assessed_consumer_scenario_count, 30, "Every direct consumer should be assessed under both candidates.");
expectEqual(migration.unassessed_consumer_scenario_count, 0, "No candidate consumer should be unassessed.");
expectEqual(migration.assessed_kernel_channel_scenario_count, 9_504, "All direct kernel channels should be retained for both candidates.");
expectEqual(migration.multiplier_encoding_changed_channel_scenario_count, 9_504, "Every direct kernel multiplier should be regenerated.");
expectEqual(migration.multiplier_shift_changed_channel_scenario_count, 7_584, "Shift changes should remain deterministic.");
expectEqual(migration.bias_code_changed_channel_scenario_count, 9_500, "Bias regeneration count should remain deterministic.");
expectEqual(migration.bias_int32_overflow_channel_scenario_count, 0, "The sample should have no candidate bias overflow.");
expectEqual(migration.add_parameter_encoding_changed_scenario_count, 24, "Pinned ADD parameter regeneration count should remain deterministic.");
expectEqual(migration.reachable_downstream_op_union_count, 55, "Reachability union should include every structurally affected downstream op.");
expectEqual(migration.maximum_downstream_edge_depth, 28, "Maximum structural impact depth should remain deterministic.");

const residual = migration.migrations.find((row) => row.source_add_op_index === 27);
const scenario = residual.scenarios.find((row) => row.design === "globally_finest_minimum_containment");
const kernel = scenario.kernel_consumers[0];
const add = scenario.add_consumers[0];
expectEqual(residual.reachable_downstream_op_count, 37, "ADD #27 downstream op radius should remain exact.");
expectEqual(residual.maximum_downstream_edge_depth, 19, "ADD #27 downstream depth should remain exact.");
expectEqual(scenario.candidate_output_scale, 0.3585545766527635, "Global candidate scale should bind the lattice design.");
expectEqual(scenario.candidate_output_zero_point, 118, "Global candidate zero-point should bind the lattice design.");
expectEqual(kernel.op_index, 28, "ADD #27 direct kernel consumer should remain graph-bound.");
expectEqual(kernel.assessed_channel_count, 384, "Every direct consumer channel should be retained.");
expectEqual(kernel.multiplier_shift_changed_channel_count, 384, "Every direct consumer shift should change in this scenario.");
expectEqual(kernel.bias_code_changed_channel_count, 384, "Every direct consumer bias should be regenerated in this scenario.");
expectEqual(kernel.maximum_absolute_bias_rebase_error_current_steps, 0.9435736048017123, "Maximum bias rebase error should remain deterministic.");
expectEqual(kernel.channel_ledger_sha256, "16a75977949402e4bbc9b1f6e0696c71b513eb7b55c9daae35a73b52b7ecd1f7", "Channel ledger digest should remain stable.");
expectEqual(kernel.top_channels[0].channel_index, 228, "Worst rebasing channel should remain stable.");
expectEqual(kernel.top_channels[0].current_bias_code, 3_683, "Worst channel current bias should remain artifact-bound.");
expectEqual(kernel.top_channels[0].candidate_bias_code_decimal, "1943", "Worst channel candidate bias should remain exact.");
expectEqual(add.op_index, 31, "ADD #27 direct residual consumer should remain graph-bound.");
expectEqual(add.changed_offset_count, 1, "Global zero-point migration should update one ADD input offset.");
expectEqual(add.changed_multiplier_encoding_count, 2, "One upstream migration should update two ADD multiplier encodings.");
expectEqual(add.candidate_parameters.left_shift, 20, "Pinned 8-bit ADD left shift should remain source-bound.");
expectEqual(add.candidate_parameters.input_multipliers[0].quantized_multiplier, 1_762_707_429, "Candidate ADD input multiplier should remain exact.");
expectEqual(add.candidate_parameters.output_multiplier.quantized_multiplier, 1_928_049_469, "Candidate ADD output multiplier should remain exact.");

const repeated = analyze_tflite_for_target(bytes, filename, "android_mid_a55").contract_migration;
expectEqual(JSON.stringify(repeated), JSON.stringify(migration), "Migration JSON should be byte-for-byte deterministic.");

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
expect(report.includes("## Contract Migration Impact Lab (DERIVED COUNTERFACTUAL RE-EXPORT)"), "Engineering report should render contract migration.");
expect(report.includes("deepbom.contract_migration.v1 / 2026-07-17.1")
  && report.includes("9,504 / 7,584")
  && report.includes("9,500 / 0")
  && report.includes(kernel.channel_ledger_sha256)
  && report.includes("bias_int_new=round_ties_away"), "Engineering report should retain exact migration counts, digest, and formula.");
expect(report.includes("counterfactual re-export impact analysis")
  && report.includes("structural behavior-impact radius only")
  && report.includes(migration.source_commit), "Engineering report should retain migration boundaries and source identity.");
const staticAnalysis = buildStaticAnalysisExport(analysis);
const quantization = buildQuantizationEvidence(analysis, identity);
expectEqual(staticAnalysis.contract_migration.schema, ANALYZER_METADATA.schemas.contractMigration, "Static evidence should retain migration schema.");
expectEqual(JSON.stringify(quantization.residual_contract_migration), JSON.stringify(staticAnalysis.contract_migration), "Static and quantization evidence should retain the same migration ledger.");
const contracts = buildQuantizationContractChecks(analysis);
expectEqual(contracts.contract_migration.status, "pass", "Sample migration contract should pass complete deterministic assessment.");
expectEqual(contracts.contract_migration.checked_candidate_scenarios, 20, "Contract summary should expose every candidate scenario.");
expectEqual(contracts.contract_migration.changed_bias_code_channel_scenarios, 9_500, "Contract summary should retain bias regeneration count.");
const finding = buildFindingsRegister(analysis).find((item) => item.finding_id === "EA-QNT-0108");
expect(finding?.observation.includes("384 kernel channel(s)")
  && finding?.observation.includes("384 changed bias code(s)")
  && finding?.recommendation.includes("Contract Migration Impact ledger"), "Residual finding should expose exact direct migration impact.");
const mlBom = buildMlBomDocument(analysis, { hash: analysis.model_sha256, fileSizeBytes: bytes.byteLength, target: analysis.target_profile });
assertCompactMlBomProjection(mlBom, {
  expect,
  expectEqual,
  omittedProperties: [
    "deepbom:model:contractMigrationSchema",
    "deepbom:model:contractMigrationKernelChannelScenarios",
    "deepbom:model:contractMigrationChangedBiasCodes",
  ],
  label: "Contract-migration compact ML-BOM",
});
const conformance = buildConformanceReport({
  analysis,
  staticAnalysis,
  quantization,
  findingsRegister: { authoritative_action_source: "findings", raw_analyzer_signals: [], findings: [] },
  runtimeResults: {},
  securityPosture: { execution_integrity: {} },
  mlBomDocument: mlBom,
  engineeringReport: report,
});
const migrationChecks = conformance.checks.filter((check) => check.id.startsWith("CF-MIGRATION-"));
expectEqual(migrationChecks.length, 4, "Conformance should expose four migration cross-output checks.");
expect(migrationChecks.every((check) => check.status === "pass"), "Every migration conformance check should pass.");

const tamperedBias = structuredClone(analysis);
tamperedBias.contract_migration.migrations.find((row) => row.source_add_op_index === 27)
  .scenarios.find((row) => row.design === "globally_finest_minimum_containment")
  .kernel_consumers[0].channel_candidate_bias_code_decimals[0] = "0";
expectThrows(() => validateContractMigration(tamperedBias, bytes), "rows", "Validator should reject a tampered bias ledger.");

const tamperedAdd = structuredClone(analysis);
tamperedAdd.contract_migration.migrations.find((row) => row.source_add_op_index === 27)
  .scenarios.find((row) => row.design === "globally_finest_minimum_containment")
  .add_consumers[0].candidate_parameters.output_multiplier.shift += 1;
expectThrows(() => validateContractMigration(tamperedAdd, bytes), "rows", "Validator should reject a tampered ADD parameter.");

const tamperedDigest = structuredClone(analysis);
tamperedDigest.contract_migration.migrations.find((row) => row.source_add_op_index === 27)
  .scenarios.find((row) => row.design === "globally_finest_minimum_containment")
  .kernel_consumers[0].channel_ledger_sha256 = "0".repeat(64);
let digestRejected = false;
try {
  await validateContractMigrationDigests(tamperedDigest, bytes);
} catch (error) {
  digestRejected = /SHA-256 mismatch/.test(error.message);
}
expect(digestRejected, "Validator should reject a tampered channel digest.");

done("Contract migration passed (10 residuals, 20 candidates, 15 direct consumers, 9,504 channel scenarios, source-backed ADD parameters, and tamper rejection).");

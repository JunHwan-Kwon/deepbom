import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";

import { analyze_tflite_for_target, initSync } from "../pkg/tflite_wasm_audit.js";
import { validatePreprocessingRealizabilityAnalysis } from "../web/lib/preprocessing-realizability.js";
import { decodeStoredRgbPng } from "../web/lib/rgb-png.js";
import { buildConformanceReport } from "../web/lib/report-conformance.js";
import { buildEngineeringReport } from "../web/lib/report-engineering.js";
import {
  buildEngineeringBundleArtifactFiles,
  buildQuantizationEvidence,
  buildRawDataArtifactFiles,
  buildStaticAnalysisExport,
} from "../web/lib/report-evidence.js";
import { buildFindingsRegister } from "../web/lib/report-findings.js";
import { buildMlBomDocument } from "../web/lib/report-mlbom.js";
import { assertCompactMlBomProjection } from "./compact-mlbom-assert.mjs";
import { ANALYZER_METADATA } from "../web/lib/report-metadata.js";
import { buildQuantizationContractChecks } from "../web/lib/report-quantization-contracts.js";
import { visualPngSpecs } from "../web/lib/visual-export.js";
import { createCheck } from "./check-assert.mjs";

const { done, expect, expectEqual } = createCheck("Pixel-to-tensor preprocessing realizability check");
const filename = "mobilenet_v2_1.0_224_quant.tflite";
initSync({ module: readFileSync("pkg/tflite_wasm_audit_bg.wasm") });
const modelBytes = new Uint8Array(readFileSync(`web/samples/${filename}`));
const analysis = analyze_tflite_for_target(modelBytes, filename, "android_mid_a55");
const validation = await validatePreprocessingRealizabilityAnalysis(analysis);
const evidence = validation.evidence;
const byId = new Map(evidence.candidates.map((candidate) => [candidate.contract_id, candidate]));

expectEqual(`${evidence.schema}:${evidence.method_version}:${evidence.status}`, "deepbom.preprocessing_realizability.v1:2026-07-18.1:assessed", "Preprocessing schema, method, and status should remain stable.");
expectEqual(`${evidence.source_witness_count}:${evidence.eligible_image_witness_count}:${evidence.candidate_contract_count}:${evidence.candidate_evaluation_count}`, "1:1:8:8", "The complete candidate matrix should remain stable.");
expectEqual(`${evidence.assessed_candidate_count}:${evidence.exact_tensor_realization_candidate_count}:${evidence.non_exact_candidate_count}`, "8:4:4", "Exact and non-exact candidate classes should remain stable.");
expectEqual(evidence.exact_contract_ids.join(","), "raw_storage_rgb,raw_storage_bgr,artifact_affine_rgb,center_128_div_128_rgb", "Exact source contracts should remain explicit and ordered.");
expectEqual(`${evidence.best_non_exact_contract_id}:${evidence.best_non_exact_unrealizable_element_count}`, "unit_interval_rgb:18", "The closest non-exact candidate should remain deterministic.");
expectEqual(evidence.candidate_conservation, "8 evaluations = 8 assessed (4 exact + 4 non-exact) + 0 not-assessed", "Candidate evaluation counts should conserve.");
expectEqual(evidence.portfolio_ledger_sha256, "35a1ca877c09dd440ed75e7cfbe1789c6b3693ea6572da5069cf6a01aadd5bb5", "Preprocessing portfolio ledger should remain stable.");

const rawRgb = byId.get("raw_storage_rgb");
expectEqual(`${rawRgb.exact_tensor_realization}:${rawRgb.exact_tensor_element_count}:${rawRgb.unrealizable_tensor_element_count}:${rawRgb.minimum_total_absolute_tensor_code_error_decimal}`, "true:150528:0:0", "Raw RGB should exactly reproduce the complete witness.");
expectEqual(rawRgb.channel_maps.map((row) => `${row.reachable_tensor_code_count}/${row.tensor_code_hole_count}/${row.collision_tensor_code_count}`).join(","), "256/0/0,256/0/0,256/0/0", "Raw RGB channel maps should be bijective.");
expectEqual(rawRgb.nearest_rgb_fixture_sha256, "89265147c9669c94eccbbdd5593623e04f1ba76190054786d88989aa6e5d3035", "Raw RGB fixture should equal the tensor byte ledger for this UINT8 RGB contract.");

const minusOne = byId.get("minus_one_to_one_rgb");
expectEqual(`${minusOne.exact_tensor_element_count}:${minusOne.unrealizable_tensor_element_count}:${minusOne.minimum_total_absolute_tensor_code_error_decimal}:${minusOne.maximum_absolute_tensor_code_error}`, "26:150502:150502:1", "The [-1,1] candidate should expose the code-128 hole across the full tensor.");
expectEqual(minusOne.channel_maps.map((row) => `${row.reachable_tensor_code_count}/${row.tensor_code_hole_count}/${row.collision_tensor_code_count}`).join(","), "255/1/1,255/1/1,255/1/1", "The [-1,1] LUT should expose one hole and one collision per channel.");
expectEqual(`${minusOne.first_unrealizable_element.target_tensor_code}:${minusOne.first_unrealizable_element.selected_source_pixel_code}:${minusOne.first_unrealizable_element.roundtrip_tensor_code}`, "128:127:127", "The first [-1,1] mismatch should be a reproducible inverse witness.");

const unit = byId.get("unit_interval_rgb");
expectEqual(`${unit.exact_tensor_element_count}:${unit.unrealizable_tensor_element_count}:${unit.minimum_total_absolute_tensor_code_error_decimal}:${unit.maximum_absolute_tensor_code_error}`, "150510:18:2304:128", "The [0,1] candidate should preserve its exact minimum-error accounting.");
expectEqual(unit.channel_maps.map((row) => `${row.reachable_tensor_code_count}/${row.tensor_code_hole_count}/${row.collision_tensor_code_count}`).join(","), "128/128/127,128/128/127,128/128/127", "The [0,1] LUT should preserve coverage, holes, and collisions.");

const exactFixture = validation.candidates.find((row) => row.candidate.contract_id === "raw_storage_rgb").fixture;
const decoded = decodeStoredRgbPng(exactFixture.png);
expectEqual(`${decoded.width}:${decoded.height}:${decoded.rgb.byteLength}:${exactFixture.png.byteLength}`, "224:224:150528:150830", "The deterministic PNG fixture should decode to the complete RGB raster.");
expectEqual(createHash("sha256").update(decoded.rgb).digest("hex"), rawRgb.nearest_rgb_fixture_sha256, "Decoded PNG pixels should match the declared fixture digest.");

const identity = {
  filename,
  hash: analysis.model_sha256,
  file_size_bytes: modelBytes.byteLength,
  format: analysis.format,
  target_profile_id: analysis.target_profile.id,
  target_label: analysis.target_profile.label,
  operator_count: analysis.operator_count,
  tensor_count: analysis.tensor_count,
  total_macs: analysis.total_macs,
};
const report = buildEngineeringReport(analysis, { identity });
expect(report.includes("## Pixel-to-Tensor Contract Lab (DERIVED EXHAUSTIVE COUNTERFACTUAL MATRIX)"), "Engineering report should render the preprocessing matrix.");
expect(report.includes("8 evaluations = 8 assessed (4 exact + 4 non-exact) + 0 not-assessed") && report.includes("unit_interval_rgb; 18 unrealizable elements") && report.includes("2,304"), "Engineering report should preserve candidate conservation and the best non-exact result.");
expect(report.includes(rawRgb.nearest_rgb_fixture_sha256) && report.includes(evidence.portfolio_ledger_sha256), "Engineering report should preserve fixture and portfolio digests.");
expect(report.includes("explicit counterfactual contracts, not observations of the production application"), "Engineering report should preserve the production-contract boundary.");

const staticAnalysis = buildStaticAnalysisExport(analysis);
const quantization = buildQuantizationEvidence(analysis, identity);
expectEqual(staticAnalysis.preprocessing_realizability.schema, ANALYZER_METADATA.schemas.preprocessingRealizability, "Static evidence should retain the preprocessing schema.");
expectEqual(JSON.stringify(quantization.preprocessing_realizability), JSON.stringify(staticAnalysis.preprocessing_realizability), "Static and quantization exports should retain one preprocessing ledger.");
const contract = buildQuantizationContractChecks(analysis).preprocessing_realizability;
expectEqual(`${contract.status}:${contract.exact_tensor_realization_candidates}:${contract.non_exact_candidates}:${contract.best_non_exact_unrealizable_elements}`, "review:4:4:18", "Quantization contracts should expose the exact/non-exact preprocessing split.");
expectEqual(contract.portfolio_ledger_sha256, evidence.portfolio_ledger_sha256, "Quantization contracts should bind the preprocessing portfolio digest.");

const findings = buildFindingsRegister(analysis);
const finding = findings.find((item) => item.finding_id === "EA-QNT-0117");
expect(Boolean(finding), "Mixed preprocessing realizability should enter the authoritative action queue.");
expectEqual(`${finding.evidence_class}:${finding.technical_priority}`, "DERIVED:Medium", "Preprocessing finding should preserve evidence and priority semantics.");
expectEqual(finding.finding_kind, "caution", "An unbound preprocessing counterfactual must not block the artifact-defect gate.");
expect(finding.observation.includes("4 contract(s)") && finding.observation.includes("18 element(s)") && finding.observation.includes(evidence.portfolio_ledger_sha256), "Preprocessing finding should preserve exact candidate, minimum mismatch, and digest evidence.");
expect(finding.observation.includes("range matching and witness realizability answer different questions")
  && finding.observation.includes("(pixel - 128) / 128")
  && finding.observation.includes("150,502 witness element(s)"), "Preprocessing finding should reconcile the range-level convention match with exact finite-code witness realizability.");
expect(finding.interpretation.includes("does not reveal which preprocessing implementation") && finding.recommendation.includes("produced tensor SHA-256"), "Preprocessing finding should preserve its interpretation and replay action.");

const mlBom = buildMlBomDocument(analysis, { hash: analysis.model_sha256, fileSizeBytes: modelBytes.byteLength, target: analysis.target_profile });
assertCompactMlBomProjection(mlBom, {
  expect,
  expectEqual,
  omittedProperties: [
    "deepbom:model:preprocessingRealizabilitySchema",
    "deepbom:model:preprocessingRealizabilityExactCandidates",
    "deepbom:model:preprocessingRealizabilityBestNonExactUnrealizableElements",
    "deepbom:model:preprocessingRealizabilityPortfolioLedgerSha256",
    "deepbom:model:preprocessingRealizabilityExactFixtureSha256",
  ],
  label: "Preprocessing-realizability compact ML-BOM",
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
const preprocessingChecks = conformance.checks.filter((check) => check.id.startsWith("CF-PR-"));
expectEqual(preprocessingChecks.length, 4, "Conformance should expose four preprocessing checks.");
expect(preprocessingChecks.every((check) => check.status === "pass"), `Every preprocessing conformance check should pass: ${JSON.stringify(preprocessingChecks)}`);
expect(visualPngSpecs({ analysis, filename }).some(([path]) => path === "visuals/preprocessing_realizability.png"), "Engineering visuals should include preprocessing realizability.");
const compact = buildEngineeringBundleArtifactFiles(analysis, { reportContext: { identity }, rawEvidenceContext: { identity }, mlBomDocument: mlBom });
expectEqual(compact.length, 3, "Preprocessing fixtures should not fragment the compact engineering bundle.");
const rawFiles = buildRawDataArtifactFiles(analysis, { rawEvidenceContext: { identity }, mlBomDocument: mlBom });
const fixtureFile = rawFiles.find((file) => file.name === "static/preprocessing_exact_rgb_fixture.png");
expectEqual(fixtureFile?.data?.byteLength, 150_830, "Raw Data should include one canonical exact RGB PNG fixture.");
expectEqual(createHash("sha256").update(decodeStoredRgbPng(fixtureFile.data).rgb).digest("hex"), rawRgb.nearest_rgb_fixture_sha256, "Raw Data fixture pixels should match the evidence digest.");

const repeated = analyze_tflite_for_target(modelBytes, filename, "android_mid_a55").preprocessing_realizability;
expectEqual(JSON.stringify(repeated), JSON.stringify(evidence), "Preprocessing realizability JSON should be byte-for-byte deterministic.");
await expectRejected(tamper((copy) => { copy.preprocessing_realizability.candidates[0].channel_maps[0].pixel_to_tensor_codes[0] = 1; }), "Preprocessing LUT mismatch", "LUT tampering should be rejected.");
await expectRejected(tamper((copy) => { copy.preprocessing_realizability.candidates[0].nearest_rgb_fixture_sha256 = "0".repeat(64); }), "Nearest RGB fixture SHA-256 mismatch", "Fixture digest tampering should be rejected.");
await expectRejected(tamper((copy) => { copy.preprocessing_realizability.candidates[0].candidate_ledger_sha256 = "0".repeat(64); }), "Preprocessing candidate ledger SHA-256 mismatch", "Candidate ledger tampering should be rejected.");
await expectRejected(tamper((copy) => { copy.preprocessing_realizability.portfolio_ledger_sha256 = "0".repeat(64); }), "Preprocessing portfolio ledger SHA-256 mismatch", "Portfolio digest tampering should be rejected.");

done("Pixel-to-tensor preprocessing realizability passed (8 candidates, 4 exact, finite 3x256 LUTs, exact/minimum-error RGB fixtures, reports, Raw Data, and tamper rejection)." );

function tamper(mutate) {
  const copy = structuredClone(analysis);
  mutate(copy);
  return validatePreprocessingRealizabilityAnalysis(copy);
}

async function expectRejected(promise, messagePart, label) {
  try {
    await promise;
    expect(false, `${label} Expected rejection containing ${messagePart}.`);
  } catch (error) {
    expect(String(error?.message || error).includes(messagePart), `${label} Expected ${messagePart}, got ${error?.message || error}.`);
  }
}

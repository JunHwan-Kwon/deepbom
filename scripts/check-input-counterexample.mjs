import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";

import { analyze_tflite_for_target, initSync } from "../pkg/tflite_wasm_audit.js";
import { validateInputCounterexampleAnalysis } from "../web/lib/input-counterexample.js";
import { buildConformanceReport } from "../web/lib/report-conformance.js";
import { buildEngineeringReport } from "../web/lib/report-engineering.js";
import {
  buildEngineeringBundleArtifactFiles,
  buildQuantizationEvidence,
  buildStaticAnalysisExport,
} from "../web/lib/report-evidence.js";
import { buildFindingsRegister } from "../web/lib/report-findings.js";
import { buildMlBomDocument } from "../web/lib/report-mlbom.js";
import { assertCompactMlBomProjection } from "./compact-mlbom-assert.mjs";
import { ANALYZER_METADATA } from "../web/lib/report-metadata.js";
import { buildQuantizationContractChecks } from "../web/lib/report-quantization-contracts.js";
import { visualPngSpecs } from "../web/lib/visual-export.js";
import { createCheck } from "./check-assert.mjs";

const { done, expect, expectEqual } = createCheck("Model input tensor ABI counterexample check");
const filename = "mobilenet_v2_1.0_224_quant.tflite";
initSync({ module: readFileSync("pkg/tflite_wasm_audit_bg.wasm") });
const modelBytes = new Uint8Array(readFileSync(`web/samples/${filename}`));
const analysis = analyze_tflite_for_target(modelBytes, filename, "android_mid_a55");
const result = await validateInputCounterexampleAnalysis(analysis);
const evidence = result.evidence;
const source = evidence.sources[0];
const witness = evidence.witnesses[0];
const reconstructed = result.witnesses[0];

expectEqual(evidence.schema, "deepbom.input_counterexample.v1", "Input witness schema should remain stable.");
expectEqual(evidence.method_version, "2026-07-18.3", "Input witness method should remain stable.");
expectEqual(evidence.status, "assessed", "All exact-local sources should be classified.");
expectEqual(`${evidence.exact_local_source_op_count}:${evidence.direct_model_input_source_op_count}:${evidence.tensor_abi_constructive_source_op_count}:${evidence.upstream_activation_unresolved_source_op_count}:${evidence.not_assessed_source_op_count}`, "52:1:1:51:0", "Source realizability classification should remain exact.");
expectEqual(`${evidence.tensor_abi_constructive_channel_count}:${evidence.tensor_abi_constructive_divergent_state_count_decimal}:${evidence.output_reachable_constructive_source_op_count}:${evidence.representative_witness_count}`, "18:2918:1:1", "Constructive source portfolio should remain exact.");
expectEqual(evidence.source_classification_conservation, "52 = 1 constructive + 51 upstream-unresolved + 0 not-assessed", "Source classification should conserve every exact-local source.");
expectEqual(evidence.portfolio_ledger_sha256, "e3dbddcfe7445128e2c763fb43acee45e0fba43f243a92f289618d1a43242ac9", "Portfolio ledger should remain deterministic.");
expectEqual(`${source.op_index}:${source.classification}:${source.exact_reachable_divergent_channel_count}:${source.exact_reachable_divergent_state_count_decimal}:${source.exact_model_output_graph_route_count_decimal}`, "0:tensor_abi_constructive:18:2918:1024", "First-layer source should be constructively bound to its structural route ledger.");
expectEqual(source.source_reachability_ledger_sha256, "67743dd329cabb5d6019924a46020b1ec59b6b69bb68f0136178184dfd1c4505", "Source should bind the exact reachability ledger.");
expectEqual(source.source_propagation_ledger_sha256, "0318b9ba433a62b8543aa3286130052cccea90769e0fe23d43106c54e2365229", "Source should bind the structural propagation ledger.");

expectEqual(`${witness.source_op_index}:${witness.source_channel_index}:${witness.source_output_coordinate.join(",")}`, "0:4:0,0,0,4", "Representative source coordinate should remain exact.");
expectEqual(`${witness.model_input_tensor_index}:${witness.model_input_shape.join(",")}:${witness.model_input_dtype}:${witness.model_input_element_count}`, "171:1,224,224,3:UINT8:150528", "Complete model-input ABI should remain exact.");
expectEqual(`${witness.full_tensor_fill_code}:${witness.sparse_override_count}:${witness.terms.length}`, "128:26:27", "Sparse construction should conserve the zero-point fill and kernel terms.");
expectEqual(witness.patch_codes_hwc.join(","), "0,0,0,0,140,0,254,255,0,0,255,0,0,0,0,0,255,0,128,255,0,0,255,0,0,255,0", "Constructive HWC patch should remain byte-exact.");
expectEqual(`${witness.dot_product_decimal}:${witness.bias_decimal}:${witness.post_bias_accumulator_decimal}:${witness.default_output_code}:${witness.single_rounding_output_code}:${witness.output_code_delta}`, "-13115:13159:44:1:0:1", "Exact dot, bias, accumulator, and output-code divergence should remain stable.");
expectEqual(witness.full_model_input_tensor_sha256, "89265147c9669c94eccbbdd5593623e04f1ba76190054786d88989aa6e5d3035", "Full model-input tensor digest should remain stable.");
expectEqual(witness.witness_ledger_sha256, "45cdbb1087d79e088d7830c0e1840daa39b420b8f29e7c78fbb7e7ba702d0ce5", "Witness ledger should remain stable.");
expectEqual(reconstructed.bytes.byteLength, 150_528, "Independent reconstruction should produce the complete input tensor.");
expectEqual(createHash("sha256").update(reconstructed.bytes).digest("hex"), witness.full_model_input_tensor_sha256, "Node SHA-256 should independently confirm the reconstructed tensor.");

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
expect(report.includes("## Model Input Tensor ABI Witness (DERIVED CONSTRUCTIVE EXISTENCE CERTIFICATE)"), "Engineering report should render the constructive input certificate.");
expect(report.includes("150,528 elements") && report.includes("26 unique sparse overrides") && report.includes("-13,115 + 13,159 = 44"), "Engineering report should preserve the complete construction and exact arithmetic.");
expect(report.includes(witness.full_model_input_tensor_sha256) && report.includes(witness.witness_ledger_sha256), "Engineering report should preserve tensor and witness digests.");
expect(report.includes("exact at the model tensor ABI") && report.includes("does not prove a declared model output changes"), "Engineering report should preserve the proof boundary.");

const staticAnalysis = buildStaticAnalysisExport(analysis);
const quantization = buildQuantizationEvidence(analysis, identity);
expectEqual(staticAnalysis.input_counterexample.schema, ANALYZER_METADATA.schemas.inputCounterexample, "Static evidence should retain the input-witness schema.");
expectEqual(JSON.stringify(quantization.input_counterexample), JSON.stringify(staticAnalysis.input_counterexample), "Static and quantization exports should retain one input-witness ledger.");
const contract = buildQuantizationContractChecks(analysis).input_counterexample;
expectEqual(`${contract.status}:${contract.tensor_abi_constructive_sources}:${contract.tensor_abi_constructive_channels}:${contract.tensor_abi_constructive_divergent_state_count_decimal}`, "review:1:18:2918", "Quantization contract should expose the constructive input portfolio.");
expectEqual(contract.portfolio_ledger_sha256, evidence.portfolio_ledger_sha256, "Quantization contract should bind the portfolio digest.");

const findings = buildFindingsRegister(analysis);
const finding = findings.find((item) => item.finding_id === "EA-QNT-0116");
expect(Boolean(finding), "Constructive model-input divergence should enter the authoritative finding queue.");
expectEqual(`${finding?.evidence_class}:${finding?.technical_priority}`, "DERIVED:Medium", "Input witness finding should retain its evidence and priority semantics.");
expect(finding?.observation.includes("150,528") && finding?.observation.includes("1,024") && finding?.observation.includes(witness.full_model_input_tensor_sha256), "Input witness finding should preserve complete input size, structural routes, and tensor digest.");
expect(finding?.interpretation.includes("model tensor ABI") && finding?.recommendation.includes("bit-for-bit"), "Input witness finding should preserve its boundary and validation action.");

const mlBom = buildMlBomDocument(analysis, { hash: analysis.model_sha256, fileSizeBytes: modelBytes.byteLength, target: analysis.target_profile });
assertCompactMlBomProjection(mlBom, {
  expect,
  expectEqual,
  omittedProperties: [
    "deepbom:model:inputCounterexampleSchema",
    "deepbom:model:inputCounterexampleConstructiveSources",
    "deepbom:model:inputCounterexampleConstructiveDivergentStates",
    "deepbom:model:inputCounterexampleFullTensorSha256",
    "deepbom:model:inputCounterexampleWitnessLedgerSha256",
  ],
  label: "Input-counterexample compact ML-BOM",
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
const inputChecks = conformance.checks.filter((check) => check.id.startsWith("CF-IW-"));
expectEqual(inputChecks.length, 4, "Conformance should expose four cross-output input-witness checks.");
expect(inputChecks.every((check) => check.status === "pass"), `Every input-witness conformance check should pass: ${JSON.stringify(inputChecks)}`);
expect(visualPngSpecs({ analysis, filename }).some(([path]) => path === "visuals/input_counterexample.png"), "Engineering visuals should include the input-witness PNG.");
const compactFiles = buildEngineeringBundleArtifactFiles(analysis, { reportContext: { identity }, rawEvidenceContext: { identity }, mlBomDocument: mlBom });
const rawWitness = compactFiles.find((file) => file.name === "input_counterexample_input.bin");
expectEqual(compactFiles.length, 3, "Constructive evidence should add exactly one raw witness member to the compact engineering bundle.");
expectEqual(rawWitness?.data?.byteLength, 150_528, "Engineering bundle should include the complete raw input witness.");
expectEqual(createHash("sha256").update(rawWitness.data).digest("hex"), witness.full_model_input_tensor_sha256, "Bundled raw witness should match its declared SHA-256.");

const repeated = analyze_tflite_for_target(modelBytes, filename, "android_mid_a55").input_counterexample;
expectEqual(JSON.stringify(repeated), JSON.stringify(evidence), "Input counterexample JSON should be byte-for-byte deterministic.");
await expectRejected(tamper((copy) => { copy.input_counterexample.witnesses[0].terms[0].input_code = 1; }), "Full tensor code mismatch", "Term-code tampering should be rejected.");
await expectRejected(tamper((copy) => { copy.input_counterexample.witnesses[0].sparse_overrides[0].input_code = 1; }), "Full model-input tensor SHA-256 mismatch", "Sparse override tampering should be rejected.");
await expectRejected(tamper((copy) => { copy.input_counterexample.witnesses[0].full_model_input_tensor_sha256 = "0".repeat(64); }), "Full model-input tensor SHA-256 mismatch", "Tensor digest tampering should be rejected.");
await expectRejected(tamper((copy) => { copy.input_counterexample.witnesses[0].witness_ledger_sha256 = "0".repeat(64); }), "Input witness ledger SHA-256 mismatch", "Witness ledger tampering should be rejected.");
await expectRejected(tamper((copy) => { copy.input_counterexample.sources[0].classification = "upstream_activation_constraint_unresolved"; }), "Input-origin classification mismatch", "Source classification tampering should be rejected.");

done("Model input tensor ABI counterexample passed (1/52 constructive sources, 18 channels, 2,918 states, exact 150,528-byte tensor reconstruction, reports, bundle, and tamper rejection)." );

function tamper(mutate) {
  const copy = structuredClone(analysis);
  mutate(copy);
  return validateInputCounterexampleAnalysis(copy);
}

async function expectRejected(promise, messagePart, label) {
  try {
    await promise;
    expect(false, `${label} Expected rejection containing ${messagePart}.`);
  } catch (error) {
    expect(String(error?.message || error).includes(messagePart), `${label} Expected ${messagePart}, got ${error?.message || error}.`);
  }
}

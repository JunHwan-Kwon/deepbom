import { existsSync, readFileSync } from "node:fs";

import {
  analyze_tflite_for_target,
  initSync,
} from "../pkg/tflite_wasm_audit.js";
import {
  buildQuantResearchCoverage,
  classifyQuantResearchArtifact,
  ensureQuantResearchCoverage,
  QUANT_RESEARCH_COVERAGE_SCHEMA,
  QUANT_RESEARCH_LABS,
} from "../web/lib/quant-research-applicability.js";
import { buildModelAtGlance } from "../web/lib/model-glance.js";
import { buildEngineeringReport } from "../web/lib/report-engineering.js";
import { buildEngineeringEvidenceDocument, buildStaticAnalysisExport } from "../web/lib/report-evidence.js";
import { buildMlBomDocument } from "../web/lib/report-mlbom.js";
import { createCheck } from "./check-assert.mjs";

const { done, expect, expectEqual } = createCheck("Quant research applicability");

const fullInteger = analysis({
  classification: "full_integer",
  quantizedComputeOps: 47,
  opStateCounts: [{ name: "full_integer", count: 47 }],
});
const fullCoverage = buildQuantResearchCoverage(fullInteger);
expectEqual(fullCoverage.schema, QUANT_RESEARCH_COVERAGE_SCHEMA, "Coverage schema should be explicit.");
expectEqual(fullCoverage.artifact_class, "full_integer", "Full-integer artifacts should retain their integer activation contract.");
expectEqual(fullCoverage.lab_count, 15, "The research denominator should contain exactly 15 named labs.");
expectEqual(fullCoverage.class_supported_lab_count, 15, "Full-integer artifacts should expose the full 15-lab class envelope.");
expectEqual(fullCoverage.not_applicable_lab_count, 0, "A full-integer artifact should not be class-excluded from a research lab.");

const dynamicRange = analysis({
  classification: "dynamic_range_or_weight_only",
  quantizedComputeOps: 0,
  opStateCounts: [{ name: "weight_only_or_dynamic_range", count: 67 }],
  zeroAssessedEvidence: true,
});
const dynamicCoverage = buildQuantResearchCoverage(dynamicRange);
expectEqual(classifyQuantResearchArtifact(dynamicRange).id, "dynamic_range", "Float activation compute with stored INT8 weights should classify as dynamic-range.");
expectEqual(dynamicCoverage.class_supported_lab_count, 1, "Dynamic-range artifacts should expose only the stored-weight integrity class envelope.");
expectEqual(dynamicCoverage.artifact_applicable_lab_count, 1, "Decoded quantized constants should make the weight integrity lab applicable.");
expectEqual(dynamicCoverage.assessed_lab_count, 1, "The stored-weight integrity lab should be assessed.");
expectEqual(dynamicCoverage.not_applicable_lab_count, 14, "Integer activation-path labs should be explicitly excluded.");
expect(dynamicCoverage.labs.filter((row) => row.id !== "weight_scale_integrity")
  .every((row) => row.status === "not_applicable" && row.reason_code === "QR-CLASS-DYNAMIC-RANGE"),
  "Every advanced dynamic-range lab should share one deterministic class gate and reason.");

const inconsistentLegacyCounter = analysis({
  classification: "full_integer",
  quantizedComputeOps: 1,
  opStateCounts: [{ name: "full_integer", count: 1 }],
});
inconsistentLegacyCounter.residual_step_response = {
  status: "partial",
  candidate_add_count: 0,
  assessed_add_count: 0,
  residual_adds: Array.from({ length: 32 }, () => ({ assessment_status: "not_assessed" })),
};
expectEqual(
  buildQuantResearchCoverage(inconsistentLegacyCounter).labs.find((row) => row.id === "residual_step_response")?.status,
  "not_assessed",
  "A zero-assessed legacy lab must remain not-assessed even when a stale scalar counter disagrees with its 32-row ledger.",
);

const floating = analysis({
  classification: "not_quantized_float",
  quantizedComputeOps: 0,
  opStateCounts: [],
  quantizedTensors: 0,
  scannedWeights: 0,
});
const floatCoverage = buildQuantResearchCoverage(floating);
expectEqual(floatCoverage.artifact_class, "float", "Float artifacts should be a first-class research class.");
expectEqual(floatCoverage.class_supported_lab_count, 0, "Float artifacts should expose a zero-lab quantization research envelope.");
expectEqual(floatCoverage.artifact_applicable_lab_count, 0, "A float artifact without quantized constants should have no applicable quant research lab.");
expectEqual(floatCoverage.not_applicable_lab_count, 15, "Every lab should be explicitly excluded when no quantized storage is present.");

const onnx = { format: "onnx", quant_research_coverage: { schema: "stale-tflite-only-object" } };
const onnxCoverage = ensureQuantResearchCoverage(onnx);
expectEqual(onnxCoverage.artifact_class, "not_applicable_format", "ONNX viewers should receive the common non-applicable quant-research policy.");
expect(!Object.hasOwn(onnx, "quant_research_coverage"), "ONNX analysis must not retain a TFLite-only quant-research computation object.");

const float16WeightOnly = analysis({
  classification: "float16_weight_storage",
  quantizedComputeOps: 0,
  opStateCounts: [{ name: "float16_constant_expansion", count: 4 }],
  quantizedTensors: 0,
  scannedWeights: 0,
});
float16WeightOnly.tensors = [{
  dtype: "FLOAT16",
  constant_buffer: true,
}];
const float16Coverage = buildQuantResearchCoverage(float16WeightOnly);
expectEqual(float16Coverage.artifact_class, "weight_only", "FLOAT16 constants expanded by Q/DQ should classify as reduced-precision weight-only.");
expectEqual(float16Coverage.class_supported_lab_count, 1, "Weight-only artifacts should retain the stored-weight integrity class envelope.");
expectEqual(float16Coverage.artifact_applicable_lab_count, 0, "An INT8 scale-grid lab should remain not applicable to FLOAT16 constants.");

for (const coverage of [fullCoverage, dynamicCoverage, floatCoverage, float16Coverage]) {
  expectEqual(coverage.labs.length, QUANT_RESEARCH_LABS.length, "Coverage rows should conserve the canonical lab denominator.");
  expectEqual(
    coverage.assessed_lab_count + coverage.partial_lab_count + coverage.not_assessed_lab_count + coverage.not_applicable_lab_count,
    coverage.lab_count,
    "Assessment statuses should conserve the lab denominator.",
  );
  expectEqual(
    coverage.class_supported_lab_count + coverage.class_excluded_lab_count,
    coverage.lab_count,
    "Class-supported and class-excluded counts should conserve the lab denominator.",
  );
}

const fullIntegerPath = "C:/Users/junhw/Downloads/main_0604_v119_4_ckpt902087_int8.tflite";
const dynamicRangePath = "C:/Users/junhw/Downloads/uw20_volume_network_0528.tflite";
if (existsSync(fullIntegerPath) || existsSync(dynamicRangePath)) {
  initSync({ module: readFileSync("pkg/tflite_wasm_audit_bg.wasm") });
}
if (existsSync(fullIntegerPath)) {
  const bytes = new Uint8Array(readFileSync(fullIntegerPath));
  const artifact = analyze_tflite_for_target(bytes, "main_0604_v119_4_ckpt902087_int8.tflite", "android_mid_a55");
  const coverage = buildQuantResearchCoverage(artifact);
  const glance = buildModelAtGlance(artifact);
  expectEqual(coverage.artifact_class, "full_integer", "The external 93-op regression artifact should classify as full-integer.");
  expectEqual(coverage.class_supported_lab_count, 15, "The full-integer regression should retain all 15 labs.");
  expectEqual(glance.delegation.padFusion.candidateCount, 3, "All three explicit PAD nodes should enter the folding-candidate inventory.");
  expectEqual(glance.delegation.padFusion.directConvolutionCandidateCount, 3, "All three PAD nodes should have exactly one convolution-family consumer.");
  expect(glance.delegation.padFusion.rows.every((row) => row.consumerFamilies.length === 1
    && row.consumerFamilies[0] === "DEPTHWISE_CONV_2D"
    && row.directConvolutionCandidate
    && row.runtimeFusionObserved === false), "A direct-convolution PAD candidate must remain distinct from observed runtime fusion.");
  expect(glance.latency.range.padFusionRecoverableUpperBoundUs > 0, "The modeled low bound should expose the direct-convolution PAD candidate upper bound.");
  expect(buildStaticAnalysisExport(artifact).accumulator_atlas != null, "Applicable full-integer proof detail must remain in the static export.");
}
if (existsSync(dynamicRangePath)) {
  const bytes = new Uint8Array(readFileSync(dynamicRangePath));
  const artifact = analyze_tflite_for_target(bytes, "uw20_volume_network_0528.tflite", "android_mid_a55");
  const coverage = buildQuantResearchCoverage(artifact);
  expectEqual(coverage.artifact_class, "dynamic_range", "The external 238-op regression artifact should classify as dynamic-range.");
  expectEqual(coverage.class_supported_lab_count, 1, "The external regression should retain only stored-weight integrity.");
  const report = buildEngineeringReport(artifact, {
    identity: {
      filename: artifact.filename,
      format: artifact.format,
      sha256: artifact.model_sha256,
      target_label: artifact.target_profile?.label,
      operator_count: artifact.operator_count,
      tensor_count: artifact.tensor_count,
      total_macs: artifact.total_macs,
    },
  });
  expect(report.includes("## Quantization Research Coverage (DERIVED)")
    && report.includes("Dynamic-range quantized")
    && report.includes("1 / 15"), "The report should state the class and 1/15 denominator.");
  expect(!report.includes("## Requantization Fidelity Lab")
    && !report.includes("## Numerical ABI Propagation Lab")
    && !/evidence rejected/i.test(report)
    && !/#undefined|#000 ADD/.test(report)
    && !/maximum p99 NaN/.test(report), "Excluded dynamic-range labs should not render invalid detail, phantom ops, rejected evidence, or numeric NaN.");
  const mlBomDocument = buildMlBomDocument(artifact, {
    hash: artifact.model_sha256,
    fileSizeBytes: artifact.file_size,
    target: artifact.target_profile,
    targetId: artifact.target_profile?.id,
  });
  const evidence = buildEngineeringEvidenceDocument(artifact, {
    reportContext: {
      identity: {
        filename: artifact.filename,
        format: artifact.format,
        sha256: artifact.model_sha256,
        target_label: artifact.target_profile?.label,
        operator_count: artifact.operator_count,
        tensor_count: artifact.tensor_count,
        total_macs: artifact.total_macs,
      },
    },
    mlBomDocument,
  });
  expectEqual(
    evidence.evidence.static_analysis.quant_research_coverage.artifact_class,
    "dynamic_range",
    "The conformance-checked engineering bundle should preserve the first-class artifact class.",
  );
  for (const key of [
    "accumulator_atlas", "requantization_fidelity", "kernel_extremum_witness",
    "channel_vitality", "rounding_equivalence", "accumulator_reachability",
    "numerical_abi_propagation", "input_counterexample", "preprocessing_realizability",
    "quantization_lattice", "contract_migration", "residual_step_response",
    "residual_contract_distortion",
  ]) {
    expectEqual(evidence.evidence.static_analysis[key], null, `Static export should suppress class-excluded ${key} detail.`);
  }
  for (const key of [
    "accumulator_headroom_atlas", "requantization_fidelity", "kernel_extremum_witness",
    "channel_vitality", "rounding_equivalence", "accumulator_reachability",
    "numerical_abi_propagation", "input_counterexample", "preprocessing_realizability",
    "residual_quantization_lattice", "residual_contract_migration", "residual_step_response",
    "residual_contract_distortion",
  ]) {
    expectEqual(evidence.evidence.quantization[key], null, `Quantization export should suppress class-excluded ${key} detail.`);
  }
  expect(evidence.evidence.metric_coverage_manifest.field_coverage.applicability_suppressed_field_pattern_count > 0
    && evidence.evidence.metric_coverage_manifest.field_coverage.unbound_field_pattern_count === 0,
  "Class-excluded source detail should be accounted as intentionally suppressed, never unbound.");
}

done(`full-integer 15/15, dynamic-range 1/15, float 0/15, denominator conservation, ${existsSync(fullIntegerPath) ? "PAD direct-consumer candidates" : "synthetic PAD policy"}, and ${existsSync(dynamicRangePath) ? "external report suppression" : "synthetic report policy"} passed`);

function analysis({
  classification,
  quantizedComputeOps,
  opStateCounts,
  quantizedTensors = 39,
  scannedWeights = 39,
  zeroAssessedEvidence = false,
}) {
  const value = {
    format: "tflite",
    quantized_tensors: quantizedTensors,
    quantization_status: {
      classification,
      quantized_compute_ops: quantizedComputeOps,
      op_state_counts: opStateCounts,
      int8_tensors: quantizedTensors,
      uint8_tensors: 0,
    },
    weight_integrity: {
      quantized_constant_tensors_scanned: scannedWeights,
    },
  };
  if (zeroAssessedEvidence) {
    for (const spec of QUANT_RESEARCH_LABS.filter((row) => row.evidenceKey && row.id !== "weight_scale_integrity")) {
      value[spec.evidenceKey] = {
        status: "partial",
        candidate_op_count: 67,
        assessed_op_count: 0,
        ops: [{ assessment_status: "not_assessed", reason_code: "LEGACY-LAB-GATE" }],
      };
    }
  }
  return value;
}

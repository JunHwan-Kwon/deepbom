import assert from "node:assert/strict";

import { deriveMacCoverage } from "../web/lib/mac-coverage.js";
import { buildMlBomCompatibilityProjection } from "../web/lib/report-mlbom-compat.js";

const fixtures = [
  {
    label: "TFLite complete MAC ledger",
    analysis: analysis("tflite", {
      total_macs: 300,
      ops: [{ index: 0, name: "CONV_2D", macs: 300 }, { index: 1, name: "RESHAPE", macs: 0 }],
      quantization_status: quant({ compute_ops: 1, quantized_compute_ops: 1, quantized_compute_mac_percent: 1 }),
    }),
    expected: { status: "assessed", compute: 1, assessed: 1, quantized: 1, quantAssessment: "assessed" },
  },
  {
    label: "ONNX explicit partial MAC ledger",
    analysis: analysis("onnx", {
      mac_assessment: { status: "not_assessed_incomplete_mac_ledger", compute_ops: 3, assessed_compute_ops: 2 },
      quantization_status: quant({ compute_ops: 3, quantized_compute_ops: 1, quantized_compute_mac_percent: null }),
    }),
    expected: { status: "not_assessed_incomplete_mac_ledger", compute: 3, assessed: 2, quantized: 1, quantAssessment: "not_assessed_mac_coverage_incomplete" },
  },
  {
    label: "Core ML decoded compute with unbound execution precision",
    analysis: analysis("coreml", {
      mac_assessment: { status: "assessed_all_decoded_compute_ops", compute_ops: 5, assessed_compute_ops: 5 },
      quantization_status: quant({ compute_ops: null, quantized_compute_ops: null, quantized_compute_mac_percent: null }),
    }),
    expected: { status: "assessed_all_decoded_compute_ops", compute: 5, assessed: 5, quantized: null, quantAssessment: "not_assessed_execution_precision_not_serialized" },
  },
  ...["gguf", "safetensors"].map((format) => ({
    label: `${format} weight container`,
    analysis: analysis(format, { quantization_status: quant({ compute_ops: 0, quantized_compute_ops: 0, quantized_compute_mac_percent: null }) }),
    expected: { status: "not_applicable_weight_container", compute: 0, assessed: 0, quantized: 0, quantAssessment: "not_applicable_weight_container" },
  })),
];

for (const fixture of fixtures) {
  const coverage = deriveMacCoverage(fixture.analysis, fixture.analysis.quantization_status);
  assert.equal(coverage.status, fixture.expected.status, `${fixture.label}: MAC status`);
  assert.equal(coverage.compute_ops, fixture.expected.compute, `${fixture.label}: compute denominator`);
  assert.equal(coverage.assessed_compute_ops, fixture.expected.assessed, `${fixture.label}: assessed numerator`);
  assert.ok(coverage.compute_ops == null || coverage.assessed_compute_ops == null
    || coverage.assessed_compute_ops <= coverage.compute_ops, `${fixture.label}: MAC coverage conservation`);

  const projection = buildMlBomCompatibilityProjection(fixture.analysis);
  const values = new Map(projection.componentProperties.map((row) => [row.name, row.value]));
  assert.equal(number(values.get("mlbom:model:macComputeOps")), fixture.expected.compute, `${fixture.label}: ML-BOM MAC denominator`);
  assert.equal(number(values.get("mlbom:model:macAssessedComputeOps")), fixture.expected.assessed, `${fixture.label}: ML-BOM MAC numerator`);
  assert.equal(number(values.get("mlbom:model:computeOperators")), fixture.expected.compute, `${fixture.label}: ML-BOM compute operators`);
  assert.equal(number(values.get("mlbom:model:quantizedComputeOperators")), fixture.expected.quantized, `${fixture.label}: ML-BOM quantized compute operators`);
  assert.equal(values.get("mlbom:model:quantizedComputeMacAssessment"), fixture.expected.quantAssessment, `${fixture.label}: quantized-MAC assessment reason`);
  if (fixture.expected.quantized != null) {
    assert.ok(fixture.expected.quantized <= fixture.expected.compute, `${fixture.label}: quantized compute count conservation`);
  }
}

const coreMl = new Map(buildMlBomCompatibilityProjection(fixtures[2].analysis).componentProperties.map((row) => [row.name, row.value]));
assert.equal(coreMl.has("mlbom:model:quantizedComputeOperators"), false, "Core ML unknown execution precision must not be coerced to zero.");
assert.equal(coreMl.get("mlbom:model:quantizedComputeOperatorAssessmentReason"),
  "serialized_weights_do_not_establish_executed_operator_precision");

console.log(`ML-BOM conservation checks passed: ${fixtures.length} format contracts, explicit MAC numerators and denominators, and no Core ML unknown-to-zero coercion.`);

function analysis(format, fields = {}) {
  return {
    format,
    filename: `fixture.${format}`,
    operator_count: fields.ops?.length ?? fields.mac_assessment?.compute_ops ?? 0,
    tensor_count: 0,
    ops: [],
    tensors: [],
    inputs: [],
    outputs: [],
    ...fields,
  };
}

function quant(fields) {
  return {
    label: "fixture quantization contract",
    classification: "fixture",
    full_integer: null,
    compute_macs: null,
    quantized_compute_macs: null,
    ...fields,
  };
}

function number(value) {
  return value == null ? null : Number(value);
}

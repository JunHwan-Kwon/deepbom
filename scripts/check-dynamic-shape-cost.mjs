import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { createHash } from "node:crypto";
import { analyzeOnnxModel } from "../web/onnx.js";
import {
  buildOnnxDynamicShapeCostContract,
  deriveTfliteBatchOneProjection,
  evaluateDynamicIntegerFormula,
} from "../web/lib/dynamic-shape-cost.js";
import { verifyDynamicShapeCostEvidence } from "../web/lib/dynamic-shape-cost-conformance.js";
import { buildEngineeringReport, buildEngineeringReportArtifacts } from "../web/lib/report-engineering.js";
import { buildFindingsRegister } from "../web/lib/report-findings.js";
import { buildMlBomDocument } from "../web/lib/report-mlbom.js";
import { buildEngineeringEvidenceDocument } from "../web/lib/report-evidence.js";
import { buildModelAtGlance } from "../web/lib/model-glance.js";
import { estimateOpBottleneck, macDistributionData, normalizeUnassessedCostValues } from "../web/lib/analysis.js";
import { formatUs } from "../web/lib/format.js";
import { buildAuditSnapshot } from "../web/lib/report-store.js";
import { evaluateOnnxDimensionExpression, parseOnnxDimensionExpression } from "../web/lib/onnx-dimension-expression.js";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

function expect(condition, message) {
  if (!condition) throw new Error(message);
}

function typeProto(dimensions) {
  return {
    shapeDimensions: dimensions.map((dimension) => typeof dimension === "string"
      ? { kind: "symbolic", parameter: dimension }
      : { kind: "value", value: dimension }),
  };
}

function tensor(index, name, shape, dtype, {
  role = "",
  constant = false,
  dimensions = shape,
  shapeDeclared = true,
  runtimeDimensionBounds = [],
} = {}) {
  return {
    index,
    name,
    shape,
    shape_declared: shapeDeclared,
    shape_signature: shape.map((dimension) => dimension >= 0 ? dimension : -1),
    dtype,
    value_kind: "tensor",
    type_proto: typeProto(dimensions),
    runtime_dimension_bounds: runtimeDimensionBounds,
    role,
    constant_buffer: constant,
  };
}

function dynamicConvAnalysis() {
  const tensors = [
    tensor(0, "input", [-1, 3, 32, 32], "FLOAT32", { role: "input", dimensions: ["batch", 3, 32, 32] }),
    tensor(1, "weight", [16, 3, 3, 3], "FLOAT32", { role: "initializer", constant: true }),
    tensor(2, "output", [-1, 16, 30, 30], "FLOAT32", { role: "output", dimensions: ["batch", 16, 30, 30] }),
  ];
  const ops = [{
    index: 0,
    name: "Conv",
    standard_domain: true,
    inputs: [0, 1],
    outputs: [2],
    output_shapes: [[-1, 16, 30, 30]],
    onnx_attributes: [],
    macs: 0,
    macs_status: "not_assessed",
    macs_reason: "dynamic output dimension",
    estimated_bytes: null,
  }];
  const contract = buildOnnxDynamicShapeCostContract(tensors, ops);
  return {
    schema: "deepbom.static_analysis.v1.68",
    format: "onnx",
    filename: "synthetic_dynamic_conv.onnx",
    file_size: 1,
    model_sha256: "0".repeat(64),
    version: 8,
    subgraphs: 1,
    graph_name: "dynamic_conv_cost",
    producer: "deepbom-test",
    onnx_ir_version: 8,
    opsets: [{ domain: "", version: 13 }],
    operator_count: 1,
    tensor_count: 3,
    tensors,
    ops,
    inputs: [tensors[0]],
    outputs: [tensors[2]],
    input_tensor_indices: [0],
    output_tensor_indices: [2],
    input_contracts: [],
    histogram: [{ name: "Conv", count: 1 }],
    stages: [],
    patterns: [],
    tensor_types: [{ name: "FLOAT32", count: 3 }],
    quantized_tensors: 0,
    per_channel_tensors: 0,
    total_macs: null,
    total_ops: null,
    mac_assessment: {
      status: "not_assessed",
      total_assessed_macs: 0,
      compute_ops: 1,
      assessed_compute_ops: 0,
      not_assessed_compute_ops: 1,
      not_assessed: [{ index: 0, name: "Conv", reason: "dynamic output dimension" }],
    },
    dynamic_shape_cost_contract: contract,
    target_profile: {},
    metadata_presence: { format: "onnx", schema: "test", status: "absent" },
    size_breakdown: { status: "assessed", file_size: 1, constant_tensor_count: 1, embedded_constant_tensor_count: 1 },
    tensor_liveness: { status: "partial", peak_bytes: null, assessed_tensor_count: 1, unassessed_tensor_count: 2 },
    weight_integrity: { status: "not_assessed", coverage_status: "not_assessed" },
    quantization_status: { classification: "float", quantized_tensor_percent: 0 },
    runtime_compat: {},
    onnx_sections_suppressed: [],
    roofline_csv: "",
    stage_mermaid: "",
  };
}

const analysis = dynamicConvAnalysis();
const contract = analysis.dynamic_shape_cost_contract;
const onnxGlance = buildModelAtGlance(analysis);
expect(onnxGlance.artifact.totalMacs === null, "an incomplete ONNX MAC ledger must remain null in the model-at-a-glance projection");
expect(onnxGlance.artifact.totalMacsEvidenceClass === "NOT_ASSESSED_INCOMPLETE_MAC_LEDGER", "an incomplete ONNX MAC ledger must carry an explicit evidence class");
const onnxMacDistribution = macDistributionData(analysis);
expect(onnxMacDistribution.coverageComplete === false && onnxMacDistribution.totalMacs === 0, "an unassessed ONNX MAC distribution must expose zero assessed subtotal without claiming complete coverage");
const tfliteMacDistribution = macDistributionData({
  format: "tflite",
  total_macs: 30,
  ops: [{ index: 0, name: "CONV_2D", macs: 30 }],
});
expect(tfliteMacDistribution.coverageComplete === true && tfliteMacDistribution.totalMacs === 30, "a TFLite complete total must remain complete without an ONNX MAC-assessment ledger");
expect(contract.status === "assessed", `dynamic Conv contract should be assessed, got ${contract.status}`);
expect(contract.symbol_count === 1, "repeated ONNX dim_param should bind one symbol");
expect(contract.symbols[0].occurrences.length === 2, "batch symbol should retain both tensor-axis occurrences");
expect(contract.tensor_formulas.find((row) => row.tensor_index === 0)?.payload_bytes_formula?.expression === "12288*D0", "input payload formula mismatch");
expect(contract.tensor_formulas.find((row) => row.tensor_index === 2)?.payload_bytes_formula?.expression === "57600*D0", "output payload formula mismatch");
expect(contract.op_formulas[0]?.macs_formula?.expression === "388800*D0", "Conv MAC polynomial mismatch");
expect(contract.total_macs_formula?.expression === "388800*D0", "total MAC polynomial mismatch");
expect(contract.liveness?.peak_live_payload_formula?.expression === "69888*D0", "symbolic peak live-payload polynomial mismatch");
expect(evaluateDynamicIntegerFormula(contract.total_macs_formula, { D0: 4 }) === 1_555_200n, "bound batch-4 MAC evaluation mismatch");

const rankNConv = buildOnnxDynamicShapeCostContract([
  tensor(0, "x3d", [-1, 2, 5, 6, 7], "FLOAT32", { dimensions: ["batch3d", 2, 5, 6, 7] }),
  tensor(1, "w3d", [4, 2, 3, 2, 2], "FLOAT32", { constant: true }),
  tensor(2, "y3d", [-1, 4, 3, 5, 6], "FLOAT32", { dimensions: ["batch3d", 4, 3, 5, 6] }),
], [{ index: 0, name: "Conv", standard_domain: true, inputs: [0, 1], outputs: [2], onnx_attributes: [], macs_status: "not_assessed" }]);
expect(rankNConv.op_formulas[0]?.macs_formula?.expression === "8640*D0", "dynamic Conv3D should preserve every spatial and reduction axis");

const outputBoundConv = buildOnnxDynamicShapeCostContract([
  tensor(0, "rank_omitted_x", [], "FLOAT32", { shapeDeclared: false }),
  tensor(1, "rank_omitted_w", [256, 322, 1], "FLOAT32", { constant: true }),
  tensor(2, "rank_omitted_y", [-1, 256, -1], "FLOAT32"),
], [{ index: 0, name: "Conv", standard_domain: true, inputs: [0, 1], outputs: [2], onnx_attributes: [{ name: "group", int_value: 1 }], macs_status: "not_assessed" }]);
expect(outputBoundConv.op_formulas[0]?.macs_formula?.expression === "82432*D0*D1", "Conv should derive exact symbolic MACs from output and weight when only the input rank is omitted");
const outputBoundConvVerification = verifyDynamicShapeCostEvidence({ analysis: { format: "onnx", tensors: [
  tensor(0, "rank_omitted_x", [], "FLOAT32", { shapeDeclared: false }),
  tensor(1, "rank_omitted_w", [256, 322, 1], "FLOAT32", { constant: true }),
  tensor(2, "rank_omitted_y", [-1, 256, -1], "FLOAT32"),
], ops: [{ index: 0, name: "Conv", standard_domain: true, inputs: [0, 1], outputs: [2], onnx_attributes: [{ name: "group", int_value: 1 }], macs_status: "not_assessed" }], dynamic_shape_cost_contract: outputBoundConv } });
expect(outputBoundConvVerification.op_formulas_valid && outputBoundConvVerification.total_macs_valid, "independent conformance should accept output/weight-derived Conv MACs when input rank is omitted");
const scalarInputConv = buildOnnxDynamicShapeCostContract([
  tensor(0, "scalar_x", [], "FLOAT32"),
  tensor(1, "scalar_w", [256, 322, 1], "FLOAT32", { constant: true }),
  tensor(2, "scalar_y", [-1, 256, -1], "FLOAT32"),
], [{ index: 0, name: "Conv", standard_domain: true, inputs: [0, 1], outputs: [2], onnx_attributes: [{ name: "group", int_value: 1 }], macs_status: "not_assessed" }]);
expect(scalarInputConv.op_formulas.length === 0, "Conv must reject an explicitly declared scalar input rank");

const dynamicConvTranspose = buildOnnxDynamicShapeCostContract([
  tensor(0, "tx", [-1, 8, 16], "FLOAT32", { dimensions: ["transpose_batch", 8, 16] }),
  tensor(1, "tw", [8, 4, 3], "FLOAT32", { constant: true }),
  tensor(2, "ty", [-1, 4, 18], "FLOAT32", { dimensions: ["transpose_batch", 4, 18] }),
], [{ index: 0, name: "ConvTranspose", standard_domain: true, inputs: [0, 1], outputs: [2], onnx_attributes: [], macs_status: "not_assessed" }]);
expect(dynamicConvTranspose.op_formulas[0]?.macs_formula?.expression === "1536*D0", "uncropped dynamic ConvTranspose should preserve the exact input-scatter MAC polynomial");
const dynamicConvTransposeVerification = verifyDynamicShapeCostEvidence({ analysis: { format: "onnx", tensors: [
  tensor(0, "tx", [-1, 8, 16], "FLOAT32", { dimensions: ["transpose_batch", 8, 16] }),
  tensor(1, "tw", [8, 4, 3], "FLOAT32", { constant: true }),
  tensor(2, "ty", [-1, 4, 18], "FLOAT32", { dimensions: ["transpose_batch", 4, 18] }),
], ops: [{ index: 0, name: "ConvTranspose", standard_domain: true, inputs: [0, 1], outputs: [2], onnx_attributes: [], macs_status: "not_assessed" }], dynamic_shape_cost_contract: dynamicConvTranspose } });
expect(dynamicConvTransposeVerification.op_formulas_valid && dynamicConvTransposeVerification.total_macs_valid, "independent conformance should reconstruct ConvTranspose symbolic MACs");

const croppedDynamicConvTranspose = buildOnnxDynamicShapeCostContract([
  tensor(0, "ctx", [-1, 8, 16], "FLOAT32", { dimensions: ["cropped_batch", 8, 16] }),
  tensor(1, "ctw", [8, 4, 3], "FLOAT32", { constant: true }),
  tensor(2, "cty", [-1, 4, 16], "FLOAT32", { dimensions: ["cropped_batch", 4, 16] }),
], [{ index: 0, name: "ConvTranspose", standard_domain: true, inputs: [0, 1], outputs: [2], onnx_attributes: [{ name: "pads", int_values: [1, 1] }], macs_status: "not_assessed" }]);
expect(croppedDynamicConvTranspose.op_formulas[0]?.formula_status === "exact_guarded_integer_expression", "cropped dynamic ConvTranspose should emit a guarded exact expression");
expect(croppedDynamicConvTranspose.op_formulas[0]?.macs_formula?.expression.includes("conv_transpose_pairs"), "cropped ConvTranspose should preserve exact spatial pair counting");
expect(croppedDynamicConvTranspose.total_macs_unresolved_op_count === 0, "cropped ConvTranspose should no longer block the total MAC expression");
expect(evaluateDynamicIntegerFormula(croppedDynamicConvTranspose.total_macs_formula, { D0: 2 }) === 2944n, "cropped ConvTranspose pair-count evaluation mismatch");
expect(evaluateDynamicIntegerFormula(croppedDynamicConvTranspose.total_macs_formula, { D0: 2, D999: 1 }) === 2944n, "irrelevant assignments must not change a guarded formula");
const croppedVerification = verifyDynamicShapeCostEvidence({ analysis: { format: "onnx", tensors: [
  tensor(0, "ctx", [-1, 8, 16], "FLOAT32", { dimensions: ["cropped_batch", 8, 16] }),
  tensor(1, "ctw", [8, 4, 3], "FLOAT32", { constant: true }),
  tensor(2, "cty", [-1, 4, 16], "FLOAT32", { dimensions: ["cropped_batch", 4, 16] }),
], ops: [{ index: 0, name: "ConvTranspose", standard_domain: true, inputs: [0, 1], outputs: [2], onnx_attributes: [{ name: "pads", int_values: [1, 1] }], macs_status: "not_assessed" }], dynamic_shape_cost_contract: croppedDynamicConvTranspose } });
expect(croppedVerification.op_formulas_valid && croppedVerification.total_macs_valid, "independent conformance should reconstruct cropped ConvTranspose pair counting and guards");

const rankOmittedConvTranspose = buildOnnxDynamicShapeCostContract([
  tensor(0, "otx", [], "FLOAT32", { shapeDeclared: false }),
  tensor(1, "otw", [2050, 1, 2048], "FLOAT32", { constant: true }),
  tensor(2, "oty", [-1, 1, -1], "FLOAT32"),
], [{ index: 0, name: "ConvTranspose", standard_domain: true, inputs: [0, 1], outputs: [2], onnx_attributes: [{ name: "strides", int_values: [320] }], macs_status: "not_assessed" }]);
expect(rankOmittedConvTranspose.op_formulas[0]?.formula_status === "exact_guarded_integer_expression", "rank-omitted ConvTranspose should emit a guarded inverse expression");
expect(rankOmittedConvTranspose.op_formulas[0]?.macs_formula?.preconditions.some((guard) => guard.kind === "divisible"), "rank-omitted ConvTranspose should expose exact divisibility");
expect(evaluateDynamicIntegerFormula(rankOmittedConvTranspose.total_macs_formula, { D0: 1, D1: 3328 }) === 20_992_000n, "rank-omitted ConvTranspose inverse evaluation mismatch");
expect(evaluateDynamicIntegerFormula(rankOmittedConvTranspose.total_macs_formula, { D0: 1, D1: 3329 }) === null, "rank-omitted ConvTranspose must reject a non-divisible output extent");
const rankOmittedVerification = verifyDynamicShapeCostEvidence({ analysis: { format: "onnx", tensors: [
  tensor(0, "otx", [], "FLOAT32", { shapeDeclared: false }),
  tensor(1, "otw", [2050, 1, 2048], "FLOAT32", { constant: true }),
  tensor(2, "oty", [-1, 1, -1], "FLOAT32"),
], ops: [{ index: 0, name: "ConvTranspose", standard_domain: true, inputs: [0, 1], outputs: [2], onnx_attributes: [{ name: "strides", int_values: [320] }], macs_status: "not_assessed" }], dynamic_shape_cost_contract: rankOmittedConvTranspose } });
expect(rankOmittedVerification.op_formulas_valid && rankOmittedVerification.total_macs_valid, "independent conformance should reconstruct rank-omitted ConvTranspose inverse guards");

const dynamicAttention = buildOnnxDynamicShapeCostContract([
  tensor(0, "q", [-1, 8, -1, 64], "FLOAT32", { dimensions: ["batch", 8, "query_tokens", 64] }),
  tensor(1, "k", [-1, 4, -1, 64], "FLOAT32", { dimensions: ["batch", 4, "kv_tokens", 64] }),
  tensor(2, "v", [-1, 4, -1, 80], "FLOAT16", { dimensions: ["batch", 4, "kv_tokens", 80] }),
  tensor(3, "past_k", [-1, 4, -1, 64], "FLOAT32", { dimensions: ["batch", 4, "past_tokens", 64] }),
  tensor(4, "past_v", [-1, 4, -1, 80], "FLOAT16", { dimensions: ["batch", 4, "past_tokens", 80] }),
  tensor(5, "y", [-1, 8, -1, 80], "FLOAT32", { dimensions: ["batch", 8, "query_tokens", 80] }),
], [{ index: 0, name: "Attention", standard_domain: true, inputs: [0, 1, 2, "", 3, 4], outputs: [5], onnx_attributes: [], macs_status: "not_assessed" }]);
expect(dynamicAttention.op_formulas[0]?.macs_formula?.expression.includes("1152"), "dynamic Attention should preserve q-head*(QK-head+V-head) as an exact coefficient");
expect(evaluateDynamicIntegerFormula(dynamicAttention.total_macs_formula, { D0: 2, D1: 16, D2: 20, D3: 5 }) === 921_600n, "dynamic Attention formula should include incoming and past KV sequence terms exactly");
const dynamicAttentionVerification = verifyDynamicShapeCostEvidence({ analysis: {
  format: "onnx",
  tensors: [
    tensor(0, "q", [-1, 8, -1, 64], "FLOAT32", { dimensions: ["batch", 8, "query_tokens", 64] }),
    tensor(1, "k", [-1, 4, -1, 64], "FLOAT32", { dimensions: ["batch", 4, "kv_tokens", 64] }),
    tensor(2, "v", [-1, 4, -1, 80], "FLOAT16", { dimensions: ["batch", 4, "kv_tokens", 80] }),
    tensor(3, "past_k", [-1, 4, -1, 64], "FLOAT32", { dimensions: ["batch", 4, "past_tokens", 64] }),
    tensor(4, "past_v", [-1, 4, -1, 80], "FLOAT16", { dimensions: ["batch", 4, "past_tokens", 80] }),
    tensor(5, "y", [-1, 8, -1, 80], "FLOAT32", { dimensions: ["batch", 8, "query_tokens", 80] }),
  ],
  ops: [{ index: 0, name: "Attention", standard_domain: true, inputs: [0, 1, 2, "", 3, 4], outputs: [5], onnx_attributes: [], macs_status: "not_assessed" }],
  dynamic_shape_cost_contract: dynamicAttention,
} });
expect(dynamicAttentionVerification.op_formulas_valid && dynamicAttentionVerification.total_macs_valid, "independent conformance should reconstruct dynamic Attention MACs");

const dynamicDeformConv = buildOnnxDynamicShapeCostContract([
  tensor(0, "deform_x", [-1, 8, -1, -1], "FLOAT32", { dimensions: ["batch", 8, "height", "width"] }),
  tensor(1, "deform_w", [12, 4, 3, 3], "FLOAT32", { constant: true }),
  tensor(2, "deform_offset", [-1, 36, -1, -1], "FLOAT32", { dimensions: ["batch", 36, "out_h", "out_w"] }),
  tensor(3, "deform_y", [-1, 12, -1, -1], "FLOAT32", { dimensions: ["batch", 12, "out_h", "out_w"] }),
], [{ index: 0, name: "DeformConv", standard_domain: true, inputs: [0, 1, 2], outputs: [3], onnx_attributes: [{ name: "group", int_value: 2 }, { name: "offset_group", int_value: 2 }], macs_status: "not_assessed" }]);
expect(evaluateDynamicIntegerFormula(dynamicDeformConv.total_macs_formula, { D0: 2, D1: 16, D2: 16, D3: 14, D4: 14 }) === 169_344n, "dynamic DeformConv should count its sampled-value/weight contraction independently of input spatial symbols");
const dynamicDeformConvVerification = verifyDynamicShapeCostEvidence({ analysis: {
  format: "onnx",
  tensors: [
    tensor(0, "deform_x", [-1, 8, -1, -1], "FLOAT32", { dimensions: ["batch", 8, "height", "width"] }),
    tensor(1, "deform_w", [12, 4, 3, 3], "FLOAT32", { constant: true }),
    tensor(2, "deform_offset", [-1, 36, -1, -1], "FLOAT32", { dimensions: ["batch", 36, "out_h", "out_w"] }),
    tensor(3, "deform_y", [-1, 12, -1, -1], "FLOAT32", { dimensions: ["batch", 12, "out_h", "out_w"] }),
  ],
  ops: [{ index: 0, name: "DeformConv", standard_domain: true, inputs: [0, 1, 2], outputs: [3], onnx_attributes: [{ name: "group", int_value: 2 }, { name: "offset_group", int_value: 2 }], macs_status: "not_assessed" }],
  dynamic_shape_cost_contract: dynamicDeformConv,
} });
expect(dynamicDeformConvVerification.op_formulas_valid && dynamicDeformConvVerification.total_macs_valid, "independent conformance should reconstruct dynamic DeformConv MACs");

const dynamicEinsum = buildOnnxDynamicShapeCostContract([
  tensor(0, "einsum_a", [-1, -1, 64], "FLOAT32", { dimensions: ["batch", "rows", 64] }),
  tensor(1, "einsum_b", [-1, 64, -1], "FLOAT32", { dimensions: ["batch", 64, "columns"] }),
  tensor(2, "einsum_y", [-1, -1, -1], "FLOAT32", { dimensions: ["batch", "rows", "columns"] }),
], [{ index: 0, name: "Einsum", standard_domain: true, inputs: [0, 1], outputs: [2], onnx_attributes: [{ name: "equation", string_value: "bij,bjk->bik" }], macs_status: "not_assessed" }]);
expect(evaluateDynamicIntegerFormula(dynamicEinsum.total_macs_formula, { D0: 2, D1: 3, D2: 5 }) === 1_920n, "dynamic two-input Einsum should multiply every unique Einstein index domain exactly");
const dynamicEinsumVerification = verifyDynamicShapeCostEvidence({ analysis: {
  format: "onnx",
  tensors: [
    tensor(0, "einsum_a", [-1, -1, 64], "FLOAT32", { dimensions: ["batch", "rows", 64] }),
    tensor(1, "einsum_b", [-1, 64, -1], "FLOAT32", { dimensions: ["batch", 64, "columns"] }),
    tensor(2, "einsum_y", [-1, -1, -1], "FLOAT32", { dimensions: ["batch", "rows", "columns"] }),
  ],
  ops: [{ index: 0, name: "Einsum", standard_domain: true, inputs: [0, 1], outputs: [2], onnx_attributes: [{ name: "equation", string_value: "bij,bjk->bik" }], macs_status: "not_assessed" }],
  dynamic_shape_cost_contract: dynamicEinsum,
} });
expect(dynamicEinsumVerification.op_formulas_valid && dynamicEinsumVerification.total_macs_valid, "independent conformance should reconstruct dynamic two-input Einsum MACs");

const orderDependentDynamicEinsum = buildOnnxDynamicShapeCostContract([
  tensor(0, "einsum3_a", [-1, 3], "FLOAT32", { dimensions: ["rows", 3] }),
  tensor(1, "einsum3_b", [3, 4], "FLOAT32"), tensor(2, "einsum3_c", [4, 5], "FLOAT32"),
  tensor(3, "einsum3_y", [-1, 5], "FLOAT32", { dimensions: ["rows", 5] }),
], [{ index: 0, name: "Einsum", standard_domain: true, inputs: [0, 1, 2], outputs: [3], onnx_attributes: [{ name: "equation", string_value: "ab,bc,cd->ad" }], macs_status: "not_assessed" }]);
expect(orderDependentDynamicEinsum.total_macs_unresolved_op_count === 1, "dynamic three-input Einsum must remain unresolved without a serialized contraction order");

const rankOmittedMatMul = buildOnnxDynamicShapeCostContract([
  tensor(0, "rankless_a", [], "FLOAT32", { shapeDeclared: false }),
  tensor(1, "rankless_b", [9, 1], "FLOAT32", { constant: true }),
  tensor(2, "rankless_y", [], "FLOAT32", { shapeDeclared: false }),
  tensor(3, "unrelated_dynamic", [-1], "FLOAT32", { dimensions: ["runtime_extent"] }),
], [{ index: 0, name: "MatMul", standard_domain: true, inputs: [0, 1], outputs: [2], onnx_attributes: [], macs_status: "not_assessed" }]);
expect(rankOmittedMatMul.total_macs_unresolved_ops.some((row) => row.reason.includes("rank is not fully serialized")), "rank-omitted MatMul residual should preserve the missing-rank contract");

const dynamicLstmTensors = [
  tensor(0, "lx", [-1, 2, 3], "FLOAT32", { dimensions: ["sequence", 2, 3] }),
  tensor(1, "lw", [1, 16, 3], "FLOAT32", { constant: true }),
  tensor(2, "lr", [1, 16, 4], "FLOAT32", { constant: true }),
  tensor(3, "ly", [-1, 1, 2, 4], "FLOAT32", { dimensions: ["sequence", 1, 2, 4] }),
];
const dynamicLstmOps = [{ index: 0, name: "LSTM", standard_domain: true, inputs: [0, 1, 2], outputs: [3], onnx_attributes: [{ name: "hidden_size", int_value: 4 }], macs_status: "not_assessed" }];
const dynamicLstm = buildOnnxDynamicShapeCostContract(dynamicLstmTensors, dynamicLstmOps);
expect(dynamicLstm.op_formulas[0]?.macs_formula?.expression === "224*D0", "dynamic LSTM should preserve exact sequence-dependent gate-contraction MACs");
const dynamicLstmVerification = verifyDynamicShapeCostEvidence({ analysis: { format: "onnx", tensors: dynamicLstmTensors, ops: dynamicLstmOps, dynamic_shape_cost_contract: dynamicLstm } });
expect(dynamicLstmVerification.op_formulas_valid && dynamicLstmVerification.total_macs_valid, "independent conformance should reconstruct dynamic LSTM MACs");

const derivedDimensionContract = buildOnnxDynamicShapeCostContract([
  tensor(0, "derived", [1, -1], "INT64", { dimensions: [1, "deepbom_expr:slice_len(v:514,v:0,s:sequence_length,1)"] }),
], []);
expect(derivedDimensionContract.symbols[0]?.source === "onnx_derived_dimension_expression", "Analyzer-derived ONNX dimensions must not be mislabeled as artifact-declared dim_param values");
expect(derivedDimensionContract.symbols[0]?.declared_name.startsWith("deepbom_expr:slice_len("), "Derived dimension provenance should retain the exact source expression");
expect(derivedDimensionContract.symbols[0]?.expression_ir?.operator === "slice_len", "Derived dimensions should expose a structured expression IR");
expect(derivedDimensionContract.symbols[0]?.expression_dependencies.includes("sequence_length"), "Expression IR should retain its source dim_param dependency");
expect(evaluateOnnxDimensionExpression(parseOnnxDimensionExpression("deepbom_expr:add(deepbom_expr:mul(s:N,v:2),v:5)"), { N: 3 }) === 11n, "Affine dimension-expression evaluation mismatch");
expect(evaluateOnnxDimensionExpression(parseOnnxDimensionExpression("deepbom_expr:ceil_div(deepbom_expr:add(s:N,v:5),v:4)"), { N: 6 }) === 3n, "Ceil-div dimension-expression evaluation mismatch");
expect(evaluateOnnxDimensionExpression(parseOnnxDimensionExpression("deepbom_expr:slice_len(s:N,v:-7,pos_inf,v:2)"), { N: 10 }) === 4n, "Positive-step Slice dimension-expression evaluation mismatch");
expect(evaluateOnnxDimensionExpression(parseOnnxDimensionExpression("deepbom_expr:range_len(v:9,v:0,v:-2)"), {}) === 5n, "Negative-step Range dimension-expression evaluation mismatch");
expect(evaluateOnnxDimensionExpression(parseOnnxDimensionExpression("deepbom_expr:broadcast_dim(s:L,s:R)"), { L: 1, R: 17 }) === 17n, "A one-sided symbolic broadcast should resolve to the non-unit extent");
expect(evaluateOnnxDimensionExpression(parseOnnxDimensionExpression("deepbom_expr:broadcast_dim(s:L,s:R)"), { L: 17, R: 17 }) === 17n, "Equal symbolic broadcast extents should resolve exactly");
expect(evaluateOnnxDimensionExpression(parseOnnxDimensionExpression("deepbom_expr:broadcast_dim(s:L,s:R)"), { L: 3, R: 17 }) === null, "Incompatible symbolic broadcast extents must fail closed");
expect(parseOnnxDimensionExpression("deepbom_expr:unknown(s:N)") == null, "Unsupported expression operators must fail closed");
const derivedDimensionVerification = verifyDynamicShapeCostEvidence({
  analysis: {
    format: "onnx",
    tensors: [tensor(0, "derived", [1, -1], "INT64", { dimensions: [1, "deepbom_expr:slice_len(v:514,v:0,s:sequence_length,1)"] })],
    ops: [],
    dynamic_shape_cost_contract: derivedDimensionContract,
  },
});
expect(derivedDimensionVerification.symbols_valid, "Independent conformance must reconstruct analyzer-derived expression identity without promoting it to artifact dim_param evidence");

const runtimeValueDimensionContract = buildOnnxDynamicShapeCostContract([
  tensor(0, "nonzero_input", [2, 3], "FLOAT32"),
  tensor(1, "nonzero_output", [2, -1], "INT64", {
    dimensions: [2, "deepbom_runtime:nnz:nonzero_input"],
    runtimeDimensionBounds: [{
      axis: 1,
      lower_bound_decimal: "0",
      upper_bound_expression: "6",
      source: "onnx_nonzero_cardinality",
    }],
  }),
], []);
expect(runtimeValueDimensionContract.symbols[0]?.source === "onnx_runtime_value_dimension", "Runtime cardinalities must not be mislabeled as artifact dim_param or analyzer-derived expressions");
expect(runtimeValueDimensionContract.symbols[0]?.lower_bound === 0, "NonZero runtime cardinality must retain its exact zero lower bound");
expect(runtimeValueDimensionContract.symbols[0]?.upper_bound_expression === "6", "NonZero runtime cardinality must retain its exact artifact-derived upper bound expression");
expect(runtimeValueDimensionContract.dimension_bounds_status === "artifact_derived_complete", "A fully bounded runtime dimension should report complete artifact-derived bounds");
const runtimeValueDimensionVerification = verifyDynamicShapeCostEvidence({ analysis: {
  format: "onnx",
  tensors: [
    tensor(0, "nonzero_input", [2, 3], "FLOAT32"),
    tensor(1, "nonzero_output", [2, -1], "INT64", {
      dimensions: [2, "deepbom_runtime:nnz:nonzero_input"],
      runtimeDimensionBounds: [{
        axis: 1,
        lower_bound_decimal: "0",
        upper_bound_expression: "6",
        source: "onnx_nonzero_cardinality",
      }],
    }),
  ],
  ops: [],
  dynamic_shape_cost_contract: runtimeValueDimensionContract,
} });
expect(runtimeValueDimensionVerification.symbols_valid, "Independent conformance must reconstruct runtime-value provenance and artifact-derived bounds");

const rankOneMatMul = buildOnnxDynamicShapeCostContract([
  tensor(0, "vector", [-1], "FLOAT32", { dimensions: ["k"] }),
  tensor(1, "matrix", [-1, 4], "FLOAT32", { constant: true, dimensions: ["k", 4] }),
  tensor(2, "product", [4], "FLOAT32"),
], [{ index: 0, name: "MatMul", standard_domain: true, inputs: [0, 1], outputs: [2], onnx_attributes: [], macs_status: "not_assessed" }]);
expect(rankOneMatMul.op_formulas[0]?.macs_formula?.expression === "4*D0", "dynamic vector-matrix MatMul should retain ONNX rank-1 promotion");

const invalidDynamicConvChannels = buildOnnxDynamicShapeCostContract([
  tensor(0, "bad_x", [-1, 3, 32, 32], "FLOAT32", { dimensions: ["bad_batch", 3, 32, 32] }),
  tensor(1, "bad_w", [16, 2, 3, 3], "FLOAT32", { constant: true }),
  tensor(2, "bad_y", [-1, 16, 30, 30], "FLOAT32", { dimensions: ["bad_batch", 16, 30, 30] }),
], [{ index: 0, name: "Conv", standard_domain: true, inputs: [0, 1], outputs: [2], onnx_attributes: [], macs_status: "not_assessed" }]);
expect(invalidDynamicConvChannels.op_formulas.length === 0, "dynamic Conv must reject a known input/weight channel mismatch before emitting a polynomial");
expect(invalidDynamicConvChannels.total_macs_unresolved_ops.some((row) => row.op_name === "Conv"), "invalid dynamic Conv must remain an explicit total-MAC blocker");

const conditionallyInvalidInput = tensor(0, "conditional_bad_x", [-1, 3, -1], "FLOAT32", { dimensions: ["batch", 3, "samples"] });
conditionallyInvalidInput.conditional_shape_contract = {
  status: "assessed_partial",
  variant_failures: [{
    status: "invalid",
    reason: "conv_input_weight_rank_mismatch",
    conditions: [{ key: "if:main_graph:node:7:condition:batch_is_one", value: "else_branch" }],
  }],
};
const conditionallyInvalidConv = buildOnnxDynamicShapeCostContract([
  conditionallyInvalidInput,
  tensor(1, "conditional_w", [4, 3, 3], "FLOAT32", { constant: true }),
  tensor(2, "conditional_y", [-1, 4, -1], "FLOAT32", { dimensions: ["batch", 4, "output_samples"] }),
], [{ index: 0, name: "Conv", standard_domain: true, inputs: [0, 1], outputs: [2], onnx_attributes: [], macs_status: "not_assessed" }]);
const conditionalBlocker = conditionallyInvalidConv.total_macs_unresolved_ops[0];
expect(conditionalBlocker?.resolution_class === "artifact_contract_conflict", "A conditionally invalid tensor branch must be classified as an artifact contract conflict, not an analyzer residual");
expect(conditionalBlocker?.root_conflicts?.[0]?.reason === "conv_input_weight_rank_mismatch", "Conditional MAC blockers must retain the exact invalid-branch reason");
expect(conditionalBlocker?.root_conflicts?.[0]?.conditions?.[0]?.value === "else_branch", "Conditional MAC blockers must retain the branch condition");

const invalidDynamicMatMulRank = buildOnnxDynamicShapeCostContract([
  tensor(0, "bad_a", [-1, 3], "FLOAT32", { dimensions: ["bad_m", 3] }),
  tensor(1, "bad_b", [3, 4], "FLOAT32", { constant: true }),
  tensor(2, "bad_product", [-1, 4, 1], "FLOAT32", { dimensions: ["bad_m", 4, 1] }),
], [{ index: 0, name: "MatMul", standard_domain: true, inputs: [0, 1], outputs: [2], onnx_attributes: [], macs_status: "not_assessed" }]);
expect(invalidDynamicMatMulRank.op_formulas.length === 0, "dynamic MatMul must reject an output rank that conflicts with ONNX rank promotion");

const tfliteBatchOnly = {
  format: "tflite",
  total_macs: 1000,
  inputs: [{ index: 0, name: "input", shape: [1, 144, 240, 3], shape_signature: [-1, 144, 240, 3] }],
  tensors: [
    { index: 0, name: "input", shape: [1, 144, 240, 3], shape_signature: [-1, 144, 240, 3] },
    { index: 1, name: "output", shape: [1, 8], shape_signature: [-1, 8] },
  ],
  dynamic_shape_cost_contract: {
    schema: "deepbom.dynamic_shape_cost_contract.v2.2",
    status: "assessed",
    format: "tflite",
    dynamic_tensor_count: 2,
    symbol_count: 2,
    symbols: [
      { symbol_id: "D0", occurrences: [{ tensor_index: 0, tensor_name: "input", axis: 0 }] },
      { symbol_id: "D1", occurrences: [{ tensor_index: 1, tensor_name: "output", axis: 0 }] },
    ],
    tensor_formulas: [
      { tensor_index: 0, shape: [1, 144, 240, 3], shape_signature: [-1, 144, 240, 3] },
      { tensor_index: 1, shape: [1, 8], shape_signature: [-1, 8] },
    ],
    total_macs_formula: {
      status: "exact_symbolic_integer_polynomial",
      terms: [{ coefficient_decimal: "1000", factors: [{ symbol_id: "D0", exponent: 1 }] }],
    },
    total_macs_formula_status: "exact_symbolic_integer_polynomial",
    liveness: {
      peak_live_payload_formula: {
        status: "exact_symbolic_integer_polynomial",
        terms: [
          { coefficient_decimal: "414720", factors: [{ symbol_id: "D0", exponent: 1 }] },
          { coefficient_decimal: "32", factors: [{ symbol_id: "D1", exponent: 1 }] },
        ],
      },
    },
  },
};
const batchProjection = deriveTfliteBatchOneProjection(tfliteBatchOnly);
expect(batchProjection.status === "assumption_bound_batch_one", "batch-only TFLite should expose an assumption-bound N=1 projection");
expect(batchProjection.projected_total_macs === 1000, "batch-only N=1 MAC projection mismatch");
expect(batchProjection.projected_peak_live_payload_bytes === 414752, "batch-only N=1 live-payload projection mismatch");
expect(batchProjection.non_batch_dynamic_axis_count === 0, "batch-only projection must report zero non-batch dynamic axes");
const batchGlance = buildModelAtGlance(tfliteBatchOnly);
expect(batchGlance.artifact.totalMacs === 1000, "batch-only dashboard should reuse the exact N=1 MAC projection");
expect(batchGlance.memory.peakLiveBytes === 414752, "batch-only dashboard should reuse the exact N=1 peak-live projection");
expect(batchGlance.artifact.totalMacsEvidenceClass === "ASSUMPTION_BOUND_N_EQ_1", "batch-only dashboard evidence class mismatch");
expect(!buildFindingsRegister(tfliteBatchOnly).some((finding) => finding.finding_id === "EA-DYN-0001"), "batch-only serialized N=1 projection should not emit the unbound-shape finding");

const tfliteSpatialDynamic = structuredClone(tfliteBatchOnly);
tfliteSpatialDynamic.inputs[0].shape_signature[1] = -1;
tfliteSpatialDynamic.dynamic_shape_cost_contract.symbols[0].occurrences[0].axis = 1;
expect(deriveTfliteBatchOneProjection(tfliteSpatialDynamic).status === "requires_explicit_shape_binding", "dynamic spatial axes must still require an explicit profile");
const spatialGlance = buildModelAtGlance(tfliteSpatialDynamic);
expect(spatialGlance.artifact.totalMacs == null, "spatially dynamic dashboard must not emit a serialized-example MAC total");
expect(spatialGlance.memory.peakLiveBytes == null, "spatially dynamic dashboard must suppress a serialized-example peak-live total");
expect(spatialGlance.latency.conservationStatus === "not_assessed_dynamic_shape", "spatially dynamic dashboard must suppress modeled latency until binding");
expect(spatialGlance.latency.totals.steadyStateUs == null, "unbound dynamic-shape steady latency must serialize as null");
expect(spatialGlance.latency.totals.coldStartUs == null, "unbound dynamic-shape cold latency must serialize as null");
expect(spatialGlance.latency.range.steadyPointUs == null, "unbound dynamic-shape latency range must serialize as null");
expect(spatialGlance.latency.range.reason.includes("explicit shape binding"), "unbound dynamic-shape latency must carry a blocker reason");
const unassessedOp = {
  macs_status: "not_assessed",
  macs_reason: "unbound H dimension",
  estimated_bytes_status: "not_assessed",
  bottleneck_compute_us: 0,
  bottleneck_memory_us: 0,
  bottleneck_packing_us: 0,
  bottleneck_break_us: 0,
  bottleneck_fallback_us: 0,
  bottleneck_total_us: 0,
};
normalizeUnassessedCostValues({ ops: [unassessedOp] });
expect(unassessedOp.bottleneck_total_us == null, "unassessed op cost must normalize to null rather than zero");
expect(unassessedOp.bottleneck_reason === "unbound H dimension", "unassessed op cost must retain its blocker reason");
expect(estimateOpBottleneck(unassessedOp).steadyStateUs == null, "unassessed bottleneck adapter must return null latency");
expect(formatUs(null) === "Not assessed", "null latency must never render as 0.00 us");
const dynamicSnapshot = buildAuditSnapshot({
  filename: "dynamic.onnx",
  format: "onnx",
  model_sha256: "a".repeat(64),
  target_profile: { id: "onnx-unbound" },
  ops: [unassessedOp],
});
expect(dynamicSnapshot.modeledCostStatus === "not_assessed", "saved snapshot must retain the unassessed modeled-cost state");
expect(dynamicSnapshot.totalEstUs == null, "saved snapshot must not coerce unassessed steady cost to zero");
expect(dynamicSnapshot.totalColdStartEstUs == null, "saved snapshot must not coerce unassessed cold cost to zero");
expect(buildFindingsRegister(tfliteSpatialDynamic).some((finding) => finding.finding_id === "EA-DYN-0001"), "dynamic spatial axes must remain in the action queue");

const anonymous = buildOnnxDynamicShapeCostContract([
  tensor(0, "a", [-1, 3], "FLOAT32", { role: "input", dimensions: [null, 3] }),
  tensor(1, "b", [-1, 3], "FLOAT32", { role: "output", dimensions: [null, 3] }),
], []);
expect(anonymous.symbol_count === 2, "anonymous ONNX dimensions must not be merged without an artifact equality constraint");

const packed = buildOnnxDynamicShapeCostContract([
  tensor(0, "packed", [-1, 3], "INT4", { role: "input", dimensions: ["batch", 3] }),
], []);
expect(packed.tensor_formulas[0]?.formula_status === "exact_symbolic_ceil_expression", "packed INT4 payload should retain a ceil expression when byte divisibility depends on the symbol");
expect(packed.tensor_formulas[0]?.payload_bytes_expression === "ceil((12*D0)/8)", "packed INT4 byte expression mismatch");

const unknownDtype = buildOnnxDynamicShapeCostContract([
  tensor(0, "opaque", [-1, 3], "STRING", { role: "input", dimensions: ["batch", 3] }),
], []);
expect(unknownDtype.status === "partial" && unknownDtype.tensor_formulas[0]?.formula_status === "not_assessed_dtype_storage_width", "unknown storage width must remain partial rather than becoming zero bytes");

const report = buildEngineeringReport(analysis);
const reportArtifacts = buildEngineeringReportArtifacts(analysis);
const mlBom = buildMlBomDocument(analysis);
const findings = buildFindingsRegister(analysis);
const verification = verifyDynamicShapeCostEvidence({ analysis, engineeringReport: report, mlBomDocument: mlBom, findingsRegister: findings });
expect(Object.values(verification).every(Boolean), `independent dynamic-shape verification failed: ${JSON.stringify(verification)}`);
expect(report.includes("388800*D0") && report.includes("69888*D0"), "Engineering Report should preserve exact dynamic MAC and live-payload formulas");
const dynamicMetric = reportArtifacts.metricCoverage?.entries?.find((entry) => entry.metric_id === "cost.dynamic_shape");
expect(dynamicMetric?.status === "assessed" && dynamicMetric.report_section === "## Dynamic Shape Cost Contract", "dynamic shape contract should be a first-class assessed metric family");
const missingDynamicReportFields = (reportArtifacts.metricCoverage?.field_coverage?.missing_required_report_field_paths || [])
  .filter((field) => field.startsWith("/dynamic_shape_cost_contract/"));
expect(!missingDynamicReportFields.length, `every required dynamic-shape decision field should be consumed by the Engineering Report: ${missingDynamicReportFields.join(", ")}`);
expect(findings.some((finding) => finding.finding_id === "EA-DYN-0001"), "dynamic shape contract should enter the authoritative action queue");

const tamperedCoefficient = structuredClone(analysis);
tamperedCoefficient.dynamic_shape_cost_contract.op_formulas[0].macs_formula.terms[0].coefficient_decimal = "388801";
tamperedCoefficient.dynamic_shape_cost_contract.op_formulas[0].macs_formula.expression = "388801*D0";
const tamperedCoefficientVerification = verifyDynamicShapeCostEvidence({
  analysis: tamperedCoefficient,
  engineeringReport: buildEngineeringReport(tamperedCoefficient),
  mlBomDocument: buildMlBomDocument(tamperedCoefficient),
  findingsRegister: buildFindingsRegister(tamperedCoefficient),
});
expect(!tamperedCoefficientVerification.op_formulas_valid, "self-consistent coefficient tampering must fail independent op recomputation");

const tamperedOccurrence = structuredClone(analysis);
tamperedOccurrence.dynamic_shape_cost_contract.symbols[0].occurrences.pop();
const tamperedOccurrenceVerification = verifyDynamicShapeCostEvidence({
  analysis: tamperedOccurrence,
  engineeringReport: buildEngineeringReport(tamperedOccurrence),
  mlBomDocument: buildMlBomDocument(tamperedOccurrence),
  findingsRegister: buildFindingsRegister(tamperedOccurrence),
});
expect(!tamperedOccurrenceVerification.symbols_valid, "symbol occurrence tampering must fail independent tensor-axis reconstruction");

const dynamicFixture = new Uint8Array(fs.readFileSync(path.join(ROOT, "scripts", "fixtures", "onnx_dynamic_conv.onnx")));
const parsedDynamic = analyzeOnnxModel(dynamicFixture, "onnx_dynamic_conv.onnx");
parsedDynamic.model_sha256 = createHash("sha256").update(dynamicFixture).digest("hex");
expect(parsedDynamic.dynamic_shape_cost_contract?.symbol_count === 1, "serialized ONNX dim_param should survive parsing as one shared cost symbol");
expect(parsedDynamic.dynamic_shape_cost_contract?.total_macs_formula?.expression === "388800*D0", "serialized ONNX dynamic Conv total formula mismatch");
const parsedIdentity = {
  filename: parsedDynamic.filename,
  format: "onnx",
  sha256: parsedDynamic.model_sha256,
  target_label: "ONNX static target posture",
};
const parsedMlBom = buildMlBomDocument(parsedDynamic, { hash: parsedDynamic.model_sha256 });
const parsedEvidence = buildEngineeringEvidenceDocument(parsedDynamic, {
  reportContext: { identity: parsedIdentity },
  rawEvidenceContext: { identity: parsedIdentity },
  mlBomDocument: parsedMlBom,
});
expect(parsedEvidence.evidence?.conformance_report?.status === "pass", "serialized dynamic ONNX should pass full Engineering Evidence conformance");

const bundleTamper = structuredClone(parsedDynamic);
bundleTamper.dynamic_shape_cost_contract.op_formulas[0].macs_formula.terms[0].coefficient_decimal = "388801";
bundleTamper.dynamic_shape_cost_contract.op_formulas[0].macs_formula.expression = "388801*D0";
let tamperError = "";
try {
  buildEngineeringEvidenceDocument(bundleTamper, {
    reportContext: { identity: parsedIdentity },
    rawEvidenceContext: { identity: parsedIdentity },
    mlBomDocument: buildMlBomDocument(bundleTamper, { hash: bundleTamper.model_sha256 }),
  });
} catch (error) {
  tamperError = String(error?.message || error);
}
expect(tamperError.includes("CF-DYNAMIC-003"), `self-consistent bundle coefficient tampering should be release-blocked by CF-DYNAMIC-003, got ${tamperError || "no error"}`);

const zeroFixture = new Uint8Array(fs.readFileSync(path.join(ROOT, "scripts", "fixtures", "onnx_zero_dim_identity.onnx")));
const parsedZero = analyzeOnnxModel(zeroFixture, "onnx_zero_dim_identity.onnx");
expect(parsedZero.inputs[0]?.shape_signature?.[0] === 0, "explicit ONNX zero dimension must remain zero in the public shape signature");
expect(parsedZero.dynamic_shape_cost_contract?.status === "not_applicable_static_shapes", "explicit zero dimension must not be classified as dynamic");

console.log("Dynamic shape cost contract checks passed.");

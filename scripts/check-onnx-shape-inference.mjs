import {
  inferOnnxShapes,
  ONNX_SHAPE_INFERENCE_OPS,
  ONNX_SHAPE_INFERENCE_SOURCE,
} from "../web/lib/onnx-shape-inference.js";
import { buildFindingsRegister } from "../web/lib/report-findings.js";
import { createCheck } from "./check-assert.mjs";

const { done, expect, expectEqual } = createCheck("ONNX shape inference check");

function tensor(name, dtype, shape, extra = {}) {
  return { name, dtype, shape: [...shape], shapeDeclared: true, ...extra };
}

function symbolicTensor(name, dtype, dimensions, extra = {}) {
  const shapeDimensions = dimensions.map((dimension) => typeof dimension === "number"
    ? { kind: "value", value: dimension, parameter: "", denotation: "", valueFieldCount: 1 }
    : { kind: "symbolic", value: null, parameter: dimension, denotation: "", valueFieldCount: 1 });
  return tensor(name, dtype, shapeDimensions.map((dimension) => dimension.kind === "value" ? dimension.value : -1), {
    typeProto: {
      kind: "tensor", dtype, shape: shapeDimensions.map((dimension) => dimension.kind === "value" ? dimension.value : -1),
      shapeDeclared: true, shapeDimensions, shapeFieldCount: 1, valueFieldsPresent: [1],
    },
    ...extra,
  });
}

function unknown(name, extra = {}) {
  return { name, dtype: "UNKNOWN", shape: [], shapeDeclared: false, ...extra };
}

function constant(name, values, shape = [values.length], dtype = "INT64") {
  return tensor(name, dtype, shape, {
    staticValuesStatus: "complete",
    staticValuesComplete: true,
    staticValues: [...values],
    staticValuesSource: "fixture",
  });
}

function canonicalIntegerTensor(name, values, shape = [values.length]) {
  return tensor(name, "INT64", shape, {
    staticValuesStatus: "complete_canonical_text_only_non_finite_or_unsafe_value",
    staticValuesComplete: false,
    staticValues: [],
    staticValuesCanonicalTextComplete: true,
    staticValuesCanonicalTexts: values.map(String),
    staticValuesSource: "fixture",
  });
}

function symbolicDimensionValuesTensor(name, parameters) {
  return tensor(name, "INT64", [parameters.length], {
    staticDimensionValuesStatus: "assessed_exact_static_shape_data",
    staticDimensionValuesComplete: true,
    staticDimensionValues: parameters.map((parameter) => typeof parameter === "number"
      ? { kind: "value", value: parameter, parameter: "", denotation: "", valueFieldCount: 1 }
      : { kind: "symbolic", value: null, parameter, denotation: "", valueFieldCount: 1 }),
    staticDimensionValuesSource: "fixture",
  });
}

function attribute({ f = null, i = null, ints = [], s = null, tensor: tensorValue = null } = {}) {
  return { f, i, ints, s, tensor: tensorValue, floats: [], strings: [], tensors: [] };
}

function node(opType, inputs, outputs, attributes = {}, domain = "") {
  return {
    opType,
    inputs,
    outputs,
    domain,
    attributes: new Map(Object.entries(attributes).map(([name, value]) => [name, attribute(value)])),
  };
}

function run(nodes, rows, opset = 21) {
  const tensorMap = new Map(rows.map((row) => [row.name, row]));
  for (const item of nodes) for (const name of [...item.inputs, ...item.outputs]) if (name && !tensorMap.has(name)) tensorMap.set(name, unknown(name));
  const imports = Array.isArray(opset) ? opset : [{ domain: "", version: opset }];
  const evidence = inferOnnxShapes({ nodes }, tensorMap, imports, (id) => ({ 1: "FLOAT32", 6: "INT32", 7: "INT64", 9: "BOOL", 10: "FLOAT16" })[id] || `TYPE_${id}`);
  return { evidence, tensorMap };
}

const shapeChain = run([
  node("Shape", ["x"], ["x_shape"]),
  node("Gather", ["x_shape", "batch_index"], ["batch_scalar"]),
  node("Unsqueeze", ["batch_scalar", "axis_zero"], ["batch_vector"]),
  node("Concat", ["batch_vector", "tail"], ["target"], { axis: { i: 0 } }),
  node("Reshape", ["x", "target"], ["reshaped"]),
], [
  tensor("x", "FLOAT32", [2, 3, 4]),
  constant("batch_index", [0], []),
  constant("axis_zero", [0], [1]),
  constant("tail", [12], [1]),
]);
expectEqual(JSON.stringify(shapeChain.tensorMap.get("x_shape").staticValues), JSON.stringify([2, 3, 4]), "Shape should propagate exact dimensions as INT64 data.");
expectEqual(JSON.stringify(shapeChain.tensorMap.get("target").staticValues), JSON.stringify([2, 12]), "Gather/Unsqueeze/Concat should conserve exact shape data.");
expectEqual(JSON.stringify(shapeChain.tensorMap.get("reshaped").shape), JSON.stringify([2, 12]), "Reshape should consume propagated shape data.");
expectEqual(shapeChain.evidence.known_node_output_count, 5, "Every output in the shape-data chain should be known.");
expectEqual(shapeChain.evidence.rule_unresolved_node_count, 0, "The shape-data chain should have no unresolved rule.");
expect(shapeChain.evidence.propagated_static_value_tensor_count >= 4, "The shape-data chain should report bounded static value propagation.");

const symbolicShapeChain = run([
  node("Shape", ["dynamic_x"], ["dynamic_shape"]),
  node("Gather", ["dynamic_shape", "dynamic_batch_index"], ["dynamic_batch_scalar"]),
  node("Unsqueeze", ["dynamic_batch_scalar", "dynamic_axis_zero"], ["dynamic_batch_vector"]),
  node("Concat", ["dynamic_batch_vector", "dynamic_tail"], ["dynamic_target"], { axis: { i: 0 } }),
  node("Reshape", ["dynamic_x", "dynamic_target"], ["dynamic_reshaped"]),
], [
  symbolicTensor("dynamic_x", "FLOAT32", ["batch_size", 3, 4]),
  constant("dynamic_batch_index", [0], []),
  constant("dynamic_axis_zero", [0], [1]),
  constant("dynamic_tail", [12], [1]),
]);
expectEqual(JSON.stringify(symbolicShapeChain.tensorMap.get("dynamic_reshaped").shape), JSON.stringify([-1, 12]), "Reshape should retain a symbolic batch dimension without inventing a numeric value.");
expectEqual(symbolicShapeChain.tensorMap.get("dynamic_reshaped").typeProto?.shapeDimensions?.[0]?.parameter, "batch_size", "The original ONNX dim_param identity should survive Shape/Gather/Unsqueeze/Concat/Reshape.");
expectEqual(symbolicShapeChain.evidence.rule_unresolved_node_count, 0, "An exactly traceable symbolic shape-data chain should not remain a local inference residual.");
expectEqual(symbolicShapeChain.evidence.shape_contract_unknown_node_output_count, 0, "Every symbolic output contract in the traceable chain should be rank/dimension complete.");
expect(symbolicShapeChain.evidence.unknown_node_output_count > 0, "Symbolically complete contracts must remain distinct from numerically concrete output shapes.");

const invalidConcatDtype = run([
  node("Concat", ["concat_float", "concat_int"], ["invalid_concat_output"], { axis: { i: 0 } }),
  node("Identity", ["invalid_concat_output"], ["invalid_concat_consumer"]),
], [tensor("concat_float", "FLOAT32", [1]), tensor("concat_int", "INT64", [1])]);
expectEqual(invalidConcatDtype.evidence.status, "fail", "Concat inputs with different element types must fail the source-backed type contract.");
expectEqual(invalidConcatDtype.evidence.semantic_contract_conflicts[0]?.reason, "concat_input_dtype_mismatch", "Concat dtype failure must retain a stable semantic reason.");
expect(invalidConcatDtype.evidence.rule_unresolved_nodes.some((row) => row.node_index === 1 && row.reason?.startsWith("blocked_by_upstream_contract_conflict:")), "A consumer of a dtype-invalid Concat must be causally blocked.");

const symbolicReshapeInference = run([
  node("Shape", ["reshape_symbolic_x"], ["reshape_symbolic_shape"]),
  node("Gather", ["reshape_symbolic_shape", "reshape_batch_index"], ["reshape_batch_scalar"]),
  node("Unsqueeze", ["reshape_batch_scalar", "reshape_axis_zero"], ["reshape_batch_vector"]),
  node("Concat", ["reshape_batch_vector", "reshape_tail"], ["reshape_symbolic_target"], { axis: { i: 0 } }),
  node("Reshape", ["reshape_symbolic_x", "reshape_symbolic_target"], ["reshape_symbolic_y"]),
], [
  symbolicTensor("reshape_symbolic_x", "FLOAT32", ["batch_size", 1280, 16, 12]),
  constant("reshape_batch_index", [0], []),
  constant("reshape_axis_zero", [0], [1]),
  constant("reshape_tail", [1280, -1], [2]),
]);
expectEqual(JSON.stringify(symbolicReshapeInference.tensorMap.get("reshape_symbolic_y").shape), JSON.stringify([-1, 1280, 192]), "Reshape must derive its -1 dimension by exact symbolic element conservation.");
expectEqual(symbolicReshapeInference.tensorMap.get("reshape_symbolic_y").typeProto?.shapeDimensions?.[0]?.parameter, "batch_size", "Reshape element conservation must retain the shared batch symbol.");
expectEqual(symbolicReshapeInference.evidence.rule_unresolved_node_count, 0, "An exactly cancellable symbolic Reshape must not remain a local inference residual.");

const conditionalReshapeInference = run([
  node("Reshape", ["conditional_reshape_x", "conditional_reshape_target"], ["conditional_reshape_y"]),
], [
  symbolicTensor("conditional_reshape_x", "FLOAT32", ["tokens", 6]),
  constant("conditional_reshape_target", [4, -1], [2]),
]);
expectEqual(conditionalReshapeInference.tensorMap.get("conditional_reshape_y").shape[0], 4, "Conditional Reshape must preserve its concrete target dimension.");
expect(conditionalReshapeInference.tensorMap.get("conditional_reshape_y").typeProto?.shapeDimensions?.[1]?.parameter?.startsWith("deepbom_expr:reshape_quotient("), "A runtime-dependent integral Reshape quotient must remain an explicit symbolic expression.");
expectEqual(conditionalReshapeInference.evidence.rule_unresolved_node_count, 0, "A conditionally integral symbolic Reshape still has a complete rank/dimension expression contract.");

const invalidConvSentinel = run([
  node("Conv", ["conv_rank_unknown", "conv_weight"], ["conv_must_not_be_scalar"]),
], [unknown("conv_rank_unknown", { dtype: "FLOAT32" }), tensor("conv_weight", "FLOAT32", [4, 3, 3, 3])]);
expectEqual(JSON.stringify(invalidConvSentinel.tensorMap.get("conv_must_not_be_scalar").shape), JSON.stringify([-1, 4, -1, -1]), "A static Conv weight rank must recover output rank and channels without inventing runtime extents.");
expectEqual(invalidConvSentinel.evidence.rule_unresolved_nodes[0]?.reason, "conv_runtime_dimensions_unbound_rank_and_output_channel_inferred", "Weight-derived Conv rank recovery must retain an explicit runtime-dimension residual.");

const weightAndAttributeRankRecovery = run([
  node("ConvTranspose", ["rankless_deconv_input", "rankless_deconv_weight"], ["rankless_deconv_output"], {
    kernel_shape: { ints: [3, 3] }, strides: { ints: [2, 2] },
  }),
  node("AveragePool", ["rankless_pool_input"], ["rankless_pool_output"], {
    kernel_shape: { ints: [2, 2] }, strides: { ints: [2, 2] },
  }),
], [
  unknown("rankless_deconv_input", { dtype: "FLOAT32" }),
  tensor("rankless_deconv_weight", "FLOAT32", [8, 4, 3, 3]),
  unknown("rankless_pool_input", { dtype: "FLOAT32" }),
]);
expectEqual(JSON.stringify(weightAndAttributeRankRecovery.tensorMap.get("rankless_deconv_output").shape), JSON.stringify([-1, 4, -1, -1]), "A static ConvTranspose weight rank must recover output rank and channels.");
expectEqual(JSON.stringify(weightAndAttributeRankRecovery.tensorMap.get("rankless_pool_output").shape), JSON.stringify([-1, -1, -1, -1]), "A Pool kernel_shape must recover output rank while preserving unknown batch, channel, and spatial extents.");
expectEqual(weightAndAttributeRankRecovery.evidence.rule_unresolved_node_count, 2, "Rank recovery must not promote anonymous runtime dimensions to complete shape contracts.");

const symbolicChannelConv = run([
  node("Conv", ["symbolic_channel_input", "symbolic_channel_weight"], ["symbolic_channel_output"], {
    kernel_shape: { ints: [3] }, pads: { ints: [1, 1] },
  }),
], [
  symbolicTensor("symbolic_channel_input", "FLOAT32", ["batch_size", "feature_size", "sequence_length"]),
  tensor("symbolic_channel_weight", "FLOAT32", [384, 80, 3]),
]);
expectEqual(JSON.stringify(symbolicChannelConv.tensorMap.get("symbolic_channel_output").shape), JSON.stringify([-1, 384, -1]), "Conv should infer its output contract when the serialized input channel is symbolic and the weight contract is concrete.");
expectEqual(symbolicChannelConv.tensorMap.get("symbolic_channel_output").typeProto?.shapeDimensions?.[2]?.parameter, "sequence_length", "Conv must preserve symbolic spatial identity when stride one and symmetric padding preserve the extent.");
expectEqual(symbolicChannelConv.evidence.rule_unresolved_node_count, 0, "A symbolic input channel must not block an otherwise complete Conv output contract.");

const anonymousSpatialContracts = run([
  node("Conv", ["anonymous_conv_input", "anonymous_conv_weight"], ["anonymous_conv_output"], {
    kernel_shape: { ints: [3, 3] }, pads: { ints: [1, 1, 1, 1] },
  }),
  node("ConvTranspose", ["anonymous_transpose_input", "anonymous_transpose_weight"], ["anonymous_transpose_output"], {
    kernel_shape: { ints: [3, 3] }, pads: { ints: [1, 1, 1, 1] }, strides: { ints: [2, 2] },
  }),
  node("MaxPool", ["anonymous_pool_input"], ["anonymous_pool_output"], {
    kernel_shape: { ints: [2, 2] }, strides: { ints: [2, 2] },
  }),
], [
  tensor("anonymous_conv_input", "FLOAT32", [-1, 3, -1, -1]),
  tensor("anonymous_conv_weight", "FLOAT32", [8, 3, 3, 3]),
  tensor("anonymous_transpose_input", "FLOAT32", [-1, 8, -1, -1]),
  tensor("anonymous_transpose_weight", "FLOAT32", [8, 4, 3, 3]),
  tensor("anonymous_pool_input", "FLOAT32", [-1, 8, -1, -1]),
]);
expectEqual(JSON.stringify(anonymousSpatialContracts.tensorMap.get("anonymous_conv_output").shape), JSON.stringify([-1, 8, -1, -1]), "Conv must preserve a rank-only contract and derive its concrete output channel without inventing anonymous dimensions.");
expectEqual(JSON.stringify(anonymousSpatialContracts.tensorMap.get("anonymous_transpose_output").shape), JSON.stringify([-1, 4, -1, -1]), "ConvTranspose must preserve rank and source-backed output channels while leaving runtime extents unbound.");
expectEqual(JSON.stringify(anonymousSpatialContracts.tensorMap.get("anonymous_pool_output").shape), JSON.stringify([-1, 8, -1, -1]), "Pool must preserve rank and channels when spatial extents are runtime-bound.");
expectEqual(anonymousSpatialContracts.evidence.rule_unresolved_node_count, 3, "Rank-only spatial contracts must remain partial while anonymous runtime dimensions are unbound.");
expectEqual(JSON.stringify(anonymousSpatialContracts.evidence.rule_unresolved_nodes.map((row) => row.reason)), JSON.stringify([
  "conv_runtime_dimensions_unbound_rank_and_output_channel_inferred",
  "conv_transpose_runtime_dimensions_unbound_rank_and_output_channel_inferred",
  "pool_runtime_dimensions_unbound_rank_and_nonspatial_axes_inferred",
]), "Rank-only inference must distinguish preserved structural facts from wholly unresolved operator contracts.");
expectEqual(anonymousSpatialContracts.evidence.shape_contract_unknown_node_output_count, 3, "Anonymous runtime dimensions must remain incomplete contracts rather than being relabeled as artifact-declared symbols.");

const zeroDimension = run([
  node("Identity", ["empty"], ["empty_out"]),
], [tensor("empty", "FLOAT32", [0, 3])]);
expectEqual(zeroDimension.evidence.status, "assessed", "An explicit zero-length dimension is known, not dynamic.");
expectEqual(JSON.stringify(zeroDimension.tensorMap.get("empty_out").shape), JSON.stringify([0, 3]), "Zero-length dimensions should be preserved exactly.");

const conditionalBroadcast = run([
  node("Add", ["broadcast_left", "broadcast_right"], ["broadcast_output"]),
], [
  symbolicTensor("broadcast_left", "FLOAT32", ["batch_size", "left_extent", 128]),
  symbolicTensor("broadcast_right", "FLOAT32", ["batch_size", "right_extent", 128]),
]);
const conditionalBroadcastAxis = conditionalBroadcast.tensorMap.get("broadcast_output").typeProto?.shapeDimensions?.[1]?.parameter || "";
expect(conditionalBroadcastAxis.startsWith("deepbom_expr:broadcast_dim("), "Distinct symbolic broadcast axes should remain an explicit conditional expression.");
expectEqual(conditionalBroadcast.evidence.rule_unresolved_node_count, 0, "A conditionally valid symbolic broadcast still has a complete output expression contract.");

const dynamicPadContracts = run([
  node("Pad", ["dynamic_pad_all_input", "dynamic_pad_all_pads", ""], ["dynamic_pad_all_output"]),
  node("Pad", ["dynamic_pad_axis_input", "dynamic_pad_axis_pads", "", "dynamic_pad_axis_axes"], ["dynamic_pad_axis_output"]),
], [
  symbolicTensor("dynamic_pad_all_input", "FLOAT32", ["batch", 32, 64]),
  tensor("dynamic_pad_all_pads", "INT64", [6]),
  symbolicTensor("dynamic_pad_axis_input", "FLOAT32", ["batch", 32, 64]),
  tensor("dynamic_pad_axis_pads", "INT64", [2]),
  constant("dynamic_pad_axis_axes", [1]),
]);
expectEqual(JSON.stringify(dynamicPadContracts.tensorMap.get("dynamic_pad_all_output").shape), JSON.stringify([-1, -1, -1]), "Runtime-bound Pad values must preserve output rank without inventing extents.");
expectEqual(JSON.stringify(dynamicPadContracts.tensorMap.get("dynamic_pad_axis_output").shape), JSON.stringify([-1, -1, 64]), "A static Pad axis must preserve dimensions outside the affected axis.");
expectEqual(JSON.stringify(dynamicPadContracts.evidence.rule_unresolved_nodes.map((row) => row.reason)), JSON.stringify([
  "pad_values_runtime_bound_preserve_rank_only",
  "pad_values_runtime_bound_preserve_rank_only",
]), "Runtime-bound Pad values must remain explicit residuals rather than being mislabeled as invalid cardinalities.");

const invalidPadCardinality = run([
  node("Pad", ["invalid_pad_input", "invalid_pad_values", ""], ["invalid_pad_output"]),
], [tensor("invalid_pad_input", "FLOAT32", [1, 8, 8]), tensor("invalid_pad_values", "INT64", [4])]);
expectEqual(invalidPadCardinality.evidence.rule_unresolved_nodes[0]?.reason, "pad_axes_or_cardinality_invalid", "A Pad vector whose declared cardinality contradicts the affected axes must fail closed.");

const expandBroadcastResult = run([
  node("Expand", ["expand_broadcast_input", "expand_broadcast_target"], ["expand_broadcast_output"]),
], [
  tensor("expand_broadcast_input", "FLOAT32", [1, 64, 1]),
  symbolicDimensionValuesTensor("expand_broadcast_target", ["batch", 1, 1]),
]);
expectEqual(JSON.stringify(expandBroadcastResult.tensorMap.get("expand_broadcast_output").shape), JSON.stringify([-1, 64, 1]), "Expand output must be the multidirectional broadcast of the input and target shapes, not the target vector verbatim.");
expectEqual(expandBroadcastResult.evidence.rule_unresolved_node_count, 0, "A conditionally valid symbolic Expand broadcast must resolve without a false target-equality requirement.");

const symbolicGelu = run([
  node("Gelu", ["gelu_input"], ["gelu_output"]),
], [symbolicTensor("gelu_input", "FLOAT32", ["batch", "tokens", 128])], 20);
expectEqual(JSON.stringify(symbolicGelu.tensorMap.get("gelu_output").shape), JSON.stringify([-1, -1, 128]), "Gelu-20 must preserve its symbolic input shape.");
expectEqual(symbolicGelu.tensorMap.get("gelu_output").typeProto?.shapeDimensions?.[1]?.parameter, "tokens", "Gelu-20 must preserve symbolic dimension identity.");
expectEqual(symbolicGelu.evidence.rule_unresolved_node_count, 0, "A source-backed Gelu-20 shape contract must resolve completely.");

const conflict = run([
  node("Identity", ["declared_in"], ["declared_out"]),
  node("Identity", ["declared_out"], ["declared_downstream"]),
], [tensor("declared_in", "FLOAT32", [2, 3]), tensor("declared_out", "FLOAT32", [2, 4])]);
expectEqual(conflict.evidence.status, "fail", "A contradictory declared output shape should fail the shape contract.");
expectEqual(conflict.evidence.declaration_conflict_count, 1, "A contradictory dimension should emit one conflict row.");
expectEqual(conflict.evidence.declaration_conflicts[0].field, "dimension_1", "The conflict should identify its exact axis.");
expectEqual(JSON.stringify(conflict.tensorMap.get("declared_out").shape), JSON.stringify([2, 4]), "Inference must not overwrite a contradictory artifact declaration.");
expectEqual(conflict.tensorMap.get("declared_out").contractStatus, "invalid", "A contradictory declared output must be sealed as an invalid contract root.");
expect(conflict.evidence.rule_unresolved_nodes.some((row) => row.node_index === 1 && row.reason?.startsWith("blocked_by_upstream_contract_conflict:")), "A consumer of a contradictory declaration must retain causal blocking evidence.");
const conflictFindings = buildFindingsRegister({
  format: "onnx",
  ops: [],
  inputs: [],
  outputs: [],
  tensors: [...conflict.tensorMap.values()],
  onnx_shape_inference: conflict.evidence,
});
const shapeConflictFinding = conflictFindings.find((item) => item.finding_id === "EA-ONX-0006");
expectEqual(shapeConflictFinding?.technical_priority, "High", "A deterministic declaration conflict should enter the action queue at High priority.");
expectEqual(shapeConflictFinding?.confidence, "high", "A combined observed/derived conflict finding should retain high method confidence.");

const cleanFindings = buildFindingsRegister({
  format: "onnx",
  ops: [],
  inputs: [],
  outputs: [],
  tensors: [...zeroDimension.tensorMap.values()],
  onnx_shape_inference: zeroDimension.evidence,
});
expect(!cleanFindings.some((item) => item.finding_id === "EA-ONX-0006"), "A conflict-free graph must not emit the declaration-conflict finding.");

const unsupportedDeclared = run([
  node("UnsupportedFixtureOp", ["known_in"], ["declared_unknown_rule"]),
], [tensor("known_in", "FLOAT32", [2, 3]), tensor("declared_unknown_rule", "FLOAT32", [2, 3])]);
expectEqual(unsupportedDeclared.evidence.status, "partial", "Declared output metadata must not turn an unsupported local rule into assessed inference coverage.");
expectEqual(unsupportedDeclared.evidence.rule_unsupported_nodes, 1, "Unsupported rules should remain explicit even when output metadata is complete.");

const missingOpset = run([
  node("Identity", ["missing_opset_in"], ["missing_opset_out"]),
], [tensor("missing_opset_in", "FLOAT32", [2, 3])], 0);
expectEqual(missingOpset.evidence.status, "fail", "A missing or invalid standard-domain opset must fail closed instead of selecting unspecified semantics.");
expectEqual(missingOpset.evidence.opset_import_contract?.status, "fail", "The OperatorSetIdProto contract should expose the invalid standard-domain import.");
expectEqual(missingOpset.evidence.rule_unresolved_nodes[0]?.reason, "opset_import_contract_invalid", "Invalid opset evidence should carry the import-contract failure reason.");

const repeatedOpset = run([
  node("Identity", ["repeated_opset_in"], ["repeated_opset_out"]),
], [tensor("repeated_opset_in", "FLOAT32", [2, 3])], [
  { domain: "", version: 19 },
  { domain: "ai.onnx", version: 19 },
]);
expectEqual(repeatedOpset.evidence.opset_import_contract.status, "pass", "Identical repeated default-domain imports are valid source records.");
expectEqual(repeatedOpset.evidence.opset_import_contract.duplicate_identical_domain_count, 1, "Repeated identical default-domain imports remain explicitly diagnosed.");
expectEqual(JSON.stringify(repeatedOpset.tensorMap.get("repeated_opset_out").shape), JSON.stringify([2, 3]), "Repeated identical imports must not suppress deterministic shape inference.");

const vectorMatMul = run([
  node("MatMul", ["left", "right"], ["dot"]),
], [tensor("left", "FLOAT32", [3]), tensor("right", "FLOAT32", [3])]);
expectEqual(JSON.stringify(vectorMatMul.tensorMap.get("dot").shape), JSON.stringify([]), "Vector-by-vector MatMul should infer a rank-0 scalar.");
expectEqual(vectorMatMul.tensorMap.get("dot").shapeDeclared, true, "An inferred scalar shape must remain distinct from an absent shape.");

const symbolicLinearAlgebra = run([
  node("Flatten", ["symbolic_image"], ["symbolic_flat"], { axis: { i: 1 } }),
  node("Gemm", ["symbolic_flat", "symbolic_classifier"], ["symbolic_logits"]),
  node("MatMul", ["symbolic_tokens", "symbolic_projection"], ["symbolic_projected"]),
], [
  symbolicTensor("symbolic_image", "FLOAT32", ["batch", 3, 4]),
  symbolicTensor("symbolic_classifier", "FLOAT32", [12, "classes"]),
  symbolicTensor("symbolic_tokens", "FLOAT32", ["batch", "tokens", 64]),
  symbolicTensor("symbolic_projection", "FLOAT32", [64, 128]),
]);
expectEqual(symbolicLinearAlgebra.tensorMap.get("symbolic_flat").typeProto?.shapeDimensions?.[0]?.parameter, "batch", "Flatten must preserve a symbolic batch dimension.");
expectEqual(JSON.stringify(symbolicLinearAlgebra.tensorMap.get("symbolic_flat").shape), JSON.stringify([-1, 12]), "Flatten must derive the concrete non-batch product without substituting the symbolic batch.");
expectEqual(symbolicLinearAlgebra.tensorMap.get("symbolic_logits").typeProto?.shapeDimensions?.[0]?.parameter, "batch", "Gemm must preserve symbolic row identity.");
expectEqual(symbolicLinearAlgebra.tensorMap.get("symbolic_logits").typeProto?.shapeDimensions?.[1]?.parameter, "classes", "Gemm must preserve symbolic output width.");
expectEqual(symbolicLinearAlgebra.tensorMap.get("symbolic_projected").typeProto?.shapeDimensions?.[0]?.parameter, "batch", "MatMul must preserve symbolic batch identity.");
expectEqual(symbolicLinearAlgebra.tensorMap.get("symbolic_projected").typeProto?.shapeDimensions?.[1]?.parameter, "tokens", "MatMul must preserve symbolic token identity.");
expectEqual(JSON.stringify(symbolicLinearAlgebra.tensorMap.get("symbolic_projected").shape), JSON.stringify([-1, -1, 128]), "MatMul must derive output width while retaining symbolic leading dimensions.");
expectEqual(symbolicLinearAlgebra.evidence.rule_unresolved_node_count, 0, "Fully traceable symbolic linear algebra must not remain unresolved.");

const layerNormalization = run([
  node("LayerNormalization", ["layer_norm_x", "layer_norm_scale", "layer_norm_bias"], ["layer_norm_y", "layer_norm_mean", "layer_norm_inv_std"], {
    axis: { i: -1 }, stash_type: { i: 1 },
  }),
], [
  symbolicTensor("layer_norm_x", "FLOAT16", ["batch", "tokens", 64]),
  tensor("layer_norm_scale", "FLOAT16", [64]),
  tensor("layer_norm_bias", "FLOAT16", [64]),
], 17);
expectEqual(JSON.stringify(layerNormalization.tensorMap.get("layer_norm_y").shape), JSON.stringify([-1, -1, 64]), "LayerNormalization Y must retain the complete input shape.");
expectEqual(layerNormalization.tensorMap.get("layer_norm_y").typeProto?.shapeDimensions?.[0]?.parameter, "batch", "LayerNormalization must preserve symbolic batch identity.");
expectEqual(layerNormalization.tensorMap.get("layer_norm_y").typeProto?.shapeDimensions?.[1]?.parameter, "tokens", "LayerNormalization must preserve symbolic token identity.");
expectEqual(JSON.stringify(layerNormalization.tensorMap.get("layer_norm_mean").shape), JSON.stringify([-1, -1, 1]), "LayerNormalization Mean must replace normalized dimensions with one.");
expectEqual(JSON.stringify(layerNormalization.tensorMap.get("layer_norm_inv_std").shape), JSON.stringify([-1, -1, 1]), "LayerNormalization InvStdDev must replace normalized dimensions with one.");
expectEqual(layerNormalization.tensorMap.get("layer_norm_mean").dtype, "FLOAT32", "LayerNormalization stash_type must determine Mean dtype.");
expectEqual(layerNormalization.evidence.rule_unresolved_node_count, 0, "A source-backed symbolic LayerNormalization contract must resolve completely.");

const layerNormalizationConflict = run([
  node("LayerNormalization", ["layer_norm_conflict_x", "layer_norm_conflict_scale"], ["layer_norm_conflict_y", "layer_norm_conflict_mean"], { axis: { i: 1 } }),
], [
  tensor("layer_norm_conflict_x", "FLOAT32", [2, 3, 4]),
  tensor("layer_norm_conflict_scale", "FLOAT32", [3, 4]),
  tensor("layer_norm_conflict_mean", "FLOAT32", [2, 3, 4]),
], 17);
expectEqual(layerNormalizationConflict.evidence.status, "fail", "A declared LayerNormalization stash shape that contradicts the source rule must fail.");
expectEqual(layerNormalizationConflict.evidence.declaration_conflicts[0]?.field, "dimension_1", "LayerNormalization conflict evidence must identify the first normalized axis.");

const invalidLayerNormalizationAxis = run([
  node("LayerNormalization", ["invalid_layer_norm_x", "invalid_layer_norm_scale"], ["invalid_layer_norm_y"], { axis: { i: 3 } }),
], [tensor("invalid_layer_norm_x", "FLOAT32", [2, 3, 4]), tensor("invalid_layer_norm_scale", "FLOAT32", [4])], 17);
expectEqual(invalidLayerNormalizationAxis.tensorMap.get("invalid_layer_norm_y").shapeDeclared, false, "An out-of-range LayerNormalization axis must not fabricate an output shape.");
expectEqual(invalidLayerNormalizationAxis.evidence.rule_unresolved_nodes[0]?.reason, "layer_normalization_axis_out_of_range", "An out-of-range LayerNormalization axis must remain an explicit residual.");

const simplifiedLayerNormalization = run([
  node("SimplifiedLayerNormalization", ["simplified_x", "simplified_scale"], ["simplified_y", "simplified_inv_std"], {
    axis: { i: 1 }, stash_type: { i: 1 },
  }),
], [
  symbolicTensor("simplified_x", "FLOAT32", ["batch", "tokens", 64]),
  tensor("simplified_scale", "FLOAT16", [64]),
], 21);
expectEqual(JSON.stringify(simplifiedLayerNormalization.tensorMap.get("simplified_y").shape), JSON.stringify([-1, -1, 64]), "ORT SimplifiedLayerNormalization Y must retain X shape.");
expectEqual(simplifiedLayerNormalization.tensorMap.get("simplified_y").dtype, "FLOAT16", "ORT SimplifiedLayerNormalization Y dtype must follow scale per the pinned schema.");
expectEqual(JSON.stringify(simplifiedLayerNormalization.tensorMap.get("simplified_inv_std").shape), JSON.stringify([-1, 1, 64]), "ORT SimplifiedLayerNormalization must reproduce the pinned schema's axis-only stash shape rule.");
expectEqual(simplifiedLayerNormalization.evidence.schema_form_rows[0]?.schema_source, "ort_contrib_standard_domain_extension", "The ORT standard-domain extension must not be relabeled as an ONNX-standard schema.");
expectEqual(simplifiedLayerNormalization.evidence.rule_unresolved_node_count, 0, "A source-backed ORT SimplifiedLayerNormalization contract must resolve completely.");

const legacyOrtLayerNormalization = run([
  node("LayerNormalization", ["legacy_layer_norm_x", "legacy_layer_norm_scale"], ["legacy_layer_norm_y", "legacy_layer_norm_mean"], { axis: { i: -1 } }),
], [tensor("legacy_layer_norm_x", "FLOAT32", [2, 3, 4]), tensor("legacy_layer_norm_scale", "FLOAT16", [4])], 16);
expectEqual(legacyOrtLayerNormalization.tensorMap.get("legacy_layer_norm_y").dtype, "FLOAT16", "ORT LayerNormalization opset 1-16 output dtype must follow scale rather than applying ONNX-17 semantics.");
expectEqual(JSON.stringify(legacyOrtLayerNormalization.tensorMap.get("legacy_layer_norm_mean").shape), JSON.stringify([2, 3, 1]), "ORT LayerNormalization opset 1-16 stash shape must follow the pinned contrib schema.");
expectEqual(legacyOrtLayerNormalization.evidence.schema_form_rows[0]?.schema_source, "ort_contrib_standard_domain_extension", "Legacy ai.onnx LayerNormalization must retain its ORT contrib identity.");

const quantizedWeightMatMul = run([
  node("MatMulNBits", ["nbits_a", "nbits_b", "nbits_scales", "", "", "nbits_bias"], ["nbits_y"], {
    K: { i: 64 }, N: { i: 96 }, bits: { i: 4 }, block_size: { i: 32 },
  }, "com.microsoft"),
  node("MatMulBnb4", ["bnb4_a", "bnb4_b", "bnb4_absmax"], ["bnb4_y"], {
    K: { i: 64 }, N: { i: 96 }, block_size: { i: 32 }, quant_type: { i: 1 }, transB: { i: 0 },
  }, "com.microsoft"),
], [
  symbolicTensor("nbits_a", "FLOAT16", ["batch", "tokens", 64]),
  tensor("nbits_b", "UINT8", [96, 2, 16]), tensor("nbits_scales", "FLOAT16", [96, 2]), tensor("nbits_bias", "FLOAT16", [96]),
  symbolicTensor("bnb4_a", "FLOAT32", ["batch", 96]),
  tensor("bnb4_b", "UINT8", [3072]), tensor("bnb4_absmax", "FLOAT32", [192]),
], [{ domain: "", version: 21 }, { domain: "com.microsoft", version: 1 }]);
expectEqual(JSON.stringify(quantizedWeightMatMul.tensorMap.get("nbits_y").shape), JSON.stringify([-1, -1, 96]), "MatMulNBits must replace the final K extent with source-declared N.");
expectEqual(quantizedWeightMatMul.tensorMap.get("nbits_y").typeProto?.shapeDimensions?.[1]?.parameter, "tokens", "MatMulNBits must preserve symbolic leading dimensions.");
expectEqual(JSON.stringify(quantizedWeightMatMul.tensorMap.get("bnb4_y").shape), JSON.stringify([-1, 64]), "MatMulBnb4 transB=0 must replace final N with K.");
expectEqual(quantizedWeightMatMul.evidence.rule_unresolved_node_count, 0, "Static ORT quantized-weight MatMul contracts must resolve completely.");
expect(quantizedWeightMatMul.evidence.schema_form_rows.every((row) => row.schema_source === "ort_contrib_schema"), "com.microsoft quantized MatMul rules must retain ORT contrib provenance.");

const invalidQuantizedWeightMatMul = run([
  node("MatMulNBits", ["invalid_nbits_a", "invalid_nbits_b", "invalid_nbits_scales"], ["invalid_nbits_y"], {
    K: { i: 64 }, N: { i: 96 }, bits: { i: 9 }, block_size: { i: 24 },
  }, "com.microsoft"),
], [tensor("invalid_nbits_a", "FLOAT32", [1, 64]), tensor("invalid_nbits_b", "UINT8", [1]), tensor("invalid_nbits_scales", "FLOAT32", [1])], [{ domain: "", version: 21 }, { domain: "com.microsoft", version: 1 }]);
expectEqual(invalidQuantizedWeightMatMul.evidence.schema_form_assessment_status, "fail", "Invalid MatMulNBits bit width and block size must fail the pinned schema contract.");
expect(invalidQuantizedWeightMatMul.evidence.schema_form_rows[0]?.reason_codes.includes("attribute_value_out_of_range:bits:2:8"), "Invalid MatMulNBits bits must remain machine-readable.");

const ortTransformerSameShape = run([
  node("FastGelu", ["fast_gelu_x"], ["fast_gelu_y"], {}, "com.microsoft"),
  node("RotaryEmbedding", ["rotary_x", "position_ids", "cos_cache", "sin_cache"], ["rotary_y"], {}, "com.microsoft"),
], [
  symbolicTensor("fast_gelu_x", "FLOAT16", ["batch", "tokens", 64]),
  symbolicTensor("rotary_x", "FLOAT16", ["batch", 8, "tokens", 64]),
  tensor("position_ids", "INT64", [2, 16]),
  tensor("cos_cache", "FLOAT16", [2048, 32]),
  tensor("sin_cache", "FLOAT16", [2048, 32]),
], [{ domain: "", version: 21 }, { domain: "com.microsoft", version: 1 }]);
expectEqual(JSON.stringify(ortTransformerSameShape.tensorMap.get("fast_gelu_y").shape), JSON.stringify([-1, -1, 64]), "ORT FastGelu must preserve the complete symbolic input shape.");
expectEqual(JSON.stringify(ortTransformerSameShape.tensorMap.get("rotary_y").shape), JSON.stringify([-1, 8, -1, 64]), "ORT RotaryEmbedding must preserve the complete symbolic input shape.");
expect(ortTransformerSameShape.evidence.schema_form_rows.every((row) => row.schema_source === "ort_contrib_transformer_schema"), "ORT transformer rules must retain transformer-schema provenance.");

const ortSkipLayerNormalization = run([
  node("SkipSimplifiedLayerNormalization", ["skip_norm_x", "skip_norm_skip", "skip_norm_gamma"], ["skip_norm_y", "skip_norm_mean", "skip_norm_inv", "skip_norm_sum"], {}, "com.microsoft"),
], [
  symbolicTensor("skip_norm_x", "FLOAT16", ["batch", "tokens", 64]),
  symbolicTensor("skip_norm_skip", "FLOAT16", ["batch", "tokens", 64]),
  tensor("skip_norm_gamma", "FLOAT16", [64]),
], [{ domain: "", version: 21 }, { domain: "com.microsoft", version: 1 }]);
expectEqual(JSON.stringify(ortSkipLayerNormalization.tensorMap.get("skip_norm_y").shape), JSON.stringify([-1, -1, 64]), "ORT SkipSimplifiedLayerNormalization output must preserve input shape.");
expectEqual(JSON.stringify(ortSkipLayerNormalization.tensorMap.get("skip_norm_mean").shape), JSON.stringify([-1, -1, 1]), "ORT skip-normalization stash output must replace the final dimension with one.");
expectEqual(ortSkipLayerNormalization.tensorMap.get("skip_norm_mean").dtype, "FLOAT32", "ORT skip-normalization stash outputs must use FLOAT32.");
expectEqual(JSON.stringify(ortSkipLayerNormalization.tensorMap.get("skip_norm_sum").shape), JSON.stringify([-1, -1, 64]), "ORT skip-normalization sum output must preserve input shape.");

const ortMultiHeadAttention = run([
  node("MultiHeadAttention", ["mha_query", "mha_key", "mha_value", "", "", "", "mha_past_key", "mha_past_value"], ["mha_output", "mha_present_key", "mha_present_value"], { num_heads: { i: 4 } }, "com.microsoft"),
], [
  symbolicTensor("mha_query", "FLOAT16", ["batch", "tokens", 64]),
  symbolicTensor("mha_key", "FLOAT16", ["batch", 3, 64]),
  symbolicTensor("mha_value", "FLOAT16", ["batch", 3, 80]),
  symbolicTensor("mha_past_key", "FLOAT16", ["batch", 4, 5, 16]),
  symbolicTensor("mha_past_value", "FLOAT16", ["batch", 4, 5, 16]),
], [{ domain: "", version: 21 }, { domain: "com.microsoft", version: 1 }]);
expectEqual(JSON.stringify(ortMultiHeadAttention.tensorMap.get("mha_output").shape), JSON.stringify([-1, -1, 80]), "ORT MultiHeadAttention output width must follow V hidden width.");
expectEqual(JSON.stringify(ortMultiHeadAttention.tensorMap.get("mha_present_key").shape), JSON.stringify([-1, 4, 8, 16]), "ORT MultiHeadAttention present-cache length must add known V and past lengths.");
expectEqual(JSON.stringify(ortMultiHeadAttention.tensorMap.get("mha_present_value").shape), JSON.stringify([-1, 4, 8, 16]), "ORT MultiHeadAttention present key/value contracts must agree.");

const ortGroupQueryAttention = run([
  node("GroupQueryAttention", ["gqa_query", "gqa_key", "gqa_value", "gqa_past_key", "gqa_past_value", "gqa_seqlens", "gqa_total"], ["gqa_output", "gqa_present_key", "gqa_present_value"], { num_heads: { i: 4 }, kv_num_heads: { i: 2 } }, "com.microsoft"),
], [
  symbolicTensor("gqa_query", "FLOAT16", ["batch", "tokens", 64]),
  symbolicTensor("gqa_key", "FLOAT16", ["batch", 4, 32]),
  symbolicTensor("gqa_value", "FLOAT16", ["batch", 4, 32]),
  symbolicTensor("gqa_past_key", "INT8", ["batch", 2, 8, 16]),
  symbolicTensor("gqa_past_value", "INT8", ["batch", 2, 8, 16]),
  tensor("gqa_seqlens", "INT32", [2]),
  constant("gqa_total", [12], [], "INT32"),
], [{ domain: "", version: 21 }, { domain: "com.microsoft", version: 1 }]);
expectEqual(JSON.stringify(ortGroupQueryAttention.tensorMap.get("gqa_output").shape), JSON.stringify([-1, -1, 64]), "ORT GroupQueryAttention with separate Q/K/V must preserve query output shape.");
expectEqual(JSON.stringify(ortGroupQueryAttention.tensorMap.get("gqa_present_key").shape), JSON.stringify([-1, 2, 12, 16]), "ORT GroupQueryAttention must apply the source-defined maximum total cache length.");
expectEqual(ortGroupQueryAttention.tensorMap.get("gqa_present_key").dtype, "INT8", "ORT GroupQueryAttention present cache must retain the past-cache dtype.");

const ortPackedGroupQueryAttention = run([
  node("GroupQueryAttention", ["packed_gqa_query", "", "", "", "", "packed_gqa_seqlens", "packed_gqa_total"], ["packed_gqa_output", "packed_gqa_present_key", "packed_gqa_present_value"], { num_heads: { i: 4 }, kv_num_heads: { i: 2 } }, "com.microsoft"),
], [
  symbolicTensor("packed_gqa_query", "FLOAT16", ["batch", "tokens", 128]),
  tensor("packed_gqa_seqlens", "INT32", [2]),
  constant("packed_gqa_total", [9], [], "INT32"),
], [{ domain: "", version: 21 }, { domain: "com.microsoft", version: 1 }]);
expectEqual(JSON.stringify(ortPackedGroupQueryAttention.tensorMap.get("packed_gqa_output").shape), JSON.stringify([-1, -1, 64]), "Packed ORT GroupQueryAttention must derive Q hidden width from head attributes.");
expectEqual(JSON.stringify(ortPackedGroupQueryAttention.tensorMap.get("packed_gqa_present_key").shape), JSON.stringify([-1, 2, 9, 16]), "Packed ORT GroupQueryAttention must derive KV cache shape without fabricating batch or token extents.");
expectEqual(ortPackedGroupQueryAttention.evidence.rule_unresolved_node_count, 0, "A fully determined packed ORT GroupQueryAttention contract must resolve completely.");

const sourceBackedResidualRules = run([
  node("CumSum", ["cumsum_x", "cumsum_axis"], ["cumsum_y"]),
  node("ScatterElements", ["scatter_data", "scatter_indices", "scatter_updates"], ["scatter_y"], { axis: { i: 1 } }),
  node("RandomNormalLike", ["random_like_x"], ["random_like_y"], { dtype: { i: 10 }, mean: { f: 0.0 }, scale: { f: 1.0 } }),
  node("RandomUniformLike", ["random_uniform_x"], ["random_uniform_y"]),
  node("NonZero", ["nonzero_x"], ["nonzero_y"]),
], [
  constant("cumsum_x", [1, 2, 3, 4, 5, 6], [2, 3], "INT64"), constant("cumsum_axis", [1], [], "INT64"),
  tensor("scatter_data", "FLOAT32", [2, 3]), tensor("scatter_indices", "INT64", [2, 3]), tensor("scatter_updates", "FLOAT32", [2, 3]),
  symbolicTensor("random_like_x", "FLOAT32", ["batch", 16]), symbolicTensor("random_uniform_x", "FLOAT32", ["batch", 8]),
  tensor("nonzero_x", "FLOAT32", []),
], 22);
expectEqual(JSON.stringify(sourceBackedResidualRules.tensorMap.get("cumsum_y").shape), JSON.stringify([2, 3]), "CumSum must preserve the input shape.");
expect(sourceBackedResidualRules.tensorMap.get("cumsum_y").staticValuesComplete !== true, "CumSum must not copy input values as if it were an identity operator.");
expectEqual(JSON.stringify(sourceBackedResidualRules.tensorMap.get("scatter_y").shape), JSON.stringify([2, 3]), "ScatterElements must preserve the data shape.");
expectEqual(sourceBackedResidualRules.tensorMap.get("random_like_y").dtype, "FLOAT16", "RandomNormalLike must honor its declared output dtype.");
expectEqual(sourceBackedResidualRules.tensorMap.get("random_like_y").typeProto?.shapeDimensions?.[0]?.parameter, "batch", "RandomNormalLike must preserve symbolic input dimensions.");
expectEqual(sourceBackedResidualRules.tensorMap.get("random_uniform_y").dtype, "FLOAT32", "RandomUniformLike without dtype must inherit the input dtype.");
expectEqual(JSON.stringify(sourceBackedResidualRules.tensorMap.get("nonzero_y").shape), JSON.stringify([0, -1]), "NonZero on a scalar must derive [rank, runtime_nnz] without fabricating the count.");
expectEqual(sourceBackedResidualRules.tensorMap.get("nonzero_y").dtype, "INT64", "NonZero indices must be INT64.");
expect(sourceBackedResidualRules.tensorMap.get("nonzero_y").typeProto?.shapeDimensions?.[1]?.parameter?.startsWith("deepbom_runtime:nnz:"), "NonZero must preserve its runtime-dependent NNZ as an explicit symbol.");
expectEqual(sourceBackedResidualRules.tensorMap.get("nonzero_y").runtimeDimensionBounds?.[0]?.lower_bound_decimal, "0", "NonZero NNZ must retain its exact non-negative lower bound.");
expectEqual(sourceBackedResidualRules.evidence.rule_unsupported_nodes, 0, "Every source-backed residual operator must be registered.");

const standardLstm = run([
  node("LSTM", ["lstm_x", "lstm_w", "lstm_r"], ["lstm_y", "lstm_yh", "lstm_yc"], { direction: { s: "bidirectional" }, hidden_size: { i: 16 }, layout: { i: 0 } }),
  node("LSTM", ["lstm_layout1_x", "lstm_layout1_w", "lstm_layout1_r"], ["lstm_layout1_y", "lstm_layout1_yh"], { hidden_size: { i: 12 }, layout: { i: 1 } }),
], [
  symbolicTensor("lstm_x", "FLOAT32", ["sequence", "batch", 32]), tensor("lstm_w", "FLOAT32", [2, 64, 32]), tensor("lstm_r", "FLOAT32", [2, 64, 16]),
  symbolicTensor("lstm_layout1_x", "FLOAT16", ["batch2", "sequence2", 24]), tensor("lstm_layout1_w", "FLOAT16", [1, 48, 24]), tensor("lstm_layout1_r", "FLOAT16", [1, 48, 12]),
], 22);
expectEqual(JSON.stringify(standardLstm.tensorMap.get("lstm_y").shape), JSON.stringify([-1, 2, -1, 16]), "LSTM layout 0 must derive sequence, direction, batch, and hidden dimensions.");
expectEqual(standardLstm.tensorMap.get("lstm_y").typeProto?.shapeDimensions?.[0]?.parameter, "sequence", "LSTM must preserve the symbolic sequence identity.");
expectEqual(JSON.stringify(standardLstm.tensorMap.get("lstm_yh").shape), JSON.stringify([2, -1, 16]), "LSTM layout 0 state output must use [direction,batch,hidden].");
expectEqual(JSON.stringify(standardLstm.tensorMap.get("lstm_layout1_y").shape), JSON.stringify([-1, -1, 1, 12]), "LSTM layout 1 must use [batch,sequence,direction,hidden].");
expectEqual(JSON.stringify(standardLstm.tensorMap.get("lstm_layout1_yh").shape), JSON.stringify([-1, 1, 12]), "LSTM layout 1 state output must use [batch,direction,hidden].");
expectEqual(standardLstm.evidence.rule_unresolved_node_count, 0, "Static LSTM shape contracts must resolve completely.");

const ortRecurrentAndFusedMatMul = run([
  node("DynamicQuantizeLSTM", ["dq_lstm_x", "dq_lstm_w", "dq_lstm_r", "", "", "", "", "", "dq_lstm_ws", "dq_lstm_wz", "dq_lstm_rs", "dq_lstm_rz"], ["dq_lstm_y", "dq_lstm_yh", "dq_lstm_yc"], { direction: { s: "bidirectional" }, hidden_size: { i: 10 } }, "com.microsoft"),
  node("FusedMatMul", ["fused_a", "fused_b"], ["fused_y"], { alpha: { f: 0.5 }, transBatchA: { i: 1 } }, "com.microsoft"),
], [
  symbolicTensor("dq_lstm_x", "FLOAT32", ["steps", "batch", 8]), tensor("dq_lstm_w", "INT8", [2, 8, 40]), tensor("dq_lstm_r", "INT8", [2, 10, 40]),
  tensor("dq_lstm_ws", "FLOAT32", [2]), tensor("dq_lstm_wz", "INT8", [2]), tensor("dq_lstm_rs", "FLOAT32", [2]), tensor("dq_lstm_rz", "INT8", [2]),
  tensor("fused_a", "FLOAT16", [2, 3, 4]), tensor("fused_b", "FLOAT16", [3, 4, 5]),
], [{ domain: "", version: 22 }, { domain: "com.microsoft", version: 1 }]);
expectEqual(JSON.stringify(ortRecurrentAndFusedMatMul.tensorMap.get("dq_lstm_y").shape), JSON.stringify([-1, 2, -1, 10]), "DynamicQuantizeLSTM must reuse the pinned ONNX RNN shape function.");
expectEqual(JSON.stringify(ortRecurrentAndFusedMatMul.tensorMap.get("dq_lstm_yh").shape), JSON.stringify([2, -1, 10]), "DynamicQuantizeLSTM state output must retain direction, batch, and hidden dimensions.");
expectEqual(JSON.stringify(ortRecurrentAndFusedMatMul.tensorMap.get("fused_y").shape), JSON.stringify([3, 2, 5]), "FusedMatMul transBatchA must reproduce the pinned ORT batch-to-matrix transform before MatMul broadcasting.");
expect(ortRecurrentAndFusedMatMul.evidence.schema_form_rows.every((row) => row.status === "pass"), "Pinned ORT recurrent and FusedMatMul schema forms must pass.");
expectEqual(ortRecurrentAndFusedMatMul.evidence.rule_unresolved_node_count, 0, "Fully static ORT recurrent and FusedMatMul contracts must resolve.");

const conditionalSymbolicBroadcast = run([
  node("Add", ["symbolic_left", "concrete_right"], ["conditional_sum"]),
], [
  symbolicTensor("symbolic_left", "FLOAT32", ["runtime_batch", 4]),
  tensor("concrete_right", "FLOAT32", [8, 1]),
]);
expectEqual(JSON.stringify(conditionalSymbolicBroadcast.tensorMap.get("conditional_sum").shape), JSON.stringify([8, 4]), "A concrete non-one broadcast dimension determines the output extent when the runtime symbolic dimension is compatible.");
expectEqual(conditionalSymbolicBroadcast.evidence.rule_unresolved_node_count, 0, "A conditionally valid symbolic-to-concrete broadcast has a deterministic output shape.");

const unresolvedSymbolicBatch = run([
  node("MatMul", ["symbolic_batch_a", "symbolic_batch_b"], ["symbolic_batch_out"]),
], [
  symbolicTensor("symbolic_batch_a", "FLOAT32", ["batch_a", 2, 3]),
  symbolicTensor("symbolic_batch_b", "FLOAT32", ["batch_b", 3, 4]),
]);
expectEqual(JSON.stringify(unresolvedSymbolicBatch.tensorMap.get("symbolic_batch_out").shape), JSON.stringify([-1, 2, 4]), "Different symbolic batch identities must not be collapsed to either input name.");
expect(unresolvedSymbolicBatch.tensorMap.get("symbolic_batch_out").typeProto?.shapeDimensions?.[0]?.parameter?.startsWith("deepbom_expr:broadcast_dim("), "Conditionally compatible symbolic batch axes must retain their exact broadcast relation.");
expectEqual(unresolvedSymbolicBatch.evidence.rule_unresolved_node_count, 0, "An explicit conditional broadcast expression is a complete symbolic output contract.");
expect(unresolvedSymbolicBatch.evidence.symbolic_dimension_method.includes("never replaced by 1"), "The evidence ledger must disclose the fail-closed symbolic-dimension method.");

const undeclaredRank = run([
  node("Identity", ["rank_unknown"], ["rank_unknown_out"]),
], [unknown("rank_unknown", { dtype: "FLOAT32" })]);
expectEqual(undeclaredRank.tensorMap.get("rank_unknown_out").shapeDeclared, false, "An absent input shape must not be inferred as a rank-0 scalar.");
expectEqual(undeclaredRank.evidence.status, "not_assessed", "Unknown rank should remain fail-closed for same-shape rules.");

const slice = run([
  node("Slice", ["codes", "starts", "ends", "axes", "steps"], ["sliced"]),
], [
  constant("codes", [10, 20, 30, 40, 50], [5]),
  constant("starts", [1]), constant("ends", [5]), constant("axes", [0]), constant("steps", [2]),
]);
expectEqual(JSON.stringify(slice.tensorMap.get("sliced").shape), JSON.stringify([2]), "Slice should compute exact output cardinality.");
expectEqual(JSON.stringify(slice.tensorMap.get("sliced").staticValues), JSON.stringify([20, 40]), "Slice should propagate exact rank-1 integer values.");

const openEndedSlice = run([
  node("Constant", [], ["slice_int64_max"], { value: { tensor: canonicalIntegerTensor("max", ["9223372036854775807"]) } }),
  node("Slice", ["open_slice_input", "open_slice_starts", "slice_int64_max", "open_slice_axes", "open_slice_steps"], ["open_slice_output"]),
], [
  tensor("open_slice_input", "FLOAT32", [2, 4, 8]),
  constant("open_slice_starts", [-3]), constant("open_slice_axes", [2]), constant("open_slice_steps", [1]),
]);
expectEqual(JSON.stringify(openEndedSlice.tensorMap.get("open_slice_output").shape), JSON.stringify([2, 4, 3]), "INT64_MAX Slice bounds should remain exact open-ended sentinels rather than becoming dynamic controls.");
expectEqual(openEndedSlice.evidence.rule_unresolved_node_count, 0, "A concrete open-ended Slice should leave no residual after safe bound clipping.");

const reverseOpenEndedSlice = run([
  node("Constant", [], ["reverse_int64_max"], { value: { tensor: canonicalIntegerTensor("max", ["9223372036854775807"]) } }),
  node("Constant", [], ["reverse_int64_min"], { value: { tensor: canonicalIntegerTensor("min", ["-9223372036854775808"]) } }),
  node("Slice", ["reverse_slice_input", "reverse_int64_max", "reverse_int64_min", "reverse_slice_axes", "reverse_slice_steps"], ["reverse_slice_output"]),
], [
  tensor("reverse_slice_input", "FLOAT32", [5]),
  constant("reverse_slice_axes", [0]), constant("reverse_slice_steps", [-1]),
]);
expectEqual(JSON.stringify(reverseOpenEndedSlice.tensorMap.get("reverse_slice_output").shape), JSON.stringify([5]), "INT64 open bounds with a negative step should derive the exact reversed extent.");

const finiteUnsafeSliceBound = run([
  node("Constant", [], ["finite_unsafe_bound"], { value: { tensor: canonicalIntegerTensor("finite", ["9007199254740993"]) } }),
  node("Slice", ["finite_unsafe_input", "finite_unsafe_start", "finite_unsafe_bound", "finite_unsafe_axis", "finite_unsafe_step"], ["finite_unsafe_output"]),
], [
  symbolicTensor("finite_unsafe_input", "FLOAT32", ["runtime_extent"]),
  constant("finite_unsafe_start", [0]), constant("finite_unsafe_axis", [0]), constant("finite_unsafe_step", [1]),
]);
expectEqual(finiteUnsafeSliceBound.tensorMap.get("finite_unsafe_output").shape[0], -1, "A finite unsafe INT64 Slice bound must not be promoted to an unbounded sentinel.");
expect(finiteUnsafeSliceBound.tensorMap.get("finite_unsafe_output").typeProto?.shapeDimensions?.[0]?.parameter?.includes("i64:9007199254740993"), "An unsafe finite Slice bound should retain its exact integer identity in the symbolic extent expression.");

const symbolicBoundSlice = run([
  node("Slice", ["symbolic_bound_input", "symbolic_bound_start", "symbolic_bound_end", "symbolic_bound_axis", "symbolic_bound_step"], ["symbolic_bound_output"]),
], [
  tensor("symbolic_bound_input", "INT64", [1, 514]),
  constant("symbolic_bound_start", [0]),
  symbolicDimensionValuesTensor("symbolic_bound_end", ["sequence_length"]),
  constant("symbolic_bound_axis", [1]), constant("symbolic_bound_step", [1]),
]);
expectEqual(JSON.stringify(symbolicBoundSlice.tensorMap.get("symbolic_bound_output").shape), JSON.stringify([1, -1]), "A dimension-valued Slice bound should produce a complete symbolic output contract.");
expect(symbolicBoundSlice.tensorMap.get("symbolic_bound_output").typeProto?.shapeDimensions?.[1]?.parameter?.startsWith("deepbom_expr:slice_len("), "A runtime Slice extent must remain an explicit exact expression.");
expectEqual(symbolicBoundSlice.evidence.rule_unresolved_node_count, 0, "An exact symbolic Slice expression should not remain a local rule residual.");

const clippedSymbolicInputSlice = run([
  node("Slice", ["clipped_symbolic_input", "clipped_symbolic_start", "clipped_symbolic_end", "clipped_symbolic_axis", "clipped_symbolic_step"], ["clipped_symbolic_output"]),
], [
  symbolicTensor("clipped_symbolic_input", "FLOAT32", ["tokens"]),
  constant("clipped_symbolic_start", [0]), constant("clipped_symbolic_end", [100]),
  constant("clipped_symbolic_axis", [0]), constant("clipped_symbolic_step", [1]),
]);
expectEqual(clippedSymbolicInputSlice.tensorMap.get("clipped_symbolic_output").shape[0], -1, "Slice 0:100 over an unbounded symbolic input must not be fabricated as extent 100.");
expect(clippedSymbolicInputSlice.tensorMap.get("clipped_symbolic_output").typeProto?.shapeDimensions?.[0]?.parameter?.startsWith("deepbom_expr:slice_len("), "Clipping against an unknown input extent must remain explicit in the symbolic contract.");

const expandShapeSelection = run([
  node("ConstantOfShape", ["expand_shape_vector_shape"], ["expand_shape_ones"], {
    value: { tensor: constant("expand_fill", [1], [1]) },
  }),
  node("Mul", ["expand_shape_ones", "expand_shape_minus_one"], ["expand_shape_negative_ones"]),
  node("Equal", ["expand_shape_target", "expand_shape_negative_ones"], ["expand_shape_use_default"]),
  node("Where", ["expand_shape_use_default", "expand_shape_ones", "expand_shape_target"], ["expand_shape_selected"]),
  node("Expand", ["expand_seed", "expand_shape_selected"], ["expanded_symbolic"]),
], [
  constant("expand_shape_vector_shape", [2]),
  constant("expand_shape_minus_one", [-1], []),
  symbolicDimensionValuesTensor("expand_shape_target", ["batch_size", "sequence_length"]),
  tensor("expand_seed", "INT64", [1, 1]),
]);
expectEqual(JSON.stringify(expandShapeSelection.tensorMap.get("expand_shape_ones").staticValues), JSON.stringify([1, 1]), "Bounded ConstantOfShape should materialize its exact scalar fill.");
expectEqual(JSON.stringify(expandShapeSelection.tensorMap.get("expand_shape_use_default").staticValues), JSON.stringify([false, false]), "A non-negative symbolic dimension can never equal the -1 Expand sentinel.");
expectEqual(expandShapeSelection.tensorMap.get("expand_shape_selected").staticDimensionValues?.[0]?.parameter, "batch_size", "Where should select the exact parameter-bound dimension vector.");
expectEqual(JSON.stringify(expandShapeSelection.tensorMap.get("expanded_symbolic").shape), JSON.stringify([-1, -1]), "Expand should consume an exactly selected symbolic target shape.");
expectEqual(expandShapeSelection.evidence.rule_unresolved_node_count, 0, "The bounded Expand target-selection pattern should be fully assessed.");

const invalidExpand = run([
  node("Expand", ["invalid_expand_input", "invalid_expand_target"], ["invalid_expand_output"]),
  node("Identity", ["invalid_expand_output"], ["invalid_expand_consumer"]),
], [
  tensor("invalid_expand_input", "FLOAT32", [2, 4]),
  constant("invalid_expand_target", [2, 3]),
]);
expectEqual(invalidExpand.evidence.status, "fail", "A statically broadcast-incompatible Expand contract must fail rather than appear unresolved.");
expectEqual(invalidExpand.evidence.semantic_contract_conflict_count, 1, "The root Expand contract violation must be counted once.");
expectEqual(invalidExpand.evidence.semantic_contract_conflicts[0]?.reason, "expand_target_not_broadcast_compatible", "The semantic conflict must retain the source operation reason.");
expect(invalidExpand.evidence.rule_unresolved_nodes.some((row) => row.node_index === 1 && row.reason?.startsWith("blocked_by_upstream_contract_conflict:")), "A downstream node must retain causal blocking instead of emitting an unrelated missing-shape residual.");

const branchConditionKey = "if:fixture:condition";
const conditionalConv = run([
  node("Conv", ["conditional_conv_input", "conditional_conv_weight"], ["conditional_conv_output"]),
  node("Identity", ["conditional_conv_output"], ["conditional_conv_consumer"]),
], [
  tensor("conditional_conv_input", "FLOAT32", [], {
    shapeDeclared: false,
    conditionalShapeContract: {
      schema: "deepbom.onnx_conditional_shape_contract.v1",
      status: "assessed_complete",
      variant_count: 2,
      condition_keys: [branchConditionKey],
    },
    conditionalShapeVariants: [
      { ...tensor("conditional_conv_input", "FLOAT32", [1, 128, 10]), conditions: [{ key: branchConditionKey, value: "then_branch" }] },
      { ...tensor("conditional_conv_input", "FLOAT32", [1, 128, 10, 2]), conditions: [{ key: branchConditionKey, value: "else_branch" }] },
    ],
  }),
  tensor("conditional_conv_weight", "FLOAT32", [256, 128, 1]),
]);
const conditionalConvOutput = conditionalConv.tensorMap.get("conditional_conv_output");
expectEqual(conditionalConvOutput.conditionalShapeContract?.status, "assessed_partial", "A conditionally valid Conv must retain viable variants without hiding a rank-incompatible branch.");
expectEqual(conditionalConvOutput.conditionalShapeContract?.variant_count, 1, "Only the source-compatible Conv branch should remain viable.");
expectEqual(conditionalConvOutput.conditionalShapeContract?.invalid_variant_count, 1, "The rank-incompatible Conv branch must be counted as a conditional invalid variant.");
expectEqual(conditionalConvOutput.conditionalShapeContract?.variant_failures?.[0]?.reason, "conv_input_weight_rank_mismatch", "Conditional failure evidence must retain the exact source-contract reason.");
expectEqual(conditionalConv.tensorMap.get("conditional_conv_consumer").conditionalShapeContract?.status, "assessed_partial", "A downstream op must inherit an upstream condition-bound failure instead of promoting the viable subset to complete.");
expectEqual(conditionalConv.evidence.partial_conditional_shape_contract_node_output_count, 2, "Partial conditional coverage must include causally dependent outputs.");
expectEqual(conditionalConv.evidence.shape_contract_unknown_node_output_count, 2, "A partial conditional contract and its dependent outputs must not be promoted to complete shape coverage.");

const unsortedSliceAxes = run([
  node("Slice", ["unsorted_slice_input", "unsorted_starts", "unsorted_ends", "unsorted_axes", "unsorted_steps"], ["unsorted_slice_output"]),
], [
  tensor("unsorted_slice_input", "FLOAT32", [2, 3]),
  constant("unsorted_starts", [1, 0]), constant("unsorted_ends", [3, 2]),
  constant("unsorted_axes", [1, 0]), constant("unsorted_steps", [1, 1]),
]);
expectEqual(JSON.stringify(unsortedSliceAxes.tensorMap.get("unsorted_slice_output").shape), JSON.stringify([2, 2]), "Slice must preserve control-vector order when axes are not sorted.");

const invalidSliceControlDtype = run([
  node("Slice", ["invalid_slice_input", "invalid_slice_starts", "invalid_slice_ends"], ["invalid_slice_output"]),
], [
  tensor("invalid_slice_input", "FLOAT32", [2, 3]),
  constant("invalid_slice_starts", [0], [1], "BOOL"),
  constant("invalid_slice_ends", [1]),
]);
expectEqual(invalidSliceControlDtype.tensorMap.get("invalid_slice_output").shapeDeclared, false, "Slice controls outside INT32/INT64 must not produce a shape contract.");
expectEqual(invalidSliceControlDtype.evidence.rule_unresolved_nodes[0]?.reason, "slice_control_tensor_contract_invalid", "Invalid Slice control dtypes should retain a deterministic reason code.");

const invalidSliceControlLength = run([
  node("Slice", ["invalid_length_input", "invalid_length_starts", "invalid_length_ends"], ["invalid_length_output"]),
], [
  tensor("invalid_length_input", "FLOAT32", [2, 3]),
  constant("invalid_length_starts", [0, 1], [1]),
  constant("invalid_length_ends", [1, 2]),
]);
expectEqual(invalidSliceControlLength.tensorMap.get("invalid_length_output").shapeDeclared, false, "A Slice control payload that conflicts with its declared length must remain fail-closed.");

const zeroStepSlice = run([
  node("Slice", ["zero_step_input", "zero_step_starts", "zero_step_ends", "zero_step_axes", "zero_step_steps"], ["zero_step_output"]),
], [
  tensor("zero_step_input", "FLOAT32", [2, 3]),
  constant("zero_step_starts", [0]), constant("zero_step_ends", [2]),
  constant("zero_step_axes", [1]), constant("zero_step_steps", [0]),
]);
expectEqual(zeroStepSlice.tensorMap.get("zero_step_output").shapeDeclared, false, "Slice step zero must not produce an output shape.");
expectEqual(zeroStepSlice.evidence.rule_unresolved_nodes[0]?.reason, "slice_step_zero_invalid", "Slice step zero should retain its schema-semantic reason code.");

const conditionalImplicitSqueeze = run([
  node("Squeeze", ["conditional_squeeze_input"], ["conditional_squeeze_output"]),
], [symbolicTensor("conditional_squeeze_input", "FLOAT32", [1, "runtime_width"])]);
expectEqual(conditionalImplicitSqueeze.tensorMap.get("conditional_squeeze_output").conditionalShapeContract?.status, "assessed_complete", "Implicit Squeeze must enumerate every finite rank outcome when a symbolic dimension may equal one.");
expectEqual(conditionalImplicitSqueeze.tensorMap.get("conditional_squeeze_output").conditionalShapeContract?.variant_count, 2, "One symbolic implicit-Squeeze axis must create exactly two condition-bound variants.");
expectEqual(conditionalImplicitSqueeze.evidence.unresolved_nonconflict_shape_contract_node_output_count, 0, "A finite implicit-Squeeze rank union must not remain an analyzer residual.");

const guardedExplicitSqueeze = run([
  node("Squeeze", ["guarded_squeeze_input", "guarded_squeeze_axis"], ["guarded_squeeze_output"]),
], [symbolicTensor("guarded_squeeze_input", "FLOAT32", [1, "runtime_width"]), constant("guarded_squeeze_axis", [1])]);
expectEqual(guardedExplicitSqueeze.tensorMap.get("guarded_squeeze_output").conditionalShapeContract?.status, "assessed_partial", "Explicit Squeeze over a symbolic extent must separate its valid and invalid runtime contracts.");
expectEqual(guardedExplicitSqueeze.tensorMap.get("guarded_squeeze_output").conditionalShapeContract?.invalid_variant_count, 1, "The non-unit explicit-Squeeze branch must be classified as INVALID rather than unassessed.");
expectEqual(guardedExplicitSqueeze.evidence.conditional_unassessed_variant_count, 0, "A guarded explicit-Squeeze contract must have no heuristic residual.");

const invalidEmptyShapeGather = run([
  node("Gather", ["empty_shape_data", "empty_shape_index"], ["invalid_gather_output"]),
], [
  tensor("empty_shape_data", "INT64", [0], {
    staticDimensionValuesStatus: "assessed_exact_static_shape_data",
    staticDimensionValuesComplete: true,
    staticDimensionValues: [],
    staticDimensionValuesSource: "fixture",
  }),
  constant("empty_shape_index", [0], []),
]);
expectEqual(invalidEmptyShapeGather.evidence.semantic_contract_conflicts[0]?.reason, "gather_index_out_of_range", "Gather over an exact empty shape vector must be an artifact contract failure.");

const invalidScalarSlice = run([
  node("Slice", ["scalar_slice_input", "scalar_slice_start", "scalar_slice_end"], ["scalar_slice_output"]),
], [tensor("scalar_slice_input", "FLOAT32", []), constant("scalar_slice_start", [0]), constant("scalar_slice_end", [1])]);
expectEqual(invalidScalarSlice.evidence.semantic_contract_conflicts[0]?.reason, "slice_control_cardinality_exceeds_rank", "Slice controls that address an axis of a scalar must be an artifact contract failure.");

const dynamicSlice = run([
  node("Slice", ["dynamic_slice_input", "dynamic_slice_starts", "dynamic_slice_ends", "dynamic_slice_axes", ""], ["dynamic_sliced"]),
  node("Shape", ["dynamic_sliced"], ["dynamic_sliced_shape"]),
  node("Gather", ["dynamic_sliced_shape", "dynamic_slice_last_axis"], ["dynamic_slice_last_extent"]),
], [
  symbolicTensor("dynamic_slice_input", "FLOAT32", ["batch_size", 4, 8]),
  tensor("dynamic_slice_starts", "INT64", [1]),
  tensor("dynamic_slice_ends", "INT64", [1]),
  constant("dynamic_slice_axes", [1]),
  constant("dynamic_slice_last_axis", [2], []),
]);
expectEqual(JSON.stringify(dynamicSlice.tensorMap.get("dynamic_sliced").shape), JSON.stringify([-1, -1, 8]), "Dynamic Slice bounds should preserve rank, unaffected dimensions, and the input symbolic identity.");
expectEqual(dynamicSlice.tensorMap.get("dynamic_sliced").typeProto?.shapeDimensions?.[0]?.parameter, "batch_size", "Dynamic Slice bounds must preserve an unaffected symbolic dimension.");
expect(dynamicSlice.tensorMap.get("dynamic_sliced").typeProto?.shapeDimensions?.[1]?.parameter?.startsWith("deepbom_expr:slice_len("), "Runtime Slice starts and ends with fixed cardinality must produce an exact evaluable slice_len contract.");
expectEqual(dynamicSlice.tensorMap.get("dynamic_slice_last_extent").staticDimensionValues?.[0]?.value, 8, "Shape consumers should recover unaffected extents after a dynamic Slice.");
expectEqual(dynamicSlice.evidence.rule_unresolved_node_count, 0, "Fixed-cardinality runtime Slice bounds must close as a symbolic contract without a false local residual.");

const dynamicSliceAxes = run([
  node("Slice", ["dynamic_axes_input", "dynamic_axes_starts", "dynamic_axes_ends", "dynamic_axes", ""], ["dynamic_axes_output"]),
], [
  tensor("dynamic_axes_input", "FLOAT32", [2, 4, 8]),
  constant("dynamic_axes_starts", [0]),
  constant("dynamic_axes_ends", [1]),
  tensor("dynamic_axes", "INT64", [1]),
]);
expectEqual(JSON.stringify(dynamicSliceAxes.tensorMap.get("dynamic_axes_output").shape), JSON.stringify([-1, -1, -1]), "An unknown Slice axis should preserve exact rank without inventing unaffected dimensions.");

const dynamicSliceSteps = run([
  node("Slice", ["dynamic_steps_input", "dynamic_steps_starts", "dynamic_steps_ends", "dynamic_steps_axes", "dynamic_steps"], ["dynamic_steps_output"]),
], [
  tensor("dynamic_steps_input", "FLOAT32", [2, 4, 8]),
  constant("dynamic_steps_starts", [0]),
  constant("dynamic_steps_ends", [3]),
  constant("dynamic_steps_axes", [1]),
  tensor("dynamic_steps", "INT64", [1]),
]);
expectEqual(JSON.stringify(dynamicSliceSteps.tensorMap.get("dynamic_steps_output").shape), JSON.stringify([2, -1, 8]), "A present dynamic steps input must not be misread as an omitted default step.");

const malformedDynamicSlice = run([
  node("Slice", ["mismatch_input", "mismatch_starts", "mismatch_ends", "mismatch_axes", ""], ["mismatch_output"]),
], [
  tensor("mismatch_input", "FLOAT32", [2, 4, 8]),
  tensor("mismatch_starts", "INT64", [2]),
  tensor("mismatch_ends", "INT64", [1]),
  tensor("mismatch_axes", "INT64", [2]),
]);
expectEqual(malformedDynamicSlice.tensorMap.get("mismatch_output").shapeDeclared, false, "Contradictory Slice control cardinalities must remain fail-closed.");
expectEqual(malformedDynamicSlice.evidence.rule_unresolved_nodes[0]?.reason, "slice_control_cardinality_mismatch", "Malformed Slice vectors should retain a deterministic reason code.");

const split = run([
  node("Split", ["split_input", "split_sizes"], ["part_a", "part_b"]),
], [constant("split_input", [1, 2, 3, 4, 5], [5]), constant("split_sizes", [2, 3], [2])]);
expectEqual(JSON.stringify(split.tensorMap.get("part_a").staticValues), JSON.stringify([1, 2]), "Split should preserve the first exact value segment.");
expectEqual(JSON.stringify(split.tensorMap.get("part_b").staticValues), JSON.stringify([3, 4, 5]), "Split should preserve the second exact value segment.");

const reduceNoop = run([
  node("ReduceSum", ["reduce_input", "empty_axes"], ["reduce_out"], { noop_with_empty_axes: { i: 1 } }),
], [tensor("reduce_input", "FLOAT32", [2, 3]), constant("empty_axes", [], [0])], 18);
expectEqual(JSON.stringify(reduceNoop.tensorMap.get("reduce_out").shape), JSON.stringify([2, 3]), "An explicitly empty axes input with noop_with_empty_axes=1 should preserve shape.");

const constantTensor = constant("attribute_value", [2, 6], [2]);
const constantNode = run([
  node("Constant", [], ["constant_out"], { value: { tensor: constantTensor } }),
  node("Reshape", ["reshape_source", "constant_out"], ["constant_reshaped"]),
], [tensor("reshape_source", "FLOAT32", [3, 4])]);
expectEqual(JSON.stringify(constantNode.tensorMap.get("constant_out").staticValues), JSON.stringify([2, 6]), "Tensor-valued Constant attributes should expose exact static data.");
expectEqual(JSON.stringify(constantNode.tensorMap.get("constant_reshaped").shape), JSON.stringify([2, 6]), "Reshape should consume a Tensor-valued Constant shape.");

const corpusResidualRules = run([
  node("DynamicQuantizeLinear", ["float_activation"], ["dynamic_q", "dynamic_scale", "dynamic_zp"]),
  node("DepthToSpace", ["depth_input"], ["depth_output"], { blocksize: { i: 2 }, mode: { s: "CRD" } }),
  node("ScatterND", ["scatter_data", "scatter_indices", "scatter_updates"], ["scatter_output"]),
  node("ConvTranspose", ["deconv_input", "deconv_weight"], ["deconv_output"], { strides: { ints: [2, 2] }, pads: { ints: [1, 1, 1, 1] } }),
  node("Range", ["range_start", "range_limit", "range_delta"], ["range_output"]),
  node("STFT", ["signal", "frame_step", "window"], ["stft_output"], { onesided: { i: 1 } }),
], [
  tensor("float_activation", "FLOAT32", [1, 3, 8, 8]),
  tensor("depth_input", "FLOAT32", [1, 12, 8, 8]),
  tensor("scatter_data", "FLOAT32", [4, 3]), tensor("scatter_indices", "INT64", [2, 1]), tensor("scatter_updates", "FLOAT32", [2, 3]),
  tensor("deconv_input", "FLOAT32", [1, 4, 8, 8]), tensor("deconv_weight", "FLOAT32", [4, 6, 3, 3]),
  constant("range_start", [0], [], "FLOAT32"), constant("range_limit", [10], [], "FLOAT32"), constant("range_delta", [2], [], "FLOAT32"),
  tensor("signal", "FLOAT32", [1, 16, 1]), constant("frame_step", [4], []), tensor("window", "FLOAT32", [8]),
]);
expectEqual(JSON.stringify(corpusResidualRules.tensorMap.get("dynamic_q").shape), JSON.stringify([1, 3, 8, 8]), "DynamicQuantizeLinear should retain activation shape.");
expectEqual(corpusResidualRules.tensorMap.get("dynamic_scale").dtype, "FLOAT32", "DynamicQuantizeLinear scale output should be a FLOAT scalar.");
expectEqual(JSON.stringify(corpusResidualRules.tensorMap.get("depth_output").shape), JSON.stringify([1, 3, 16, 16]), "DepthToSpace should conserve NCHW cardinality.");
expectEqual(JSON.stringify(corpusResidualRules.tensorMap.get("scatter_output").shape), JSON.stringify([4, 3]), "ScatterND output should retain data shape.");
expectEqual(JSON.stringify(corpusResidualRules.tensorMap.get("deconv_output").shape), JSON.stringify([1, 6, 15, 15]), "ConvTranspose should apply stride, effective kernel, and pads exactly.");
expectEqual(JSON.stringify(corpusResidualRules.tensorMap.get("range_output").staticValues), JSON.stringify([0, 2, 4, 6, 8]), "Range should propagate bounded exact values.");
expectEqual(JSON.stringify(corpusResidualRules.tensorMap.get("stft_output").shape), JSON.stringify([1, 3, 5, 2]), "STFT should derive frame and one-sided frequency cardinality.");
expectEqual(corpusResidualRules.evidence.rule_unsupported_nodes, 0, "Every corpus-prioritized rule should be registered.");
expectEqual(corpusResidualRules.evidence.rule_unresolved_node_count, 0, "Every fully static corpus-prioritized rule should resolve.");

const symbolicRange = run([
  node("Range", ["symbolic_range_start", "symbolic_range_limit", "symbolic_range_delta"], ["symbolic_range_output"]),
], [
  constant("symbolic_range_start", [0], []),
  tensor("symbolic_range_limit", "INT64", [], {
    staticDimensionValuesStatus: "assessed_exact_static_shape_data",
    staticDimensionValuesComplete: true,
    staticDimensionValues: [{ kind: "symbolic", value: null, parameter: "sequence_length", denotation: "", valueFieldCount: 1 }],
    staticDimensionValuesSource: "fixture",
  }),
  constant("symbolic_range_delta", [1], []),
]);
expectEqual(JSON.stringify(symbolicRange.tensorMap.get("symbolic_range_output").shape), JSON.stringify([-1]), "Dimension-valued Range controls should produce an exact symbolic rank-1 contract.");
expect(symbolicRange.tensorMap.get("symbolic_range_output").typeProto?.shapeDimensions?.[0]?.parameter?.startsWith("deepbom_expr:range_len("), "Dynamic Range cardinality must remain an explicit exact expression.");
expectEqual(symbolicRange.evidence.rule_unresolved_node_count, 0, "A Range bounded by exact symbolic shape data should not remain a local residual.");

const unboundRangeDelta = run([
  node("Range", ["unbound_delta_start", "unbound_delta_limit", "unbound_delta"], ["unbound_delta_output"]),
], [
  constant("unbound_delta_start", [0], []), constant("unbound_delta_limit", [10], []),
  symbolicDimensionValuesTensor("unbound_delta", ["runtime_step"], []),
]);
expectEqual(unboundRangeDelta.tensorMap.get("unbound_delta_output").shapeDeclared, false, "A symbolic Range delta that may be zero must remain unassessed.");
expectEqual(unboundRangeDelta.evidence.rule_unresolved_nodes[0]?.reason, "range_delta_not_artifact_bound", "An unbound Range delta should retain a deterministic conditional-validity reason.");

const symbolicConvTranspose = run([
  node("ConvTranspose", ["symbolic_deconv_input", "symbolic_deconv_weight"], ["symbolic_deconv_output"], {
    kernel_shape: { ints: [4, 4] }, strides: { ints: [2, 2] }, pads: { ints: [1, 1, 1, 1] },
  }),
  node("BatchNormalization", ["symbolic_deconv_output", "bn_scale", "bn_bias", "bn_mean", "bn_var"], ["symbolic_deconv_bn"]),
], [
  symbolicTensor("symbolic_deconv_input", "FLOAT32", ["batch_size", 1280, 16, 12]),
  tensor("symbolic_deconv_weight", "FLOAT32", [1280, 256, 4, 4]),
  tensor("bn_scale", "FLOAT32", [256]), tensor("bn_bias", "FLOAT32", [256]),
  tensor("bn_mean", "FLOAT32", [256]), tensor("bn_var", "FLOAT32", [256]),
]);
expectEqual(JSON.stringify(symbolicConvTranspose.tensorMap.get("symbolic_deconv_output").shape), JSON.stringify([-1, 256, 32, 24]), "ConvTranspose should preserve a symbolic batch and derive exact spatial extents.");
expectEqual(symbolicConvTranspose.tensorMap.get("symbolic_deconv_output").typeProto?.shapeDimensions?.[0]?.parameter, "batch_size", "ConvTranspose must retain the input batch symbol.");
expectEqual(JSON.stringify(symbolicConvTranspose.tensorMap.get("symbolic_deconv_bn").shape), JSON.stringify([-1, 256, 32, 24]), "A resolved symbolic ConvTranspose contract should propagate through same-shape consumers.");
expectEqual(symbolicConvTranspose.evidence.rule_unresolved_node_count, 0, "A symbolic batch alone must not block ConvTranspose inference.");

const symbolicChannelConvTranspose = run([
  node("ConvTranspose", ["symbolic_channel_deconv_input", "symbolic_channel_deconv_weight"], ["symbolic_channel_deconv_output"], {
    strides: { ints: [2] }, pads: { ints: [1, 1] },
  }),
], [
  symbolicTensor("symbolic_channel_deconv_input", "FLOAT32", ["batch_size", "feature_size", "sequence_length"]),
  tensor("symbolic_channel_deconv_weight", "FLOAT32", [80, 24, 3]),
]);
expectEqual(JSON.stringify(symbolicChannelConvTranspose.tensorMap.get("symbolic_channel_deconv_output").shape), JSON.stringify([-1, 24, -1]), "ConvTranspose should infer its output contract when the serialized input channel is symbolic and the weight contract is concrete.");
expectEqual(symbolicChannelConvTranspose.evidence.rule_unresolved_node_count, 0, "A symbolic input channel must not block an otherwise complete ConvTranspose output contract.");

const dilationPermittedOutputPadding = run([
  node("ConvTranspose", ["dilated_deconv_input", "dilated_deconv_weight"], ["dilated_deconv_output"], {
    strides: { ints: [1] }, dilations: { ints: [2] }, output_padding: { ints: [1] },
  }),
], [
  tensor("dilated_deconv_input", "FLOAT32", [1, 2, 4]),
  tensor("dilated_deconv_weight", "FLOAT32", [2, 3, 3]),
]);
expectEqual(JSON.stringify(dilationPermittedOutputPadding.tensorMap.get("dilated_deconv_output").shape), JSON.stringify([1, 3, 9]), "ConvTranspose output_padding may be smaller than dilation even when it is not smaller than stride.");
expectEqual(dilationPermittedOutputPadding.evidence.rule_unresolved_node_count, 0, "A source-valid dilation-bounded output_padding contract must resolve.");

const symbolicStft = run([
  node("STFT", ["dynamic_signal", "dynamic_frame_step", "dynamic_window"], ["dynamic_stft"], { onesided: { i: 1 } }),
  node("Transpose", ["dynamic_stft"], ["dynamic_stft_transposed"], { perm: { ints: [0, 3, 2, 1] } }),
], [
  symbolicTensor("dynamic_signal", "FLOAT32", ["batch_size", "audio_samples", 1]),
  constant("dynamic_frame_step", [4], []),
  tensor("dynamic_window", "FLOAT32", [8]),
]);
const symbolicStftDimensions = symbolicStft.tensorMap.get("dynamic_stft").typeProto?.shapeDimensions || [];
expectEqual(symbolicStft.evidence.rule_unresolved_node_count, 0, "A symbolic signal length with static STFT controls must preserve a complete output shape contract.");
expectEqual(symbolicStftDimensions[0]?.parameter, "batch_size", "STFT must preserve the symbolic batch identity.");
expectEqual(symbolicStftDimensions[1]?.parameter, "deepbom_expr:add(floor_div(deepbom_expr:add(s:audio_samples,-8),4),1)", "STFT must derive the exact symbolic frame-count expression.");
expectEqual(JSON.stringify(symbolicStft.tensorMap.get("dynamic_stft_transposed").shape), JSON.stringify([-1, 2, 5, -1]), "Downstream Transpose must preserve the symbolic STFT contract without fabricating a signal length.");
expectEqual(symbolicStft.tensorMap.get("dynamic_stft_transposed").typeProto?.shapeDimensions?.[3]?.parameter, symbolicStftDimensions[1]?.parameter, "Downstream shape propagation must retain the exact STFT frame-count identity.");

const hugeCardinality = run([
  node("Size", ["huge_input"], ["huge_size"]),
  node("Reshape", ["huge_input", "safe_huge_target"], ["safe_huge_reshape"]),
  node("Reshape", ["huge_input", "unsafe_huge_target"], ["unsafe_huge_reshape"]),
], [
  tensor("huge_input", "FLOAT32", [Number.MAX_SAFE_INTEGER, 2]),
  constant("safe_huge_target", [-1, 2]),
  constant("unsafe_huge_target", [-1, 1]),
]);
expect(hugeCardinality.evidence.rule_unresolved_nodes.some((row) => row.op_name === "Size" && row.reason === "size_output_outside_exact_static_integer_range"), "Size must not round a logical cardinality above JavaScript's exact-integer range.");
expectEqual(JSON.stringify(hugeCardinality.tensorMap.get("safe_huge_reshape").shape), JSON.stringify([Number.MAX_SAFE_INTEGER, 2]), "Reshape should use exact BigInt cardinality arithmetic when the inferred dimension remains representable.");
expect(hugeCardinality.evidence.rule_unresolved_nodes.some((row) => row.op_name === "Reshape" && row.reason === "reshape_inferred_dimension_outside_exact_static_integer_range"), "Reshape must fail closed when the inferred dimension itself is outside the exact static dimension domain.");

const symbolicResize = run([
  node("Resize", ["resize_input", "", "", "resize_sizes"], ["resize_output"]),
  node("Transpose", ["resize_output"], ["resize_transposed"], { perm: { ints: [0, 2, 3, 1] } }),
], [
  symbolicTensor("resize_input", "FLOAT32", ["batch_size", 64, "source_height", "source_width"]),
  tensor("resize_sizes", "INT64", [4], {
    staticDimensionValuesStatus: "assessed_exact_symbolic_shape_data",
    staticDimensionValuesComplete: true,
    staticDimensionValues: [
      { kind: "symbolic", value: null, parameter: "batch_size", denotation: "", valueFieldCount: 1 },
      { kind: "value", value: 128, parameter: "", denotation: "", valueFieldCount: 1 },
      { kind: "symbolic", value: null, parameter: "deepbom_expr:div(s:height,v:14)", denotation: "", valueFieldCount: 1 },
      { kind: "symbolic", value: null, parameter: "deepbom_expr:div(s:width,v:14)", denotation: "", valueFieldCount: 1 },
    ],
    staticDimensionValuesSource: "fixture",
  }),
]);
expectEqual(JSON.stringify(symbolicResize.tensorMap.get("resize_output").shape), JSON.stringify([-1, 128, -1, -1]), "Resize sizes should preserve a complete symbolic output rank and dimension contract.");
expectEqual(symbolicResize.tensorMap.get("resize_output").typeProto?.shapeDimensions?.[2]?.parameter, "deepbom_expr:div(s:height,v:14)", "Resize must preserve the exact symbolic size identity.");
expectEqual(JSON.stringify(symbolicResize.tensorMap.get("resize_transposed").shape), JSON.stringify([-1, -1, -1, 128]), "Downstream operators should retain the symbolic Resize rank instead of losing the graph contract.");
expectEqual(symbolicResize.evidence.rule_unresolved_node_count, 0, "A complete symbolic Resize sizes vector must resolve without runtime numeric binding.");

expect(ONNX_SHAPE_INFERENCE_OPS.size >= 119, "The pinned local rule inventory should retain the expanded operator set.");
expectEqual(ONNX_SHAPE_INFERENCE_SOURCE.commit, "be2b5fde82d9c8874f3d19328bdfe3b6962dc67b", "Shape inference should bind the pinned ONNX commit.");
expect(ONNX_SHAPE_INFERENCE_SOURCE.documents.every((row) => /^[a-f0-9]{64}$/.test(row.sha256)), "Every shape-inference source document should carry a SHA-256.");

const attention4d = run([
  node("Attention", ["attn_q", "attn_k", "attn_v", "", "attn_pk", "attn_pv"], ["attn_y", "attn_present_k", "attn_present_v", "attn_qk"]),
], [
  tensor("attn_q", "FLOAT32", [2, 8, 16, 64]), tensor("attn_k", "FLOAT32", [2, 4, 20, 64]),
  tensor("attn_v", "FLOAT16", [2, 4, 20, 80]), tensor("attn_pk", "FLOAT32", [2, 4, 5, 64]),
  tensor("attn_pv", "FLOAT16", [2, 4, 5, 80]),
], 24);
expectEqual(JSON.stringify(attention4d.tensorMap.get("attn_y").shape), JSON.stringify([2, 8, 16, 80]), "Attention must derive its 4D value projection output shape.");
expectEqual(JSON.stringify(attention4d.tensorMap.get("attn_present_k").shape), JSON.stringify([2, 4, 25, 64]), "Attention must add incoming and past key sequence lengths exactly.");
expectEqual(JSON.stringify(attention4d.tensorMap.get("attn_present_v").shape), JSON.stringify([2, 4, 25, 80]), "Attention must preserve the value-head width in its updated cache.");
expectEqual(JSON.stringify(attention4d.tensorMap.get("attn_qk").shape), JSON.stringify([2, 8, 16, 25]), "Attention must expose the exact QK score shape when requested.");
expectEqual(attention4d.evidence.rule_unresolved_node_count, 0, "A fully static Attention contract must close without a shape residual.");

const attention3d = run([
  node("Attention", ["attn3_q", "attn3_k", "attn3_v"], ["attn3_y"], {
    q_num_heads: { i: 8 }, kv_num_heads: { i: 4 },
  }),
], [
  tensor("attn3_q", "FLOAT32", [2, 16, 512]), tensor("attn3_k", "FLOAT32", [2, 20, 256]),
  tensor("attn3_v", "FLOAT16", [2, 20, 320]),
], 24);
expectEqual(JSON.stringify(attention3d.tensorMap.get("attn3_y").shape), JSON.stringify([2, 16, 640]), "Attention must derive a 3D output width from query heads and value-head width.");

const invalidAttentionCache = run([
  node("Attention", ["bad_attn_q", "bad_attn_k", "bad_attn_v", "", "bad_attn_pk", "bad_attn_pv"], ["bad_attn_y"]),
], [
  tensor("bad_attn_q", "FLOAT32", [1, 4, 2, 8]), tensor("bad_attn_k", "FLOAT32", [1, 2, 3, 8]),
  tensor("bad_attn_v", "FLOAT32", [1, 2, 3, 8]), tensor("bad_attn_pk", "FLOAT32", [1, 2, 1, 8]),
  tensor("bad_attn_pv", "FLOAT32", [1, 2, 1, 8]),
], 24);
expectEqual(invalidAttentionCache.evidence.semantic_contract_conflicts[0]?.reason, "attention_past_present_cache_contract_incomplete", "Attention past and present caches must be used as a complete pair contract.");

const deformConvShape = run([
  node("DeformConv", ["deform_x", "deform_w", "deform_offset", "deform_bias", "deform_mask"], ["deform_y"], {
    group: { i: 2 }, offset_group: { i: 2 },
  }),
], [
  tensor("deform_x", "FLOAT32", [2, 8, 16, 16]), tensor("deform_w", "FLOAT32", [12, 4, 3, 3]),
  tensor("deform_offset", "FLOAT32", [2, 36, 14, 14]), tensor("deform_bias", "FLOAT32", [12]),
  tensor("deform_mask", "FLOAT32", [2, 18, 14, 14]),
], 22);
expectEqual(JSON.stringify(deformConvShape.tensorMap.get("deform_y").shape), JSON.stringify([2, 12, 14, 14]), "DeformConv must share the Conv spatial shape contract after validating offset and mask cardinalities.");
expectEqual(deformConvShape.evidence.rule_unresolved_node_count, 0, "A complete DeformConv auxiliary contract must close statically.");

const invalidDeformOffset = run([
  node("DeformConv", ["bad_deform_x", "bad_deform_w", "bad_deform_offset"], ["bad_deform_y"], { offset_group: { i: 2 } }),
], [
  tensor("bad_deform_x", "FLOAT32", [1, 4, 8, 8]), tensor("bad_deform_w", "FLOAT32", [6, 4, 3, 3]),
  tensor("bad_deform_offset", "FLOAT32", [1, 18, 6, 6]),
], 22);
expectEqual(invalidDeformOffset.evidence.semantic_contract_conflicts[0]?.reason, "deform_conv_offset_shape_mismatch", "DeformConv must reject an offset channel count that omits offset_group multiplicity.");

const einsumShape = run([
  node("Einsum", ["einsum_a", "einsum_b"], ["einsum_y"], { equation: { s: "bij,bjk->bik" } }),
  node("Einsum", ["einsum_scalar", "einsum_vector"], ["einsum_scaled"], { equation: { s: ",i->i" } }),
], [
  tensor("einsum_a", "FLOAT32", [2, 3, 4]), tensor("einsum_b", "FLOAT32", [2, 4, 5]),
  tensor("einsum_scalar", "FLOAT32", []), tensor("einsum_vector", "FLOAT32", [7]),
], 12);
expectEqual(JSON.stringify(einsumShape.tensorMap.get("einsum_y").shape), JSON.stringify([2, 3, 5]), "Einsum must derive explicit-output contraction shapes.");
expectEqual(JSON.stringify(einsumShape.tensorMap.get("einsum_scaled").shape), JSON.stringify([7]), "Einsum must accept the empty subscript of a scalar operand.");

const invalidEinsum = run([
  node("Einsum", ["bad_einsum_a", "bad_einsum_b"], ["bad_einsum_y"], { equation: { s: "bij,bjk->bik" } }),
], [tensor("bad_einsum_a", "FLOAT32", [2, 3, 4]), tensor("bad_einsum_b", "FLOAT32", [2, 6, 5])], 12);
expectEqual(invalidEinsum.evidence.semantic_contract_conflicts[0]?.reason, "einsum_label_dimension_mismatch", "Einsum must reject incompatible dimensions bound to the same label.");

done(`${ONNX_SHAPE_INFERENCE_OPS.size} pinned local rules, shape-data propagation, zero dimensions, conflict rejection, views, slicing, reductions, attention, deformable convolution, and Einstein contractions`);

import {
  BACKEND_PLACEMENT_STATES,
  buildBackendPlacementProjection,
} from "./backend-placement-projection.js";
import { canonicalJson } from "./report-utils.js";
import { sha256TextHex } from "./sha256-sync.js";

const TF_COMMIT = "87bbf65b8d23d3f06912b1b2183587e1884bc45c";
const LITERT_COMMIT = "e8b1ea421cf527584589213c8f5127a8ae8cc997";

const COREML_SOURCE = Object.freeze({
  repository: "https://github.com/tensorflow/tensorflow",
  commit: TF_COMMIT,
  support_path: "tensorflow/lite/delegates/coreml/coreml_delegate.mm",
  support_sha256: "c09aa345581353ac6be2dfb3a5f7be218ed976f3388a5d4dad1e88a4bda0f0f1",
  builder_path: "tensorflow/lite/delegates/coreml/builders/op_builder.cc",
  builder_sha256: "e2f024d0d74befd5dc06f8cf731fe33d1556c6b46cd457a0ccd85019e465156f",
  partition_helper_header_path: "tensorflow/lite/delegates/utils.h",
  partition_helper_header_sha256: "390a88c51785f87c1d4c31c27427e6246605d450fc4edfa534c68277c6cb8cc9",
  partition_helper_source_path: "tensorflow/lite/delegates/utils.cc",
  partition_helper_source_sha256: "7026cd7e00360b3e528a8a53d91b4ca1f7fb05ccbcad9c6f784fc6df02913464",
});

const COREML_OPS = Object.freeze({
  ADD: 1, AVERAGE_POOL_2D: 1, CONCATENATION: 1, CONV_2D: 1,
  DEPTHWISE_CONV_2D: 2, FULLY_CONNECTED: 6, HARD_SWISH: 1, LOGISTIC: 1,
  MAX_POOL_2D: 1, MEAN: 1, MIRROR_PAD: 1, MUL: 1, PAD: 1, PADV2: 1,
  RELU: 1, RELU_N1_TO_1: 1, RELU6: 1, RESHAPE: 1, RESIZE_BILINEAR: 1,
  SOFTMAX: 1, TANH: 1, TRANSPOSE_CONV: 1,
});

const QUALCOMM_SOURCE = Object.freeze({
  repository: "https://github.com/google-ai-edge/LiteRT",
  commit: LITERT_COMMIT,
  support_path: "litert/vendors/qualcomm/compiler/Qualcomm_QNN_Compiler.md",
  support_sha256: "873a6d442010aa1e448b90bfa5601424627e00fa4475cf012eea32fd87205e3f",
  compiler_path: "litert/vendors/qualcomm/compiler/qnn_compiler_plugin.cc",
  compiler_sha256: "2c71cf954ac906786b1f56bc053a570708cccc0e7cc25e56e704777d2c251623",
  dispatch_path: "litert/vendors/qualcomm/dispatch/dispatch_api.cc",
  dispatch_sha256: "87ef37e5071b86d4bb7fa63fdf9bb5465b205a1290998310f1e84a5aaaa51b12",
  supported_soc_path: "litert/vendors/qualcomm/supported_soc.csv",
  supported_soc_sha256: "a9b77234cadaae2b40a99018d3e2f4fc7cbac7267125599a3b09e90f6b9050f7",
});

const QUALCOMM_OPS = new Set([
  "ABS", "ADD", "ARG_MAX", "ARG_MIN", "AVERAGE_POOL_2D", "BATCH_MATMUL",
  "BATCH_TO_SPACE_ND", "BROADCAST_TO", "CAST", "CEIL", "CONCATENATION",
  "CONV_2D", "CONV_3D", "COS", "CUMSUM", "DEPTH_TO_SPACE", "DEPTHWISE_CONV_2D",
  "DEQUANTIZE", "DIV", "DYNAMIC_UPDATE_SLICE", "ELU", "EMBEDDING_LOOKUP", "EQUAL",
  "EXP", "FLOOR", "FLOOR_DIV", "FLOOR_MOD", "FULLY_CONNECTED", "GATHER", "GATHER_ND",
  "GELU", "GREATER", "GREATER_EQUAL", "HARD_SWISH", "L2_NORMALIZATION", "L2_POOL_2D",
  "LEAKY_RELU", "LESS", "LESS_EQUAL", "LOG", "LOGICAL_AND", "LOGICAL_NOT", "LOGICAL_OR",
  "LOGISTIC", "MAXIMUM", "MAX_POOL_2D", "MEAN", "MINIMUM", "MIRROR_PAD", "MUL", "NEG",
  "NOT_EQUAL", "ONE_HOT", "PACK", "PAD", "PADV2", "POW", "PRELU", "QUANTIZE",
  "REDUCE_ALL", "REDUCE_ANY", "REDUCE_MAX", "REDUCE_MIN", "RELU", "RELU_0_TO_1",
  "RELU6", "RELU_N1_TO_1", "RESHAPE", "RESIZE_BILINEAR", "RESIZE_NEAREST_NEIGHBOR",
  "REVERSE_V2", "ROUND", "RSQRT", "SELECT", "SELECT_V2", "SIGN", "SIN", "SLICE",
  "SOFTMAX", "SPACE_TO_BATCH_ND", "SPACE_TO_DEPTH", "SPLIT", "SPLIT_V", "SQRT", "SQUARE",
  "SQUARED_DIFFERENCE", "STABLEHLO_COMPOSITE", "STRIDED_SLICE", "SUB", "SUM", "TANH",
  "TILE", "TOPK_V2", "TRANSPOSE", "TRANSPOSE_CONV", "UNPACK",
]);

export const TFLITE_COREML_RULEPACK_SHA256 = sourceDigest("tflite_coreml", COREML_SOURCE, Object.keys(COREML_OPS));
export const LITERT_QUALCOMM_RULEPACK_SHA256 = sourceDigest("litert_qualcomm_qnn", QUALCOMM_SOURCE, [...QUALCOMM_OPS]);

export function buildTfliteAdditionalSourceProfiles(analysis) {
  if (String(analysis?.format || "").toLowerCase() !== "tflite" || !Array.isArray(analysis?.ops)
    || !Array.isArray(analysis?.tensors)) return [];
  return [
    buildBackendPlacementProjection({
      analysis,
      profileId: "tflite_coreml_delegate",
      label: "TFLite Core ML delegate",
      evidenceClass: "SOURCE_PINNED_ARTIFACT_PRECHECK",
      rows: coreMlRows(analysis),
      source: { ...COREML_SOURCE, rulepack_sha256: TFLITE_COREML_RULEPACK_SHA256 },
      interpretationBoundary: "Pinned TensorFlow Core ML delegate registration, FP16GraphPartitionHelper, serialized op version, constant-FP16 DEQUANTIZE full-versus-partial delegation rule, and visible first-input dtype precheck. Op-specific validators, selected Apple OS/Core ML version and options, delegate build, compiled partition, device selection, execution, and latency remain unresolved.",
    }),
    buildBackendPlacementProjection({
      analysis,
      profileId: "litert_qualcomm_qnn",
      label: "LiteRT Qualcomm QNN compiler",
      evidenceClass: "SOURCE_PINNED_ARTIFACT_PRECHECK",
      rows: qualcommRows(analysis),
      source: { ...QUALCOMM_SOURCE, rulepack_sha256: LITERT_QUALCOMM_RULEPACK_SHA256 },
      interpretationBoundary: "Pinned LiteRT Qualcomm compiler legalization registry precheck. Per-op legalization predicates, selected QNN SDK/plugin build, SoC/backend capability, compiled QNN graph, dispatch, execution, transfer, and latency remain unresolved until identity-bound evidence is imported.",
    }),
  ];
}

export function tfliteAcceleratorSourceManifest() {
  return Object.freeze({
    schema: "deepbom.tflite_additional_accelerator_source_manifest.v1",
    profiles: [
      { id: "tflite_coreml_delegate", source: COREML_SOURCE, rulepack_sha256: TFLITE_COREML_RULEPACK_SHA256, registered_op_count: Object.keys(COREML_OPS).length },
      { id: "litert_qualcomm_qnn", source: QUALCOMM_SOURCE, rulepack_sha256: LITERT_QUALCOMM_RULEPACK_SHA256, registered_op_count: QUALCOMM_OPS.size },
    ],
  });
}

function coreMlRows(analysis) {
  const rows = analysis.ops.map((op, position) => {
    const opIndex = opIndexOf(op, position);
    const opName = String(op?.name || "UNKNOWN");
    if (opName === "DEQUANTIZE") return coreMlDequantizeRow(analysis, op, opIndex);
    const maximumVersion = COREML_OPS[opName];
    if (maximumVersion == null) return excluded(opIndex, "builtin_operator_not_registered_in_pinned_coreml_delegate_source");
    const version = Number(op?.version ?? 1);
    if (!Number.isSafeInteger(version) || version < 1) return unresolved(opIndex, "serialized_op_version_invalid_or_missing");
    if (version > maximumVersion) return excluded(opIndex, `serialized_op_version_${version}_exceeds_coreml_maximum_${maximumVersion}`);
    const inputPosition = opName === "TRANSPOSE_CONV" ? 2 : 0;
    const dtype = inputDtype(analysis, op, inputPosition);
    if (dtype && dtype !== "FLOAT32") return excluded(opIndex, `coreml_delegate_first_runtime_input_dtype_${dtype}_is_not_float32`);
    const predicates = [
      "selected_apple_os_coreml_version_and_delegate_build",
      "coreml_op_specific_validator_and_partition_acceptance",
    ];
    if (!dtype) predicates.push("first_runtime_input_dtype_not_resolved");
    return candidate(opIndex, predicates);
  });
  const fp16DequantRows = rows.filter((row) => row.fp16_constant_dequantize === true);
  if (!fp16DequantRows.length) return rows;
  const otherRows = rows.filter((row) => row.fp16_constant_dequantize !== true);
  const hasDefinitePartialDelegation = otherRows.some((row) => row.state === BACKEND_PLACEMENT_STATES.DEFINITE_EXCLUSION);
  const hasUnresolvedPartitionMembership = otherRows.some((row) => row.state === BACKEND_PLACEMENT_STATES.UNRESOLVED);
  return rows.map((row) => {
    if (row.fp16_constant_dequantize !== true) return row;
    if (hasDefinitePartialDelegation) {
      return excluded(row.op_index, "fp16_constant_dequantize_kept_on_cpu_during_partial_coreml_delegation");
    }
    if (hasUnresolvedPartitionMembership) {
      return unresolved(row.op_index, "fp16_constant_dequantize_full_delegation_condition_unresolved");
    }
    return candidate(row.op_index, [
      "all_non_dequantize_nodes_must_be_accepted_by_the_selected_coreml_delegate_build",
      "coreml_delegate_full_graph_partition_and_option_limits",
    ]);
  });
}

function coreMlDequantizeRow(analysis, op, opIndex) {
  const tensor = inputTensor(analysis, op, 0);
  if (!tensor) return unresolved(opIndex, "dequantize_input_tensor_contract_not_resolved");
  const dtype = String(tensor.dtype || "").toUpperCase();
  if (!dtype) return unresolved(opIndex, "dequantize_input_dtype_not_resolved");
  if (dtype !== "FLOAT16") return excluded(opIndex, `dequantize_input_dtype_${dtype}_is_not_constant_float16`);
  if (tensor.constant_buffer === false) return excluded(opIndex, "float16_dequantize_input_is_not_a_serialized_constant");
  if (tensor.constant_buffer !== true) return unresolved(opIndex, "float16_dequantize_constant_storage_not_resolved");
  return {
    op_index: opIndex,
    state: BACKEND_PLACEMENT_STATES.UNRESOLVED,
    reason_codes: [],
    unresolved_predicates: ["fp16_constant_dequantize_full_delegation_condition_pending"],
    fp16_constant_dequantize: true,
  };
}

function qualcommRows(analysis) {
  return analysis.ops.map((op, position) => {
    const opIndex = opIndexOf(op, position);
    const opName = String(op?.name || "UNKNOWN");
    return QUALCOMM_OPS.has(opName)
      ? candidate(opIndex, ["qnn_legalization_predicates_selected_sdk_plugin_soc_and_backend"])
      : excluded(opIndex, "litert_opcode_not_registered_in_pinned_qualcomm_compiler_support_registry");
  });
}

function inputDtype(analysis, op, position) {
  return String(inputTensor(analysis, op, position)?.dtype || "").toUpperCase() || null;
}

function inputTensor(analysis, op, position) {
  const index = Number(op?.inputs?.[position]);
  if (!Number.isSafeInteger(index) || index < 0) return null;
  return analysis.tensors.find((row, fallback) => Number(row?.index ?? fallback) === index) || null;
}

function candidate(opIndex, predicates) {
  return { op_index: opIndex, state: BACKEND_PLACEMENT_STATES.CONDITIONALLY_ELIGIBLE, reason_codes: [], unresolved_predicates: predicates };
}

function excluded(opIndex, reason) {
  return { op_index: opIndex, state: BACKEND_PLACEMENT_STATES.DEFINITE_EXCLUSION, reason_codes: [reason], unresolved_predicates: [] };
}

function unresolved(opIndex, reason) {
  return { op_index: opIndex, state: BACKEND_PLACEMENT_STATES.UNRESOLVED, reason_codes: [], unresolved_predicates: [reason] };
}

function opIndexOf(op, position) {
  const value = Number(op?.index);
  return Number.isSafeInteger(value) ? value : position;
}

function sourceDigest(id, source, ops) {
  return sha256TextHex(canonicalJson({ schema: "deepbom.source_rulepack_identity.v1", id, source, registered_ops: [...ops].sort() }));
}

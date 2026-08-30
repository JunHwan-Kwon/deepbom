import { buildOnnxDomainAnalysis } from "./lib/onnx-domain-analysis.js";
import { buildOnnxDynamicShapeCostContract } from "./lib/dynamic-shape-cost.js";
import { inferOnnxShapesWithReachableScopes } from "./lib/onnx-extended-shape-inference.js";
import { buildOnnxSparseTensorContract } from "./lib/onnx-sparse-tensor.js";
import { buildOnnxTypeProtoContract, isDenseTensorValue } from "./lib/onnx-type-proto.js";
import { applyGraphTopology } from "./lib/graph-topology.js";
import { classifyOnnxMacOperation, isOnnxAlgorithmDependentArithmetic, isOnnxMacBearingOperation } from "./lib/onnx-operation-cost.js";
import { parseOnnxEinsumEquation } from "./lib/onnx-einsum-contract.js";
import { exactNonnegativeRatio } from "./lib/exact-rational.js";

const textDecoder = new TextDecoder();
const textEncoder = new TextEncoder();
const MAX_ONNX_NODES = 100_000;
const MAX_ONNX_INITIALIZERS = 100_000;
const MAX_ONNX_TENSORS = 250_000;
export const MAX_ONNX_DECODED_ELEMENTS = 100_000_000;
const MAX_ONNX_INITIALIZER_BYTES = 1_073_741_824;
const MAX_ONNX_QUANT_PARAMETER_ELEMENTS = 65_536;
const MAX_ONNX_FUNCTIONS = 10_000;
const MAX_ONNX_METADATA_PROPERTIES = 10_000;
const MAX_ONNX_METADATA_TEXT_BYTES = 4_194_304;
const MAX_ONNX_EXTERNAL_DATA_ENTRIES = 100_000;
const MAX_ONNX_EXTERNAL_DATA_TEXT_BYTES = 4_194_304;
const MAX_ONNX_EXTERNAL_FILE_COUNT = 1_024;
const MAX_ONNX_EXTERNAL_FILE_BYTES = 536_870_912;
const MAX_ONNX_EXTERNAL_AGGREGATE_BYTES = 1_073_741_824;
const MAX_ONNX_EP_CONDITION_INTEGER_ELEMENTS = 4_096;
const MAX_ONNX_QUANTIZATION_ANNOTATIONS = 100_000;
const MAX_ONNX_QUANTIZATION_ANNOTATION_ENTRIES = 200_000;

const TENSOR_TYPES = {
  0: { name: "UNDEFINED", bits: 0 },
  1: { name: "FLOAT32", bits: 32 },
  2: { name: "UINT8", bits: 8 },
  3: { name: "INT8", bits: 8 },
  4: { name: "UINT16", bits: 16 },
  5: { name: "INT16", bits: 16 },
  6: { name: "INT32", bits: 32 },
  7: { name: "INT64", bits: 64 },
  8: { name: "STRING", bits: 0 },
  9: { name: "BOOL", bits: 8 },
  10: { name: "FLOAT16", bits: 16 },
  11: { name: "FLOAT64", bits: 64 },
  12: { name: "UINT32", bits: 32 },
  13: { name: "UINT64", bits: 64 },
  14: { name: "COMPLEX64", bits: 64 },
  15: { name: "COMPLEX128", bits: 128 },
  16: { name: "BFLOAT16", bits: 16 },
  17: { name: "FLOAT8E4M3FN", bits: 8 },
  18: { name: "FLOAT8E4M3FNUZ", bits: 8 },
  19: { name: "FLOAT8E5M2", bits: 8 },
  20: { name: "FLOAT8E5M2FNUZ", bits: 8 },
  21: { name: "UINT4", bits: 4 },
  22: { name: "INT4", bits: 4 },
  23: { name: "FLOAT4E2M1", bits: 4 },
  24: { name: "FLOAT8E8M0", bits: 8 },
  25: { name: "UINT2", bits: 2 },
  26: { name: "INT2", bits: 2 },
};
const TENSOR_TYPE_BY_NAME = new Map(Object.values(TENSOR_TYPES).map((type) => [type.name, type]));
const ONNX_TENSOR_TYPE_SOURCE = Object.freeze({
  release: "v1.21.0",
  commit: "be2b5fde82d9c8874f3d19328bdfe3b6962dc67b",
  ref: "https://raw.githubusercontent.com/onnx/onnx/be2b5fde82d9c8874f3d19328bdfe3b6962dc67b/onnx/onnx.in.proto",
  sha256: "f4cbc198df3a0f3f4519d4d38cd2262e8f84057583b7313e2d0f981b3f68c213",
});

export const ONNX_OPERATION_COST_SOURCE = Object.freeze({
  repository: "onnx/onnx",
  release: "v1.21.0",
  commit: "be2b5fde82d9c8874f3d19328bdfe3b6962dc67b",
  documents: Object.freeze([
    Object.freeze({
      role: "neural_network_operator_schemas",
      path: "onnx/defs/nn/defs.cc",
      source_ref: "https://raw.githubusercontent.com/onnx/onnx/be2b5fde82d9c8874f3d19328bdfe3b6962dc67b/onnx/defs/nn/defs.cc",
      sha256: "1619dd419d2eaa1da3ad4155206d58d86432829a534d5a8c587269abf5c1df02",
    }),
    Object.freeze({
      role: "matrix_operator_schemas",
      path: "onnx/defs/math/defs.cc",
      source_ref: "https://raw.githubusercontent.com/onnx/onnx/be2b5fde82d9c8874f3d19328bdfe3b6962dc67b/onnx/defs/math/defs.cc",
      sha256: "0428224a3cb2b5aabf87dab3dfca94988c3a913d73b6f39fa295980060b97594",
    }),
  ]),
});

const RUNTIME_REVIEW_OPS = new Map([
  ["Flatten", ["VIEW_OR_MATERIALIZATION_CANDIDATE", "A view may be aliased or materialized depending on the selected runtime kernel."]],
  ["Reshape", ["VIEW_OR_MATERIALIZATION_CANDIDATE", "A view may be aliased or materialized depending on the selected runtime kernel."]],
  ["Squeeze", ["VIEW_OR_MATERIALIZATION_CANDIDATE", "A view may be aliased or materialized depending on the selected runtime kernel."]],
  ["Unsqueeze", ["VIEW_OR_MATERIALIZATION_CANDIDATE", "A view may be aliased or materialized depending on the selected runtime kernel."]],
  ["Transpose", ["LAYOUT_MATERIALIZATION_CANDIDATE", "Layout conversion can require materialization; provider assignment is not inferred."]],
  ["Concat", ["COPY_OR_ALLOCATION_CANDIDATE", "Concatenation commonly allocates an output buffer; exact traffic is runtime-dependent."]],
  ["Expand", ["BROADCAST_MATERIALIZATION_CANDIDATE", "Broadcasting may be represented as a view or materialized by a runtime kernel."]],
  ["Pad", ["COPY_OR_ALLOCATION_CANDIDATE", "Padding produces a larger logical tensor and can require allocation."]],
  ["Resize", ["RESAMPLING_KERNEL", "Resampling is a non-MAC transform whose kernel and temporary-buffer behavior are runtime-dependent."]],
  ["Slice", ["VIEW_OR_MATERIALIZATION_CANDIDATE", "A slice may be aliased or copied depending on strides and runtime support."]],
  ["Split", ["VIEW_OR_MATERIALIZATION_CANDIDATE", "Split outputs may alias or materialize depending on the runtime."]],
  ["Tile", ["COPY_OR_ALLOCATION_CANDIDATE", "Tiling expands logical payload and generally requires materialization."]],
  ["Gather", ["INDEXING_KERNEL", "Indexed reads are non-MAC data movement whose performance depends on access locality."]],
  ["GatherElements", ["INDEXING_KERNEL", "Indexed reads are non-MAC data movement whose performance depends on access locality."]],
  ["GatherND", ["INDEXING_KERNEL", "Indexed reads are non-MAC data movement whose performance depends on access locality."]],
  ["Softmax", ["OUTPUT_REDUCTION_KERNEL", "Softmax is a reduction and output transform, not a structural or view operation."]],
  ["ReduceMean", ["REDUCTION_KERNEL", "Reduction cost and temporary storage depend on axes, shape, and runtime kernel."]],
  ["ArgMax", ["REDUCTION_KERNEL", "Reduction cost depends on axes, shape, and runtime kernel."]],
  ["ArgMin", ["REDUCTION_KERNEL", "Reduction cost depends on axes, shape, and runtime kernel."]],
  ["TopK", ["SELECTION_KERNEL", "Selection cost and temporary storage depend on K, shape, and runtime kernel."]],
  ["NonMaxSuppression", ["SELECTION_KERNEL", "Selection cost is data- and runtime-dependent."]],
  ["Cast", ["ELEMENTWISE_CONVERSION_KERNEL", "Dtype conversion can materialize a full output tensor."]],
  ["ConstantOfShape", ["TENSOR_GENERATION_KERNEL", "The runtime creates and fills a tensor from a shape input."]],
  ["Shape", ["SHAPE_METADATA_KERNEL", "Shape extraction is metadata-oriented and is not a provider-assignment signal."]],
]);
const ONNX_NCHW_INPUT_SEMANTIC_OPS = new Set([
  "AveragePool", "BatchNormalization", "Conv", "ConvInteger", "ConvTranspose", "GlobalAveragePool",
  "GlobalLpPool", "GlobalMaxPool", "InstanceNormalization", "LpPool", "MaxPool",
  "QLinearConv",
]);
export function analyzeOnnxModel(bytes, filename, targetProfile = null, options = {}) {
  const applicableTargetProfile = normalizeOnnxTargetProfile(targetProfile);
  const model = parseOnnxModel(bytes);
  const graph = model.graph;
  if (!graph) {
    throw new Error("ONNX model does not contain a graph");
  }
  const onnxTypeProtoContract = buildOnnxTypeProtoContract(graph, model.functions);
  validateOnnxGraphLimits(graph, model.functions, model.metadataProps);
  const sparseTensorScopes = collectOnnxSparseTensorScopes(graph, model.functions);
  const tensorPayloadScopes = collectOnnxTensorPayloadScopes(graph, model.functions);
  const externalDataResolution = resolveOnnxExternalDataPayloads(tensorPayloadScopes, options?.externalDataFiles || []);
  validateOnnxGraphLimits(graph, model.functions, model.metadataProps);
  const onnxSparseTensorContract = buildOnnxSparseTensorContract(sparseTensorScopes);

  const tensorMap = new Map();
  const initializerNames = new Set([
    ...graph.initializers.map((tensor) => tensor.name),
    ...graph.sparseInitializers.map((sparse) => sparse.values?.name),
  ].filter(Boolean));

  for (const input of graph.inputs) {
    upsertTensor(tensorMap, input.name, { dtype: input.dtype, shape: input.shape, shapeDeclared: input.shapeDeclared, valueKind: input.valueKind, typeProto: input.typeProto, role: "input" });
  }
  for (const output of graph.outputs) {
    upsertTensor(tensorMap, output.name, { dtype: output.dtype, shape: output.shape, shapeDeclared: output.shapeDeclared, valueKind: output.valueKind, typeProto: output.typeProto, role: "output" });
  }
  for (const value of graph.valueInfo) {
    upsertTensor(tensorMap, value.name, { dtype: value.dtype, shape: value.shape, shapeDeclared: value.shapeDeclared, valueKind: value.valueKind, typeProto: value.typeProto });
  }
  for (const initializer of graph.initializers) {
    const external = isExternalInitializer(initializer);
    const payloadAvailable = !external || initializer.externalPayloadVerified === true;
    const initializerElements = onnxInitializerElementCount(initializer);
    const storedBytes = Number(initializer.storedDataBytes || 0);
    const rawBytes = payloadAvailable ? Number(initializer.rawDataBytes || 0) : 0;
    const projectedFp16Bytes = isFloatDtype(initializer.dtype) ? safeExactProduct([initializerElements, 2]) : storedBytes;
    const projectedInt8Bytes = isFloatDtype(initializer.dtype) ? initializerElements : storedBytes;
    const integerValues = summarizeConditionIntegerInitializer(initializer, payloadAvailable);
    const staticValues = summarizeStaticTensor(initializer, payloadAvailable);
    upsertTensor(tensorMap, initializer.name, {
      dtype: initializer.dtype,
      shape: initializer.shape,
      shapeDeclared: true,
      valueKind: "tensor",
      role: "initializer",
      initializerBytes: payloadAvailable ? storedBytes : 0,
      initializerElements,
      initializerStoredElements: initializerElements,
      initializerEmbeddedBytes: external ? 0 : storedBytes,
      initializerEmbeddedStoredElements: external ? 0 : initializerElements,
      initializerVerifiedExternalBytes: external && payloadAvailable ? storedBytes : 0,
      initializerVerifiedExternalStoredElements: external && payloadAvailable ? initializerElements : 0,
      initializerAvailableBytes: payloadAvailable ? storedBytes : 0,
      initializerAvailableStoredElements: payloadAvailable ? initializerElements : 0,
      initializerEmbeddedFloatBytes: !external && isFloatDtype(initializer.dtype) ? storedBytes : 0,
      initializerProjectedEmbeddedFp16Bytes: !external ? projectedFp16Bytes : 0,
      initializerProjectedEmbeddedInt8Bytes: !external ? projectedInt8Bytes : 0,
      initializerExternalComponentCount: external ? 1 : 0,
      initializerVerifiedExternalComponentCount: external && payloadAvailable ? 1 : 0,
      initializerRawDataBytes: rawBytes,
      initializerRawZeroBytes: !payloadAvailable || !initializer.rawData
        ? 0
        : initializer.rawData.reduce((count, value) => count + (value === 0 ? 1 : 0), 0),
      initializerTypedDataBytes: payloadAvailable ? (initializer.typedDataBytes || 0) : 0,
      externalDataEntries: initializer.externalDataEntries || 0,
      externalData: initializer.externalData || [],
      dataLocation: initializer.dataLocation || 0,
      externalPayloadStatus: initializer.externalPayloadStatus || (isExternalInitializer(initializer) ? "not_supplied" : "not_applicable"),
      externalPayloadVerified: initializer.externalPayloadVerified === true,
      externalEmbeddedPayloadConflict: initializer.externalEmbeddedPayloadConflict === true,
      externalSidecarPath: initializer.externalSidecarPath || "",
      externalSidecarBytes: initializer.externalSidecarBytes || 0,
      externalSidecarSha256: initializer.externalSidecarSha256 || "",
      externalSidecarSha1: initializer.externalSidecarSha1 || "",
      initializerIntegerValuesStatus: integerValues.status,
      initializerIntegerValuesComplete: integerValues.complete,
      initializerIntegerValues: integerValues.values,
      initializerIntegerValuesExactComplete: integerValues.exactComplete,
      initializerIntegerValuesExactDecimals: integerValues.exactDecimals,
      staticValuesStatus: staticValues.status,
      staticValuesComplete: staticValues.complete,
      staticValues: staticValues.values,
      staticValuesCanonicalTextComplete: staticValues.canonicalTextComplete === true,
      staticValuesCanonicalTexts: staticValues.canonicalTexts || [],
      staticValuesSource: "initializer",
    });
  }
  for (const sparse of graph.sparseInitializers) upsertSparseInitializer(tensorMap, sparse);
  for (const node of graph.nodes) {
    for (const name of [...node.inputs, ...node.outputs]) {
      if (name) upsertTensor(tensorMap, name, {});
    }
  }
  const onnxDomainAnalysis = buildOnnxDomainAnalysis(model);
  const shapeInference = inferOnnxShapesWithReachableScopes(graph, tensorMap, model, tensorTypeName, onnxDomainAnalysis, assessOnnxScopeIntrinsicCost);
  const onnxQuantizationBinding = bindOnnxQuantization(graph, tensorMap, model.opsets, model.irVersion, model.functions);
  const externalInterfaceNames = new Set([
    ...graph.inputs.map((value) => value.name),
    ...graph.outputs.map((value) => value.name),
  ].filter(Boolean));

  const tensors = [...tensorMap.values()].map((tensor, index) => {
    const negativeZeroIndices = staticNegativeZeroIndices(tensor.staticValues);
    return {
      index,
      name: tensor.name,
      dtype: tensor.dtype || "UNKNOWN",
      value_kind: tensor.valueKind || "unresolved",
      type_proto: tensor.typeProto || null,
      shape: Array.isArray(tensor.shape) ? tensor.shape : [],
      shape_declared: tensor.shapeDeclared === true,
      conditional_shape_contract: tensor.conditionalShapeContract || null,
      conditional_shape_variants: (tensor.conditionalShapeVariants || []).map((variant) => ({
        dtype: variant.dtype || "UNKNOWN",
        value_kind: variant.valueKind || "tensor",
        type_proto: variant.typeProto || null,
        shape: Array.isArray(variant.shape) ? [...variant.shape] : [],
        shape_declared: variant.shapeDeclared === true,
        conditions: structuredClone(variant.conditions || []),
      })),
      runtime_dimension_bounds: structuredClone(tensor.runtimeDimensionBounds || []),
      contract_status: tensor.contractStatus || "assessed",
      contract_conflict: tensor.contractConflict || null,
      shape_signature: Array.isArray(tensor.shape) ? tensor.shape.map((dim) => Number.isSafeInteger(Number(dim)) && Number(dim) >= 0 ? Number(dim) : -1) : [],
      quant_scales: tensor.quantScaleValues?.length || 0,
      quant_zero_points: tensor.quantZeroPointValues?.length || 0,
      scale_sample: tensor.quantScaleValues?.slice(0, 256) || [],
      zero_point_sample: tensor.quantZeroPointValues?.slice(0, 256) || [],
      interface_scale_values: externalInterfaceNames.has(tensor.name) ? [...(tensor.quantScaleValues || [])] : undefined,
      interface_zero_point_values: externalInterfaceNames.has(tensor.name) ? [...(tensor.quantZeroPointValues || [])] : undefined,
      quantized_dimension: Number.isInteger(tensor.quantizedDimension) ? tensor.quantizedDimension : 0,
      quantization_parameterization: tensor.quantizationParameterization || "none",
      quantization_axis_source: tensor.quantizationAxisSource || "",
      quantization_block_size: tensor.quantizationBlockSize || null,
      quantization_scale_tensor_shape: tensor.quantizationScaleTensorShape || [],
      quantization_zero_point_tensor_shape: tensor.quantizationZeroPointTensorShape || [],
      quantization_cardinality_status: tensor.quantizationCardinalityStatus || "",
      quantization_cardinality_detail: tensor.quantizationCardinalityDetail || "",
      quantization_parameter_role: tensor.quantizationParameterRole || "",
      quantization_binding_status: tensor.quantizationBindingStatus || "not_bound",
      onnx_quantization_bindings: (tensor.onnxQuantizationBindings || []).map((item) => ({ op_index: item.op_index, op_name: item.op_name, role: item.role, status: item.status })),
      initializer_bytes: tensor.initializerBytes || 0,
    initializer_elements: tensor.initializerElements || 0,
    initializer_stored_elements: tensor.initializerStoredElements ?? tensor.initializerElements ?? 0,
    initializer_embedded_bytes: tensor.initializerEmbeddedBytes ?? 0,
    initializer_embedded_stored_elements: tensor.initializerEmbeddedStoredElements ?? 0,
    initializer_verified_external_bytes: tensor.initializerVerifiedExternalBytes ?? 0,
    initializer_verified_external_stored_elements: tensor.initializerVerifiedExternalStoredElements ?? 0,
    initializer_available_bytes: tensor.initializerAvailableBytes ?? tensor.initializerBytes ?? 0,
    initializer_available_stored_elements: tensor.initializerAvailableStoredElements ?? tensor.initializerElements ?? 0,
    buffer_data_length: ["initializer", "sparse_initializer"].includes(tensor.role)
      && (Number(tensor.externalDataEntries || 0) === 0 || tensor.externalPayloadVerified === true)
      ? Number(tensor.initializerAvailableBytes ?? tensor.initializerBytes ?? 0)
      : null,
    buffer_data_status: !["initializer", "sparse_initializer"].includes(tensor.role)
      ? "not_applicable"
      : Number(tensor.externalDataEntries || 0) === 0
        ? "observed_embedded_initializer_payload"
        : tensor.externalPayloadVerified === true
          ? "verified_external_initializer_payload"
          : "not_assessed_external_payload_unavailable",
    initializer_embedded_float_bytes: tensor.initializerEmbeddedFloatBytes ?? 0,
    initializer_projected_embedded_fp16_bytes: tensor.initializerProjectedEmbeddedFp16Bytes ?? 0,
    initializer_projected_embedded_int8_bytes: tensor.initializerProjectedEmbeddedInt8Bytes ?? 0,
    initializer_external_component_count: tensor.initializerExternalComponentCount ?? 0,
    initializer_verified_external_component_count: tensor.initializerVerifiedExternalComponentCount ?? 0,
    initializer_storage_kind: tensor.initializerStorageKind || (tensor.role === "initializer" ? "tensor_proto" : ""),
    initializer_raw_data_bytes: tensor.initializerRawDataBytes || 0,
    initializer_raw_zero_bytes: tensor.initializerRawZeroBytes || 0,
    initializer_typed_data_bytes: tensor.initializerTypedDataBytes || 0,
    external_data_entries: tensor.externalDataEntries || 0,
    external_data: (tensor.externalData || []).map((entry) => ({ key: entry.key, value: entry.value })),
    data_location: tensor.dataLocation || 0,
    external_payload_status: tensor.externalPayloadStatus || "not_applicable",
    external_payload_verified: tensor.externalPayloadVerified === true,
    external_embedded_payload_conflict: tensor.externalEmbeddedPayloadConflict === true,
    external_sidecar_path: tensor.externalSidecarPath || "",
    external_sidecar_bytes: tensor.externalSidecarBytes || 0,
    external_sidecar_sha256: tensor.externalSidecarSha256 || "",
    external_sidecar_sha1: tensor.externalSidecarSha1 || "",
    initializer_integer_values_status: tensor.initializerIntegerValuesStatus || "not_applicable_non_integer_initializer",
    initializer_integer_values_complete: tensor.initializerIntegerValuesComplete === true,
    initializer_integer_values: tensor.initializerIntegerValues || [],
    initializer_integer_values_exact_complete: tensor.initializerIntegerValuesExactComplete === true,
    initializer_integer_values_exact_decimals: tensor.initializerIntegerValuesExactDecimals || [],
    static_values_status: tensor.staticValuesStatus || "not_assessed",
    static_values_complete: tensor.staticValuesComplete === true,
    static_values: jsonSafeStaticValues(tensor.staticValues),
    static_values_canonical_text_complete: tensor.staticValuesCanonicalTextComplete === true,
    static_values_canonical_texts: [...(tensor.staticValuesCanonicalTexts || [])],
    static_values_negative_zero_count: negativeZeroIndices.length,
    static_values_negative_zero_indices: negativeZeroIndices,
    static_values_source: tensor.staticValuesSource || "",
    static_dimension_values_status: tensor.staticDimensionValuesStatus || "not_assessed",
    static_dimension_values_complete: tensor.staticDimensionValuesComplete === true,
    static_dimension_values: structuredClone(tensor.staticDimensionValues || []),
    static_dimension_values_source: tensor.staticDimensionValuesSource || "",
    sequence_length_status: tensor.sequenceLengthStatus || "not_applicable",
    sequence_length: tensor.sequenceLength ?? null,
    sequence_element_inventory_status: tensor.sequenceElementInventoryStatus || "not_applicable",
    sequence_element_type_count: Array.isArray(tensor.sequenceElementTypes) ? tensor.sequenceElementTypes.length : 0,
    sequence_element_types: Array.isArray(tensor.sequenceElementTypes) ? tensor.sequenceElementTypes : [],
    optional_presence_status: tensor.optionalPresenceStatus || "not_applicable",
    optional_presence: typeof tensor.optionalPresence === "boolean" ? tensor.optionalPresence : null,
    sparse_nnz: tensor.sparseNnz ?? null,
    sparse_index_encoding: tensor.sparseIndexEncoding || "",
    sparse_value_elements: tensor.sparseValueElements ?? null,
    sparse_index_elements: tensor.sparseIndexElements ?? null,
    constant_buffer: tensor.role === "initializer" || tensor.role === "sparse_initializer",
    role: tensor.role || "",
    };
  });
  const tensorIdByName = new Map(tensors.map((tensor) => [tensor.name, tensor.index]));

  const ops = graph.nodes.map((node, index) => buildOnnxOp(node, index, tensorMap, tensorIdByName));
  const graphTopology = applyGraphTopology(ops);
  const quantBindingsByOp = new Map();
  for (const binding of onnxQuantizationBinding.bindings) {
    const rows = quantBindingsByOp.get(binding.op_index) || [];
    rows.push(binding);
    quantBindingsByOp.set(binding.op_index, rows);
  }
  for (const op of ops) {
    const bindings = quantBindingsByOp.get(op.index) || [];
    op.onnx_quantization_binding_count = bindings.length;
    op.onnx_quantization_contract_status = bindings.some((item) => item.status === "fail")
      ? "fail"
      : bindings.some((item) => item.status.startsWith("not_assessed")) ? "not_assessed" : bindings.length ? "pass" : "not_applicable";
  }
  const histogram = countBy(ops.map((op) => op.name));
  const runtimeReviewWatchlist = countBy(ops.filter((op) => op.standard_domain && RUNTIME_REVIEW_OPS.has(op.name)).map((op) => op.name))
    .map((item) => {
      const [reasonCode, reason] = RUNTIME_REVIEW_OPS.get(item.name);
      return {
        ...item,
        reason_code: reasonCode,
        evidence_class: "OBSERVED",
        review_class: "HEURISTIC",
        reason,
      };
    });
  const exactMacTotals = summarizeOnnxAssessedMacs(ops.filter((op) => op.macs_status === "assessed").map((op) => op.macs_decimal ?? op.macs));
  const computeOps = ops.filter((op) => isOnnxMacBearingOperation(op.name, op.standard_domain));
  const assessedComputeOps = computeOps.filter((op) => op.macs_status === "assessed");
  const unassessedComputeOps = computeOps.filter((op) => op.macs_status === "not_assessed");
  const completeMacTotals = projectOnnxCompleteMacTotals(exactMacTotals, unassessedComputeOps.length);
  const totalMacs = completeMacTotals.total_macs;
  const totalOps = completeMacTotals.total_ops;
  const algorithmDependentArithmeticOps = ops.filter((op) => isOnnxAlgorithmDependentArithmetic(op.name, op.standard_domain));
  const macAssessment = {
    status: unassessedComputeOps.length === 0 ? "assessed" : assessedComputeOps.length > 0 ? "partially_assessed" : "not_assessed",
    ...exactMacTotals,
    compute_ops: computeOps.length,
    assessed_compute_ops: assessedComputeOps.length,
    not_assessed_compute_ops: unassessedComputeOps.length,
    not_assessed: unassessedComputeOps.map((op) => ({ index: op.index, name: op.name, reason: op.macs_reason })),
    algorithm_dependent_arithmetic_ops: algorithmDependentArithmeticOps.length,
    algorithm_dependent_arithmetic: algorithmDependentArithmeticOps.map((op) => ({
      index: op.index,
      name: op.name,
      reason: "The ONNX operator contract fixes the mathematical transform but does not select a direct, FFT, or backend-specific implementation, so no implementation-independent MAC count exists.",
    })),
    metric_scope: "nominal tensor-contraction MACs",
    detail: `${assessedComputeOps.length}/${computeOps.length} nominal tensor-contraction op(s) were deterministically assessed from compatible tensor shapes. The total excludes ${unassessedComputeOps.length} unassessed contraction op(s) and separately inventories ${algorithmDependentArithmeticOps.length} algorithm-dependent transform op(s).`,
  };
  const tensorTypes = countBy(tensors.map((tensor) => tensor.dtype || "UNKNOWN"));
  const quantizedTensors = countQuantizedTensors(tensors);
  const perChannelTensors = countPerChannelQuantizers(tensors);
  const inputs = selectModelInputs(graph.inputs, initializerNames, tensors, tensorIdByName);
  const outputs = graph.outputs.map((output) => tensors[tensorIdByName.get(output.name)]).filter(Boolean);
  const inputContracts = buildOnnxInputContracts(inputs, ops);
  const dynamicShapeCostContract = buildOnnxDynamicShapeCostContract(tensors, ops);
  for (const op of ops) {
    op.mac_percent = op.macs_status === "assessed"
      ? exactNonnegativeRatio(op.macs_decimal ?? op.macs, exactMacTotals.total_assessed_macs_decimal)
      : null;
    op.mac_percent_basis = "share of assessed compute MACs";
    op.row_working_set_ratio = op.row_working_set_bytes != null && Number(applicableTargetProfile?.l1_data_bytes || 0) > 0
      ? op.row_working_set_bytes / applicableTargetProfile.l1_data_bytes : null;
    op.row_working_set_ratio_status = op.row_working_set_status !== "assessed"
      ? op.row_working_set_status : Number(applicableTargetProfile?.l1_data_bytes || 0) > 0 ? "assessed" : "not_assessed_target_l1_unavailable";
  }
  const quantizationStatus = classifyOnnxQuantization(ops, tensors, inputs, outputs, quantizedTensors);
  const stages = buildStages(ops);
  const rooflineCsv = buildRooflineCsv(ops);
  const stageMermaid = buildStageMermaid(stages);
  const onnxExternalData = buildOnnxExternalDataEvidence(tensorPayloadScopes, externalDataResolution);
  const initializerAnalysis = analyzeOnnxInitializers(graph, tensorMap, onnxSparseTensorContract);
  const sizeBreakdown = buildOnnxSizeBreakdown(bytes, graph, tensors, initializerAnalysis);
  const tensorLiveness = computeOnnxTensorLiveness(ops, tensors, inputs, outputs);
  const metadataPresence = buildOnnxMetadataPresence(model, graph);
  const weightIntegrity = buildOnnxWeightIntegrity(graph, tensors, ops, initializerAnalysis);
  const suppressedSections = [
    "Execution-provider assignment prediction",
    "Provider partition-break cost model",
    "ONNX Runtime minimum-version derivation",
    "Machine-verifiable preprocessing semantics (ONNX ModelProto metadata_props are untyped key/value declarations)",
    "weight-packing warmup estimates until ONNX Runtime EP behavior is modeled",
  ];

  return {
    format: "onnx",
    filename,
    file_size: bytes.byteLength,
    target_profile: applicableTargetProfile,
    version: model.irVersion || 0,
    subgraphs: 1,
    graph_name: graph.name || "",
    producer: model.producer || "",
    onnx_ir_version: model.irVersion || 0,
    opsets: model.opsets || [],
    onnx_domain_analysis: onnxDomainAnalysis,
    onnx_shape_inference: shapeInference,
    onnx_type_proto_contract: onnxTypeProtoContract,
    onnx_sparse_tensor_contract: onnxSparseTensorContract,
    onnx_tensor_data_type_contract: buildOnnxTensorDataTypeContract(),
    onnx_sections_suppressed: suppressedSections,
    operator_count: ops.length,
    tensor_count: tensors.length,
    onnx_external_data_tensor_count: onnxExternalData.tensor_count,
    onnx_external_data: onnxExternalData,
    tensor_types: tensorTypes,
    quantized_tensors: quantizedTensors,
    per_channel_tensors: perChannelTensors,
    quantization_status: quantizationStatus,
    onnx_quantization_binding: onnxQuantizationBinding,
    total_macs: totalMacs,
    total_macs_decimal: completeMacTotals.total_macs_decimal,
    total_ops: totalOps,
    total_ops_decimal: completeMacTotals.total_ops_decimal,
    mac_assessment: macAssessment,
    inputs,
    outputs,
    input_tensor_indices: inputs.map((tensor) => tensor.index),
    output_tensor_indices: outputs.map((tensor) => tensor.index),
    input_contracts: inputContracts,
    dynamic_shape_cost_contract: dynamicShapeCostContract,
    graph_topology: graphTopology,
    tensors,
    ops,
    histogram,
    runtime_review_watchlist: runtimeReviewWatchlist,
    stages,
    roofline_csv: rooflineCsv,
    stage_mermaid: stageMermaid,
    size_breakdown: sizeBreakdown,
    tensor_liveness: tensorLiveness,
    weight_integrity: weightIntegrity,
    metadata_presence: metadataPresence,
    runtime_compat: {
      min_runtime_version: "",
      derived_min_runtime_version: "",
      effective_min_runtime_version: "",
      max_op_version: Math.max(0, ...model.opsets.map((opset) => Number(opset.version || 0))),
      version_driving_ops: model.opsets.map((opset) => `${opset.domain || "ai.onnx"}:${opset.version || "unknown"}`),
      runtime_version_basis: "ONNX IR/opset observed; ONNX Runtime minimum-version matrix is not bundled in this analyzer build.",
      detail: "Use pinned ONNX Runtime release notes or session creation on the target runtime to confirm support.",
    },
    ort_compatibility_assessment_status: "not_loaded",
    ort_compatibility_evidence_schema: "",
    ort_compatibility_evidence_access: "research_authorization_required",
    xnnpack_assumption: "NOT_ASSESSABLE_ONNX: Actual ONNX Runtime execution-provider assignment requires a selected runtime/build/device and profiling/session evidence; source registration candidates are assessed separately when the protected ORT rulepack is loaded.",
    xnnpack_chains: [],
    xnnpack_chain_breaks: 0,
    xnnpack_effective_chain_breaks: 0,
    xnnpack_structural_chain_breaks: 0,
    xnnpack_zero_mac_chain_breaks: 0,
    delegated_mac_percent: null,
    fallback_byte_percent: null,
    conv_packing_warn_ops: 0,
    fc_packing_warn_ops: 0,
    markdown: buildMarkdown(filename, model, graph, ops, tensors, inputs, totalMacs, quantizedTensors, quantizationStatus, macAssessment),
  };
}

function normalizeOnnxTargetProfile(targetProfile) {
  if (!targetProfile || typeof targetProfile !== "object") return null;
  return {
    id: targetProfile.id || "",
    label: targetProfile.label || "",
    profile_sha256: targetProfile.profile_sha256 || "",
    l1_data_bytes: Number(targetProfile.l1_data_bytes || 0),
    applicability: "Static L1 working-set ratio reference only; TFLite delegation, packing, channel-tail, throughput, bandwidth, ridge, and XNNPACK selector assumptions are not applicable.",
  };
}

function buildOnnxInputContracts(inputs, ops) {
  return (inputs || []).map((tensor) => {
    const shape = Array.isArray(tensor.shape) ? [...tensor.shape] : [];
    const semanticConsumer = shape.length === 4
      ? (ops || []).find((op) => op.standard_domain === true
        && op.inputs?.[0] === tensor.index
        && ONNX_NCHW_INPUT_SEMANTIC_OPS.has(op.name))
      : null;
    const layout = semanticConsumer ? "NCHW" : null;
    const channelAxis = semanticConsumer ? 1 : null;
    const channels = channelAxis == null ? null : Number.isInteger(shape[channelAxis]) ? shape[channelAxis] : null;
    const layoutStatus = shape.length !== 4
      ? "not_applicable_non_4d_input"
      : semanticConsumer ? "derived_nchw_from_direct_consumer_semantics"
        : "not_assessed_no_direct_layout_semantic_consumer";
    const layoutEvidenceClass = shape.length !== 4 ? "NOT_APPLICABLE" : semanticConsumer ? "DERIVED" : "NOT_ASSESSABLE";
    const layoutReason = shape.length !== 4
      ? `Rank ${shape.length} input does not carry a four-dimensional image-layout contract.`
      : semanticConsumer
        ? `Graph input T${tensor.index} is activation input 0 of ai.onnx ${semanticConsumer.name} #${String(semanticConsumer.index).padStart(3, "0")}; the pinned operator tensor semantics use N,C,spatial dimensions.`
        : "Rank alone does not determine layout, and no supported direct standard-domain activation-input consumer fixes the channel axis.";
    const range = inputScalarQuantizedRange(tensor);
    const risks = [];
    if (tensor.dtype === "FLOAT32" && !range.is_quantized) {
      risks.push("FLOAT32 normalization range is unknown and must match the training pipeline exactly.");
    }
    if (!layout && shape.length === 4) {
      risks.push("Input layout and channel axis are not determined by the supported direct-consumer semantics.");
    }
    if (channels === 3) {
      risks.push("Channel order (RGB vs BGR) cannot be determined from graph structure; a mismatch may silently degrade task performance.");
    }
    return {
      schema: "deepbom.input_tensor_contract.v1",
      tensor_index: tensor.index,
      name: tensor.name || "",
      shape,
      dtype: tensor.dtype || "UNKNOWN",
      is_quantized: range.is_quantized,
      expected_range_low: range.low,
      expected_range_high: range.high,
      range_note: range.note,
      tensor_numerical_contract_status: range.status,
      source_data_to_tensor_preprocessing_status: "not_embedded_in_artifact",
      layout,
      layout_status: layoutStatus,
      layout_evidence_class: layoutEvidenceClass,
      layout_source_op_index: semanticConsumer?.index ?? null,
      layout_source_op_name: semanticConsumer?.name ?? null,
      layout_reason: layoutReason,
      channel_axis: channelAxis,
      channels,
      risks,
    };
  });
}

function inputScalarQuantizedRange(tensor) {
  const scaleCount = Number(tensor.quant_scales || 0);
  const zeroPointCount = Number(tensor.quant_zero_points || 0);
  const isQuantized = scaleCount > 0;
  if (!isQuantized) return {
    is_quantized: false,
    low: null,
    high: null,
    status: "not_embedded_in_artifact",
    note: `${tensor.dtype || "UNKNOWN"} input has no artifact-bound scalar real range`,
  };
  const bounds = inputQuantizedCodeRange(tensor.dtype);
  const scale = tensor.scale_sample?.[0];
  const zeroPoint = tensor.zero_point_sample?.[0];
  if (scaleCount !== 1 || zeroPointCount !== 1) return {
    is_quantized: true,
    low: null,
    high: null,
    status: "not_assessed_non_scalar_input_quantization",
    note: `${tensor.dtype || "UNKNOWN"} input declares ${scaleCount} scale(s) and ${zeroPointCount} zero point(s); no scalar real range is emitted`,
  };
  if (!bounds) return {
    is_quantized: true,
    low: null,
    high: null,
    status: "not_assessed_unsupported_quantized_input_dtype",
    note: `${tensor.dtype || "UNKNOWN"} is not in the bounded integer input-range table`,
  };
  if (!Number.isFinite(scale) || scale <= 0 || !Number.isSafeInteger(zeroPoint)
    || zeroPoint < bounds[0] || zeroPoint > bounds[1]) return {
    is_quantized: true,
    low: null,
    high: null,
    status: "invalid_or_incomplete_quantization_metadata",
    note: `${tensor.dtype || "UNKNOWN"} scalar scale or zero point is missing, non-finite, non-positive, or outside the dtype code domain`,
  };
  const low = scale * (bounds[0] - zeroPoint);
  const high = scale * (bounds[1] - zeroPoint);
  return {
    is_quantized: true,
    low,
    high,
    status: "known_from_artifact_quantization_metadata",
    note: `${tensor.dtype} dequantized code domain [${low.toPrecision(9)}, ${high.toPrecision(9)}] from scalar scale ${scale.toPrecision(9)} and zero point ${zeroPoint}`,
  };
}

function inputQuantizedCodeRange(dtype) {
  return ({
    UINT8: [0, 255], INT8: [-128, 127], UINT16: [0, 65_535], INT16: [-32_768, 32_767],
    INT32: [-2_147_483_648, 2_147_483_647], INT4: [-8, 7], UINT4: [0, 15], UINT2: [0, 3], INT2: [-2, 1],
  })[String(dtype || "").toUpperCase()] || null;
}

function buildOnnxTensorDataTypeContract() {
  const types = Object.entries(TENSOR_TYPES)
    .map(([id, type]) => ({ id: Number(id), name: type.name, storage_bits: type.bits }))
    .sort((left, right) => left.id - right.id);
  return {
    schema: "deepbom.onnx_tensor_data_type_contract.v1",
    status: "complete_for_pinned_onnx_release",
    evidence_class: "SOURCE_PINNED_AND_IMPLEMENTATION_TESTED",
    source_release: ONNX_TENSOR_TYPE_SOURCE.release,
    source_commit: ONNX_TENSOR_TYPE_SOURCE.commit,
    source_ref: ONNX_TENSOR_TYPE_SOURCE.ref,
    source_sha256: ONNX_TENSOR_TYPE_SOURCE.sha256,
    concrete_data_type_count: types.filter((type) => type.id > 0).length,
    fixed_width_numeric_data_type_count: types.filter((type) => type.id > 0 && type.name !== "STRING" && type.storage_bits > 0).length,
    packed_data_type_count: types.filter((type) => type.storage_bits > 0 && type.storage_bits < 8).length,
    raw_numeric_decoder_count: types.filter((type) => type.id > 0 && type.name !== "STRING" && type.storage_bits > 0).length,
    typed_numeric_decoder_count: types.filter((type) => type.id > 0 && type.name !== "STRING" && type.storage_bits > 0).length,
    packed_data_types: types.filter((type) => type.storage_bits > 0 && type.storage_bits < 8).map((type) => type.name),
    packing_rule: "UINT4/INT4/FLOAT4E2M1 use two LSB-first elements per byte; UINT2/INT2 use four LSB-first elements per byte; payload bytes = ceil(element_count*storage_bits/8).",
    numerical_integrity_projection: "Real numeric values are decoded exactly from raw_data and typed TensorProto fields. COMPLEX64/COMPLEX128 integrity metrics use each complex element magnitude while payload identity remains byte/value exact.",
    types,
  };
}

function parseOnnxModel(bytes) {
  const model = {
    graph: null,
    irVersion: 0,
    producer: "",
    producerVersion: "",
    domain: "",
    modelVersion: 0,
    docString: "",
    metadataProps: [],
    opsets: [],
    functions: [],
  };
  for (const field of readFields(bytes)) {
    if (field.no === 1 && field.wire === 0) model.irVersion = toSafeNumber(field.value);
    if (field.no === 2 && field.wire === 2) model.producer = decodeString(field.bytes);
    if (field.no === 3 && field.wire === 2) model.producerVersion = decodeString(field.bytes);
    if (field.no === 4 && field.wire === 2) model.domain = decodeString(field.bytes);
    if (field.no === 5 && field.wire === 0) model.modelVersion = toSafeSignedNumber(field.value, 64);
    if (field.no === 6 && field.wire === 2) model.docString = decodeString(field.bytes);
    if (field.no === 7 && field.wire === 2) model.graph = parseGraph(field.bytes);
    if (field.no === 8 && field.wire === 2) model.opsets.push(parseOpset(field.bytes));
    if (field.no === 14 && field.wire === 2) model.metadataProps.push(parseStringStringEntry(field.bytes));
    if (field.no === 25 && field.wire === 2) model.functions.push(parseFunction(field.bytes));
  }
  return model;
}

function parseGraph(bytes) {
  const graph = { name: "", docString: "", nodes: [], inputs: [], outputs: [], valueInfo: [], initializers: [], sparseInitializers: [], quantizationAnnotations: [], metadataProps: [] };
  for (const field of readFields(bytes)) {
    if (field.no === 1 && field.wire === 2) graph.nodes.push(parseNode(field.bytes));
    if (field.no === 2 && field.wire === 2) graph.name = decodeString(field.bytes);
    if (field.no === 5 && field.wire === 2) graph.initializers.push(parseTensor(field.bytes));
    if (field.no === 10 && field.wire === 2) graph.docString = decodeString(field.bytes);
    if (field.no === 11 && field.wire === 2) graph.inputs.push(parseValueInfo(field.bytes));
    if (field.no === 12 && field.wire === 2) graph.outputs.push(parseValueInfo(field.bytes));
    if (field.no === 13 && field.wire === 2) graph.valueInfo.push(parseValueInfo(field.bytes));
    if (field.no === 14 && field.wire === 2) graph.quantizationAnnotations.push(parseTensorAnnotation(field.bytes));
    if (field.no === 15 && field.wire === 2) graph.sparseInitializers.push(parseSparseTensor(field.bytes));
    if (field.no === 16 && field.wire === 2) graph.metadataProps.push(parseStringStringEntry(field.bytes));
  }
  for (const tensor of graph.initializers) annotateParsedInitializerValues(tensor);
  for (const sparse of graph.sparseInitializers) annotateParsedSparseTensorValues(sparse, "sparse_initializer");
  return graph;
}

function parseTensorAnnotation(bytes) {
  const annotation = { tensorName: "", quantParameterTensorNames: [] };
  for (const field of readFields(bytes)) {
    if (field.no === 1 && field.wire === 2) annotation.tensorName = decodeString(field.bytes);
    if (field.no === 2 && field.wire === 2) annotation.quantParameterTensorNames.push(parseStringStringEntry(field.bytes));
  }
  return annotation;
}

function annotateParsedInitializerValues(tensor) {
  const payloadAvailable = !isExternalInitializer(tensor);
  annotateParsedTensorValues(tensor, payloadAvailable, "initializer");
}

function annotateParsedTensorValues(tensor, payloadAvailable, source) {
  const integers = summarizeConditionIntegerInitializer(tensor, payloadAvailable);
  const values = summarizeStaticTensor(tensor, payloadAvailable);
  tensor.initializerIntegerValuesStatus = integers.status;
  tensor.initializerIntegerValuesComplete = integers.complete;
  tensor.initializerIntegerValues = integers.values;
  tensor.initializerIntegerValuesExactComplete = integers.exactComplete;
  tensor.initializerIntegerValuesExactDecimals = integers.exactDecimals;
  tensor.staticValuesStatus = values.status;
  tensor.staticValuesComplete = values.complete;
  tensor.staticValues = values.values;
  tensor.staticValuesCanonicalTextComplete = values.canonicalTextComplete === true;
  tensor.staticValuesCanonicalTexts = values.canonicalTexts || [];
  tensor.staticValuesSource = source;
}

function annotateParsedSparseTensorValues(sparse, source) {
  if (sparse?.values) annotateParsedTensorValues(sparse.values, !isExternalInitializer(sparse.values), `${source}_values`);
  if (sparse?.indices) annotateParsedTensorValues(sparse.indices, !isExternalInitializer(sparse.indices), `${source}_indices`);
}

function parseStringStringEntry(bytes) {
  const entry = { key: "", value: "" };
  for (const field of readFields(bytes)) {
    if (field.no === 1 && field.wire === 2) entry.key = decodeString(field.bytes);
    if (field.no === 2 && field.wire === 2) entry.value = decodeString(field.bytes);
  }
  return entry;
}

function parseNode(bytes) {
  const node = { name: "", opType: "", domain: "", overload: "", inputs: [], outputs: [], attributes: new Map(), duplicateAttributeNames: [] };
  for (const field of readFields(bytes)) {
    if (field.no === 1 && field.wire === 2) node.inputs.push(decodeString(field.bytes));
    if (field.no === 2 && field.wire === 2) node.outputs.push(decodeString(field.bytes));
    if (field.no === 3 && field.wire === 2) node.name = decodeString(field.bytes);
    if (field.no === 4 && field.wire === 2) node.opType = decodeString(field.bytes);
    if (field.no === 5 && field.wire === 2) {
      const attr = parseAttribute(field.bytes);
      if (attr.name) {
        if (node.attributes.has(attr.name)) node.duplicateAttributeNames.push(attr.name);
        node.attributes.set(attr.name, attr);
      }
    }
    if (field.no === 7 && field.wire === 2) node.domain = decodeString(field.bytes);
    if (field.no === 8 && field.wire === 2) node.overload = decodeString(field.bytes);
  }
  return node;
}

function parseAttribute(bytes) {
  const attr = {
    name: "", ints: [], intExactDecimals: [], floats: [], strings: [], i: null, iExactDecimal: "",
    f: null, s: null, type: 0, tensor: null, tensors: [], graph: null, graphs: [],
    sparseTensor: null, sparseTensors: [], typeProto: null, typeProtos: [],
    refAttrName: "", valueTypesPresent: [],
  };
  for (const field of readFields(bytes)) {
    if (field.no === 1 && field.wire === 2) attr.name = decodeString(field.bytes);
    if (field.no === 2 && field.wire === 5) { attr.f = field.float32; markAttributeValueType(attr, 1); }
    if (field.no === 3 && field.wire === 0) {
      const exact = signedVarint(field.value, 64);
      attr.iExactDecimal = exact.toString();
      attr.i = safeBigIntNumber(exact);
      markAttributeValueType(attr, 2);
    }
    if (field.no === 4 && field.wire === 2) { attr.s = decodeString(field.bytes); markAttributeValueType(attr, 3); }
    if (field.no === 5 && field.wire === 2) { attr.tensor = parseTensor(field.bytes); markAttributeValueType(attr, 4); }
    if (field.no === 6 && field.wire === 2) { attr.graph = parseGraph(field.bytes); markAttributeValueType(attr, 5); }
    if (field.no === 7 && field.wire === 5) { attr.floats.push(field.float32); markAttributeValueType(attr, 6); }
    if (field.no === 7 && field.wire === 2) { attr.floats.push(...readPackedFloat32(field.bytes)); markAttributeValueType(attr, 6); }
    if (field.no === 8 && field.wire === 0) { appendExactAttributeInt(attr, field.value); markAttributeValueType(attr, 7); }
    if (field.no === 8 && field.wire === 2) { readPackedVarints(field.bytes).forEach((value) => appendExactAttributeInt(attr, value)); markAttributeValueType(attr, 7); }
    if (field.no === 9 && field.wire === 2) { attr.strings.push(decodeString(field.bytes)); markAttributeValueType(attr, 8); }
    if (field.no === 10 && field.wire === 2) { attr.tensors.push(parseTensor(field.bytes)); markAttributeValueType(attr, 9); }
    if (field.no === 11 && field.wire === 2) { attr.graphs.push(parseGraph(field.bytes)); markAttributeValueType(attr, 10); }
    if (field.no === 14 && field.wire === 2) { attr.typeProto = parseType(field.bytes); markAttributeValueType(attr, 13); }
    if (field.no === 15 && field.wire === 2) { attr.typeProtos.push(parseType(field.bytes)); markAttributeValueType(attr, 14); }
    if (field.no === 20 && field.wire === 0) attr.type = toSafeNumber(field.value);
    if (field.no === 21 && field.wire === 2) attr.refAttrName = decodeString(field.bytes);
    if (field.no === 22 && field.wire === 2) { attr.sparseTensor = parseSparseTensor(field.bytes); markAttributeValueType(attr, 11); }
    if (field.no === 23 && field.wire === 2) { attr.sparseTensors.push(parseSparseTensor(field.bytes)); markAttributeValueType(attr, 12); }
  }
  for (const tensor of [attr.tensor, ...attr.tensors].filter(Boolean)) {
    annotateParsedTensorValues(tensor, !isExternalInitializer(tensor), "attribute_tensor");
  }
  for (const sparse of [attr.sparseTensor, ...attr.sparseTensors].filter(Boolean)) annotateParsedSparseTensorValues(sparse, "attribute_sparse_tensor");
  return attr;
}

function markAttributeValueType(attribute, type) {
  if (!attribute.valueTypesPresent.includes(type)) attribute.valueTypesPresent.push(type);
}

function appendExactAttributeInt(attribute, value) {
  const exact = signedVarint(value, 64);
  attribute.intExactDecimals.push(exact.toString());
  attribute.ints.push(safeBigIntNumber(exact));
}

function parseFunction(bytes) {
  const fn = {
    name: "", domain: "", overload: "", docString: "", inputs: [], outputs: [], attributes: [],
    attributeProtos: [], valueInfo: [], metadataProps: [], nodes: [], opsets: [],
  };
  for (const field of readFields(bytes)) {
    if (field.no === 1 && field.wire === 2) fn.name = decodeString(field.bytes);
    if (field.no === 4 && field.wire === 2) fn.inputs.push(decodeString(field.bytes));
    if (field.no === 5 && field.wire === 2) fn.outputs.push(decodeString(field.bytes));
    if (field.no === 6 && field.wire === 2) fn.attributes.push(decodeString(field.bytes));
    if (field.no === 7 && field.wire === 2) fn.nodes.push(parseNode(field.bytes));
    if (field.no === 8 && field.wire === 2) fn.docString = decodeString(field.bytes);
    if (field.no === 9 && field.wire === 2) fn.opsets.push(parseOpset(field.bytes));
    if (field.no === 10 && field.wire === 2) fn.domain = decodeString(field.bytes);
    if (field.no === 11 && field.wire === 2) fn.attributeProtos.push(parseAttribute(field.bytes));
    if (field.no === 12 && field.wire === 2) fn.valueInfo.push(parseValueInfo(field.bytes));
    if (field.no === 13 && field.wire === 2) fn.overload = decodeString(field.bytes);
    if (field.no === 14 && field.wire === 2) fn.metadataProps.push(parseStringStringEntry(field.bytes));
  }
  return fn;
}

function parseTensor(bytes) {
  const tensor = {
    name: "",
    dtype: "UNKNOWN",
    shape: [],
    shapeDeclared: true,
    rawDataBytes: 0,
    typedDataBytes: 0,
    typedElementCount: 0,
    externalDataEntries: 0,
    externalData: [],
    dataLocation: 0,
    rawData: null,
    typedValues: [],
    stringValues: [],
    stringValueCount: 0,
    typedDecodeError: "",
  };
  for (const field of readFields(bytes)) {
    if (field.no === 1 && field.wire === 0) tensor.shape.push(toSafeNumber(field.value));
    if (field.no === 1 && field.wire === 2) tensor.shape.push(...readPackedVarints(field.bytes).map(toSafeNumber));
    if (field.no === 2 && field.wire === 0) tensor.dtype = tensorTypeName(toSafeNumber(field.value));
    if (field.no === 4 && field.wire === 5) {
      addTypedDataBytes(tensor, 1, 4);
      tensor.typedValues.push(field.float32);
    }
    if (field.no === 4 && field.wire === 2) {
      const values = readPackedFloat32(field.bytes);
      addTypedDataBytes(tensor, values.length, 4);
      appendValues(tensor.typedValues, values);
    }
    if (field.no === 5 && field.wire === 0) {
      addTypedDataBytes(tensor, 1, 4);
      tensor.typedValues.push(Number(signedVarint(field.value, 32)));
    }
    if (field.no === 5 && field.wire === 2) {
      const values = readPackedVarints(field.bytes).map((value) => Number(signedVarint(value, 32)));
      addTypedDataBytes(tensor, values.length, 4);
      appendValues(tensor.typedValues, values);
    }
    if (field.no === 6 && field.wire === 2) {
      addTypedDataBytes(tensor, 1, field.bytes.byteLength);
      tensor.stringValues.push(decodeString(field.bytes));
      tensor.stringValueCount += 1;
    }
    if (field.no === 7 && field.wire === 0) {
      addTypedDataBytes(tensor, 1, 8);
      tensor.typedValues.push(signedVarint(field.value, 64));
    }
    if (field.no === 7 && field.wire === 2) {
      const values = readPackedVarints(field.bytes).map((value) => signedVarint(value, 64));
      addTypedDataBytes(tensor, values.length, 8);
      appendValues(tensor.typedValues, values);
    }
    if (field.no === 8 && field.wire === 2) tensor.name = decodeString(field.bytes);
    if (field.no === 9 && field.wire === 2) {
      tensor.rawDataBytes = field.bytes.byteLength;
      tensor.rawData = field.bytes;
    }
    if (field.no === 10 && field.wire === 1) {
      addTypedDataBytes(tensor, 1, 8);
      tensor.typedValues.push(new DataView(field.bytes.buffer, field.bytes.byteOffset, 8).getFloat64(0, true));
    }
    if (field.no === 10 && field.wire === 2) {
      const values = readPackedFloat64(field.bytes);
      addTypedDataBytes(tensor, values.length, 8);
      appendValues(tensor.typedValues, values);
    }
    if (field.no === 11 && field.wire === 0) {
      addTypedDataBytes(tensor, 1, 8);
      tensor.typedValues.push(field.value);
    }
    if (field.no === 11 && field.wire === 2) {
      const values = readPackedVarints(field.bytes);
      addTypedDataBytes(tensor, values.length, 8);
      appendValues(tensor.typedValues, values);
    }
    if (field.no === 13 && field.wire === 2) tensor.externalData.push(parseStringStringEntry(field.bytes));
    if (field.no === 14 && field.wire === 0) tensor.dataLocation = toSafeNumber(field.value);
  }
  finalizeTypedTensorPayload(tensor);
  tensor.externalDataEntries = tensor.externalData.length;
  tensor.storedDataBytes = tensor.rawDataBytes || tensor.typedDataBytes;
  return tensor;
}

function parseSparseTensor(bytes) {
  const sparse = { values: null, indices: null, dims: [] };
  for (const field of readFields(bytes)) {
    if (field.no === 1 && field.wire === 2) sparse.values = parseTensor(field.bytes);
    if (field.no === 2 && field.wire === 2) sparse.indices = parseTensor(field.bytes);
    if (field.no === 3 && field.wire === 0) sparse.dims.push(toSafeNumber(field.value));
    if (field.no === 3 && field.wire === 2) sparse.dims.push(...readPackedVarints(field.bytes).map(toSafeNumber));
  }
  return sparse;
}

function finalizeTypedTensorPayload(tensor) {
  if (tensor.rawData || tensor.stringValueCount > 0 || !tensor.typedValues.length) return;
  const expected = shapeElementCount(tensor.shape, true);
  const trimPackedPadding = (values, valuesPerSlot) => {
    if (expected > 0 && values.length >= expected && values.length - expected < valuesPerSlot) return values.slice(0, expected);
    return values;
  };
  if (["UINT4", "INT4", "FLOAT4E2M1"].includes(tensor.dtype)) {
    const unpacked = [];
    for (const raw of tensor.typedValues) {
      const packed = Number(raw) & 0xff;
      unpacked.push(decodePackedCode(packed & 0x0f, tensor.dtype));
      unpacked.push(decodePackedCode((packed >> 4) & 0x0f, tensor.dtype));
    }
    tensor.typedValues = trimPackedPadding(unpacked, 2);
  } else if (["UINT2", "INT2"].includes(tensor.dtype)) {
    const unpacked = [];
    for (const raw of tensor.typedValues) {
      const packed = Number(raw) & 0xff;
      for (let shift = 0; shift < 8; shift += 2) unpacked.push(decodePackedCode((packed >> shift) & 0x03, tensor.dtype));
    }
    tensor.typedValues = trimPackedPadding(unpacked, 4);
  } else if (["FLOAT16", "BFLOAT16", "FLOAT8E4M3FN", "FLOAT8E4M3FNUZ", "FLOAT8E5M2", "FLOAT8E5M2FNUZ", "FLOAT8E8M0"].includes(tensor.dtype)) {
    tensor.typedValues = tensor.typedValues.map((raw) => decodeTypedBitPattern(Number(raw), tensor.dtype));
  } else if (["COMPLEX64", "COMPLEX128"].includes(tensor.dtype)) {
    if (tensor.typedValues.length % 2 !== 0) tensor.typedDecodeError = `${tensor.dtype} typed payload has an odd component count.`;
    const complex = [];
    for (let index = 0; index + 1 < tensor.typedValues.length; index += 2) {
      complex.push(Math.hypot(Number(tensor.typedValues[index]), Number(tensor.typedValues[index + 1])));
    }
    tensor.typedValues = complex;
  }
  tensor.typedElementCount = tensor.typedValues.length;
  const logicalBytes = dtypePayloadBytes(tensor.dtype, tensor.typedElementCount);
  if (logicalBytes != null) tensor.typedDataBytes = logicalBytes;
}

function addTypedDataBytes(tensor, count, bytesPerElement) {
  const n = Math.max(0, Number(count || 0));
  tensor.typedElementCount += n;
  tensor.typedDataBytes += n * Math.max(0, Number(bytesPerElement || 0));
}

function appendValues(target, values) {
  for (const value of values) target.push(value);
}

function validateOnnxGraphLimits(graph, functions = [], modelMetadata = []) {
  const graphScopes = collectGraphScopeEntries(graph, functions);
  const totalNodeCount = graphScopes.reduce((sumNodes, entry) => sumNodes + (entry.graph.nodes || []).length, 0)
    + functions.reduce((sumNodes, fn) => sumNodes + (fn.nodes || []).length, 0);
  if (totalNodeCount > MAX_ONNX_NODES) throw new Error(`ONNX graph/function node count ${totalNodeCount} exceeds safety limit ${MAX_ONNX_NODES}.`);
  if (functions.length > MAX_ONNX_FUNCTIONS) throw new Error(`ONNX local function count ${functions.length} exceeds safety limit ${MAX_ONNX_FUNCTIONS}.`);
  const graphs = graphScopes.map((entry) => entry.graph);
  const quantizationAnnotations = graphs.flatMap((scope) => scope.quantizationAnnotations || []);
  const quantizationAnnotationEntryCount = quantizationAnnotations.reduce((sum, annotation) => sum + (annotation.quantParameterTensorNames || []).length, 0);
  if (quantizationAnnotations.length > MAX_ONNX_QUANTIZATION_ANNOTATIONS) throw new Error(`ONNX quantization annotation count ${quantizationAnnotations.length} exceeds safety limit ${MAX_ONNX_QUANTIZATION_ANNOTATIONS}.`);
  if (quantizationAnnotationEntryCount > MAX_ONNX_QUANTIZATION_ANNOTATION_ENTRIES) throw new Error(`ONNX quantization annotation parameter-entry count ${quantizationAnnotationEntryCount} exceeds safety limit ${MAX_ONNX_QUANTIZATION_ANNOTATION_ENTRIES}.`);
  const graphInitializers = graphs.flatMap((scope) => scope.initializers || []);
  const graphSparseInitializers = graphs.flatMap((scope) => scope.sparseInitializers || []);
  if (graphInitializers.length + graphSparseInitializers.length > MAX_ONNX_INITIALIZERS) throw new Error(`ONNX dense+sparse initializer count ${graphInitializers.length + graphSparseInitializers.length} exceeds safety limit ${MAX_ONNX_INITIALIZERS}.`);
  const directNodes = [
    ...graphs.flatMap((scope) => scope.nodes || []),
    ...functions.flatMap((fn) => fn.nodes || []),
  ];
  const attributeTensors = directNodes.flatMap((node) => [...(node.attributes?.values?.() || [])]
    .flatMap((attribute) => [attribute.tensor, ...(attribute.tensors || [])].filter(Boolean)));
  const attributeSparseTensors = directNodes.flatMap((node) => [...(node.attributes?.values?.() || [])]
    .flatMap((attribute) => [attribute.sparseTensor, ...(attribute.sparseTensors || [])].filter(Boolean)));
  const functionDefaultTensors = functions.flatMap((fn) => (fn.attributeProtos || [])
    .flatMap((attribute) => [attribute.tensor, ...(attribute.tensors || [])].filter(Boolean)));
  const functionDefaultSparseTensors = functions.flatMap((fn) => (fn.attributeProtos || [])
    .flatMap((attribute) => [attribute.sparseTensor, ...(attribute.sparseTensors || [])].filter(Boolean)));
  const sparseTensors = [...graphSparseInitializers, ...attributeSparseTensors, ...functionDefaultSparseTensors];
  const sparsePayloadTensors = sparseTensors.flatMap((sparse) => [sparse.values, sparse.indices].filter(Boolean));
  const payloadTensors = [...graphInitializers, ...attributeTensors, ...functionDefaultTensors, ...sparsePayloadTensors];
  const tensorCount = graphs.reduce((sum, scope) => sum + (scope.inputs || []).length + (scope.outputs || []).length
    + (scope.valueInfo || []).length + (scope.initializers || []).length + (scope.sparseInitializers || []).length * 2, 0)
    + functions.reduce((sum, fn) => sum + (fn.inputs || []).length + (fn.outputs || []).length + (fn.valueInfo || []).length, 0)
    + attributeTensors.length + functionDefaultTensors.length + (attributeSparseTensors.length + functionDefaultSparseTensors.length) * 2;
  if (tensorCount > MAX_ONNX_TENSORS) throw new Error(`ONNX tensor declaration count ${tensorCount} exceeds safety limit ${MAX_ONNX_TENSORS}.`);
  let initializerBytes = 0;
  let decodedElements = 0;
  let externalDataEntries = 0;
  let externalDataTextBytes = 0;
  for (const tensor of payloadTensors) {
    initializerBytes += Number(tensor.storedDataBytes || 0);
    // A TensorProto may declare a very large logical shape while keeping its
    // payload in external_data.  Bound the values actually materialized by
    // this parser; the logical cardinality is still overflow-checked when the
    // external-data reference is validated below.
    const elements = materializedTensorElementCount(tensor);
    decodedElements += elements;
    if (!Number.isSafeInteger(initializerBytes) || initializerBytes > MAX_ONNX_INITIALIZER_BYTES) throw new Error(`ONNX embedded initializer bytes exceed safety limit ${MAX_ONNX_INITIALIZER_BYTES}.`);
    if (!Number.isSafeInteger(decodedElements) || decodedElements > MAX_ONNX_DECODED_ELEMENTS) throw new Error(`ONNX decoded initializer elements exceed safety limit ${MAX_ONNX_DECODED_ELEMENTS}.`);
    externalDataEntries += (tensor.externalData || []).length;
    externalDataTextBytes += (tensor.externalData || []).reduce((total, entry) => total
      + textEncoder.encode(String(entry.key || "")).byteLength
      + textEncoder.encode(String(entry.value || "")).byteLength, 0);
    if (externalDataEntries > MAX_ONNX_EXTERNAL_DATA_ENTRIES) throw new Error(`ONNX external_data entry count ${externalDataEntries} exceeds safety limit ${MAX_ONNX_EXTERNAL_DATA_ENTRIES}.`);
    if (externalDataTextBytes > MAX_ONNX_EXTERNAL_DATA_TEXT_BYTES) throw new Error(`ONNX external_data text bytes ${externalDataTextBytes} exceed safety limit ${MAX_ONNX_EXTERNAL_DATA_TEXT_BYTES}.`);
  }
  const metadata = [...(modelMetadata || []), ...graphs.flatMap((scope) => scope.metadataProps || []), ...functions.flatMap((fn) => fn.metadataProps || [])];
  if (metadata.length > MAX_ONNX_METADATA_PROPERTIES) throw new Error(`ONNX model and FunctionProto metadata property count ${metadata.length} exceeds safety limit ${MAX_ONNX_METADATA_PROPERTIES}.`);
  const metadataBytes = metadata.reduce((total, entry) => total
    + textEncoder.encode(String(entry.key || "")).byteLength
    + textEncoder.encode(String(entry.value || "")).byteLength, 0);
  if (metadataBytes > MAX_ONNX_METADATA_TEXT_BYTES) throw new Error(`ONNX model and FunctionProto metadata text bytes ${metadataBytes} exceed safety limit ${MAX_ONNX_METADATA_TEXT_BYTES}.`);
}

function materializedTensorElementCount(tensor) {
  const typedElements = Number(tensor?.typedElementCount || 0);
  if (!Number.isSafeInteger(typedElements) || typedElements < 0) throw new Error("Unsafe ONNX decoded typed-element count.");
  if (typedElements > 0) return typedElements;
  if (tensor?.rawData instanceof Uint8Array || Number(tensor?.rawDataBytes || 0) > 0) return checkedShapeElementCount(tensor.shape);
  return 0;
}

function collectGraphScopeEntries(root, functions) {
  const entries = [];
  const visitGraph = (graph, scope) => {
    entries.push({ scope, graph });
    visitNodes(graph.nodes || [], scope);
  };
  const visitNodes = (nodes, prefix) => {
    for (const [nodeIndex, node] of (nodes || []).entries()) {
      for (const [attributeName, attribute] of node.attributes || []) {
        const graphs = [attribute.graph, ...(attribute.graphs || [])].filter(Boolean);
        for (const [graphIndex, nested] of graphs.entries()) {
          const suffix = graphs.length === 1 ? attributeName : `${attributeName}[${graphIndex}]`;
          visitGraph(nested, `${prefix}/node:${nodeIndex}/attribute:${suffix}`);
        }
      }
    }
  };
  visitGraph(root, "main_graph");
  for (const [functionIndex, fn] of (functions || []).entries()) {
    const functionScope = onnxFunctionScope(functionIndex, fn);
    visitNodes(fn.nodes || [], functionScope);
    for (const [attributeIndex, attribute] of (fn.attributeProtos || []).entries()) {
      const attributeName = attribute.name || `attribute_${attributeIndex}`;
      const graphs = [attribute.graph, ...(attribute.graphs || [])].filter(Boolean);
      for (const [graphIndex, nested] of graphs.entries()) {
        const suffix = graphs.length === 1 ? attributeName : `${attributeName}[${graphIndex}]`;
        visitGraph(nested, `${functionScope}/default_attribute:${suffix}`);
      }
    }
  }
  return entries;
}

function collectOnnxTensorPayloadScopes(root, functions) {
  const entries = [];
  const appendSparsePayloads = (sparse, scope, rolePrefix) => {
    if (sparse?.values) entries.push({ scope: `${scope}/values`, role: `${rolePrefix}_values`, tensor: sparse.values });
    if (sparse?.indices) entries.push({ scope: `${scope}/indices`, role: `${rolePrefix}_indices`, tensor: sparse.indices });
  };
  const appendNodeAttributeTensors = (nodes, prefix) => {
    for (const [nodeIndex, node] of (nodes || []).entries()) {
      for (const [attributeName, attribute] of node.attributes || []) {
        const tensors = [attribute.tensor, ...(attribute.tensors || [])].filter(Boolean);
        for (const [tensorIndex, tensor] of tensors.entries()) {
          const suffix = tensors.length === 1 ? attributeName : `${attributeName}[${tensorIndex}]`;
          entries.push({ scope: `${prefix}/node:${nodeIndex}/attribute:${suffix}`, role: "node_attribute_tensor", tensor });
        }
        const sparseTensors = [attribute.sparseTensor, ...(attribute.sparseTensors || [])].filter(Boolean);
        for (const [sparseIndex, sparse] of sparseTensors.entries()) {
          const suffix = sparseTensors.length === 1 ? attributeName : `${attributeName}[${sparseIndex}]`;
          appendSparsePayloads(sparse, `${prefix}/node:${nodeIndex}/attribute:${suffix}`, "node_attribute_sparse_tensor");
        }
      }
    }
  };
  for (const { scope, graph } of collectGraphScopeEntries(root, functions)) {
    for (const tensor of graph.initializers || []) entries.push({ scope, role: "graph_initializer", tensor });
    for (const [index, sparse] of (graph.sparseInitializers || []).entries()) {
      appendSparsePayloads(sparse, `${scope}/sparse_initializer:${sparse.values?.name || index}`, "graph_sparse_initializer");
    }
    appendNodeAttributeTensors(graph.nodes || [], scope);
  }
  for (const [functionIndex, fn] of (functions || []).entries()) {
    const functionScope = onnxFunctionScope(functionIndex, fn);
    appendNodeAttributeTensors(fn.nodes || [], functionScope);
    for (const [attributeIndex, attribute] of (fn.attributeProtos || []).entries()) {
      const tensors = [attribute.tensor, ...(attribute.tensors || [])].filter(Boolean);
      const attributeName = attribute.name || `attribute_${attributeIndex}`;
      for (const [tensorIndex, tensor] of tensors.entries()) {
        const suffix = tensors.length === 1 ? attributeName : `${attributeName}[${tensorIndex}]`;
        entries.push({ scope: `${functionScope}/default_attribute:${suffix}`, role: "function_default_attribute_tensor", tensor });
      }
      const sparseTensors = [attribute.sparseTensor, ...(attribute.sparseTensors || [])].filter(Boolean);
      for (const [sparseIndex, sparse] of sparseTensors.entries()) {
        const suffix = sparseTensors.length === 1 ? attributeName : `${attributeName}[${sparseIndex}]`;
        appendSparsePayloads(sparse, `${functionScope}/default_attribute:${suffix}`, "function_default_attribute_sparse_tensor");
      }
    }
  }
  return entries;
}

function collectOnnxSparseTensorScopes(root, functions) {
  const entries = [];
  const appendNodeSparseTensors = (nodes, prefix) => {
    for (const [nodeIndex, node] of (nodes || []).entries()) {
      for (const [attributeName, attribute] of node.attributes || []) {
        const sparseTensors = [attribute.sparseTensor, ...(attribute.sparseTensors || [])].filter(Boolean);
        for (const [index, sparse] of sparseTensors.entries()) {
          const suffix = sparseTensors.length === 1 ? attributeName : `${attributeName}[${index}]`;
          entries.push({ scope: `${prefix}/node:${nodeIndex}/attribute:${suffix}`, role: "node_attribute_sparse_tensor", sparse });
        }
      }
    }
  };
  for (const { scope, graph } of collectGraphScopeEntries(root, functions)) {
    for (const [index, sparse] of (graph.sparseInitializers || []).entries()) {
      entries.push({ scope: `${scope}/sparse_initializer:${sparse.values?.name || index}`, role: "graph_sparse_initializer", sparse });
    }
    appendNodeSparseTensors(graph.nodes || [], scope);
  }
  for (const [functionIndex, fn] of (functions || []).entries()) {
    const functionScope = onnxFunctionScope(functionIndex, fn);
    appendNodeSparseTensors(fn.nodes || [], functionScope);
    for (const [attributeIndex, attribute] of (fn.attributeProtos || []).entries()) {
      const sparseTensors = [attribute.sparseTensor, ...(attribute.sparseTensors || [])].filter(Boolean);
      const attributeName = attribute.name || `attribute_${attributeIndex}`;
      for (const [index, sparse] of sparseTensors.entries()) {
        const suffix = sparseTensors.length === 1 ? attributeName : `${attributeName}[${index}]`;
        entries.push({ scope: `${functionScope}/default_attribute:${suffix}`, role: "function_default_attribute_sparse_tensor", sparse });
      }
    }
  }
  return entries;
}

function onnxFunctionScope(index, fn) {
  const domain = String(fn?.domain || "ai.onnx").trim() || "ai.onnx";
  return `function:${index}:${domain}::${String(fn?.name || "")}::${String(fn?.overload || "")}`;
}

function countGraphNodes(graph) { return countNodesWithNestedGraphs(graph?.nodes || []); }
function countNodesWithNestedGraphs(nodes) {
  let count = (nodes || []).length;
  for (const node of nodes || []) {
    for (const attribute of node.attributes?.values?.() || []) {
      if (attribute.graph) count += countGraphNodes(attribute.graph);
      for (const graph of attribute.graphs || []) count += countGraphNodes(graph);
    }
  }
  return count;
}

function checkedShapeElementCount(shape) {
  if (!Array.isArray(shape)) return 0;
  if (!shape.length) return 1;
  let product = 1;
  for (const rawDim of shape) {
    const dim = Number(rawDim);
    if (!Number.isSafeInteger(dim) || dim < 0) throw new Error(`Unsafe ONNX tensor dimension: ${rawDim}.`);
    if (dim === 0) return 0;
    if (product > Number.MAX_SAFE_INTEGER / dim) throw new Error("ONNX tensor element-count multiplication overflow.");
    product *= dim;
  }
  return product;
}

function onnxInitializerElementCount(tensor) {
  if (Number(tensor?.typedElementCount || 0) > 0) return Number(tensor.typedElementCount);
  if (Array.isArray(tensor?.shape)) return shapeElementCount(tensor.shape, true);
  const bits = dtypeStorageBits(tensor?.dtype);
  const bytes = Number(tensor?.storedDataBytes || 0);
  return bits >= 8 && bytes > 0 && (bytes * 8) % bits === 0 ? bytes * 8 / bits : 0;
}

function parseValueInfo(bytes) {
  const value = { name: "", dtype: "UNKNOWN", shape: [], shapeDeclared: false, valueKind: "unresolved", typeProto: null };
  for (const field of readFields(bytes)) {
    if (field.no === 1 && field.wire === 2) value.name = decodeString(field.bytes);
    if (field.no === 2 && field.wire === 2) {
      const type = parseType(field.bytes);
      value.typeProto = type;
      value.valueKind = type.kind || "unresolved";
      value.dtype = type.dtype || value.dtype;
      value.shape = type.shape || value.shape;
      value.shapeDeclared = type.shapeDeclared === true;
    }
  }
  return value;
}

function parseType(bytes, depth = 0) {
  if (depth > 64) throw new Error("ONNX TypeProto nesting depth exceeds safety limit 64.");
  const type = {
    kind: "undefined", dtype: "UNKNOWN", shape: [], shapeDeclared: false, shapeDimensions: [],
    denotation: "", valueFieldsPresent: [],
  };
  for (const field of readFields(bytes)) {
    if (field.no === 1 && field.wire === 2) {
      const tensorType = parseTensorType(field.bytes);
      assignTypeBranch(type, {
        kind: "tensor", valueFieldsPresent: [1], dtype: tensorType.dtype, shape: tensorType.shape,
        shapeDeclared: tensorType.shapeDeclared, shapeDimensions: tensorType.shapeDimensions,
        shapeFieldCount: tensorType.shapeFieldCount,
        elementTypeId: tensorType.elementTypeId, elementTypeName: tensorType.elementTypeName,
        elementTypeFieldCount: tensorType.elementTypeFieldCount,
      });
    }
    if (field.no === 4 && field.wire === 2) assignTypeBranch(type, parseWrappedType(field.bytes, "sequence", depth));
    if (field.no === 5 && field.wire === 2) assignTypeBranch(type, parseMapType(field.bytes, depth));
    if (field.no === 6 && field.wire === 2) type.denotation = decodeString(field.bytes);
    if (field.no === 7 && field.wire === 2) assignTypeBranch(type, parseOpaqueType(field.bytes));
    if (field.no === 8 && field.wire === 2) {
      const sparseType = parseTensorType(field.bytes);
      assignTypeBranch(type, {
        kind: "sparse_tensor",
        valueFieldsPresent: [8],
        dtype: sparseType.dtype,
        shape: sparseType.shape,
        shapeDeclared: sparseType.shapeDeclared,
        shapeDimensions: sparseType.shapeDimensions,
        shapeFieldCount: sparseType.shapeFieldCount,
        elementTypeId: sparseType.elementTypeId,
        elementTypeName: sparseType.elementTypeName,
        elementTypeFieldCount: sparseType.elementTypeFieldCount,
      });
    }
    if (field.no === 9 && field.wire === 2) assignTypeBranch(type, parseWrappedType(field.bytes, "optional", depth));
  }
  return type;
}

function assignTypeBranch(target, branch) {
  const valueFieldsPresent = [...(target.valueFieldsPresent || []), ...(branch.valueFieldsPresent || [])];
  Object.assign(target, branch);
  target.valueFieldsPresent = valueFieldsPresent;
}

function parseTensorType(bytes) {
  const tensorType = { dtype: "UNKNOWN", shape: [], shapeDeclared: false, shapeDimensions: [], shapeFieldCount: 0, elementTypeId: 0, elementTypeFieldCount: 0 };
  for (const field of readFields(bytes)) {
    if (field.no === 1 && field.wire === 0) {
      tensorType.elementTypeId = toSafeNumber(field.value);
      tensorType.elementTypeFieldCount += 1;
      tensorType.dtype = tensorTypeName(tensorType.elementTypeId);
    }
    if (field.no === 2 && field.wire === 2) {
      tensorType.shapeFieldCount += 1;
      tensorType.shapeDimensions = parseShape(field.bytes);
      tensorType.shape = tensorType.shapeDimensions.map((dimension) => dimension.kind === "value" ? dimension.value : -1);
      tensorType.shapeDeclared = true;
    }
  }
  tensorType.elementTypeName = tensorType.dtype;
  return tensorType;
}

function parseShape(bytes) {
  const shape = [];
  for (const field of readFields(bytes)) {
    if (field.no === 1 && field.wire === 2) shape.push(parseDimension(field.bytes));
  }
  return shape;
}

function parseDimension(bytes) {
  const dimension = { kind: "unknown", value: null, parameter: "", denotation: "", valueFieldCount: 0 };
  for (const field of readFields(bytes)) {
    if (field.no === 1 && field.wire === 0) {
      dimension.kind = "value";
      dimension.value = toSafeNumber(field.value);
      dimension.valueFieldCount += 1;
    }
    if (field.no === 2 && field.wire === 2) {
      dimension.kind = "symbolic";
      dimension.parameter = decodeString(field.bytes);
      dimension.valueFieldCount += 1;
    }
    if (field.no === 3 && field.wire === 2) dimension.denotation = decodeString(field.bytes);
  }
  return dimension;
}

function parseWrappedType(bytes, kind, depth) {
  const parsed = { kind, valueFieldsPresent: [kind === "sequence" ? 4 : 9], elementType: null, childFieldCount: 0 };
  for (const field of readFields(bytes)) {
    if (field.no === 1 && field.wire === 2) {
      parsed.elementType = parseType(field.bytes, depth + 1);
      parsed.childFieldCount += 1;
    }
  }
  return parsed;
}

function parseMapType(bytes, depth) {
  const parsed = { kind: "map", valueFieldsPresent: [5], keyTypeId: 0, keyTypeName: "UNDEFINED", keyFieldCount: 0, valueType: null, childFieldCount: 0 };
  for (const field of readFields(bytes)) {
    if (field.no === 1 && field.wire === 0) {
      parsed.keyTypeId = toSafeNumber(field.value);
      parsed.keyTypeName = tensorTypeName(parsed.keyTypeId);
      parsed.keyFieldCount += 1;
    }
    if (field.no === 2 && field.wire === 2) {
      parsed.valueType = parseType(field.bytes, depth + 1);
      parsed.childFieldCount += 1;
    }
  }
  return parsed;
}

function parseOpaqueType(bytes) {
  const parsed = { kind: "opaque", valueFieldsPresent: [7], domain: "", name: "" };
  for (const field of readFields(bytes)) {
    if (field.no === 1 && field.wire === 2) parsed.domain = decodeString(field.bytes);
    if (field.no === 2 && field.wire === 2) parsed.name = decodeString(field.bytes);
  }
  return parsed;
}

function parseOpset(bytes) {
  const opset = { domain: "", version: 0 };
  for (const field of readFields(bytes)) {
    if (field.no === 1 && field.wire === 2) opset.domain = decodeString(field.bytes);
    if (field.no === 2 && field.wire === 0) opset.version = toSafeNumber(field.value);
  }
  return opset;
}

function readFields(bytes) {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const fields = [];
  let offset = 0;
  while (offset < bytes.byteLength) {
    const tag = readVarint(view, offset);
    offset = tag.offset;
    const no = Number(tag.value >> 3n);
    const wire = Number(tag.value & 7n);
    if (wire === 0) {
      const value = readVarint(view, offset);
      offset = value.offset;
      fields.push({ no, wire, value: value.value });
    } else if (wire === 1) {
      ensureFieldBytes(bytes, offset, 8, no, wire);
      fields.push({ no, wire, bytes: bytes.slice(offset, offset + 8) });
      offset += 8;
    } else if (wire === 2) {
      const length = readVarint(view, offset);
      offset = length.offset;
      const end = offset + toSafeNumber(length.value);
      ensureFieldBytes(bytes, offset, end - offset, no, wire);
      fields.push({ no, wire, bytes: bytes.slice(offset, end) });
      offset = end;
    } else if (wire === 5) {
      ensureFieldBytes(bytes, offset, 4, no, wire);
      fields.push({ no, wire, float32: view.getFloat32(offset, true), bytes: bytes.slice(offset, offset + 4) });
      offset += 4;
    } else {
      throw new Error(`Unsupported ONNX protobuf wire type ${wire}`);
    }
  }
  return fields;
}

function ensureFieldBytes(bytes, offset, length, fieldNo, wire) {
  if (length < 0 || offset + length > bytes.byteLength) {
    throw new Error(`Malformed ONNX protobuf: field ${fieldNo} wire ${wire} exceeds message bounds`);
  }
}

function readVarint(view, offset) {
  let shift = 0n;
  let value = 0n;
  let cursor = offset;
  while (cursor < view.byteLength) {
    const byte = view.getUint8(cursor++);
    value |= BigInt(byte & 0x7f) << shift;
    if ((byte & 0x80) === 0) return { value, offset: cursor };
    shift += 7n;
  }
  throw new Error("Unterminated ONNX protobuf varint");
}

function readPackedVarints(bytes) {
  const values = [];
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  let offset = 0;
  while (offset < bytes.byteLength) {
    const value = readVarint(view, offset);
    values.push(value.value);
    offset = value.offset;
  }
  return values;
}

function readPackedFloat32(bytes) {
  const values = [];
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  for (let offset = 0; offset + 4 <= bytes.byteLength; offset += 4) {
    values.push(view.getFloat32(offset, true));
  }
  return values;
}

function readPackedFloat64(bytes) {
  const values = [];
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  for (let offset = 0; offset + 8 <= bytes.byteLength; offset += 8) values.push(view.getFloat64(offset, true));
  return values;
}

function signedVarint(value, bits) {
  const width = BigInt(bits);
  const sign = 1n << (width - 1n);
  const mask = (1n << width) - 1n;
  const normalized = value & mask;
  return (normalized & sign) !== 0n ? normalized - (1n << width) : normalized;
}

function broadcastShape(left, right) {
  if (!Array.isArray(left) || !Array.isArray(right)) return null;
  const out = [];
  const len = Math.max(left.length, right.length);
  for (let i = 0; i < len; i++) {
    const a = left[left.length - 1 - i] ?? 1;
    const b = right[right.length - 1 - i] ?? 1;
    if (!Number.isSafeInteger(a) || !Number.isSafeInteger(b) || a < 0 || b < 0
      || (a !== b && a !== 1 && b !== 1)) return null;
    out.unshift(a === 1 ? b : b === 1 ? a : a);
  }
  return out;
}

function tensorShapeKnown(shape, shapeDeclared = Array.isArray(shape) && shape.length > 0) {
  return Array.isArray(shape) && shapeDeclared === true
    && shape.every((dim) => Number.isSafeInteger(Number(dim)) && Number(dim) >= 0);
}

function knownTensorShape(tensor) {
  return !tensorContractBlocksDeterministicCost(tensor)
    && tensorShapeKnown(tensor?.shape, tensor?.shapeDeclared ?? tensor?.shape_declared);
}

function tensorContractBlocksDeterministicCost(tensor) {
  return tensor?.contractStatus === "invalid" || tensor?.contract_status === "invalid"
    || tensor?.conditionalShapeContract?.status === "assessed_partial"
    || tensor?.conditional_shape_contract?.status === "assessed_partial";
}

function safeExactProduct(values) {
  let product = 1n;
  const maximum = BigInt(Number.MAX_SAFE_INTEGER);
  for (const value of values) {
    if (!Number.isSafeInteger(value) || value < 0) return null;
    product *= BigInt(value);
    if (product > maximum) return null;
  }
  return Number(product);
}

function exactNonnegativeProduct(values) {
  let product = 1n;
  for (const value of values) {
    if (!Number.isSafeInteger(value) || value < 0) return null;
    product *= BigInt(value);
  }
  return product;
}

function exactNonnegativeInteger(value) {
  if (typeof value === "bigint") return value >= 0n ? value : null;
  if (Number.isSafeInteger(value) && value >= 0) return BigInt(value);
  const text = value == null ? "" : String(value);
  return /^(?:0|[1-9]\d*)$/.test(text) ? BigInt(text) : null;
}

function exactNonnegativeSum(values) {
  let total = 0n;
  for (const value of values) {
    const exact = exactNonnegativeInteger(value);
    if (exact == null) return null;
    total += exact;
  }
  return total;
}

export function summarizeOnnxAssessedMacs(values) {
  const macs = exactNonnegativeSum(values);
  if (macs == null) throw new Error("ONNX assessed MAC subtotal contains an invalid nonnegative integer.");
  const operations = macs * 2n;
  const macMirror = safeBigIntNumber(macs);
  const operationMirror = safeBigIntNumber(operations);
  return {
    total_assessed_macs: macMirror,
    total_assessed_macs_decimal: macs.toString(),
    total_assessed_ops: operationMirror,
    total_assessed_ops_decimal: operations.toString(),
    safe_number_mirror_status: macMirror == null || operationMirror == null ? "exact_decimal_only" : "safe_integer_mirrors_available",
  };
}

export function projectOnnxCompleteMacTotals(assessedTotals, unassessedComputeOpCount) {
  const unresolved = Number(unassessedComputeOpCount);
  if (!Number.isSafeInteger(unresolved) || unresolved < 0) {
    throw new Error("ONNX unassessed compute-op count must be a nonnegative safe integer.");
  }
  const complete = unresolved === 0;
  return {
    total_macs: complete ? assessedTotals?.total_assessed_macs ?? null : null,
    total_macs_decimal: complete ? assessedTotals?.total_assessed_macs_decimal ?? null : null,
    total_ops: complete ? assessedTotals?.total_assessed_ops ?? null : null,
    total_ops_decimal: complete ? assessedTotals?.total_assessed_ops_decimal ?? null : null,
  };
}

function bindOnnxQuantization(graph, tensorMap, opsets, irVersion, functions = []) {
  const initializers = new Map(graph.initializers.map((tensor) => [tensor.name, tensor]));
  const importedOpsets = new Map((opsets || []).map((entry) => [normalizeOnnxDomain(entry.domain), Number(entry.version || 0)]));
  const bindings = [];
  const boundaryEdges = [];
  const integerComputeWithoutScale = [];
  for (const [nodeIndex, node] of graph.nodes.entries()) {
    const domain = normalizeOnnxDomain(node.domain);
    const importedOpset = importedOpsets.get(domain) || null;
    if (domain !== "ai.onnx") continue;
    if (node.opType === "QuantizeLinear" || node.opType === "DequantizeLinear") {
      const dataTensorName = node.opType === "QuantizeLinear" ? node.outputs[0] : node.inputs[0];
      const binding = buildOnnxQuantBinding({
        node,
        nodeIndex,
        importedOpset,
        role: node.opType === "QuantizeLinear" ? "quantized_output" : "quantized_input",
        tensorName: dataTensorName,
        scaleName: node.inputs[1],
        zeroPointName: node.inputs[2],
        zeroPointOptional: true,
        axisMode: "attribute",
        tensorMap,
        initializers,
      });
      bindings.push(binding);
      applyOnnxQuantBinding(tensorMap, binding);
      const input = tensorMap.get(node.inputs[0]);
      const output = tensorMap.get(node.outputs[0]);
      boundaryEdges.push({
        op_index: nodeIndex,
        op_name: node.opType,
        input_tensor_name: node.inputs[0] || "",
        output_tensor_name: node.outputs[0] || "",
        input_payload_bytes: tensorPayloadBytes(input),
        output_payload_bytes: tensorPayloadBytes(output),
        payload_status: tensorPayloadBytes(input) != null && tensorPayloadBytes(output) != null ? "assessed" : "not_assessed_unknown_shape_or_dtype",
        materialization_status: "not_assessed_static_graph_only",
      });
      continue;
    }
    if (node.opType === "QLinearConv") {
      for (const spec of [
        { role: "activation_input", tensorName: node.inputs[0], scaleName: node.inputs[1], zeroPointName: node.inputs[2], axisMode: "scalar" },
        { role: "kernel_weight", tensorName: node.inputs[3], scaleName: node.inputs[4], zeroPointName: node.inputs[5], axisMode: "fixed_zero" },
        { role: "activation_output", tensorName: node.outputs[0], scaleName: node.inputs[6], zeroPointName: node.inputs[7], axisMode: "scalar" },
      ]) {
        const binding = buildOnnxQuantBinding({ node, nodeIndex, importedOpset, ...spec, zeroPointOptional: false, tensorMap, initializers });
        bindings.push(binding);
        applyOnnxQuantBinding(tensorMap, binding);
      }
      continue;
    }
    if (node.opType === "QLinearMatMul") {
      for (const spec of [
        { role: "activation_input", tensorName: node.inputs[0], scaleName: node.inputs[1], zeroPointName: node.inputs[2], axisMode: "scalar" },
        { role: "kernel_weight", tensorName: node.inputs[3], scaleName: node.inputs[4], zeroPointName: node.inputs[5], axisMode: "scalar" },
        { role: "activation_output", tensorName: node.outputs[0], scaleName: node.inputs[6], zeroPointName: node.inputs[7], axisMode: "scalar" },
      ]) {
        const binding = buildOnnxQuantBinding({ node, nodeIndex, importedOpset, ...spec, zeroPointOptional: false, tensorMap, initializers });
        bindings.push(binding);
        applyOnnxQuantBinding(tensorMap, binding);
      }
      continue;
    }
    if (node.opType === "ConvInteger" || node.opType === "MatMulInteger") {
      integerComputeWithoutScale.push({
        op_index: nodeIndex,
        op_name: node.opType,
        status: "integer_arithmetic_observed_real_scale_not_embedded_in_operator",
        evidence_class: "OBSERVED",
      });
    }
  }
  const annotationTensorCounts = new Map();
  for (const annotation of graph.quantizationAnnotations || []) {
    annotationTensorCounts.set(annotation.tensorName, (annotationTensorCounts.get(annotation.tensorName) || 0) + 1);
  }
  const annotationBindings = (graph.quantizationAnnotations || []).map((annotation, annotationIndex) => buildOnnxAnnotationBinding({
    annotation,
    annotationIndex,
    duplicateTensorAnnotation: annotationTensorCounts.get(annotation.tensorName) > 1,
    irVersion,
    tensorMap,
    initializers,
    operatorBindings: bindings,
  }));
  for (const binding of annotationBindings) {
    bindings.push(binding);
    applyOnnxQuantBinding(tensorMap, binding);
  }
  const scopedGraphs = collectGraphScopeEntries(graph, functions);
  const nestedGraphAnnotations = scopedGraphs.slice(1).flatMap(({ scope, graph: nested }) => (nested.quantizationAnnotations || []).map((annotation) => ({
    scope,
    tensor_name: annotation.tensorName,
    parameter_entry_count: (annotation.quantParameterTensorNames || []).length,
  })));
  const invalid = bindings.filter((item) => item.status === "fail");
  const unresolved = bindings.filter((item) => item.status.startsWith("not_assessed"));
  return {
    schema: "deepbom.onnx_quantization_binding.v1.1",
    status: invalid.length ? "fail" : unresolved.length || nestedGraphAnnotations.length ? "partial" : bindings.length ? "pass" : "not_applicable",
    evidence_class: bindings.length || integerComputeWithoutScale.length || nestedGraphAnnotations.length ? "DERIVED" : "NOT_APPLICABLE",
    binding_count: bindings.length,
    valid_binding_count: bindings.filter((item) => item.status === "pass").length,
    invalid_binding_count: invalid.length,
    unresolved_binding_count: unresolved.length,
    explicit_qdq_boundary_count: boundaryEdges.length,
    boundary_edges: boundaryEdges,
    integer_compute_without_real_scale_count: integerComputeWithoutScale.length,
    integer_compute_without_real_scale: integerComputeWithoutScale,
    graph_scope_count: scopedGraphs.length,
    main_graph_annotation_count: annotationBindings.length,
    valid_annotation_count: annotationBindings.filter((item) => item.status === "pass").length,
    invalid_annotation_count: annotationBindings.filter((item) => item.status === "fail").length,
    unresolved_annotation_count: annotationBindings.filter((item) => item.status.startsWith("not_assessed")).length,
    nested_graph_annotation_count: nestedGraphAnnotations.length,
    nested_graph_annotations: nestedGraphAnnotations,
    annotation_scope_status: nestedGraphAnnotations.length ? "main_graph_bound_nested_graph_annotations_inventoried_not_bound" : "all_serialized_graph_annotations_bound",
    annotation_source_ref: ONNX_TENSOR_TYPE_SOURCE.ref,
    annotation_source_sha256: ONNX_TENSOR_TYPE_SOURCE.sha256,
    bindings,
    method: "ONNX standard-domain Q/DQ and QLinear operator inputs plus main-GraphProto TensorAnnotation mappings are bound to embedded TensorProto scale/zero-point values; reference identity, scale positivity, dtype range, axis/cardinality where defined, duplicate annotations, and numerical conflicts are checked without runtime inference. Nested GraphProto annotations are counted by scope and remain explicitly unbound.",
  };
}

function buildOnnxAnnotationBinding({ annotation, annotationIndex, duplicateTensorAnnotation, irVersion, tensorMap, initializers, operatorBindings }) {
  const entries = (annotation.quantParameterTensorNames || []).map((entry) => ({ key: String(entry.key || ""), value: String(entry.value || "") }));
  const keyCounts = new Map();
  for (const entry of entries) keyCounts.set(entry.key, (keyCounts.get(entry.key) || 0) + 1);
  const tensorName = String(annotation.tensorName || "");
  const tensor = tensorMap.get(tensorName);
  const scaleName = entries.find((entry) => entry.key === "SCALE_TENSOR")?.value || "";
  const zeroPointName = entries.find((entry) => entry.key === "ZERO_POINT_TENSOR")?.value || "";
  if (tensorMap.get(scaleName)) tensorMap.get(scaleName).quantizationParameterRole = "scale";
  if (tensorMap.get(zeroPointName)) tensorMap.get(zeroPointName).quantizationParameterRole = "zero_point";
  const scale = decodeOnnxQuantParameter(initializers.get(scaleName), scaleName, "scale", tensorMap.get(scaleName));
  const zeroPoint = zeroPointName
    ? decodeOnnxQuantParameter(initializers.get(zeroPointName), zeroPointName, "zero_point", tensorMap.get(zeroPointName))
    : { status: "not_assessed_missing_parameter", values: [], dtype: "UNKNOWN", shape: [] };
  const failures = [];
  const unresolved = [];
  if (Number(irVersion || 0) < 5) failures.push(`quantization_annotation_requires_ir_version_5_or_later_observed_${Number(irVersion || 0)}`);
  if (!tensorName) failures.push("annotation_tensor_name_is_empty");
  if (!tensor) failures.push("annotation_tensor_name_does_not_resolve_in_main_graph");
  if (duplicateTensorAnnotation) failures.push("duplicate_tensor_annotation_for_same_tensor_name");
  if (!entries.length) unresolved.push("not_assessed_annotation_has_no_parameter_entries");
  if (entries.some((entry) => !entry.key || !entry.value)) failures.push("annotation_parameter_key_or_tensor_name_is_empty");
  for (const [key, count] of keyCounts) if (key && count > 1) failures.push(`duplicate_quant_parameter_key_${key}`);
  const unknownKeys = [...keyCounts.keys()].filter((key) => key && !["SCALE_TENSOR", "ZERO_POINT_TENSOR"].includes(key));
  if (unknownKeys.length) unresolved.push(`not_assessed_unrecognized_quant_parameter_keys_${unknownKeys.join("_")}`);
  if (!scaleName) unresolved.push("not_assessed_missing_SCALE_TENSOR_mapping");
  if (scale.status !== "assessed") unresolved.push(`scale:${scale.status}`);
  if (!zeroPointName) unresolved.push("not_assessed_missing_ZERO_POINT_TENSOR_mapping");
  else if (zeroPoint.status !== "assessed") unresolved.push(`zero_point:${zeroPoint.status}`);
  if (scale.status === "assessed" && !isFloatDtype(scale.dtype)) failures.push(`scale_tensor_dtype_${scale.dtype}_is_not_floating_point`);
  if (scale.status === "assessed" && scale.values.some((value) => !(Number(value) > 0) || !Number.isFinite(Number(value)))) failures.push("scale_values_must_be_finite_and_positive");
  const legalRange = integerDtypeRange(zeroPoint.dtype);
  if (zeroPoint.status === "assessed" && !legalRange) failures.push("zero_point_dtype_must_be_supported_integer_type");
  if (zeroPoint.status === "assessed" && tensor?.dtype && tensor.dtype !== "UNKNOWN" && zeroPoint.dtype !== tensor.dtype) failures.push(`zero_point_dtype_${zeroPoint.dtype}_does_not_match_tensor_dtype_${tensor.dtype}`);
  if (legalRange && zeroPoint.values.some((value) => !Number.isInteger(Number(value)) || Number(value) < legalRange[0] || Number(value) > legalRange[1])) failures.push("zero_point_value_is_not_representable_by_declared_dtype");
  if (scale.status === "assessed" && zeroPoint.status === "assessed" && !sameShape(scale.shape, zeroPoint.shape)) failures.push("scale_zero_point_shape_mismatch");

  const prior = operatorBindings.filter((binding) => binding.tensor_name === tensorName && binding.scale_values?.length && ["pass", "fail"].includes(binding.status));
  const equivalent = scale.status === "assessed" && zeroPoint.status === "assessed"
    ? prior.filter((binding) => sameNumericArray(binding.scale_values, scale.values) && sameNumericArray(binding.zero_point_values, zeroPoint.values))
    : [];
  if (prior.length && scale.status === "assessed" && zeroPoint.status === "assessed" && !equivalent.length) failures.push("annotation_numerical_contract_conflicts_with_operator_binding");
  let axis = null;
  let axisSource = "not_applicable_per_tensor";
  let blockSize = null;
  let parameterization = scale.values.length === 1 ? "per_tensor" : scale.values.length > 1 ? "per_axis_or_blocked_axis_unbound" : "unresolved";
  let cardinality = scale.values.length === 1
    ? { status: "pass", reason: "scalar_scale" }
    : { status: "not_assessed_annotation_axis_unbound", reason: "TensorAnnotation does not serialize axis or block size" };
  if (equivalent.length) {
    const signatures = new Set(equivalent.map((binding) => `${binding.axis ?? "null"}:${binding.block_size ?? "null"}:${binding.parameterization}`));
    if (signatures.size === 1) {
      axis = equivalent[0].axis;
      axisSource = "cross_checked_operator_contract";
      blockSize = equivalent[0].block_size;
      parameterization = equivalent[0].parameterization;
      cardinality = assessOnnxQuantCardinality(tensor, scale.shape, scale.values.length, parameterization, axis, blockSize || 0);
    } else {
      unresolved.push("not_assessed_operator_bindings_disagree_on_axis_or_block_size");
    }
  } else if (scale.values.length > 1) {
    unresolved.push("not_assessed_annotation_does_not_serialize_axis_or_block_size");
  }
  if (cardinality.status === "fail") failures.push(cardinality.reason);
  if (cardinality.status.startsWith("not_assessed") && !unresolved.includes(cardinality.status)) unresolved.push(cardinality.status);
  const status = failures.length ? "fail" : unresolved.length ? "not_assessed_annotation_contract_incomplete" : "pass";
  return {
    binding_source: "graph_quantization_annotation",
    annotation_index: annotationIndex,
    annotation_entries: entries,
    op_index: null,
    op_name: "GraphProto.TensorAnnotation",
    domain: "ai.onnx",
    imported_opset: null,
    role: "graph_quantization_annotation",
    tensor_name: tensorName,
    tensor_dtype: tensor?.dtype || "UNKNOWN",
    tensor_shape: tensor?.shape || [],
    scale_tensor_name: scaleName,
    scale_tensor_dtype: scale.dtype,
    scale_tensor_shape: scale.shape,
    scale_values: scale.values,
    zero_point_tensor_name: zeroPointName,
    zero_point_tensor_dtype: zeroPoint.dtype,
    zero_point_tensor_shape: zeroPoint.shape,
    zero_point_values: zeroPoint.values,
    scale_count: scale.values.length,
    zero_point_count: zeroPoint.values.length,
    axis,
    axis_source: axisSource,
    block_size: blockSize,
    parameterization,
    cardinality_status: cardinality.status,
    cardinality_detail: cardinality.reason,
    operator_cross_check_status: !prior.length ? "not_applicable_no_operator_binding" : equivalent.length ? "equivalent_numerical_contract" : "conflict",
    status,
    reasons: [...failures, ...unresolved],
    evidence_class: status === "pass" || status === "fail" ? "DERIVED" : "NOT_ASSESSABLE",
    source_ref: ONNX_TENSOR_TYPE_SOURCE.ref,
    source_sha256: ONNX_TENSOR_TYPE_SOURCE.sha256,
  };
}

function buildOnnxQuantBinding({ node, nodeIndex, importedOpset, role, tensorName, scaleName, zeroPointName, zeroPointOptional, axisMode, tensorMap, initializers }) {
  const tensor = tensorMap.get(tensorName);
  if (tensorMap.get(scaleName)) tensorMap.get(scaleName).quantizationParameterRole = "scale";
  if (tensorMap.get(zeroPointName)) tensorMap.get(zeroPointName).quantizationParameterRole = "zero_point";
  const scale = decodeOnnxQuantParameter(initializers.get(scaleName), scaleName, "scale", tensorMap.get(scaleName));
  const zeroPoint = zeroPointName
    ? decodeOnnxQuantParameter(initializers.get(zeroPointName), zeroPointName, "zero_point", tensorMap.get(zeroPointName))
    : zeroPointOptional ? { status: "defaulted", values: [0], dtype: tensor?.dtype || "UINT8", shape: [] } : { status: "not_assessed_missing_parameter", values: [], dtype: "UNKNOWN", shape: [] };
  const scaleValueCount = scale.values.length;
  const zeroPointValueCount = zeroPoint.values.length;
  const scaleCount = scaleValueCount || scale.declaredCount || 0;
  const zeroPointCount = zeroPointValueCount || zeroPoint.declaredCount || 0;
  const blockSize = Math.max(0, attrInt(node, "block_size", 0));
  let axis = null;
  let axisSource = "not_applicable_per_tensor";
  if (axisMode === "fixed_zero") {
    axis = 0;
    axisSource = "operator_semantics";
  } else if (axisMode === "attribute" && (scaleCount > 1 || blockSize > 0)) {
    const explicit = node.attributes.has("axis");
    axis = attrInt(node, "axis", 1);
    axisSource = explicit ? "attribute" : importedOpset != null && importedOpset >= 13 ? "schema_default_axis_1" : "unresolved_before_opset_13";
  }
  if (axis != null && tensor?.shape?.length && axis < 0) axis += tensor.shape.length;
  const parameterization = blockSize > 0 ? "blocked" : scaleCount > 1 ? "per_axis" : scaleCount === 1 ? "per_tensor" : "unresolved";
  const reasons = [];
  if (scale.status !== "assessed") reasons.push(`scale:${scale.status}`);
  if (!["assessed", "defaulted"].includes(zeroPoint.status)) reasons.push(`zero_point:${zeroPoint.status}`);
  const allowedScaleDtypes = ["QLinearConv", "QLinearMatMul"].includes(node.opType) || (importedOpset != null && importedOpset < 19)
    ? ["FLOAT32"]
    : ["FLOAT16", "FLOAT32", "BFLOAT16"];
  if (scale.dtype !== "UNKNOWN" && !allowedScaleDtypes.includes(scale.dtype)) reasons.push(`scale_dtype_${scale.dtype}_not_allowed_for_${node.opType}_opset_${importedOpset ?? "unknown"}`);
  if (scale.status === "assessed" && scale.values.some((value) => !(Number(value) > 0) || !Number.isFinite(Number(value)))) reasons.push("scale_values_must_be_finite_and_positive");
  const legalRange = integerDtypeRange(zeroPoint.dtype || tensor?.dtype);
  if (zeroPoint.dtype !== "UNKNOWN" && !integerDtypeRange(zeroPoint.dtype)) reasons.push("zero_point_dtype_must_be_supported_integer_type");
  if (zeroPoint.dtype !== "UNKNOWN" && tensor?.dtype && tensor.dtype !== "UNKNOWN" && zeroPoint.dtype !== tensor.dtype) reasons.push(`zero_point_dtype_${zeroPoint.dtype}_does_not_match_tensor_dtype_${tensor.dtype}`);
  if (zeroPoint.values.some((value) => !Number.isInteger(Number(value)))) reasons.push("zero_point_values_must_be_integers");
  if (legalRange && zeroPoint.values.some((value) => Number(value) < legalRange[0] || Number(value) > legalRange[1])) reasons.push("zero_point_out_of_dtype_range");
  if (scaleCount > 0 && zeroPointCount > 0 && zeroPoint.status !== "defaulted" && zeroPointCount !== scaleCount) reasons.push("scale_zero_point_cardinality_mismatch");
  if (scale.shapeDeclared && zeroPoint.shapeDeclared && !sameShape(scale.shape, zeroPoint.shape)) reasons.push("scale_zero_point_shape_mismatch");
  if (axisMode === "scalar" && scaleCount > 1) reasons.push("operator_requires_scalar_scale");
  if (parameterization === "per_axis" && scale.shape.length !== 1) reasons.push("per_axis_scale_must_be_rank_one");
  if (blockSize > 0 && (node.opType !== "QuantizeLinear" && node.opType !== "DequantizeLinear" || importedOpset == null || importedOpset < 21)) reasons.push("block_size_not_supported_by_operator_opset");
  if (axisSource === "unresolved_before_opset_13") reasons.push("per_axis_axis_not_defined_by_imported_opset");
  const cardinality = assessOnnxQuantCardinality(tensor, scale.shape, scaleCount, parameterization, axis, blockSize);
  if (cardinality.status === "fail") reasons.push(cardinality.reason);
  const unresolved = scale.status !== "assessed" || !["assessed", "defaulted"].includes(zeroPoint.status) || cardinality.status.startsWith("not_assessed");
  return {
    binding_source: "operator_quantization_contract",
    op_index: nodeIndex,
    op_name: node.opType,
    domain: normalizeOnnxDomain(node.domain),
    imported_opset: importedOpset,
    role,
    tensor_name: tensorName || "",
    tensor_dtype: tensor?.dtype || "UNKNOWN",
    tensor_shape: tensor?.shape || [],
    scale_tensor_name: scaleName || "",
    scale_tensor_dtype: scale.dtype,
    scale_tensor_shape: scale.shape,
    scale_values: scale.values,
    scale_value_count: scaleValueCount,
    zero_point_tensor_name: zeroPointName || "",
    zero_point_tensor_dtype: zeroPoint.dtype,
    zero_point_tensor_shape: zeroPoint.shape,
    zero_point_values: zeroPoint.values,
    zero_point_value_count: zeroPointValueCount,
    scale_count: scaleCount,
    zero_point_count: zeroPointCount,
    scale_cardinality_source: scaleValueCount ? "decoded_values" : scale.declaredCount ? "declared_tensor_shape" : "unresolved",
    zero_point_cardinality_source: zeroPoint.status === "defaulted" ? "schema_default" : zeroPointValueCount ? "decoded_values" : zeroPoint.declaredCount ? "declared_tensor_shape" : "unresolved",
    axis,
    axis_source: axisSource,
    block_size: blockSize || null,
    parameterization,
    cardinality_status: cardinality.status,
    cardinality_detail: cardinality.reason,
    status: reasons.length ? unresolvedQuantBindingStatus(scale, zeroPoint, reasons) : "pass",
    reasons,
    structure_evidence_class: "DERIVED",
    value_evidence_class: scale.status === "assessed" && zeroPoint.status === "assessed" ? "OBSERVED"
      : scale.status === "assessed" && zeroPoint.status === "defaulted" ? "DERIVED" : "RUNTIME_REQUIRED",
    evidence_class: unresolved ? "NOT_ASSESSABLE" : "DERIVED",
  };
}

function unresolvedQuantBindingStatus(scale, zeroPoint, reasons) {
  const unresolvedOnly = reasons.every((reason) => reason.includes("not_assessed") || reason.includes("missing_parameter"));
  if (!unresolvedOnly) return "fail";
  if (scale.status === "not_assessed_runtime_parameter" || zeroPoint.status === "not_assessed_runtime_parameter") {
    return "not_assessed_runtime_parameter_values";
  }
  return "not_assessed_missing_parameter";
}

function decodeOnnxQuantParameter(tensor, name, _kind, declaredTensor = null) {
  if (!name) return { status: "not_assessed_missing_parameter", values: [], dtype: "UNKNOWN", shape: [], shapeDeclared: false, declaredCount: null };
  if (!tensor && declaredTensor) {
    const shapeDeclared = declaredTensor.shapeDeclared === true;
    const declaredCount = tensorShapeKnown(declaredTensor.shape, shapeDeclared)
      ? shapeElementCount(declaredTensor.shape, shapeDeclared) : null;
    return {
      status: "not_assessed_runtime_parameter",
      values: [],
      dtype: declaredTensor.dtype || "UNKNOWN",
      shape: declaredTensor.shape || [],
      shapeDeclared,
      declaredCount,
    };
  }
  if (!tensor) return { status: "not_assessed_missing_parameter", values: [], dtype: "UNKNOWN", shape: [], shapeDeclared: false, declaredCount: null };
  if (isExternalInitializer(tensor) && tensor.externalPayloadVerified !== true) return { status: "not_assessed_external_data", values: [], dtype: tensor.dtype, shape: tensor.shape || [], shapeDeclared: true, declaredCount: onnxInitializerElementCount(tensor) };
  const expected = onnxInitializerElementCount(tensor);
  if (!Number.isSafeInteger(expected) || expected < 0) return { status: "not_assessed_invalid_cardinality", values: [], dtype: tensor.dtype, shape: tensor.shape || [], shapeDeclared: true, declaredCount: null };
  if (expected > MAX_ONNX_QUANT_PARAMETER_ELEMENTS) return { status: "not_assessed_parameter_limit", values: [], dtype: tensor.dtype, shape: tensor.shape || [], shapeDeclared: true, declaredCount: expected };
  const values = [];
  const decoded = forEachOnnxInitializerValue(tensor, (value) => values.push(typeof value === "bigint" ? Number(value) : Number(value)));
  if (!decoded.ok) return { status: "not_assessed_decode_error", values: [], dtype: tensor.dtype, shape: tensor.shape || [], shapeDeclared: true, declaredCount: expected, reason: decoded.reason };
  if (expected > 0 && values.length !== expected) return { status: "not_assessed_element_count_mismatch", values: [], dtype: tensor.dtype, shape: tensor.shape || [], shapeDeclared: true, declaredCount: expected };
  return { status: "assessed", values, dtype: tensor.dtype, shape: tensor.shape || [], shapeDeclared: true, declaredCount: values.length };
}

function assessOnnxQuantCardinality(tensor, scaleShape, scaleCount, parameterization, axis, blockSize) {
  if (!scaleCount) return { status: "not_assessed_missing_scale", reason: "scale_parameter_not_decoded" };
  if (parameterization === "per_tensor") return { status: "pass", reason: "scalar_scale" };
  const shape = tensor?.shape || [];
  if (!tensorShapeKnown(shape, tensor?.shapeDeclared)) return { status: "not_assessed_unknown_tensor_shape", reason: "tensor_shape_is_not_fully_known" };
  if (!Number.isInteger(axis) || axis < 0 || axis >= shape.length) return { status: "fail", reason: "quantization_axis_out_of_range" };
  if (parameterization === "per_axis") {
    return scaleCount === shape[axis]
      ? { status: "pass", reason: `scale_count_matches_axis_${axis}` }
      : { status: "fail", reason: `scale_count_${scaleCount}_does_not_match_axis_${axis}_dimension_${shape[axis]}` };
  }
  if (!Array.isArray(scaleShape) || scaleShape.length !== shape.length || !(blockSize > 0)) {
    return { status: "fail", reason: "blocked_quantization_requires_rank_matched_scale_shape_and_positive_block_size" };
  }
  const expectedAxis = Math.ceil(shape[axis] / blockSize);
  const matches = scaleShape.every((dim, index) => Number(dim) === (index === axis ? expectedAxis : shape[index]));
  return matches
    ? { status: "pass", reason: `blocked_scale_shape_matches_axis_${axis}_block_${blockSize}` }
    : { status: "fail", reason: "blocked_scale_shape_does_not_match_tensor_shape_and_block_size" };
}

function applyOnnxQuantBinding(tensorMap, binding) {
  if (!binding.tensor_name) return;
  const tensor = tensorMap.get(binding.tensor_name);
  if (!tensor) return;
  const prior = tensor.onnxQuantizationBindings || [];
  const structural = ["per_tensor", "per_axis", "blocked"].includes(binding.parameterization)
    && binding.cardinality_status !== "fail";
  const comparable = binding.scale_values.length > 0 && ["pass", "fail"].includes(binding.status);
  const previous = prior.find((item) => ["per_tensor", "per_axis", "blocked"].includes(item.parameterization));
  const structuralConflict = previous && structural && (previous.axis !== binding.axis
    || previous.block_size !== binding.block_size
    || previous.parameterization !== binding.parameterization);
  const numericalConflict = previous?.scale_values?.length > 0 && comparable && (!sameNumericArray(previous.scale_values, binding.scale_values)
    || !sameNumericArray(previous.zero_point_values, binding.zero_point_values)
  );
  const conflict = structuralConflict || numericalConflict;
  tensor.onnxQuantizationBindings = [...prior, binding];
  tensor.quantizationBindingStatus = conflict ? "conflict" : binding.status;
  if (conflict) return;
  if ((comparable || structural) && !tensor.quantizationParameterization) {
    if (comparable) {
      tensor.quantScaleValues = [...binding.scale_values];
      tensor.quantZeroPointValues = [...binding.zero_point_values];
    }
    tensor.quantizedDimension = binding.axis ?? 0;
    tensor.quantizationParameterization = binding.parameterization;
    tensor.quantizationAxisSource = binding.axis_source;
    tensor.quantizationBlockSize = binding.block_size;
    tensor.quantizationScaleTensorShape = [...binding.scale_tensor_shape];
    tensor.quantizationZeroPointTensorShape = [...binding.zero_point_tensor_shape];
    tensor.quantizationCardinalityStatus = binding.cardinality_status;
    tensor.quantizationCardinalityDetail = binding.cardinality_detail;
  }
}

function sameNumericArray(left, right) {
  return left.length === right.length && left.every((value, index) => Object.is(Number(value), Number(right[index])));
}

function integerDtypeRange(dtype) {
  if (dtype === "INT2") return [-2, 1];
  if (dtype === "UINT2") return [0, 3];
  if (dtype === "INT4") return [-8, 7];
  if (dtype === "UINT4") return [0, 15];
  if (dtype === "INT8") return [-128, 127];
  if (dtype === "UINT8") return [0, 255];
  if (dtype === "INT16") return [-32768, 32767];
  if (dtype === "UINT16") return [0, 65535];
  if (dtype === "INT32") return [-2147483648, 2147483647];
  if (dtype === "UINT32") return [0, 4294967295];
  return null;
}

function normalizeOnnxDomain(domain) {
  const value = String(domain || "").trim();
  return !value || value === "ai.onnx" ? "ai.onnx" : value;
}

function isStandardOnnxNode(node) {
  return Boolean(node) && normalizeOnnxDomain(node.domain) === "ai.onnx";
}

function analyzeOnnxInitializers(graph, tensorMap, sparseTensorContract) {
  const tensorResults = [];
  const duplicateGroups = new Map();
  const embeddedDuplicateGroups = new Map();
  const sparseDuplicateGroups = new Map();
  const embeddedSparseDuplicateGroups = new Map();
  let availableDuplicateBytes = 0;
  let embeddedDuplicateBytes = 0;
  let sparseDuplicateAnalysisComplete = true;
  let decodedElements = 0;
  let storedWeightValuesDecoded = 0;
  let implicitZeroElements = 0;
  let assessedTensors = 0;
  let unassessedTensors = 0;
  let nanTensors = 0;
  let infTensors = 0;
  let allZeroTensors = 0;
  let nearZeroElements = 0;
  let finiteElements = 0;
  let maxAbs = 0;
  let largeMagnitudeTensors = 0;
  let highSparsityTensors = 0;
  let zeroKernelSliceTensors = 0;
  let zeroKernelSliceCount = 0;
  let eligibleKernelTensors = 0;
  let outputChannelsEvaluated = 0;
  const zeroKernelSliceDetails = [];
  const logicalInitializers = buildOnnxLogicalInitializerIndex(graph);
  const kernelTrackerIndex = buildOnnxKernelTrackerIndex(graph.nodes, logicalInitializers, tensorMap);

  const accumulate = (result) => {
    tensorResults.push(result);
    if (result.status !== "assessed") {
      unassessedTensors += 1;
      return;
    }
    assessedTensors += 1;
    decodedElements += result.elements_scanned;
    storedWeightValuesDecoded += result.stored_weight_values_decoded ?? result.elements_scanned;
    implicitZeroElements += result.implicit_zero_elements || 0;
    nearZeroElements += result.near_zero_elements;
    finiteElements += result.finite_elements;
    maxAbs = Math.max(maxAbs, result.max_abs_value);
    if (result.nan_count > 0) nanTensors += 1;
    if (result.inf_count > 0) infTensors += 1;
    if (result.all_zero) allZeroTensors += 1;
    if (result.max_abs_value > 1e4) largeMagnitudeTensors += 1;
    if (result.sparsity > 0.5) highSparsityTensors += 1;
    if (result.output_channels_evaluated > 0) eligibleKernelTensors += 1;
    outputChannelsEvaluated += result.output_channels_evaluated;
    if (result.zero_kernel_slice_count > 0) {
      zeroKernelSliceTensors += 1;
      zeroKernelSliceCount += result.zero_kernel_slice_count;
      zeroKernelSliceDetails.push(...result.zero_kernel_slice_details);
    }
  };

  for (const tensor of graph.initializers) {
    const result = scanOnnxInitializer(tensor, kernelTrackerIndex.get(tensor.name) || []);
    accumulate(result);

    if (result.status !== "assessed") continue;
    const key = initializerPayloadHashKey(tensor);
    const candidates = duplicateGroups.get(key) || [];
    const duplicateOf = candidates.find((candidate) => sameInitializerPayload(candidate, tensor));
    if (duplicateOf) availableDuplicateBytes += Number(tensor.storedDataBytes || 0);
    else {
      candidates.push(tensor);
      duplicateGroups.set(key, candidates);
    }
    if (!isExternalInitializer(tensor)) {
      const embeddedCandidates = embeddedDuplicateGroups.get(key) || [];
      const embeddedDuplicateOf = embeddedCandidates.find((candidate) => sameInitializerPayload(candidate, tensor));
      if (embeddedDuplicateOf) embeddedDuplicateBytes += Number(tensor.storedDataBytes || 0);
      else {
        embeddedCandidates.push(tensor);
        embeddedDuplicateGroups.set(key, embeddedCandidates);
      }
    }
  }

  const sparseRows = new Map((sparseTensorContract?.rows || [])
    .filter((row) => row.tensor_role === "graph_sparse_initializer")
    .map((row) => [row.sparse_tensor_name, row]));
  for (const sparse of graph.sparseInitializers || []) {
    const name = sparse.values?.name || "";
    const contractRow = sparseRows.get(name);
    accumulate(scanOnnxSparseInitializer(sparse, contractRow, kernelTrackerIndex.get(name) || []));
    const components = [sparse.values, sparse.indices].filter(Boolean);
    const payloadAvailable = components.length === 2 && components.every((tensor) => !isExternalInitializer(tensor) || tensor.externalPayloadVerified === true);
    const canonical = contractRow?.status === "pass" && payloadAvailable ? canonicalSparseInitializer(sparse) : null;
    if (!canonical) {
      sparseDuplicateAnalysisComplete = false;
      continue;
    }
    const storageBytes = components.reduce((total, tensor) => total + Number(tensor.storedDataBytes || 0), 0);
    appendCanonicalSparseDuplicateCandidate(sparseDuplicateGroups, canonical, storageBytes);
    if (components.every((tensor) => !isExternalInitializer(tensor))) {
      appendCanonicalSparseDuplicateCandidate(embeddedSparseDuplicateGroups, canonical, storageBytes);
    }
  }
  availableDuplicateBytes += canonicalSparseDuplicateSavings(sparseDuplicateGroups);
  embeddedDuplicateBytes += canonicalSparseDuplicateSavings(embeddedSparseDuplicateGroups);

  const sparseComponents = (graph.sparseInitializers || []).flatMap((sparse) => [sparse.values, sparse.indices].filter(Boolean));
  const allInitializerPayloads = [...graph.initializers, ...sparseComponents];
  const embeddedConstantBytes = allInitializerPayloads.reduce((sumBytes, tensor) => sumBytes + (isExternalInitializer(tensor) ? 0 : Number(tensor.storedDataBytes || 0)), 0);
  const availableConstantBytes = allInitializerPayloads.reduce((sumBytes, tensor) => sumBytes
    + (!isExternalInitializer(tensor) || tensor.externalPayloadVerified === true ? Number(tensor.storedDataBytes || 0) : 0), 0);
  return {
    status: assessedTensors > 0 ? "assessed" : "not_assessed",
    coverage_status: unassessedTensors > 0 ? "partial" : "complete",
    assessed_tensors: assessedTensors,
    unassessed_tensors: unassessedTensors,
    elements_scanned: decodedElements,
    logical_elements_assessed: decodedElements,
    stored_weight_values_decoded: storedWeightValuesDecoded,
    implicit_zero_elements: implicitZeroElements,
    dense_initializer_tensors: graph.initializers.length,
    sparse_initializer_tensors: graph.sparseInitializers.length,
    nan_tensors: assessedTensors ? nanTensors : null,
    inf_tensors: assessedTensors ? infTensors : null,
    all_zero_tensors: assessedTensors ? allZeroTensors : null,
    max_abs_weight: assessedTensors ? maxAbs : null,
    large_magnitude_tensors: assessedTensors ? largeMagnitudeTensors : null,
    mean_sparsity: finiteElements ? nearZeroElements / finiteElements : assessedTensors ? 0 : null,
    high_sparsity_tensors: assessedTensors ? highSparsityTensors : null,
    eligible_kernel_tensors_scanned: assessedTensors ? eligibleKernelTensors : null,
    output_channels_evaluated: assessedTensors ? outputChannelsEvaluated : null,
    zero_kernel_slice_tensors: assessedTensors ? zeroKernelSliceTensors : null,
    zero_kernel_slice_count: assessedTensors ? zeroKernelSliceCount : null,
    zero_kernel_slice_details: assessedTensors ? zeroKernelSliceDetails : null,
    unique_constant_bytes: embeddedConstantBytes - embeddedDuplicateBytes,
    duplicate_constant_bytes: embeddedDuplicateBytes,
    available_unique_constant_bytes: availableConstantBytes - availableDuplicateBytes,
    available_duplicate_constant_bytes: availableDuplicateBytes,
    duplicate_analysis_status: sparseDuplicateAnalysisComplete ? "assessed" : "partial_unassessed_sparse_initializer",
    tensor_results: tensorResults,
  };
}

function scanOnnxInitializer(tensor, kernelTrackers) {
  if (isExternalInitializer(tensor) && tensor.externalPayloadVerified !== true) {
    return { tensor_name: tensor.name, status: "not_assessed", reason: `Initializer external_data payload status is ${tensor.externalPayloadStatus || "not_supplied"}; no values were decoded.` };
  }
  if (tensor.dtype === "STRING") {
    return { tensor_name: tensor.name, status: "not_assessed", reason: "String initializer numerical integrity metrics are not applicable." };
  }
  const expectedElements = onnxInitializerElementCount(tensor);
  const quantTracker = kernelTrackers.find((tracker) => tracker.quant_scales.length) || null;
  let elementsScanned = 0;
  let finiteElements = 0;
  let nearZeroElements = 0;
  let nanCount = 0;
  let infCount = 0;
  let maxAbsValue = 0;
  let allZero = true;
  const decoded = forEachOnnxInitializerValue(tensor, (value, index) => {
    elementsScanned += 1;
    const numeric = typeof value === "bigint" ? Number(value) : Number(value);
    if (Number.isNaN(numeric)) {
      nanCount += 1;
      allZero = false;
      markOnnxKernelValueNonZero(kernelTrackers, index);
      return;
    }
    if (!Number.isFinite(numeric)) {
      infCount += 1;
      allZero = false;
      markOnnxKernelValueNonZero(kernelTrackers, index);
      return;
    }
    finiteElements += 1;
    const quantChannel = quantTracker
      ? Math.floor(index / quantTracker.quant_stride_after_axis) % quantTracker.quant_channel_count
      : 0;
    const zeroPoint = quantTracker?.quant_zero_points[Math.min(quantChannel, quantTracker.quant_zero_points.length - 1)] || 0;
    const scale = quantTracker?.quant_scales[Math.min(quantChannel, quantTracker.quant_scales.length - 1)] || 1;
    const decodedNumeric = quantTracker ? scale * (numeric - zeroPoint) : numeric;
    const abs = Math.abs(decodedNumeric);
    maxAbsValue = Math.max(maxAbsValue, abs);
    const nearZero = abs < 1e-8;
    if (nearZero) nearZeroElements += 1;
    if (decodedNumeric !== 0) allZero = false;
    for (const tracker of kernelTrackers) {
      const channel = Math.floor(index / tracker.stride_after_axis) % tracker.channel_count;
      if (!nearZero) tracker.zero_channels[channel] = false;
    }
  });
  if (!decoded.ok) return { tensor_name: tensor.name, status: "not_assessed", reason: decoded.reason };
  if (expectedElements > 0 && elementsScanned !== expectedElements) {
    return {
      tensor_name: tensor.name,
      status: "not_assessed",
      reason: `TensorProto element-count mismatch: shape requires ${expectedElements}, payload decodes to ${elementsScanned}.`,
      expected_elements: expectedElements,
      decoded_elements: elementsScanned,
    };
  }
  const zeroKernelSliceDetails = buildOnnxZeroKernelSliceDetails(tensor, kernelTrackers);
  return {
    tensor_name: tensor.name,
    dtype: tensor.dtype,
    shape: tensor.shape,
    status: "assessed",
    storage_kind: "tensor_proto",
    elements_scanned: elementsScanned,
    stored_weight_values_decoded: elementsScanned,
    implicit_zero_elements: 0,
    finite_elements: finiteElements,
    nan_count: nanCount,
    inf_count: infCount,
    all_zero: elementsScanned > 0 && allZero,
    near_zero_elements: nearZeroElements,
    sparsity: finiteElements ? nearZeroElements / finiteElements : 0,
    max_abs_value: maxAbsValue,
    output_channels_evaluated: kernelTrackers.reduce((count, tracker) => count + tracker.channel_count, 0),
    zero_kernel_slice_count: zeroKernelSliceDetails.reduce((count, detail) => count + detail.channel_count, 0),
    zero_kernel_slice_details: zeroKernelSliceDetails,
  };
}

function buildOnnxZeroKernelSliceDetails(tensor, kernelTrackers) {
  return kernelTrackers.flatMap((tracker) => {
    const channels = tracker.zero_channels.flatMap((isZero, channel) => isZero ? [channel] : []);
    return channels.length ? [{
      tensor_name: tensor.name,
      dtype: tensor.dtype,
      shape: tensor.shape,
      tensor_shape: tensor.shape,
      consumer_op_index: tracker.op.index,
      consumer_op_name: tracker.op.opType,
      output_axis: tracker.output_axis,
      kernel_output_channels: tracker.channel_count,
      channel_count: channels.length,
      channels: channels.slice(0, 256),
      bias_tensor_name: tracker.bias?.name || null,
      bias_dtype: tracker.bias?.dtype || null,
      bias_value_sample: tracker.bias ? channels.slice(0, 16).map((channel) => onnxLogicalInitializerValueAt(tracker.bias, channel)) : [],
      bias_nonzero_for_flagged_channels: tracker.bias
        ? channels.some((channel) => Math.abs(Number(onnxLogicalInitializerValueAt(tracker.bias, channel))) >= 1e-8)
        : false,
      fused_activation: "none; ONNX Conv/Gemm/MatMul does not encode a fused activation attribute",
      functional_inactivity_status: "not_assessed",
      functional_inactivity_reason: "Kernel values alone do not establish functional inactivity; bias, activation, residual paths, and representative outputs were not evaluated.",
    }] : [];
  });
}

function scanOnnxSparseInitializer(sparse, contractRow, kernelTrackers) {
  const values = sparse?.values;
  const name = values?.name || "";
  if (!values) return { tensor_name: name, storage_kind: "sparse_tensor_proto", status: "not_assessed", reason: "SparseTensorProto values TensorProto is missing." };
  if (contractRow?.status !== "pass") {
    return {
      tensor_name: name,
      storage_kind: "sparse_tensor_proto",
      status: "not_assessed",
      reason: contractRow
        ? `SparseTensorProto contract status is ${contractRow.status}: ${(contractRow.reason_codes || []).join(", ") || contractRow.index_content_status || "incomplete payload"}.`
        : "SparseTensorProto contract record was not available for the graph initializer.",
    };
  }
  if (isExternalInitializer(values) && values.externalPayloadVerified !== true) {
    return { tensor_name: name, storage_kind: "sparse_tensor_proto", status: "not_assessed", reason: `Sparse values external_data payload status is ${values.externalPayloadStatus || "not_supplied"}; no stored values were decoded.` };
  }
  if (values.dtype === "STRING") {
    return { tensor_name: name, storage_kind: "sparse_tensor_proto", status: "not_assessed", reason: "String sparse initializer numerical integrity metrics are not applicable." };
  }
  const linearIndices = decodeSparseLinearIndices(sparse);
  if (!linearIndices.ok) return { tensor_name: name, storage_kind: "sparse_tensor_proto", status: "not_assessed", reason: linearIndices.reason };
  const denseElements = shapeElementCount(sparse.dims || [], true);
  const nnz = onnxInitializerElementCount(values);
  if (!Number.isSafeInteger(denseElements) || denseElements < nnz) {
    return { tensor_name: name, storage_kind: "sparse_tensor_proto", status: "not_assessed", reason: "Sparse logical cardinality is invalid or smaller than NNZ." };
  }
  let storedValuesDecoded = 0;
  let finiteStoredValues = 0;
  let nanCount = 0;
  let infCount = 0;
  const quantTracker = kernelTrackers.find((tracker) => tracker.quant_scales.length) || null;
  const implicitZeros = denseElements - nnz;
  const implicitEffect = assessSparseImplicitValues(sparse.dims || [], linearIndices.values, quantTracker, kernelTrackers);
  let nearZeroElements = implicitEffect.near_zero_elements;
  let maxAbsValue = implicitEffect.max_abs_value;
  let allZero = implicitEffect.all_zero;
  const decoded = forEachOnnxInitializerValue(values, (value, storedIndex) => {
    const logicalIndex = linearIndices.values[storedIndex];
    storedValuesDecoded += 1;
    const numeric = typeof value === "bigint" ? Number(value) : Number(value);
    if (Number.isNaN(numeric)) {
      nanCount += 1;
      allZero = false;
      markOnnxKernelValueNonZero(kernelTrackers, logicalIndex);
      return;
    }
    if (!Number.isFinite(numeric)) {
      infCount += 1;
      allZero = false;
      markOnnxKernelValueNonZero(kernelTrackers, logicalIndex);
      return;
    }
    finiteStoredValues += 1;
    const quantChannel = quantTracker
      ? Math.floor(logicalIndex / quantTracker.quant_stride_after_axis) % quantTracker.quant_channel_count
      : 0;
    const zeroPoint = quantTracker?.quant_zero_points[Math.min(quantChannel, quantTracker.quant_zero_points.length - 1)] || 0;
    const scale = quantTracker?.quant_scales[Math.min(quantChannel, quantTracker.quant_scales.length - 1)] || 1;
    const decodedNumeric = quantTracker ? scale * (numeric - zeroPoint) : numeric;
    const abs = Math.abs(decodedNumeric);
    maxAbsValue = Math.max(maxAbsValue, abs);
    const nearZero = abs < 1e-8;
    if (nearZero) nearZeroElements += 1;
    if (decodedNumeric !== 0) allZero = false;
    if (!nearZero) markOnnxKernelValueNonZero(kernelTrackers, logicalIndex);
  });
  if (!decoded.ok) return { tensor_name: name, storage_kind: "sparse_tensor_proto", status: "not_assessed", reason: decoded.reason };
  if (storedValuesDecoded !== nnz || storedValuesDecoded !== linearIndices.values.length) {
    return {
      tensor_name: name,
      storage_kind: "sparse_tensor_proto",
      status: "not_assessed",
      reason: `Sparse values/index cardinality mismatch: NNZ ${nnz}, decoded values ${storedValuesDecoded}, decoded indices ${linearIndices.values.length}.`,
    };
  }
  const finiteElements = implicitZeros + finiteStoredValues;
  const logical = { name, dtype: values.dtype, shape: [...(sparse.dims || [])] };
  const zeroKernelSliceDetails = buildOnnxZeroKernelSliceDetails(logical, kernelTrackers);
  return {
    tensor_name: name,
    dtype: values.dtype,
    shape: [...(sparse.dims || [])],
    status: "assessed",
    storage_kind: "sparse_tensor_proto",
    elements_scanned: denseElements,
    stored_weight_values_decoded: storedValuesDecoded,
    implicit_zero_elements: implicitZeros,
    finite_elements: finiteElements,
    nan_count: nanCount,
    inf_count: infCount,
    all_zero: denseElements > 0 && allZero,
    near_zero_elements: nearZeroElements,
    sparsity: finiteElements ? nearZeroElements / finiteElements : 0,
    max_abs_value: maxAbsValue,
    output_channels_evaluated: kernelTrackers.reduce((count, tracker) => count + tracker.channel_count, 0),
    zero_kernel_slice_count: zeroKernelSliceDetails.reduce((count, detail) => count + detail.channel_count, 0),
    zero_kernel_slice_details: zeroKernelSliceDetails,
    method: "Exact SparseTensorProto logical scan: stored values were decoded at validated row-major indices and every absent logical element was counted as zero without allocating a dense tensor.",
  };
}

function markOnnxKernelValueNonZero(trackers, index) {
  for (const tracker of trackers) {
    const channel = Math.floor(index / tracker.stride_after_axis) % tracker.channel_count;
    tracker.zero_channels[channel] = false;
  }
}

function assessSparseImplicitValues(shape, storedLinearIndices, quantTracker, kernelTrackers) {
  const denseElements = shapeElementCount(shape, true);
  const implicitElements = denseElements - storedLinearIndices.length;
  if (!quantTracker?.quant_scales?.length) {
    return { near_zero_elements: implicitElements, max_abs_value: 0, all_zero: true };
  }
  const qAxis = quantTracker.quant_axis;
  const qChannels = qAxis == null ? 1 : Math.max(1, Number(shape[qAxis]) || 1);
  const qStride = qAxis == null ? denseElements : quantTracker.quant_stride_after_axis;
  const storedByQ = Array(qChannels).fill(0);
  for (const index of storedLinearIndices) {
    const channel = qAxis == null ? 0 : Math.floor(index / qStride) % qChannels;
    storedByQ[channel] += 1;
  }
  const logicalPerQ = qAxis == null ? denseElements : denseElements / qChannels;
  let nearZeroElements = 0;
  let maxAbsValue = 0;
  let allZero = true;
  const nonZeroImplicitQ = new Set();
  for (let channel = 0; channel < qChannels; channel += 1) {
    const absent = logicalPerQ - storedByQ[channel];
    if (!(absent > 0)) continue;
    const scale = Number(quantTracker.quant_scales[Math.min(channel, quantTracker.quant_scales.length - 1)] ?? 1);
    const zeroPoint = Number(quantTracker.quant_zero_points[Math.min(channel, Math.max(0, quantTracker.quant_zero_points.length - 1))] ?? 0);
    const decodedZeroCode = scale * (0 - zeroPoint);
    const abs = Math.abs(decodedZeroCode);
    maxAbsValue = Math.max(maxAbsValue, abs);
    if (abs < 1e-8) nearZeroElements += absent;
    else {
      allZero = false;
      nonZeroImplicitQ.add(channel);
    }
  }
  if (nonZeroImplicitQ.size) markSparseImplicitKernelChannelsNonZero(shape, storedLinearIndices, qAxis, nonZeroImplicitQ, kernelTrackers);
  return { near_zero_elements: nearZeroElements, max_abs_value: maxAbsValue, all_zero: allZero };
}

function markSparseImplicitKernelChannelsNonZero(shape, storedLinearIndices, qAxis, nonZeroImplicitQ, kernelTrackers) {
  for (const tracker of kernelTrackers) {
    const outputAxis = tracker.output_axis;
    const outputChannels = tracker.channel_count;
    const outputStride = tracker.stride_after_axis;
    const outputSliceElements = shapeElementCount(shape, true) / outputChannels;
    const storedByOutput = Array(outputChannels).fill(0);
    for (const index of storedLinearIndices) storedByOutput[Math.floor(index / outputStride) % outputChannels] += 1;
    if (qAxis == null) {
      for (let output = 0; output < outputChannels; output += 1) {
        if (storedByOutput[output] < outputSliceElements) tracker.zero_channels[output] = false;
      }
      continue;
    }
    if (qAxis === outputAxis) {
      for (const output of nonZeroImplicitQ) {
        if (storedByOutput[output] < outputSliceElements) tracker.zero_channels[output] = false;
      }
      continue;
    }
    const qChannels = Math.max(1, Number(shape[qAxis]) || 1);
    const qStride = tracker.quant_stride_after_axis;
    const pairCapacity = shapeElementCount(shape, true) / (outputChannels * qChannels);
    const pairCounts = new Map();
    for (const index of storedLinearIndices) {
      const output = Math.floor(index / outputStride) % outputChannels;
      const quant = Math.floor(index / qStride) % qChannels;
      if (!nonZeroImplicitQ.has(quant)) continue;
      const key = output * qChannels + quant;
      pairCounts.set(key, (pairCounts.get(key) || 0) + 1);
    }
    const fullyStoredNonZeroQByOutput = Array(outputChannels).fill(0);
    for (const [key, count] of pairCounts) {
      if (count === pairCapacity) fullyStoredNonZeroQByOutput[Math.floor(key / qChannels)] += 1;
    }
    for (let output = 0; output < outputChannels; output += 1) {
      if (fullyStoredNonZeroQByOutput[output] < nonZeroImplicitQ.size) tracker.zero_channels[output] = false;
    }
  }
}

function buildOnnxLogicalInitializerIndex(graph) {
  const byName = new Map();
  for (const tensor of graph.initializers || []) {
    if (tensor.name) byName.set(tensor.name, { kind: "dense", name: tensor.name, dtype: tensor.dtype, shape: tensor.shape || [], tensor });
  }
  for (const sparse of graph.sparseInitializers || []) {
    const name = sparse.values?.name || "";
    if (name) byName.set(name, { kind: "sparse", name, dtype: sparse.values?.dtype || "UNKNOWN", shape: sparse.dims || [], tensor: sparse.values, sparse });
  }
  return byName;
}

function buildOnnxKernelTrackerIndex(nodes, initializerByName, tensorMap) {
  const trackersByTensor = new Map();
  const producerByOutput = new Map();
  for (const node of nodes) for (const output of node.outputs || []) if (output) producerByOutput.set(output, node);
  for (const [index, op] of nodes.entries()) {
    if (!isStandardOnnxNode(op)) continue;
    const kernelInput = ["QLinearConv", "QLinearMatMul"].includes(op.opType) ? 3 : 1;
    const kernelName = op.inputs?.[kernelInput];
    const dequantProducer = producerByOutput.get(kernelName);
    const tensorName = isStandardOnnxNode(dequantProducer) && dequantProducer?.opType === "DequantizeLinear" ? dequantProducer.inputs?.[0] : kernelName;
    const initializer = initializerByName.get(tensorName);
    if (!initializer) continue;
    const shape = initializer.shape || [];
    let outputAxis = null;
    if (["Conv", "QLinearConv", "ConvInteger"].includes(op.opType) && shape.length >= 2) outputAxis = 0;
    if (op.opType === "Gemm" && shape.length === 2) outputAxis = attrInt(op, "transB", 0) ? 0 : 1;
    if (["MatMul", "QLinearMatMul", "MatMulInteger"].includes(op.opType) && shape.length >= 2) outputAxis = shape.length - 1;
    if (outputAxis == null || !(shape[outputAxis] > 0)) continue;
    const bound = tensorMap?.get(initializer.name) || {};
    const quantAxis = Number.isInteger(bound.quantizedDimension) ? bound.quantizedDimension : outputAxis;
    const quantShapeAxis = quantAxis >= 0 && quantAxis < shape.length ? quantAxis : outputAxis;
    const strideAfterAxis = shapeElementCount(shape.slice(outputAxis + 1), true);
    const quantStrideAfterAxis = quantShapeAxis == null ? shapeElementCount(shape, true) : shapeElementCount(shape.slice(quantShapeAxis + 1), true);
    if (strideAfterAxis == null || quantStrideAfterAxis == null) continue;
    const biasInput = op.opType === "QLinearConv" ? 8 : ["Conv", "Gemm"].includes(op.opType) ? 2 : null;
    const trackers = trackersByTensor.get(tensorName) || [];
    trackers.push({
      op: { ...op, index },
      bias: biasInput == null ? null : initializerByName.get(op.inputs?.[biasInput]) || null,
      output_axis: outputAxis,
      channel_count: shape[outputAxis],
      stride_after_axis: strideAfterAxis,
      zero_channels: Array(shape[outputAxis]).fill(true),
      quant_axis: bound.quantizationBindingStatus === "pass" && (bound.quantScaleValues || []).length > 1 ? quantShapeAxis : null,
      quant_scales: bound.quantizationBindingStatus === "pass" ? bound.quantScaleValues || [] : [],
      quant_zero_points: bound.quantizationBindingStatus === "pass" ? bound.quantZeroPointValues || [] : [],
      quant_channel_count: quantShapeAxis == null ? 1 : Math.max(1, shape[quantShapeAxis] || 1),
      quant_stride_after_axis: quantStrideAfterAxis,
    });
    trackersByTensor.set(tensorName, trackers);
  }
  return trackersByTensor;
}

function decodeSparseLinearIndices(sparse) {
  const values = sparse?.values;
  const indices = sparse?.indices;
  const dims = [...(sparse?.dims || [])].map(Number);
  const nnz = onnxInitializerElementCount(values);
  const denseElements = shapeElementCount(dims, true);
  if (!indices || indices.staticValuesComplete !== true) return { ok: false, values: [], reason: "Sparse index payload is not completely decoded." };
  if (!Number.isSafeInteger(nnz) || !Number.isSafeInteger(denseElements) || nnz < 0 || nnz > denseElements) return { ok: false, values: [], reason: "Sparse NNZ or logical dense cardinality is invalid." };
  const source = indices.staticValues || [];
  let linear;
  if (indices.shape?.length === 1 && Number(indices.shape[0]) === nnz) {
    linear = source.map(Number);
  } else if (indices.shape?.length === 2 && Number(indices.shape[0]) === nnz && Number(indices.shape[1]) === dims.length) {
    if (source.length !== nnz * dims.length) return { ok: false, values: [], reason: "Sparse coordinate-index payload cardinality does not match [NNZ, rank]." };
    linear = [];
    for (let row = 0; row < nnz; row += 1) {
      let offset = 0;
      for (let axis = 0; axis < dims.length; axis += 1) {
        const coordinate = Number(source[row * dims.length + axis]);
        if (!Number.isSafeInteger(coordinate) || coordinate < 0 || coordinate >= dims[axis]) return { ok: false, values: [], reason: "Sparse coordinate index is outside the logical dense shape." };
        if (offset > Math.floor((Number.MAX_SAFE_INTEGER - coordinate) / Math.max(1, dims[axis]))) return { ok: false, values: [], reason: "Sparse coordinate linearization exceeds the safe-integer range." };
        offset = offset * dims[axis] + coordinate;
      }
      linear.push(offset);
    }
  } else {
    return { ok: false, values: [], reason: "Sparse index tensor shape is neither [NNZ] nor [NNZ, rank]." };
  }
  if (linear.length !== nnz) return { ok: false, values: [], reason: "Sparse index payload cardinality does not match NNZ." };
  for (let index = 0; index < linear.length; index += 1) {
    const value = linear[index];
    if (!Number.isSafeInteger(value) || value < 0 || value >= denseElements) return { ok: false, values: [], reason: "Sparse linear index is outside the logical dense tensor." };
    if (index > 0 && value <= linear[index - 1]) return { ok: false, values: [], reason: "Sparse indices are not strictly ascending and unique." };
  }
  return { ok: true, values: linear, reason: "" };
}

function onnxLogicalInitializerValueAt(initializer, targetIndex) {
  if (!initializer) return null;
  if (initializer.kind === "dense") return onnxInitializerValueAt(initializer.tensor, targetIndex);
  if (initializer.kind !== "sparse") return null;
  const decoded = decodeSparseLinearIndices(initializer.sparse);
  if (!decoded.ok) return null;
  let low = 0;
  let high = decoded.values.length - 1;
  while (low <= high) {
    const middle = Math.floor((low + high) / 2);
    const index = decoded.values[middle];
    if (index === targetIndex) return onnxInitializerValueAt(initializer.tensor, middle);
    if (index < targetIndex) low = middle + 1;
    else high = middle - 1;
  }
  return 0;
}

function onnxInitializerValueAt(tensor, targetIndex) {
  let found = null;
  const decoded = forEachOnnxInitializerValue(tensor, (value, index) => {
    if (index === targetIndex) found = typeof value === "bigint" ? Number(value) : Number(value);
  });
  return decoded.ok ? found : null;
}

function summarizeConditionIntegerInitializer(tensor, payloadAvailable) {
  const integerDtypes = new Set(["BOOL", "INT2", "UINT2", "INT4", "UINT4", "INT8", "UINT8", "INT16", "UINT16", "INT32", "UINT32", "INT64", "UINT64"]);
  const incomplete = (status) => ({ status, complete: false, values: [], exactComplete: false, exactDecimals: [] });
  if (!integerDtypes.has(tensor?.dtype)) return incomplete("not_applicable_non_integer_initializer");
  if (!payloadAvailable) return incomplete("not_assessed_payload_unavailable");
  const expected = onnxInitializerElementCount(tensor);
  if (!Number.isSafeInteger(expected) || expected < 0) return incomplete("not_assessed_invalid_cardinality");
  if (expected > MAX_ONNX_EP_CONDITION_INTEGER_ELEMENTS) return incomplete("not_assessed_element_limit");
  const values = [];
  const exactDecimals = [];
  let unsafeInteger = false;
  const decoded = forEachOnnxInitializerValue(tensor, (value) => {
    if (typeof value === "bigint") {
      exactDecimals.push(value.toString());
      const exact = safeBigIntNumber(value);
      if (exact == null) unsafeInteger = true;
      else values.push(exact);
      return;
    }
    if (!Number.isSafeInteger(value)) {
      unsafeInteger = true;
      return;
    }
    values.push(value);
    exactDecimals.push(String(value));
  });
  if (!decoded.ok) return incomplete("not_assessed_decode_failed");
  const exactComplete = exactDecimals.length === decoded.count && decoded.count === expected;
  if (unsafeInteger || values.length !== decoded.count || decoded.count !== expected) {
    return {
      status: unsafeInteger ? "not_assessed_outside_safe_integer" : "not_assessed_cardinality_mismatch",
      complete: false,
      values: [],
      exactComplete,
      exactDecimals: exactComplete ? exactDecimals : [],
    };
  }
  return { status: "complete", complete: true, values, exactComplete, exactDecimals };
}

function summarizeStaticTensor(tensor, payloadAvailable) {
  if (!tensor) {
    return { status: "not_applicable_non_numeric_tensor", complete: false, values: [] };
  }
  if (!payloadAvailable) return { status: "not_assessed_payload_unavailable", complete: false, values: [] };
  const expected = onnxInitializerElementCount(tensor);
  if (!Number.isSafeInteger(expected) || expected < 0) return { status: "not_assessed_invalid_cardinality", complete: false, values: [] };
  if (expected > MAX_ONNX_EP_CONDITION_INTEGER_ELEMENTS) return { status: "not_assessed_element_limit", complete: false, values: [] };
  if (tensor.dtype === "STRING") {
    const values = Array.isArray(tensor.stringValues) ? tensor.stringValues : [];
    return values.length === expected
      ? { status: "complete", complete: true, values: [...values] }
      : { status: "not_assessed_cardinality_mismatch", complete: false, values: [] };
  }
  if (!(dtypeStorageBits(tensor.dtype) > 0)) {
    return { status: "not_applicable_non_numeric_tensor", complete: false, values: [] };
  }
  const values = [];
  const canonicalTexts = [];
  let unsafe = false;
  const decoded = forEachOnnxInitializerValue(tensor, (value) => {
    const numeric = typeof value === "bigint" ? safeBigIntNumber(value) : Number(value);
    if (numeric == null) {
      unsafe = true;
      canonicalTexts.push(typeof value === "bigint" ? value.toString() : String(value));
    } else {
      canonicalTexts.push(canonicalOnnxFloatText(numeric));
      if (!Number.isFinite(numeric)) unsafe = true;
      else values.push(numeric);
    }
  });
  if (!decoded.ok) return { status: "not_assessed_decode_failed", complete: false, values: [] };
  if (unsafe || values.length !== decoded.count || decoded.count !== expected) {
    const canonicalTextComplete = canonicalTexts.length === decoded.count && decoded.count === expected;
    return {
      status: unsafe ? "complete_canonical_text_only_non_finite_or_unsafe_value" : "not_assessed_cardinality_mismatch",
      complete: false,
      values: [],
      canonicalTextComplete,
      canonicalTexts: canonicalTextComplete ? canonicalTexts : [],
    };
  }
  return { status: "complete", complete: true, values, canonicalTextComplete: false, canonicalTexts: [] };
}

function jsonSafeStaticValues(values) {
  return Array.isArray(values) ? values.map((value) => Object.is(value, -0) ? 0 : value) : [];
}

function staticNegativeZeroIndices(values) {
  if (!Array.isArray(values)) return [];
  const indices = [];
  values.forEach((value, index) => {
    if (Object.is(value, -0)) indices.push(index);
  });
  return indices;
}

function forEachOnnxInitializerValue(tensor, visitor) {
  if (tensor.rawData) {
    const expected = shapeElementCount(tensor.shape, true);
    const expectedBytes = dtypePayloadBytes(tensor.dtype, expected);
    if (expectedBytes == null) return { ok: false, reason: `Unsupported raw_data dtype ${tensor.dtype}.` };
    if (tensor.rawData.byteLength !== expectedBytes) return { ok: false, reason: `raw_data byte length ${tensor.rawData.byteLength} does not match ${expected} ${tensor.dtype} element(s), which require ${expectedBytes} byte(s).` };
    const bits = dtypeStorageBits(tensor.dtype);
    if (bits < 8) {
      for (let index = 0; index < expected; index += 1) {
        const bitOffset = index * bits;
        const packed = tensor.rawData[Math.floor(bitOffset / 8)];
        const code = packed >> (bitOffset % 8) & ((1 << bits) - 1);
        visitor(decodePackedCode(code, tensor.dtype), index);
      }
      return { ok: true, count: expected };
    }
    const bytesPerElement = dtypeBytesExact(tensor.dtype);
    if (!(bytesPerElement > 0)) return { ok: false, reason: `Unsupported fixed-width raw_data dtype ${tensor.dtype}.` };
    const view = new DataView(tensor.rawData.buffer, tensor.rawData.byteOffset, tensor.rawData.byteLength);
    for (let index = 0; index < expected; index += 1) visitor(readOnnxRawValue(view, index * bytesPerElement, tensor.dtype), index);
    return { ok: true, count: expected };
  }
  if (tensor.typedDecodeError) return { ok: false, reason: tensor.typedDecodeError };
  if (tensor.typedValues.length) {
    tensor.typedValues.forEach(visitor);
    return { ok: true, count: tensor.typedValues.length };
  }
  const expected = shapeElementCount(tensor.shape);
  return expected === 0
    ? { ok: true, count: 0 }
    : { ok: false, reason: "TensorProto has no embedded raw_data or supported typed scalar payload." };
}

function dtypeBytesExact(dtype) {
  const bits = dtypeStorageBits(dtype);
  return bits >= 8 && bits % 8 === 0 ? bits / 8 : 0;
}

function dtypeStorageBits(dtype) {
  return Number(TENSOR_TYPE_BY_NAME.get(dtype)?.bits || 0);
}

function dtypePayloadBytes(dtype, elements) {
  const count = Number(elements);
  const bits = dtypeStorageBits(dtype);
  if (!Number.isSafeInteger(count) || count < 0 || !(bits > 0) || count > Math.floor(Number.MAX_SAFE_INTEGER / bits)) return null;
  const payload = Math.ceil(count * bits / 8);
  return Number.isSafeInteger(payload) ? payload : null;
}

function readOnnxRawValue(view, offset, dtype) {
  switch (dtype) {
    case "FLOAT32": return view.getFloat32(offset, true);
    case "FLOAT64": return view.getFloat64(offset, true);
    case "FLOAT16": return float16ToNumber(view.getUint16(offset, true));
    case "BFLOAT16": return bfloat16ToNumber(view.getUint16(offset, true));
    case "INT8": return view.getInt8(offset);
    case "UINT8": return view.getUint8(offset);
    case "INT16": return view.getInt16(offset, true);
    case "UINT16": return view.getUint16(offset, true);
    case "INT32": return view.getInt32(offset, true);
    case "UINT32": return view.getUint32(offset, true);
    case "INT64": return view.getBigInt64(offset, true);
    case "UINT64": return view.getBigUint64(offset, true);
    case "BOOL": return view.getUint8(offset) ? 1 : 0;
    case "COMPLEX64": return Math.hypot(view.getFloat32(offset, true), view.getFloat32(offset + 4, true));
    case "COMPLEX128": return Math.hypot(view.getFloat64(offset, true), view.getFloat64(offset + 8, true));
    case "FLOAT8E4M3FN":
    case "FLOAT8E4M3FNUZ":
    case "FLOAT8E5M2":
    case "FLOAT8E5M2FNUZ":
    case "FLOAT8E8M0": return decodeTypedBitPattern(view.getUint8(offset), dtype);
    default: return Number.NaN;
  }
}

function decodeTypedBitPattern(raw, dtype) {
  const code = Number(raw) & (dtype === "FLOAT16" || dtype === "BFLOAT16" ? 0xffff : 0xff);
  if (dtype === "FLOAT16") return float16ToNumber(code);
  if (dtype === "BFLOAT16") return bfloat16ToNumber(code);
  if (dtype === "FLOAT8E8M0") return code === 0xff ? Number.NaN : 2 ** (code - 127);
  const config = {
    FLOAT8E4M3FN: { exponentBits: 4, mantissaBits: 3, bias: 7, finiteOnly: true, unsignedZero: false },
    FLOAT8E4M3FNUZ: { exponentBits: 4, mantissaBits: 3, bias: 8, finiteOnly: true, unsignedZero: true },
    FLOAT8E5M2: { exponentBits: 5, mantissaBits: 2, bias: 15, finiteOnly: false, unsignedZero: false },
    FLOAT8E5M2FNUZ: { exponentBits: 5, mantissaBits: 2, bias: 16, finiteOnly: true, unsignedZero: true },
  }[dtype];
  if (!config) return Number.NaN;
  if (config.unsignedZero && code === 0x80) return Number.NaN;
  const sign = code & 0x80 ? -1 : 1;
  const exponentMask = (1 << config.exponentBits) - 1;
  const mantissaMask = (1 << config.mantissaBits) - 1;
  const exponent = code >> config.mantissaBits & exponentMask;
  const mantissa = code & mantissaMask;
  if (!config.finiteOnly && exponent === exponentMask) return mantissa === 0 ? sign * Number.POSITIVE_INFINITY : Number.NaN;
  if (!config.unsignedZero && config.finiteOnly && exponent === exponentMask && mantissa === mantissaMask) return Number.NaN;
  if (exponent === 0) {
    if (mantissa === 0) return sign < 0 ? -0 : 0;
    return sign * 2 ** (1 - config.bias) * (mantissa / (1 << config.mantissaBits));
  }
  return sign * 2 ** (exponent - config.bias) * (1 + mantissa / (1 << config.mantissaBits));
}

function decodePackedCode(code, dtype) {
  if (dtype === "INT4") return code & 0x08 ? code - 0x10 : code;
  if (dtype === "INT2") return code & 0x02 ? code - 0x04 : code;
  if (dtype === "FLOAT4E2M1") {
    const sign = code & 0x08 ? -1 : 1;
    const exponent = code >> 1 & 0x03;
    const mantissa = code & 0x01;
    if (exponent === 0) return mantissa ? sign * 0.5 : sign < 0 ? -0 : 0;
    return sign * 2 ** (exponent - 1) * (1 + mantissa / 2);
  }
  return code;
}

function float16ToNumber(bits) {
  const sign = (bits & 0x8000) ? -1 : 1;
  const exponent = (bits >> 10) & 0x1f;
  const fraction = bits & 0x03ff;
  if (exponent === 0x1f) return fraction ? Number.NaN : sign * Number.POSITIVE_INFINITY;
  if (exponent === 0) return sign * 2 ** -14 * (fraction / 1024);
  return sign * 2 ** (exponent - 15) * (1 + fraction / 1024);
}

function bfloat16ToNumber(bits) {
  const buffer = new ArrayBuffer(4);
  const view = new DataView(buffer);
  view.setUint32(0, bits << 16, true);
  return view.getFloat32(0, true);
}

function initializerPayloadHashKey(tensor) {
  let hash = 0xcbf29ce484222325n;
  const updateByte = (byte) => {
    hash ^= BigInt(byte & 0xff);
    hash = BigInt.asUintN(64, hash * 0x100000001b3n);
  };
  if (tensor.rawData) tensor.rawData.forEach(updateByte);
  else {
    for (const value of tensor.typedValues) {
      const text = `${typeof value}:${String(value)};`;
      for (let index = 0; index < text.length; index += 1) updateByte(text.charCodeAt(index));
    }
  }
  return `${tensor.dtype}|${(tensor.shape || []).join("x")}|${tensor.storedDataBytes || 0}|${hash.toString(16).padStart(16, "0")}`;
}

function sameInitializerPayload(left, right) {
  if (left.dtype !== right.dtype || String(left.shape) !== String(right.shape)) return false;
  if (left.rawData || right.rawData) {
    if (!(left.rawData && right.rawData) || left.rawData.byteLength !== right.rawData.byteLength) return false;
    for (let index = 0; index < left.rawData.byteLength; index += 1) if (left.rawData[index] !== right.rawData[index]) return false;
    return true;
  }
  if (left.typedValues.length !== right.typedValues.length) return false;
  return left.typedValues.every((value, index) => Object.is(value, right.typedValues[index]) || value === right.typedValues[index]);
}

function canonicalSparseInitializer(sparse) {
  const decoded = decodeSparseLinearIndices(sparse);
  if (!decoded.ok || !sparse?.values) return null;
  let indexHash = 0xcbf29ce484222325n;
  for (const value of decoded.values) {
    const text = `${value};`;
    for (let index = 0; index < text.length; index += 1) {
      indexHash ^= BigInt(text.charCodeAt(index) & 0xff);
      indexHash = BigInt.asUintN(64, indexHash * 0x100000001b3n);
    }
  }
  const dims = [...(sparse.dims || [])].map(Number);
  return {
    key: `${dims.join("x")}|${initializerPayloadHashKey(sparse.values)}|${indexHash.toString(16).padStart(16, "0")}`,
    dims,
    linear_indices: decoded.values,
    values: sparse.values,
  };
}

function sameCanonicalSparseInitializer(left, right) {
  return sameShape(left.dims, right.dims)
    && sameShape(left.linear_indices, right.linear_indices)
    && sameInitializerPayload(left.values, right.values);
}

function appendCanonicalSparseDuplicateCandidate(groups, canonical, storageBytes) {
  const equivalenceClasses = groups.get(canonical.key) || [];
  const existing = equivalenceClasses.find((entry) => sameCanonicalSparseInitializer(entry.canonical, canonical));
  if (existing) {
    existing.total_bytes += storageBytes;
    existing.minimum_bytes = Math.min(existing.minimum_bytes, storageBytes);
    existing.count += 1;
  } else {
    equivalenceClasses.push({ canonical, total_bytes: storageBytes, minimum_bytes: storageBytes, count: 1 });
    groups.set(canonical.key, equivalenceClasses);
  }
}

function canonicalSparseDuplicateSavings(groups) {
  let savings = 0;
  for (const equivalenceClasses of groups.values()) {
    for (const entry of equivalenceClasses) if (entry.count > 1) savings += entry.total_bytes - entry.minimum_bytes;
  }
  return savings;
}

function buildOnnxSizeBreakdown(bytes, graph, tensors, initializerAnalysis) {
  const constantTensors = tensors.filter((tensor) => tensor.role === "initializer" || tensor.constant_buffer);
  const embeddedConstantTensors = constantTensors.filter((tensor) => Number(tensor.initializer_external_component_count || 0) === 0);
  const externalDataTensorCount = sum(constantTensors.map((tensor) => Number(tensor.initializer_external_component_count || 0)));
  const verifiedExternalComponentCount = sum(constantTensors.map((tensor) => Number(tensor.initializer_verified_external_component_count || 0)));
  const constantBytes = sum(constantTensors.map((tensor) => Number(tensor.initializer_embedded_bytes || 0)));
  const storedElements = sum(constantTensors.map((tensor) => Number(tensor.initializer_embedded_stored_elements || 0)));
  const verifiedExternalBytes = sum(constantTensors.map((tensor) => Number(tensor.initializer_verified_external_bytes || 0)));
  const verifiedExternalElements = sum(constantTensors.map((tensor) => Number(tensor.initializer_verified_external_stored_elements || 0)));
  const availableInitializerBytes = sum(constantTensors.map((tensor) => Number(tensor.initializer_available_bytes || 0)));
  const availableInitializerElements = sum(constantTensors.map((tensor) => Number(tensor.initializer_available_stored_elements || 0)));
  const logicalInitializerElements = sum(constantTensors.map((tensor) => Number(tensor.initializer_elements || 0)));
  const theoreticalFp16Bytes = sum(constantTensors.map((tensor) => Number(tensor.initializer_projected_embedded_fp16_bytes || 0)));
  const theoreticalInt8Bytes = sum(constantTensors.map((tensor) => Number(tensor.initializer_projected_embedded_int8_bytes || 0)));
  const rawDataBytes = sum(constantTensors.map((tensor) => Number(tensor.initializer_raw_data_bytes || 0)));
  const rawZeroByteCount = sum(constantTensors.map((tensor) => Number(tensor.initializer_raw_zero_bytes || 0)));
  const externalCoverageComplete = verifiedExternalComponentCount === externalDataTensorCount;
  const rawByteRatioAssessed = availableInitializerBytes > 0 && externalCoverageComplete && rawDataBytes === availableInitializerBytes;
  const rawZeroBytes = rawByteRatioAssessed ? rawZeroByteCount : null;
  const zeroConstantByteRatio = rawByteRatioAssessed ? rawZeroBytes / availableInitializerBytes : null;
  const zeroByteMetric = rawByteRatioAssessed
    ? { status: "assessed", value: zeroConstantByteRatio, reason: "Exact 0x00 bytes divided by every available initializer raw_data byte; all external references are supplied and verified." }
    : !externalCoverageComplete
      ? { status: "not_assessed_external_data", value: null, reason: "At least one external_data initializer payload is not supplied and verified, so a whole-artifact raw zero-byte ratio is withheld." }
      : availableInitializerBytes === 0
        ? { status: "not_applicable_no_available_initializer_payload", value: null, reason: "The graph contains no non-empty embedded or verified external initializer payload." }
        : { status: "not_assessed_typed_tensor_fields", value: null, reason: "At least one available initializer uses a typed TensorProto field, so exact serialized payload-byte zeros are not reconstructed." };
  return {
    file_size: bytes.byteLength,
    constant_tensor_count: graph.initializers.length + graph.sparseInitializers.length,
    dense_initializer_count: graph.initializers.length,
    sparse_initializer_count: graph.sparseInitializers.length,
    embedded_constant_tensor_count: embeddedConstantTensors.length,
    constant_bytes: constantBytes,
    stored_scalar_elements: storedElements,
    verified_external_payload_bytes: verifiedExternalBytes,
    verified_external_scalar_elements: verifiedExternalElements,
    available_initializer_bytes: availableInitializerBytes,
    available_initializer_scalar_elements: availableInitializerElements,
    logical_initializer_elements: logicalInitializerElements,
    unique_constant_bytes: initializerAnalysis.duplicate_analysis_status === "assessed" ? initializerAnalysis.unique_constant_bytes : null,
    duplicate_constant_bytes: initializerAnalysis.duplicate_analysis_status === "assessed" ? initializerAnalysis.duplicate_constant_bytes : null,
    available_unique_constant_bytes: initializerAnalysis.duplicate_analysis_status === "assessed" ? initializerAnalysis.available_unique_constant_bytes : null,
    available_duplicate_constant_bytes: initializerAnalysis.duplicate_analysis_status === "assessed" ? initializerAnalysis.available_duplicate_constant_bytes : null,
    duplicate_initializer_analysis: {
      status: initializerAnalysis.duplicate_analysis_status,
      value: initializerAnalysis.duplicate_analysis_status === "assessed" ? {
        unique_constant_bytes: initializerAnalysis.unique_constant_bytes,
        duplicate_constant_bytes: initializerAnalysis.duplicate_constant_bytes,
      } : null,
      reason: initializerAnalysis.duplicate_analysis_status === "assessed"
        ? "Dense TensorProto payload equality and canonical SparseTensorProto equality were assessed exactly. Sparse comparison normalizes linear/coordinate indices to strictly ascending row-major linear indices, then compares dims, dtype, and every stored value; hashes only select candidates. For equivalent sparse encodings, unique bytes retain the smallest storage representation and duplicate bytes are total group storage minus that representative."
        : "At least one SparseTensorProto lacked a valid complete values+indices payload, so whole-inventory duplicate and unique byte totals are withheld rather than emitting a partial subtotal.",
    },
    metadata_bytes: null,
    structure_overhead_bytes: Math.max(0, bytes.byteLength - constantBytes),
    float_constant_bytes: sum(constantTensors.map((tensor) => Number(tensor.initializer_embedded_float_bytes || 0))),
    theoretical_fp16_constant_bytes: theoreticalFp16Bytes,
    theoretical_int8_constant_bytes: theoreticalInt8Bytes,
    zero_constant_byte_ratio: zeroConstantByteRatio,
    metrics: {
      metadata_bytes: {
        status: "not_assessed",
        value: null,
        reason: "ONNX protobuf metadata bytes are not separable from graph/schema overhead without a field-preserving serialization pass.",
      },
      theoretical_fp16_constant_bytes: {
        status: "assessed",
        value: theoreticalFp16Bytes,
        reason: "Embedded FLOAT initializer elements x 2 bytes; embedded non-FLOAT initializer payload bytes unchanged. External data excluded.",
      },
      theoretical_int8_constant_bytes: {
        status: "assessed",
        value: theoreticalInt8Bytes,
        reason: "Embedded FLOAT initializer elements x 1 byte; embedded non-FLOAT initializer payload bytes unchanged. Quantization parameters, bias policy, alignment, accuracy, and runtime packing excluded.",
      },
      zero_constant_byte_ratio: zeroByteMetric,
    },
    external_data_tensor_count: externalDataTensorCount,
    detail: "ONNX initializer storage counts conserve dense TensorProto payloads and every SparseTensorProto values/indices component separately. Embedded and independently verified external component bytes are reported apart from the logical dense element cardinality of sparse initializers. Protobuf tag/varint encoding remains part of the exact model file size rather than initializer payload. FP16/INT8 values remain embedded-only scalar-width projections, not converted artifact sizes.",
  };
}

function computeOnnxTensorLiveness(ops, tensors, inputs, outputs) {
  const producer = new Map();
  const lastUse = new Map();
  const outputIds = new Set(outputs.map((tensor) => tensor.index));
  for (const input of inputs) producer.set(input.index, -1);
  for (const op of ops) {
    for (const id of op.outputs || []) {
      if (id >= 0) producer.set(id, op.index);
    }
    for (const id of op.inputs || []) {
      if (id >= 0 && !tensors[id]?.constant_buffer) lastUse.set(id, Math.max(lastUse.get(id) ?? -1, op.index));
    }
  }
  for (const id of outputIds) lastUse.set(id, Math.max(lastUse.get(id) ?? -1, ops.length));
  let peakBytes = 0;
  let peakAtOp = 0;
  let peakAtOpName = "";
  for (let i = 0; i <= ops.length; i++) {
    let live = 0;
    for (const tensor of tensors) {
      if (tensor.constant_buffer) continue;
      const first = producer.get(tensor.index);
      const last = lastUse.get(tensor.index);
      if (first == null || last == null) continue;
      const payload = tensorPayloadBytes(tensor);
      if (payload != null && first <= i && last >= i) live += payload;
    }
    if (live > peakBytes) {
      peakBytes = live;
      peakAtOp = Math.min(i, Math.max(0, ops.length - 1));
      peakAtOpName = ops[peakAtOp]?.name || "";
    }
  }
  const liveValues = tensors.filter((tensor) => !tensor.constant_buffer && producer.has(tensor.index) && lastUse.has(tensor.index));
  const nonDenseValues = liveValues.filter((tensor) => !isDenseTensorValue(tensor)).map((tensor) => ({
    tensor_index: tensor.index,
    tensor_name: tensor.name,
    value_kind: tensor.value_kind || tensor.valueKind || "non_tensor",
    reason: `value kind ${tensor.value_kind || tensor.valueKind || "non_tensor"} has no dense activation-payload projection`,
  }));
  const activationTensors = liveValues.filter(isDenseTensorValue);
  const unassessedTensors = activationTensors.filter((tensor) => tensorPayloadBytes(tensor) == null).map((tensor) => ({
    tensor_index: tensor.index,
    tensor_name: tensor.name,
    reason: !tensorShapeKnown(tensor.shape, tensor.shape_declared) ? "shape is not fully known" : `dtype ${tensor.dtype || "UNKNOWN"} has no supported byte width`,
  }));
  const assessedTensorCount = activationTensors.length - unassessedTensors.length;
  const residualCount = unassessedTensors.length + nonDenseValues.length;
  const status = residualCount === 0 && assessedTensorCount > 0 ? "assessed" : assessedTensorCount > 0 ? "partial" : "not_assessed";
  return {
    status,
    evidence_class: "DERIVED",
    peak_bytes: status === "not_assessed" ? null : peakBytes,
    peak_bytes_status: status === "partial" ? "assessed_tensor_lower_bound" : status,
    peak_at_op: peakAtOp,
    peak_at_op_name: peakAtOpName,
    assessed: status !== "not_assessed",
    assessed_tensor_count: assessedTensorCount,
    unassessed_tensor_count: unassessedTensors.length,
    unknown_activation_tensors: unassessedTensors.length,
    unassessed_tensors: unassessedTensors,
    non_dense_value_count: nonDenseValues.length,
    non_dense_values: nonDenseValues,
    method: "Static ONNX producer-to-last-consumer liveness sweep over dense tensor values. Dense values with unknown shape/dtype and declared non-dense values are excluded separately; either residual makes the emitted dense-tensor subtotal a lower bound rather than being substituted with zero.",
  };
}

function isExternalInitializer(tensor) {
  return Number(tensor?.dataLocation || 0) === 1 || Number(tensor?.externalDataEntries || 0) > 0 || (tensor?.externalData || []).length > 0;
}

function resolveOnnxExternalDataPayloads(tensorPayloadScopes, suppliedFiles) {
  if (!Array.isArray(suppliedFiles)) throw new Error("ONNX external data files must be an array.");
  if (suppliedFiles.length > MAX_ONNX_EXTERNAL_FILE_COUNT) throw new Error(`ONNX external data file count ${suppliedFiles.length} exceeds safety limit ${MAX_ONNX_EXTERNAL_FILE_COUNT}.`);
  const filesByPath = new Map();
  let aggregateBytes = 0;
  for (const item of suppliedFiles) {
    const path = normalizeExternalDataPath(item?.path || item?.name || "");
    if (classifyExternalDataLocation(path) !== "safe_relative_path") throw new Error(`Unsafe supplied ONNX external data path: ${path || "(missing)"}.`);
    if (filesByPath.has(path)) throw new Error(`Duplicate supplied ONNX external data path: ${path}.`);
    const bytes = item?.bytes instanceof Uint8Array
      ? item.bytes
      : item?.bytes instanceof ArrayBuffer ? new Uint8Array(item.bytes) : null;
    if (!bytes) throw new Error(`Supplied ONNX external data ${path} has no byte payload.`);
    if (bytes.byteLength > MAX_ONNX_EXTERNAL_FILE_BYTES) throw new Error(`ONNX external data file ${path} exceeds ${MAX_ONNX_EXTERNAL_FILE_BYTES} bytes.`);
    aggregateBytes += bytes.byteLength;
    if (!Number.isSafeInteger(aggregateBytes) || aggregateBytes > MAX_ONNX_EXTERNAL_AGGREGATE_BYTES) throw new Error(`ONNX external data aggregate exceeds ${MAX_ONNX_EXTERNAL_AGGREGATE_BYTES} bytes.`);
    const sha256 = String(item?.sha256 || "").toLowerCase();
    const sha1 = String(item?.sha1 || "").toLowerCase();
    if (!/^[a-f0-9]{64}$/.test(sha256)) throw new Error(`Supplied ONNX external data ${path} is missing a computed SHA-256.`);
    if (sha1 && !/^[a-f0-9]{40}$/.test(sha1)) throw new Error(`Supplied ONNX external data ${path} has an invalid SHA-1.`);
    filesByPath.set(path, { path, bytes, sha256, sha1, used: false });
  }

  const referenced = (tensorPayloadScopes || []).filter((entry) => isExternalInitializer(entry.tensor));
  for (const { scope, role, tensor } of referenced) {
    tensor.externalScope = scope;
    tensor.externalTensorRole = role;
    tensor.externalEmbeddedPayloadConflict = Number(tensor.storedDataBytes || 0) > 0;
    const reference = parseOnnxExternalDataReference(tensor);
    tensor.externalPayloadStatus = reference.reference_status === "malformed_reference" ? "malformed_reference" : "not_supplied";
    tensor.externalPayloadVerified = false;
    tensor.externalExpectedPayloadBytes = reference.expected_payload_bytes;
    if (reference.reference_status === "malformed_reference") continue;
    const file = filesByPath.get(reference.normalized_location);
    if (!file) continue;
    file.used = true;
    tensor.externalSidecarPath = file.path;
    tensor.externalSidecarBytes = file.bytes.byteLength;
    tensor.externalSidecarSha256 = file.sha256;
    tensor.externalSidecarSha1 = file.sha1;
    const end = reference.length == null ? file.bytes.byteLength : reference.offset + reference.length;
    if (!Number.isSafeInteger(end) || reference.offset > file.bytes.byteLength || end > file.bytes.byteLength) {
      tensor.externalPayloadStatus = "range_out_of_bounds";
      continue;
    }
    const payloadBytes = end - reference.offset;
    if (reference.expected_payload_bytes != null && payloadBytes !== reference.expected_payload_bytes) {
      tensor.externalPayloadStatus = "payload_size_mismatch";
      continue;
    }
    if (reference.checksum) {
      if (!/^[a-f0-9]{40}$/i.test(reference.checksum)) {
        tensor.externalPayloadStatus = "invalid_checksum_declaration";
        continue;
      }
      if (!file.sha1 || file.sha1 !== reference.checksum.toLowerCase()) {
        tensor.externalPayloadStatus = "checksum_mismatch";
        continue;
      }
    }
    tensor.rawData = file.bytes.subarray(reference.offset, end);
    tensor.rawDataBytes = payloadBytes;
    tensor.typedDataBytes = 0;
    tensor.typedElementCount = 0;
    tensor.typedValues = [];
    tensor.storedDataBytes = payloadBytes;
    tensor.externalPayloadStatus = "verified";
    tensor.externalPayloadVerified = true;
  }
  for (const { role, tensor } of tensorPayloadScopes || []) {
    const payloadAvailable = !isExternalInitializer(tensor) || tensor.externalPayloadVerified === true;
    annotateParsedTensorValues(tensor, payloadAvailable, role === "graph_initializer" ? "initializer" : "attribute_tensor");
  }
  const files = [...filesByPath.values()];
  return {
    supplied_file_count: files.length,
    supplied_file_bytes: aggregateBytes,
    used_file_count: files.filter((file) => file.used).length,
    unused_file_count: files.filter((file) => !file.used).length,
    files: files.map(({ path, bytes, sha256, sha1, used }) => ({ path, byte_length: bytes.byteLength, sha256, sha1, used })),
  };
}

function parseOnnxExternalDataReference(tensor) {
  const entries = (tensor.externalData || []).map((entry) => ({ key: String(entry.key || ""), value: String(entry.value || "") }));
  const byKey = new Map();
  const duplicateKeys = [];
  for (const entry of entries) {
    if (byKey.has(entry.key)) duplicateKeys.push(entry.key);
    else byKey.set(entry.key, entry.value);
  }
  const location = byKey.get("location") || "";
  const normalizedLocation = normalizeExternalDataPath(location);
  const checksum = byKey.get("checksum") || "";
  const checksumValid = !checksum || /^[a-f0-9]{40}$/i.test(checksum);
  const offset = parseExternalDataInteger(byKey.get("offset"), 0);
  const length = parseExternalDataInteger(byKey.get("length"), null);
  const pathStatus = classifyExternalDataLocation(location);
  const embeddedPayloadConflict = tensor.externalPayloadVerified === true
    ? Boolean(tensor.externalEmbeddedPayloadConflict)
    : Number(tensor.storedDataBytes || 0) > 0;
  const dataLocationMismatch = Number(tensor.dataLocation || 0) !== 1;
  const rangeEnd = length.value == null || offset.value == null ? null : offset.value + length.value;
  const rangeSafe = rangeEnd == null || Number.isSafeInteger(rangeEnd);
  const expectedPayloadBytes = dtypePayloadBytes(tensor.dtype, checkedShapeElementCount(tensor.shape));
  const malformed = !location || duplicateKeys.length > 0 || pathStatus !== "safe_relative_path"
    || !offset.valid || !length.valid || !checksumValid || embeddedPayloadConflict || dataLocationMismatch || !rangeSafe;
  return {
    tensor_name: tensor.name || "",
    dtype: tensor.dtype || "UNKNOWN",
    shape: tensor.shape || [],
    data_location: Number(tensor.dataLocation || 0),
    entry_count: entries.length,
    entries,
    duplicate_keys: duplicateKeys,
    location,
    normalized_location: normalizedLocation,
    location_status: pathStatus,
    offset: offset.value,
    offset_status: offset.status,
    length: length.value,
    length_status: length.status,
    range_end: rangeSafe ? rangeEnd : null,
    expected_payload_bytes: expectedPayloadBytes,
    checksum,
    checksum_status: checksumValid ? (checksum ? "declared_sha1" : "not_declared") : "invalid_declaration",
    basepath: byKey.get("basepath") || "",
    unknown_keys: [...byKey.keys()].filter((key) => !["location", "offset", "length", "checksum", "basepath"].includes(key)),
    embedded_payload_conflict: embeddedPayloadConflict,
    data_location_mismatch: dataLocationMismatch,
    reference_status: malformed ? "malformed_reference" : "declared_payload_not_supplied",
  };
}

function buildOnnxExternalDataEvidence(tensorPayloadScopes, resolution) {
  const referenced = (tensorPayloadScopes || []).filter((entry) => isExternalInitializer(entry.tensor));
  const sourceTensorDeclarations = referenced.map(({ scope, role, tensor }) => ({
    scope,
    tensor_role: role,
    name: tensor.name || "",
    dtype: tensor.dtype || "UNKNOWN",
    shape: [...(tensor.shape || [])],
    data_location: Number(tensor.dataLocation || 0),
    external_data_entries: Number(tensor.externalDataEntries || 0),
    external_data: (tensor.externalData || []).map((entry) => ({ key: String(entry.key || ""), value: String(entry.value || "") })),
    external_embedded_payload_conflict: tensor.externalEmbeddedPayloadConflict === true,
    external_payload_verified: tensor.externalPayloadVerified === true,
    initializer_bytes: !isExternalInitializer(tensor) || tensor.externalPayloadVerified === true ? Number(tensor.storedDataBytes || 0) : 0,
    initializer_raw_data_bytes: !isExternalInitializer(tensor) || tensor.externalPayloadVerified === true ? Number(tensor.rawDataBytes || 0) : 0,
  }));
  const rows = referenced.map(({ scope, role, tensor }) => {
    const row = parseOnnxExternalDataReference(tensor);
    const payloadStatus = tensor.externalPayloadStatus || "not_supplied";
    const supplied = Boolean(tensor.externalSidecarPath);
    const verified = tensor.externalPayloadVerified === true;
    return {
      ...row,
      scope,
      tensor_role: role,
      reference_status: row.reference_status === "malformed_reference"
        ? "malformed_reference" : verified ? "verified_reference_and_payload" : supplied ? "payload_verification_failed" : "declared_payload_not_supplied",
      payload_status: payloadStatus,
      payload_bytes: verified ? Number(tensor.rawDataBytes || 0) : null,
      sidecar_path: tensor.externalSidecarPath || "",
      sidecar_bytes: supplied ? Number(tensor.externalSidecarBytes || 0) : null,
      sidecar_sha256: tensor.externalSidecarSha256 || "",
      sidecar_sha1: tensor.externalSidecarSha1 || "",
      checksum_status: !row.checksum ? "not_declared"
        : !/^[a-f0-9]{40}$/i.test(row.checksum) ? "invalid_declaration"
          : verified ? "verified" : payloadStatus === "checksum_mismatch" ? "mismatch" : supplied ? "not_verified" : "not_assessed_payload_not_supplied",
    };
  });
  const entryCount = rows.reduce((total, row) => total + row.entry_count, 0);
  const malformedReferenceCount = rows.filter((row) => row.reference_status === "malformed_reference").length;
  const unsafeLocationCount = rows.filter((row) => row.location_status !== "safe_relative_path").length;
  const missingLocationCount = rows.filter((row) => !row.location).length;
  const duplicateKeyCount = rows.reduce((total, row) => total + row.duplicate_keys.length, 0);
  const invalidRangeCount = rows.filter((row) => row.offset_status === "invalid" || row.length_status === "invalid" || (row.length != null && row.range_end == null)).length;
  const invalidChecksumCount = rows.filter((row) => row.checksum_status === "invalid_declaration").length;
  const embeddedPayloadConflictCount = rows.filter((row) => row.embedded_payload_conflict).length;
  const dataLocationMismatchCount = rows.filter((row) => row.data_location_mismatch).length;
  const everyLengthKnown = rows.length > 0 && rows.every((row) => row.length_status === "declared");
  const declaredPayloadBytes = everyLengthKnown
    ? rows.reduce((total, row) => total + Number(row.length || 0), 0)
    : null;
  const suppliedPayloadCount = rows.filter((row) => row.sidecar_path).length;
  const verifiedPayloadCount = rows.filter((row) => row.payload_status === "verified").length;
  const payloadVerificationFailedCount = rows.filter((row) => row.reference_status === "payload_verification_failed").length;
  const rangeOutOfBoundsCount = rows.filter((row) => row.payload_status === "range_out_of_bounds").length;
  const payloadSizeMismatchCount = rows.filter((row) => row.payload_status === "payload_size_mismatch").length;
  const checksumMismatchCount = rows.filter((row) => row.payload_status === "checksum_mismatch").length;
  const verifiedPayloadBytes = rows.reduce((total, row) => total + Number(row.payload_bytes || 0), 0);
  const status = rows.length === 0 ? "assessed_absent"
    : malformedReferenceCount > 0 ? "malformed_reference"
      : payloadVerificationFailedCount > 0 ? "payload_verification_failed"
        : verifiedPayloadCount === rows.length ? "verified_payloads"
          : verifiedPayloadCount > 0 ? "partial_payload_coverage"
            : "not_assessed_payload_not_supplied";
  return {
    schema: "deepbom.onnx_external_data.v1.3",
    status,
    evidence_class: rows.length && verifiedPayloadCount < rows.length ? "OBSERVED/NOT_ASSESSABLE" : rows.length ? "OBSERVED/DERIVED" : "OBSERVED",
    tensor_count: rows.length,
    source_tensor_declaration_count: sourceTensorDeclarations.length,
    source_tensor_declarations: sourceTensorDeclarations,
    entry_count: entryCount,
    malformed_reference_count: malformedReferenceCount,
    unsafe_location_count: unsafeLocationCount,
    missing_location_count: missingLocationCount,
    duplicate_key_count: duplicateKeyCount,
    invalid_range_count: invalidRangeCount,
    invalid_checksum_count: invalidChecksumCount,
    embedded_payload_conflict_count: embeddedPayloadConflictCount,
    data_location_mismatch_count: dataLocationMismatchCount,
    declared_payload_bytes: declaredPayloadBytes,
    supplied_payload_count: suppliedPayloadCount,
    verified_payload_count: verifiedPayloadCount,
    payload_verification_failed_count: payloadVerificationFailedCount,
    range_out_of_bounds_count: rangeOutOfBoundsCount,
    payload_size_mismatch_count: payloadSizeMismatchCount,
    checksum_mismatch_count: checksumMismatchCount,
    verified_payload_bytes: verifiedPayloadBytes,
    supplied_file_count: Number(resolution?.supplied_file_count || 0),
    supplied_file_bytes: Number(resolution?.supplied_file_bytes || 0),
    used_file_count: Number(resolution?.used_file_count || 0),
    unused_file_count: Number(resolution?.unused_file_count || 0),
    supplied_files: resolution?.files || [],
    tensors: rows,
    detail: rows.length === 0
      ? "No TensorProto external_data reference is present."
      : `${rows.length} external TensorProto reference(s) across all parsed graph/function scopes: ${verifiedPayloadCount} payload range(s) verified from ${Number(resolution?.used_file_count || 0)} SHA-bound sidecar file(s), ${payloadVerificationFailedCount} verification failure(s), ${rows.length - suppliedPayloadCount} not supplied.`,
  };
}

function parseExternalDataInteger(value, absentValue) {
  if (value == null || value === "") return { valid: true, value: absentValue, status: "absent" };
  const text = String(value);
  if (!/^(0|[1-9][0-9]*)$/.test(text)) return { valid: false, value: null, status: "invalid" };
  const parsed = BigInt(text);
  if (parsed > BigInt(Number.MAX_SAFE_INTEGER)) return { valid: false, value: null, status: "invalid" };
  return { valid: true, value: Number(parsed), status: "declared" };
}

function classifyExternalDataLocation(location) {
  const value = String(location || "");
  if (!value) return "missing";
  if (value.includes("\0")) return "unsafe_nul";
  const normalized = normalizeExternalDataPath(value);
  if (/^[a-z][a-z0-9+.-]*:/i.test(normalized)) return "unsafe_absolute_or_uri";
  if (normalized.startsWith("/")) return "unsafe_path_escape";
  if (normalized.split("/").some((part) => !part || part === "." || part === "..")) return "unsafe_noncanonical_segment";
  return "safe_relative_path";
}

function normalizeExternalDataPath(path) {
  return String(path || "").replace(/\\/g, "/").replace(/^(?:\.\/)+/, "");
}

function buildOnnxMetadataPresence(model, graph) {
  const metadataProperties = [
    ...(model.metadataProps || []).map((entry) => ({ ...entry, scope: "model" })),
    ...(graph.metadataProps || []).map((entry) => ({ ...entry, scope: "graph" })),
  ].map((entry) => ({
    key: String(entry.key || ""),
    value: String(entry.value || ""),
    scope: entry.scope,
  }));
  if (metadataProperties.length > MAX_ONNX_METADATA_PROPERTIES) {
    throw new Error(`ONNX metadata property count ${metadataProperties.length} exceeds safety limit ${MAX_ONNX_METADATA_PROPERTIES}.`);
  }
  const metadataTextBytes = [
    model.producer,
    model.producerVersion,
    model.domain,
    model.docString,
    graph.name,
    graph.docString,
    ...metadataProperties.flatMap((entry) => [entry.key, entry.value]),
  ].reduce((total, value) => total + textEncoder.encode(String(value || "")).byteLength, 0);
  if (metadataTextBytes > MAX_ONNX_METADATA_TEXT_BYTES) {
    throw new Error(`ONNX decoded metadata text bytes ${metadataTextBytes} exceed safety limit ${MAX_ONNX_METADATA_TEXT_BYTES}.`);
  }
  const descriptions = [
    model.docString ? `model: ${model.docString}` : "",
    graph.docString ? `graph: ${graph.docString}` : "",
  ].filter(Boolean);
  return {
    format: "onnx",
    schema: "deepbom.artifact_metadata.v1.4",
    status: "assessed",
    graph_input_count: graph.inputs.length,
    graph_output_count: graph.outputs.length,
    graph_input_names: graph.inputs.map((input) => input.name).filter(Boolean),
    graph_output_names: graph.outputs.map((output) => output.name).filter(Boolean),
    has_model_metadata: metadataProperties.length > 0 || descriptions.length > 0,
    metadata_entries: metadataProperties.map((entry) => entry.key).filter(Boolean),
    metadata_property_count: metadataProperties.length,
    metadata_properties: metadataProperties,
    metadata_text_bytes: metadataTextBytes,
    has_description: descriptions.length > 0,
    description: descriptions.join(" / "),
    model_doc_string: model.docString || "",
    graph_doc_string: graph.docString || "",
    producer_name: model.producer || "",
    producer_version: model.producerVersion || "",
    model_domain: model.domain || "",
    model_version: model.modelVersion,
    documented_preprocessing: false,
    preprocessing_contract_status: metadataProperties.length
      ? "not_assessed_untyped_metadata_properties"
      : "absent_no_standard_preprocessing_contract",
    output_semantics_documented: false,
    output_label_file_count: 0,
    detail: `ONNX ModelProto/GraphProto metadata parsed: producer ${model.producer || "not declared"}${model.producerVersion ? ` @ ${model.producerVersion}` : ""}; model/graph doc_string ${descriptions.length}/2 present; metadata_props ${metadataProperties.length} (model ${metadataProperties.filter((entry) => entry.scope === "model").length}, graph ${metadataProperties.filter((entry) => entry.scope === "graph").length}); decoded metadata text ${metadataTextBytes} B. ONNX metadata_props are untyped key/value declarations and are not promoted to a machine-verifiable preprocessing or output-label contract.`,
  };
}

function buildOnnxWeightIntegrity(graph, tensors, ops, initializerAnalysis) {
  const initializerTensors = tensors.filter((tensor) => tensor.role === "initializer" || tensor.constant_buffer);
  const initializerElementCounts = initializerTensors.map((tensor) => Number.isSafeInteger(tensor.initializer_elements)
    ? tensor.initializer_elements : shapeElementCount(tensor.shape, tensor.shape_declared ?? tensor.shapeDeclared ?? (Array.isArray(tensor.shape) && tensor.shape.length > 0)));
  const initializerElements = initializerElementCounts.every(Number.isSafeInteger) ? sum(initializerElementCounts) : null;
  const assessed = initializerAnalysis.status === "assessed";
  const tensorByName = new Map(tensors.map((tensor) => [tensor.name, tensor]));
  const quantGrid = analyzeOnnxQuantizedKernelGrid(graph, tensors);
  const details = (initializerAnalysis.zero_kernel_slice_details || []).map((detail) => {
    const consumer = ops.find((op) => op.index === detail.consumer_op_index);
    const consumerOutputs = new Set(consumer?.outputs || []);
    const downstream = ops.filter((op) => op.inputs?.some((input) => consumerOutputs.has(input)));
    const flaggedChannels = Number(detail.channel_count || 0);
    const outputChannels = Number(detail.kernel_output_channels || 0);
    const consumerMacs = consumer?.macs_status === "assessed" ? Number(consumer.macs || 0) : null;
    const sliceMacs = consumerMacs != null && outputChannels > 0
      ? consumerMacs * flaggedChannels / outputChannels
      : null;
    const consumerMacShare = consumer?.mac_percent == null ? null : Number(consumer.mac_percent);
    const sliceMacShare = consumerMacShare != null && outputChannels > 0
      ? consumerMacShare * flaggedChannels / outputChannels
      : null;
    const tensor = tensorByName.get(detail.tensor_name);
    return {
      ...detail,
      dtype: detail.dtype || tensor?.dtype || "UNKNOWN",
      shape: detail.shape || detail.tensor_shape || tensor?.shape || [],
      consumer_ops: consumer ? [`#${consumer.index} ${consumer.name}`] : [],
      consumer_mac_percent: consumerMacShare,
      zero_slice_static_macs: sliceMacs,
      zero_slice_arithmetic_share: sliceMacShare,
      arithmetic_share_basis: sliceMacShare == null
        ? "not assessed because the direct consumer MAC count or kernel output-channel count is unavailable"
        : `direct consumer static MACs multiplied by flagged output slices / kernel output channels (${flaggedChannels}/${outputChannels}); arithmetic-waste proxy only, not latency or accuracy impact`,
      next_consumers: downstream.slice(0, 6).map((op) => `#${op.index} ${op.name}`),
      residual_path: downstream.some((op) => op.name === "Add")
        ? `direct residual Add consumer(s): ${downstream.filter((op) => op.name === "Add").map((op) => `#${op.index}`).join(", ")}`
        : "no direct Add consumer observed",
      functional_status: detail.bias_nonzero_for_flagged_channels
        ? "NOT_INACTIVE: corresponding non-zero bias was decoded; kernel slice remains a numerical-structure finding"
        : "NOT_ASSESSABLE: kernel slice and decoded bias alone do not establish functional inactivity",
    };
  });
  const assessedMetric = (value, reason) => assessed
    ? { status: "assessed", value, reason }
    : { status: "not_assessed", value: null, reason };
  return {
    status: initializerAnalysis.status,
    coverage_status: initializerAnalysis.coverage_status,
    evidence_class: assessed ? "OBSERVED" : "NOT_ASSESSABLE",
    weight_tensors_scanned: assessed ? initializerAnalysis.assessed_tensors : null,
    quantized_constant_tensors_scanned: quantGrid.assessed_tensors,
    elements_scanned: assessed ? initializerAnalysis.elements_scanned : null,
    logical_elements_assessed: assessed ? initializerAnalysis.logical_elements_assessed : null,
    stored_weight_values_decoded: assessed ? initializerAnalysis.stored_weight_values_decoded : null,
    implicit_zero_elements: assessed ? initializerAnalysis.implicit_zero_elements : null,
    dense_initializer_tensors: initializerAnalysis.dense_initializer_tensors,
    sparse_initializer_tensors: initializerAnalysis.sparse_initializer_tensors,
    eligible_kernel_tensors_scanned: initializerAnalysis.eligible_kernel_tensors_scanned,
    output_channels_evaluated: initializerAnalysis.output_channels_evaluated,
    nan_tensors: initializerAnalysis.nan_tensors,
    inf_tensors: initializerAnalysis.inf_tensors,
    all_zero_tensors: initializerAnalysis.all_zero_tensors,
    zero_kernel_slice_tensors: initializerAnalysis.zero_kernel_slice_tensors,
    zero_kernel_slice_count: initializerAnalysis.zero_kernel_slice_count,
    max_abs_weight: initializerAnalysis.max_abs_weight,
    large_magnitude_tensors: initializerAnalysis.large_magnitude_tensors,
    mean_sparsity: initializerAnalysis.mean_sparsity,
    high_sparsity_tensors: initializerAnalysis.high_sparsity_tensors,
    zero_kernel_slice_details: assessed ? details : null,
    low_grid_utilization_tensors: quantGrid.low_utilization_tensors,
    saturated_quantized_tensors: quantGrid.saturated_tensors,
    min_grid_utilization: quantGrid.minimum_grid_utilization,
    max_saturation_percent: quantGrid.maximum_saturation_ratio,
    quant_grid_details: quantGrid.details,
    quant_grid_detail: quantGrid.detail,
    metrics: {
      nan_tensors: assessedMetric(initializerAnalysis.nan_tensors, "Decoded embedded or verified-external initializer values; validated SparseTensorProto implicit elements are exact zeros."),
      inf_tensors: assessedMetric(initializerAnalysis.inf_tensors, "Decoded embedded or verified-external initializer values; validated SparseTensorProto implicit elements are exact zeros."),
      all_zero_tensors: assessedMetric(initializerAnalysis.all_zero_tensors, "Every logical initializer scalar is exactly zero, including specification-defined SparseTensorProto implicit zeros."),
      zero_kernel_slice_tensors: initializerAnalysis.eligible_kernel_tensors_scanned > 0
        ? assessedMetric(initializerAnalysis.zero_kernel_slice_tensors, "Conv/Gemm/MatMul output-axis kernel slices were evaluated; functional inactivity is not inferred.")
        : { status: "not_applicable", value: null, reason: "No supported embedded Conv/Gemm/MatMul kernel layout was available." },
      zero_kernel_slice_count: initializerAnalysis.eligible_kernel_tensors_scanned > 0
        ? assessedMetric(initializerAnalysis.zero_kernel_slice_count, "Count of output-axis kernel slices whose decoded values are all below |x| < 1e-8.")
        : { status: "not_applicable", value: null, reason: "No supported embedded Conv/Gemm/MatMul kernel layout was available." },
      max_abs_weight: assessedMetric(initializerAnalysis.max_abs_weight, "Maximum finite absolute decoded initializer scalar."),
      mean_sparsity: assessedMetric(initializerAnalysis.mean_sparsity, "Near-zero logical scalar count divided by finite logical scalar count; near-zero means |x| < 1e-8 and validated SparseTensorProto absent elements are exact zeros."),
      min_grid_utilization: quantGrid.assessed_tensors
        ? { status: "assessed", value: quantGrid.minimum_grid_utilization, reason: "Unique logical integer levels divided by the legal 8-bit dtype level count for quantized kernel initializers with bound scale/zero-point parameters; validated sparse absent entries contribute integer code 0." }
        : { status: "not_applicable", value: null, reason: "No available 8-bit kernel initializer had a valid Q/DQ or QLinear parameter binding." },
      max_saturation_percent: quantGrid.assessed_tensors
        ? { status: "assessed", value: quantGrid.maximum_saturation_ratio, reason: "Logical qmin/qmax endpoint element count divided by logical kernel initializer elements; validated sparse absent entries contribute integer code 0." }
        : { status: "not_applicable", value: null, reason: "No available 8-bit kernel initializer had a valid Q/DQ or QLinear parameter binding." },
    },
    initializer_tensors_present: graph.initializers.length + graph.sparseInitializers.length,
    initializer_elements_present: initializerElements,
    initializer_tensors_unassessed: initializerAnalysis.unassessed_tensors,
    initializer_value_decoding: assessed ? "implemented for dense TensorProto plus validated SparseTensorProto logical tensors using embedded and verified external raw_data or typed numeric fields" : "no supported complete numeric initializer payload was available",
    tensor_results: initializerAnalysis.tensor_results,
    detail: assessed
      ? `${initializerAnalysis.assessed_tensors} available ONNX initializer tensor(s) were assessed across ${initializerAnalysis.logical_elements_assessed} logical scalar element(s): ${initializerAnalysis.stored_weight_values_decoded} stored value(s) decoded and ${initializerAnalysis.implicit_zero_elements} validated SparseTensorProto implicit zero(s) reconstructed without densification. ${initializerAnalysis.unassessed_tensors} incomplete-external, invalid-sparse, or unsupported tensor(s) remain not assessed.`
      : `${graph.initializers.length + graph.sparseInitializers.length} logical initializer tensor(s) were inventoried, but no supported complete numeric payload was available for value integrity checks.`,
  };
}

function analyzeOnnxQuantizedKernelGrid(graph, tensors) {
  const initializerByName = buildOnnxLogicalInitializerIndex(graph);
  const producerByOutput = new Map();
  for (const node of graph.nodes) for (const output of node.outputs || []) if (output) producerByOutput.set(output, node);
  const kernelNames = new Set();
  for (const node of graph.nodes) {
    if (!isStandardOnnxNode(node)) continue;
    if (["QLinearConv", "QLinearMatMul"].includes(node.opType) && node.inputs[3]) kernelNames.add(node.inputs[3]);
    if (!["Conv", "Gemm", "MatMul"].includes(node.opType)) continue;
    const candidate = producerByOutput.get(node.inputs[1]);
    const dequant = isStandardOnnxNode(candidate) && candidate?.opType === "DequantizeLinear" ? candidate : null;
    if (dequant?.inputs?.[0]) kernelNames.add(dequant.inputs[0]);
  }
  const tensorByName = new Map(tensors.map((tensor) => [tensor.name, tensor]));
  const details = [];
  for (const name of kernelNames) {
    const initializer = initializerByName.get(name);
    const tensor = tensorByName.get(name);
    const legal = integerDtypeRange(initializer?.dtype);
    if (!initializer || !tensor || !["INT8", "UINT8"].includes(initializer.dtype) || !legal || !(tensor.quant_scales > 0) || tensor.quantization_binding_status !== "pass") continue;
    const levels = new Set();
    let elements = 0;
    let storedValuesDecoded = 0;
    let implicitZeroElements = 0;
    let endpointElements = 0;
    if (initializer.kind === "sparse") {
      const indices = decodeSparseLinearIndices(initializer.sparse);
      if (!indices.ok) continue;
      const denseElements = shapeElementCount(initializer.shape, true);
      if (denseElements == null) continue;
      implicitZeroElements = denseElements - indices.values.length;
      elements = implicitZeroElements;
      if (implicitZeroElements > 0) {
        levels.add(0);
        if (legal[0] === 0 || legal[1] === 0) endpointElements += implicitZeroElements;
      }
    }
    const decoded = forEachOnnxInitializerValue(initializer.tensor, (value) => {
      const numeric = Number(value);
      levels.add(numeric);
      elements += 1;
      storedValuesDecoded += 1;
      if (numeric === legal[0] || numeric === legal[1]) endpointElements += 1;
    });
    if (!decoded.ok || !elements) continue;
    const levelCount = legal[1] - legal[0] + 1;
    const utilization = levels.size / levelCount;
    const saturation = endpointElements / elements;
    details.push({
      tensor_name: name,
      dtype: initializer.dtype,
      shape: initializer.shape || [],
      storage_kind: initializer.kind === "sparse" ? "sparse_tensor_proto" : "tensor_proto",
      elements_scanned: elements,
      stored_values_decoded: storedValuesDecoded,
      implicit_zero_elements: implicitZeroElements,
      unique_integer_levels: levels.size,
      legal_integer_levels: levelCount,
      grid_utilization: utilization,
      endpoint_elements: endpointElements,
      saturation_ratio: saturation,
      low_utilization_review: elements >= 256 && utilization < 0.25,
      saturation_review: saturation > 0.01,
      threshold_class: "HEURISTIC",
      evidence_class: "DERIVED",
      formula: "grid_utilization=unique(logical q)/legal_dtype_levels; saturation=count(logical q==qmin or q==qmax)/logical N; validated SparseTensorProto absent entries contribute integer code 0",
    });
  }
  return {
    assessed_tensors: details.length,
    low_utilization_tensors: details.filter((item) => item.low_utilization_review).length,
    saturated_tensors: details.filter((item) => item.saturation_review).length,
    minimum_grid_utilization: details.length ? Math.min(...details.map((item) => item.grid_utilization)) : null,
    maximum_saturation_ratio: details.length ? Math.max(...details.map((item) => item.saturation_ratio)) : null,
    details,
    detail: details.length
      ? `${details.length} available ONNX quantized kernel initializer(s) were assessed over logical integer codes, including validated SparseTensorProto implicit code 0; low-utilization threshold <25% with logical N>=256, saturation threshold >1%.`
      : "No available 8-bit ONNX kernel initializer had a valid Q/DQ or QLinear parameter binding.",
  };
}

function buildOnnxOp(node, index, tensorMap, tensorIdByName) {
  const inputTensors = node.inputs.map((name) => tensorMap.get(name)).filter(Boolean);
  const outputTensors = node.outputs.map((name) => tensorMap.get(name)).filter(Boolean);
  const payloads = [...inputTensors, ...outputTensors].map(tensorPayloadBytes);
  const payloadAssessed = payloads.every((value) => value != null);
  const assessedPayloadBytes = sum(payloads.filter((value) => value != null));
  const estimatedBytes = payloadAssessed ? assessedPayloadBytes : null;
  const standardDomain = isStandardOnnxNode(node);
  const macAssessment = estimateOnnxMacs(node, tensorMap);
  const cachePayload = estimateCachePayload(node, tensorMap);
  const rowWorkingSet = cachePayload.status === "assessed"
    ? {
        value: cachePayload.logical_row_payload_bytes,
        status: "assessed",
        reason: "The input-strip plus output-row logical payload from deepbom.cache_payload.v1.",
      }
    : {
        value: null,
        status: cachePayload.status,
        reason: cachePayload.boundary,
  };
  const macs = macAssessment.value;
  const totalOps = macAssessment.status === "assessed" && macs != null ? macs * 2 : null;
  const intensity = macAssessment.status === "assessed" && estimatedBytes > 0
    ? exactNonnegativeRatio(BigInt(macAssessment.value_decimal) * 2n, estimatedBytes)
    : null;
  const staticBound = standardDomain ? classifyStaticBound(node.opType, macAssessment.status, macs, intensity, macAssessment.value_decimal) : "not-assessed";
  const quantizedPath = onnxOpHasQuantSignal(node, inputTensors, outputTensors);
  const quantizedComputePath = onnxOpHasQuantizedComputePath(node, inputTensors, outputTensors);
  const quantization = classifyOnnxOpQuantization(node, inputTensors, outputTensors);
  return {
    index,
    name: node.opType || "UNKNOWN",
    graph_node_name: node.name || "",
    domain: normalizeOnnxDomain(node.domain),
    standard_domain: standardDomain,
    input_names: [...node.inputs],
    output_names: [...node.outputs],
    inputs: node.inputs.map((name) => tensorIdByName.get(name) ?? -1),
    outputs: node.outputs.map((name) => tensorIdByName.get(name) ?? -1),
    onnx_attributes: serializeOnnxAttributes(node),
    output_shapes: outputTensors.map((tensor) => tensor.shape || []),
    macs,
    macs_decimal: macAssessment.value_decimal,
    macs_status: macAssessment.status,
    macs_reason: macAssessment.reason,
    estimated_bytes: estimatedBytes,
    estimated_bytes_status: payloadAssessed ? "assessed" : "not_assessed",
    estimated_bytes_reason: payloadAssessed
      ? "sum(input and output logical tensor payload bytes)"
      : `${payloads.filter((value) => value == null).length} input/output tensor payload(s) lack a fully known shape or dtype; assessed subtotal ${assessedPayloadBytes} B is not promoted to a complete op total`,
    assessed_payload_bytes: assessedPayloadBytes,
    unassessed_payload_tensor_count: payloads.filter((value) => value == null).length,
    intensity_ops_per_byte: intensity,
    intensity_status: macAssessment.status !== "assessed" || !payloadAssessed
      ? "not_assessed"
      : estimatedBytes > 0 ? intensity == null ? "not_assessed_exact_ratio_outside_numeric_range" : "assessed" : "not_applicable_zero_logical_bytes",
    row_working_set_bytes: rowWorkingSet.value,
    row_working_set_status: rowWorkingSet.status,
    row_working_set_reason: rowWorkingSet.reason,
    cache_payload: cachePayload,
    static_bound_guess: staticBound,
    static_action: standardDomain ? staticAction(node.opType, staticBound) : "Custom-domain operator semantics are not inferred; use the bound custom-op registry and runtime evidence.",
    quantized_path: quantizedPath,
    quantized_compute_path: quantizedComputePath,
    quantization_state: quantization.state,
    quantization_detail: quantization.detail,
  };
}

function serializeOnnxAttributes(node) {
  return [...(node.attributes?.values?.() || [])]
    .sort((left, right) => left.name.localeCompare(right.name))
    .map((attribute) => ({
      name: attribute.name,
      type: Number(attribute.type || 0),
      int_value: Number.isSafeInteger(attribute.i) ? attribute.i : null,
      int_value_exact_decimal: attribute.iExactDecimal || "",
      float_value: Number.isFinite(attribute.f) ? jsonSafeOnnxFloat(attribute.f) : null,
      float_value_text: attribute.valueTypesPresent.includes(1) ? canonicalOnnxFloatText(attribute.f) : "",
      string_value: attribute.s,
      int_values: attribute.ints.every(Number.isSafeInteger) ? [...attribute.ints] : [],
      int_values_exact_decimal: [...attribute.intExactDecimals],
      float_values: attribute.floats.map((value) => Number.isFinite(value) ? jsonSafeOnnxFloat(value) : null),
      float_values_text: attribute.floats.map(canonicalOnnxFloatText),
      string_values: [...attribute.strings],
      tensor_value: serializeOnnxAttributeTensor(attribute.tensor),
      has_graph: Boolean(attribute.graph),
      graph_count: attribute.graphs.length,
    }));
}

function serializeOnnxAttributeTensor(tensor) {
  if (!tensor) return null;
  const negativeZeroIndices = staticNegativeZeroIndices(tensor.staticValues);
  const exactValues = [];
  const expected = onnxInitializerElementCount(tensor);
  const bounded = Number.isSafeInteger(expected) && expected >= 0 && expected <= MAX_ONNX_EP_CONDITION_INTEGER_ELEMENTS;
  const decoded = bounded && tensor.dtype === "STRING" && tensor.staticValuesComplete === true
    && Array.isArray(tensor.staticValues) && tensor.staticValues.length === expected
    ? (tensor.staticValues.forEach((value) => exactValues.push(String(value))), { ok: true, reason: "", count: expected })
    : bounded
      ? forEachOnnxInitializerValue(tensor, (value) => exactValues.push(canonicalOnnxAttributeTensorValue(value)))
      : { ok: false, reason: "not_assessed_element_limit", count: 0 };
  const exactValuesComplete = decoded.ok && Number.isSafeInteger(expected) && decoded.count === expected;
  return {
    dtype: tensor.dtype || "UNKNOWN",
    shape: Array.isArray(tensor.shape) ? [...tensor.shape] : [],
    shape_declared: tensor.shapeDeclared === true,
    raw_data_bytes: Number(tensor.rawDataBytes || 0),
    typed_data_bytes: Number(tensor.typedDataBytes || 0),
    external_data_entries: Number(tensor.externalDataEntries || 0),
    data_location: Number(tensor.dataLocation || 0),
    integer_values_exact_complete: tensor.initializerIntegerValuesExactComplete === true,
    integer_values_exact_decimals: [...(tensor.initializerIntegerValuesExactDecimals || [])],
    static_values_status: tensor.staticValuesStatus || "not_assessed",
    static_values_complete: tensor.staticValuesComplete === true,
    static_values: jsonSafeStaticValues(tensor.staticValues),
    static_values_negative_zero_count: negativeZeroIndices.length,
    static_values_negative_zero_indices: negativeZeroIndices,
    static_values_source: tensor.staticValuesSource || "attribute_tensor",
    exact_values_status: exactValuesComplete ? "complete" : decoded.reason || "not_assessed",
    exact_values_complete: exactValuesComplete,
    exact_values_text: exactValuesComplete ? exactValues : [],
  };
}

function canonicalOnnxAttributeTensorValue(value) {
  if (typeof value === "bigint") return value.toString();
  if (typeof value === "string") return value;
  return canonicalOnnxFloatText(Number(value));
}

function jsonSafeOnnxFloat(value) {
  return Object.is(value, -0) ? 0 : value;
}

function canonicalOnnxFloatText(value) {
  if (Number.isNaN(value)) return "NaN";
  if (value === Number.POSITIVE_INFINITY) return "Infinity";
  if (value === Number.NEGATIVE_INFINITY) return "-Infinity";
  if (Object.is(value, -0)) return "-0";
  return String(value);
}

function onnxOpHasQuantSignal(node, inputTensors, outputTensors) {
  if (isStandardOnnxNode(node) && ["QuantizeLinear", "DequantizeLinear", "QLinearConv", "QLinearMatMul", "MatMulInteger", "ConvInteger"].includes(node.opType)) {
    return true;
  }
  return [...inputTensors, ...outputTensors].some((tensor) => isDenseTensorValue(tensor) && ["INT8", "UINT8"].includes(tensor?.dtype));
}

function onnxOpHasQuantizedComputePath(node, inputTensors, outputTensors) {
  if (!isStandardOnnxNode(node)) return false;
  if (["QLinearConv", "QLinearMatMul", "MatMulInteger", "ConvInteger"].includes(node.opType)) return true;
  if (!["Conv", "Gemm", "MatMul"].includes(node.opType)) return false;
  const input0 = inputTensors[0];
  const output0 = outputTensors[0];
  return isDenseTensorValue(input0) && isDenseTensorValue(output0)
    && ["INT8", "UINT8"].includes(input0?.dtype) && ["INT8", "UINT8"].includes(output0?.dtype);
}

function classifyOnnxOpQuantization(node, inputTensors, outputTensors) {
  const input0 = inputTensors[0];
  const input1 = inputTensors[1];
  const output0 = outputTensors[0];
  const input0Q = isDenseTensorValue(input0) && is8BitDtype(input0?.dtype);
  const input1Q = isDenseTensorValue(input1) && is8BitDtype(input1?.dtype);
  const output0Q = isDenseTensorValue(output0) && is8BitDtype(output0?.dtype);
  const any8Bit = [...inputTensors, ...outputTensors].some((tensor) => isDenseTensorValue(tensor) && is8BitDtype(tensor?.dtype));
  if (!isStandardOnnxNode(node)) return {
    state: any8Bit ? "quant_signal_only" : "none",
    detail: `${normalizeOnnxDomain(node.domain)}:${node.opType || "UNKNOWN"} custom-domain semantics were not inferred; tensor dtypes are observed only`,
  };
  const computeLike = ["Conv", "Gemm", "MatMul", "QLinearConv", "QLinearMatMul", "MatMulInteger", "ConvInteger"].includes(node.opType);
  let state = "none";
  if (node.opType === "QuantizeLinear" || node.opType === "DequantizeLinear") {
    state = "quant_boundary";
  } else if (["QLinearConv", "QLinearMatMul", "MatMulInteger", "ConvInteger"].includes(node.opType)) {
    state = "quantized_compute";
  } else if (computeLike && input0Q && output0Q) {
    state = "quantized_compute";
  } else if (computeLike && input1Q && !input0Q && !output0Q) {
    state = "weight_only_or_dynamic_range";
  } else if (computeLike && any8Bit) {
    state = "mixed_or_hybrid_compute";
  } else if (!computeLike && input0Q && output0Q) {
    state = "quantized_data_movement";
  } else if (any8Bit) {
    state = "quant_signal_only";
  }
  const detail = state === "none"
    ? "no ONNX quantization signal for this op"
    : `${node.opType} quant state=${state}; input0=${input0?.dtype || "missing"} input1=${input1?.dtype || "missing"} output0=${output0?.dtype || "missing"}`;
  return { state, detail };
}

export function estimateOnnxMacs(node, tensorMap) {
  const input = tensorMap.get(node.inputs[0]);
  const weightInputIndex = ["QLinearConv", "QLinearMatMul"].includes(node.opType) ? 3 : 1;
  const weight = tensorMap.get(node.inputs[weightInputIndex]);
  const output = tensorMap.get(node.outputs[0]);
  const assessed = (value, reason) => {
    const exact = exactNonnegativeInteger(value);
    if (exact == null) return { value: null, value_decimal: null, status: "not_assessed", reason: `${reason} The derived MAC cardinality is not a valid nonnegative integer.` };
    return { value: safeBigIntNumber(exact), value_decimal: exact.toString(), status: "assessed", reason };
  };
  const notAssessed = (reason) => ({ value: null, value_decimal: null, status: "not_assessed", reason });
  const costClass = classifyOnnxMacOperation(node.opType, isStandardOnnxNode(node));
  if (costClass === "non_standard_domain") return notAssessed(`${normalizeOnnxDomain(node.domain)}:${node.opType || "UNKNOWN"} is not an ai.onnx operator; MAC semantics were not inferred from its name.`);
  if (costClass === "known_mac_bearing_unimplemented") return notAssessed(`${node.opType} is MAC-bearing, but its source-backed nominal MAC rule is not implemented.`);
  if (costClass === "algorithm_dependent_arithmetic") return notAssessed(`${node.opType} has algorithm-dependent arithmetic cost; the ONNX contract does not select a direct, FFT, or backend-specific implementation.`);
  if (costClass === "unclassified") return notAssessed(`${node.opType || "UNKNOWN"} is absent from the pinned ONNX operation-cost classification.`);
  if (costClass === "source_classified_non_mac") return { value: 0, value_decimal: "0", status: "not_applicable", reason: "The pinned nominal tensor-contraction MAC metric classifies this operator as non-MAC." };
  const invalidContract = [...(node.inputs || []), ...(node.outputs || [])]
    .filter(Boolean)
    .map((name) => ({ name, tensor: tensorMap.get(name) }))
    .find(({ tensor }) => tensorContractBlocksDeterministicCost(tensor));
  if (invalidContract) {
    return notAssessed(`${node.opType} nominal MACs are blocked by invalid tensor contract ${invalidContract.name}; no stale declared shape is used.`);
  }
  if ([input, weight, output].filter(Boolean).some((tensor) => !isDenseTensorValue(tensor))) {
    return notAssessed(`${node.opType} MACs require dense tensor values; a declared non-dense TypeProto participates in this signature.`);
  }
  if (node.opType === "Attention") {
    const key = tensorMap.get(node.inputs[1]);
    const valueTensor = tensorMap.get(node.inputs[2]);
    const pastKey = tensorMap.get(node.inputs[4]);
    const pastValue = tensorMap.get(node.inputs[5]);
    if (![input, key, valueTensor, output].every((tensor) => tensor && knownTensorShape(tensor))) {
      return notAssessed("Attention nominal MACs require static Q, K, V, and Y shapes.");
    }
    const rank = input.shape.length;
    if (![3, 4].includes(rank) || key.shape.length !== rank || valueTensor.shape.length !== rank) return notAssessed("Attention Q, K, and V must share source-compatible rank 3 or 4.");
    let batch, queryHeads, kvHeads, querySequence, keySequence, qkHeadSize, valueHeadSize, expectedOutput;
    if (rank === 4) {
      [batch, queryHeads, querySequence, qkHeadSize] = input.shape;
      kvHeads = key.shape[1]; keySequence = key.shape[2]; valueHeadSize = valueTensor.shape[3];
      expectedOutput = [batch, queryHeads, querySequence, valueHeadSize];
      if (key.shape[0] !== batch || valueTensor.shape[0] !== batch || key.shape[1] !== valueTensor.shape[1]
        || key.shape[2] !== valueTensor.shape[2] || key.shape[3] !== qkHeadSize
        || queryHeads < kvHeads || queryHeads % kvHeads !== 0) return notAssessed("Attention 4D Q/K/V head or sequence dimensions are incompatible.");
    } else {
      batch = input.shape[0]; querySequence = input.shape[1]; keySequence = key.shape[1];
      queryHeads = attrInt(node, "q_num_heads", 0); kvHeads = attrInt(node, "kv_num_heads", 0);
      if (queryHeads <= 0 || kvHeads <= 0 || queryHeads < kvHeads || queryHeads % kvHeads !== 0
        || input.shape[2] % queryHeads !== 0 || key.shape[2] % kvHeads !== 0 || valueTensor.shape[2] % kvHeads !== 0) {
        return notAssessed("Attention 3D hidden widths and q_num_heads/kv_num_heads are incompatible.");
      }
      qkHeadSize = input.shape[2] / queryHeads;
      valueHeadSize = valueTensor.shape[2] / kvHeads;
      if (key.shape[0] !== batch || valueTensor.shape[0] !== batch || key.shape[1] !== valueTensor.shape[1]
        || key.shape[2] / kvHeads !== qkHeadSize) return notAssessed("Attention 3D Q/K/V batch, sequence, or head dimensions are incompatible.");
      expectedOutput = [batch, querySequence, queryHeads * valueHeadSize];
    }
    if (!sameShape(output.shape, expectedOutput)) return notAssessed("Attention Y shape is incompatible with the source-defined Q and V contract.");
    let pastSequence = 0;
    if (pastKey || pastValue) {
      if (!(pastKey && pastValue && knownTensorShape(pastKey) && knownTensorShape(pastValue))
        || pastKey.shape.length !== 4 || pastValue.shape.length !== 4
        || pastKey.shape[0] !== batch || pastValue.shape[0] !== batch || pastKey.shape[1] !== kvHeads || pastValue.shape[1] !== kvHeads
        || pastKey.shape[2] !== pastValue.shape[2] || pastKey.shape[3] !== qkHeadSize || pastValue.shape[3] !== valueHeadSize) {
        return notAssessed("Attention past key/value cache contracts are incomplete or incompatible.");
      }
      pastSequence = pastKey.shape[2];
    }
    const totalSequence = pastSequence + keySequence;
    const value = exactNonnegativeProduct([batch, queryHeads, querySequence, totalSequence, qkHeadSize + valueHeadSize]);
    return assessed(value, "B*q_num_heads*q_sequence_length*total_sequence_length*(qk_head_size+v_head_size), summing the two source-defined dense Attention MatMul contractions; mask, softmax, and backend pruning are outside nominal MACs.");
  }
  if (node.opType === "DeformConv") {
    const offset = tensorMap.get(node.inputs[2]);
    const bias = tensorMap.get(node.inputs[3]);
    const mask = tensorMap.get(node.inputs[4]);
    if (![input, weight, offset, output].every((tensor) => tensor && knownTensorShape(tensor))) return notAssessed("DeformConv requires static X, W, offset, and Y shapes.");
    const x = input.shape, w = weight.shape, y = output.shape, rank = x.length, spatialRank = rank - 2;
    const group = attrInt(node, "group", 1), offsetGroup = attrInt(node, "offset_group", 1);
    if (rank < 3 || w.length !== rank || y.length !== rank || offset.shape.length !== rank || group <= 0 || offsetGroup <= 0
      || x[0] !== y[0] || x[1] !== w[1] * group || x[1] % offsetGroup !== 0 || y[1] !== w[0] || w[0] % group !== 0
      || w.slice(2).some((dimension) => dimension <= 0)) return notAssessed("DeformConv X/W/Y ranks, channels, groups, or kernel dimensions are incompatible.");
    const kernelVolume = w.slice(2).reduce((product, dimension) => product * dimension, 1);
    const expectedOffset = [x[0], offsetGroup * kernelVolume * spatialRank, ...y.slice(2)];
    if (!sameShape(offset.shape, expectedOffset)) return notAssessed("DeformConv offset shape is incompatible with offset_group, kernel volume, spatial rank, and Y.");
    if (bias && (!knownTensorShape(bias) || !sameShape(bias.shape, [w[0]]))) return notAssessed("DeformConv optional bias shape is incompatible with output channels.");
    if (mask) {
      const expectedMask = [x[0], offsetGroup * kernelVolume, ...y.slice(2)];
      if (!knownTensorShape(mask) || !sameShape(mask.shape, expectedMask)) return notAssessed("DeformConv optional mask shape is incompatible with offset_group, kernel volume, and Y.");
    }
    const value = exactNonnegativeProduct([...y, ...w.slice(1)]);
    return assessed(value, "product(Y shape)*product(W shape excluding output-channel axis), counting the source-defined sampled-value/weight contraction only; interpolation arithmetic is excluded from nominal tensor-contraction MACs.");
  }
  if (node.opType === "Einsum") {
    const inputs = node.inputs.filter(Boolean).map((name) => tensorMap.get(name));
    if (!inputs.length || inputs.some((tensor) => !tensor || !knownTensorShape(tensor))) return notAssessed("Einsum nominal MACs require every operand shape to be static.");
    const parsed = parseOnnxEinsumEquation(attrString(node, "equation", ""), inputs.map((tensor) => tensor.shape.length));
    if (parsed.status !== "assessed") return notAssessed(`Einsum equation contract is invalid: ${parsed.reason}.`);
    if (inputs.length === 1) return { value: 0, value_decimal: "0", status: "not_applicable", reason: "A one-input Einsum has no binary tensor contraction under the nominal MAC metric." };
    if (inputs.length !== 2) return notAssessed("Einsum with more than two operands has contraction-order-dependent multiplication work; ONNX does not serialize an evaluation path.");
    const dimensions = new Map();
    for (let inputIndex = 0; inputIndex < inputs.length; inputIndex += 1) {
      for (let axis = 0; axis < parsed.operands[inputIndex].length; axis += 1) {
        const label = parsed.operands[inputIndex][axis], value = inputs[inputIndex].shape[axis];
        const previous = dimensions.get(label);
        if (label.startsWith("@ellipsis:")) {
          if (previous != null && previous !== 1 && value !== 1 && previous !== value) return notAssessed("Einsum ellipsis dimensions are not broadcast-compatible.");
          dimensions.set(label, Math.max(previous ?? 1, value));
        } else if (previous != null && previous !== value) return notAssessed(`Einsum label ${label} has incompatible dimensions.`);
        else dimensions.set(label, value);
      }
    }
    const expectedOutput = parsed.output.map((label) => dimensions.get(label));
    if (!sameShape(output.shape, expectedOutput)) return notAssessed("Einsum output shape is incompatible with the parsed equation and operands.");
    const value = exactNonnegativeProduct(parsed.all_labels.map((label) => dimensions.get(label)));
    return assessed(value, "Product of every unique index-domain extent for a two-input Einsum, equal to one pairwise product/accumulation term per complete Einstein index assignment.");
  }
  if (node.opType === "Conv" || node.opType === "QLinearConv" || node.opType === "ConvInteger") {
    if (!input || !weight || !output) return notAssessed(`${node.opType} input, weight, or output tensor metadata is missing.`);
    const group = attrInt(node, "group", 1);
    const outShape = output.shape || [];
    const inShape = input?.shape || [];
    const weightShape = weight.shape || [];
    if (inShape.length < 3 || outShape.length !== inShape.length || weightShape.length !== inShape.length
      || !knownTensorShape(input) || !knownTensorShape(weight) || !knownTensorShape(output)) {
      return notAssessed(`${node.opType} MACs require known equal-rank N,C,D1...Dn input/output and M,C/group,k1...kn weight shapes.`);
    }
    const groupedInputChannels = safeExactProduct([weightShape[1], group]);
    if (!Number.isSafeInteger(group) || group <= 0 || groupedInputChannels == null || weightShape.slice(0, 2).some((value) => value <= 0)
      || weightShape.slice(2).some((value) => value <= 0)
      || inShape[0] !== outShape[0] || outShape[1] !== weightShape[0]
      || inShape[1] !== groupedInputChannels || weightShape[0] % group !== 0) {
      return notAssessed(`${node.opType} tensor channels, batch, or group dimensions are incompatible.`);
    }
    const value = exactNonnegativeProduct([...outShape, ...weightShape.slice(1)]);
    return assessed(value, "product(output shape) * product(weight shape excluding output-channel axis) from compatible ONNX N,C,D1...Dn and M,C/group,k1...kn tensors.");
  }
  if (node.opType === "ConvTranspose") {
    if (!input || !weight || !output) return notAssessed("ConvTranspose input, weight, or output tensor metadata is missing.");
    const x = input.shape || [], w = weight.shape || [], y = output.shape || [], rank = x.length, spatialRank = rank - 2;
    const group = attrInt(node, "group", 1);
    if (rank < 3 || w.length !== rank || y.length !== rank || !knownTensorShape(input) || !knownTensorShape(weight) || !knownTensorShape(output)
      || !Number.isSafeInteger(group) || group <= 0 || x[0] !== y[0] || x[1] !== w[0] || x[1] % group !== 0 || y[1] !== w[1] * group
      || w.slice(1).some((value) => value <= 0)) return notAssessed("ConvTranspose requires compatible known N,C,D1...Dn, C,M/group,k1...kn, and N,M,O1...On tensors with a positive group.");
    const strides = exactSpatialAttribute(node, "strides", spatialRank, 1, false);
    const dilations = exactSpatialAttribute(node, "dilations", spatialRank, 1, false);
    const outputPadding = exactSpatialAttribute(node, "output_padding", spatialRank, 0, true);
    const pads = convTransposePads(node, x.slice(2), w.slice(2), y.slice(2), strides, dilations, outputPadding);
    if (!strides || !dilations || !outputPadding || !pads) return notAssessed("ConvTranspose spatial attributes or serialized output shape are inconsistent with the ONNX contract.");
    const pairs = x.slice(2).map((size, axis) => convTransposeAxisPairs(size, w[axis + 2], strides[axis], dilations[axis], pads[axis], y[axis + 2]));
    if (pairs.some((value) => value == null)) return notAssessed("ConvTranspose contributing-pair count exceeds the bounded exact arithmetic path.");
    let value = BigInt(x[0]) * BigInt(x[1]) * BigInt(w[1]);
    for (const pair of pairs) value *= pair;
    return assessed(value, "Exact contributing input/kernel pairs per spatial axis times N*C*(M/group), after ONNX pad, stride, dilation, output-padding, output-shape, and group validation.");
  }
  if (["RNN", "GRU", "LSTM"].includes(node.opType)) {
    const recurrent = tensorMap.get(node.inputs[2]);
    const x = input?.shape || [], w = weight?.shape || [], r = recurrent?.shape || [];
    const gates = node.opType === "LSTM" ? 4 : node.opType === "GRU" ? 3 : 1;
    const hidden = attrInt(node, "hidden_size", 0), layout = attrInt(node, "layout", 0), direction = attrString(node, "direction", "forward");
    const directions = direction === "bidirectional" ? 2 : ["forward", "reverse"].includes(direction) ? 1 : 0;
    if (![input, weight, recurrent].every((tensor) => tensor && isDenseTensorValue(tensor) && knownTensorShape(tensor))
      || x.length !== 3 || w.length !== 3 || r.length !== 3 || ![0, 1].includes(layout) || !directions || hidden <= 0
      || w[0] !== directions || r[0] !== directions || w[1] !== gates * hidden || r[1] !== gates * hidden
      || w[2] !== x[2] || r[2] !== hidden) return notAssessed(`${node.opType} MACs require source-compatible X, W, R, hidden_size, direction, layout, and gate dimensions.`);
    const sequence = x[layout ? 1 : 0], batch = x[layout ? 0 : 1];
    if (output && knownTensorShape(output)) {
      const expected = layout ? [batch, sequence, directions, hidden] : [sequence, directions, batch, hidden];
      if (!sameShape(output.shape, expected)) return notAssessed(`${node.opType} Y shape is incompatible with direction, layout, and hidden_size.`);
    }
    const value = exactNonnegativeProduct([sequence, batch, directions, gates, hidden, x[2] + hidden]);
    return assessed(value, `sequence*batch*directions*${gates}*hidden_size*(input_size+hidden_size), counting source-defined input and recurrent matrix contractions.`);
  }
  if (node.opType === "Gemm") {
    if (!input || !weight || !output) return notAssessed("Gemm input, weight, or output tensor metadata is missing.");
    const a = input.shape || [];
    const b = weight.shape || [];
    if (a.length !== 2 || b.length !== 2 || !knownTensorShape(input) || !knownTensorShape(weight) || !knownTensorShape(output)) {
      return notAssessed("Gemm MACs require known rank-2 A and B shapes.");
    }
    const transAValue = attrInt(node, "transA", 0);
    const transBValue = attrInt(node, "transB", 0);
    if (![0, 1].includes(transAValue) || ![0, 1].includes(transBValue)) return notAssessed("Gemm transA/transB must be 0 or 1.");
    const transA = transAValue === 1;
    const transB = transBValue === 1;
    const m = transA ? a[1] : a[0];
    const kA = transA ? a[0] : a[1];
    const kB = transB ? b[1] : b[0];
    const n = transB ? b[0] : b[1];
    if (kA !== kB || !sameShape(output.shape, [m, n])) return notAssessed("Gemm inner dimensions or output shape are incompatible after transpose attributes.");
    const value = exactNonnegativeProduct([m, n, kA]);
    return assessed(value, "M*N*K from compatible Gemm A/B/output shapes after transA/transB.");
  }
  if (["MatMul", "QLinearMatMul", "MatMulInteger"].includes(node.opType)) {
    if (!input || !weight || !output) return notAssessed(`${node.opType} input, weight, or output tensor metadata is missing.`);
    const originalA = input.shape || [];
    const originalB = weight.shape || [];
    if (originalA.length < 1 || originalB.length < 1 || !knownTensorShape(input) || !knownTensorShape(weight) || !knownTensorShape(output)) {
      return notAssessed(`${node.opType} MACs require known rank-1-or-greater input and output shapes.`);
    }
    const vectorA = originalA.length === 1;
    const vectorB = originalB.length === 1;
    const a = vectorA ? [1, ...originalA] : originalA;
    const b = vectorB ? [...originalB, 1] : originalB;
    if (a.at(-1) !== b.at(-2)) return notAssessed(`${node.opType} inner dimensions are incompatible.`);
    const batchShape = broadcastShape(a.slice(0, -2), b.slice(0, -2));
    if (batchShape === null) return notAssessed(`${node.opType} batch dimensions are not broadcast-compatible.`);
    const expectedOutput = [...batchShape, a.at(-2), b.at(-1)];
    if (vectorA) expectedOutput.splice(-2, 1);
    if (vectorB) expectedOutput.pop();
    if (!sameShape(output.shape, expectedOutput)) return notAssessed(`${node.opType} output shape is incompatible with ONNX MatMul rank promotion and batch broadcasting.`);
    const value = exactNonnegativeProduct([...batchShape, a.at(-2), b.at(-1), a.at(-1)]);
    return assessed(value, "broadcast_batch*M*N*K after ONNX rank-1 promotion, batch broadcasting, and output-shape validation.");
  }
  return notAssessed(`${node.opType} MAC estimation is not implemented for this ONNX operator signature.`);
}

function assessOnnxScopeIntrinsicCost(graph, tensorMap) {
  const computeRows = [];
  const payloadRows = [];
  const macValues = [];
  const payloadValues = [];
  for (const [index, node] of (graph?.nodes || []).entries()) {
    if (isOnnxMacBearingOperation(node.opType, isStandardOnnxNode(node))) {
      const assessment = estimateOnnxMacs(node, tensorMap);
      if (assessment.status === "assessed") macValues.push(assessment.value_decimal);
      else computeRows.push({ node_index: index, op_name: node.opType || "UNKNOWN", reason: assessment.reason });
    }
    const names = [...(node.inputs || []), ...(node.outputs || [])].filter(Boolean);
    const tensors = names.map((name) => tensorMap.get(name));
    const bytes = tensors.map(tensorPayloadBytes);
    if (tensors.length !== names.length || bytes.some((value) => value == null)) {
      payloadRows.push({ node_index: index, op_name: node.opType || "UNKNOWN", reason: "one or more operator I/O values lack a known dense shape and storage dtype" });
    } else {
      const total = exactNonnegativeSum(bytes);
      if (total == null) payloadRows.push({ node_index: index, op_name: node.opType || "UNKNOWN", reason: "operator I/O payload subtotal contains an invalid value" });
      else payloadValues.push(total);
    }
  }
  const computeCount = (graph?.nodes || []).filter((node) => isOnnxMacBearingOperation(node.opType, isStandardOnnxNode(node))).length;
  const macTotal = exactNonnegativeSum(macValues);
  const payloadTotal = exactNonnegativeSum(payloadValues);
  return {
    schema: "deepbom.onnx_scope_intrinsic_cost.v1",
    status: computeRows.length || payloadRows.length ? "partial" : "assessed",
    evidence_class: "SOURCE_PINNED_AND_DERIVED",
    source_release: ONNX_OPERATION_COST_SOURCE.release,
    source_commit: ONNX_OPERATION_COST_SOURCE.commit,
    source_documents: ONNX_OPERATION_COST_SOURCE.documents.map((source) => ({
      role: source.role,
      source_ref: source.source_ref,
      sha256: source.sha256,
    })),
    operator_count: graph?.nodes?.length || 0,
    mac_compute_operator_count: computeCount,
    assessed_nominal_mac_operator_count: computeCount - computeRows.length,
    unassessed_nominal_mac_operator_count: computeRows.length,
    complete_nominal_macs: computeRows.length ? null : safeBigIntNumber(macTotal),
    complete_nominal_macs_decimal: computeRows.length ? null : macTotal.toString(),
    assessed_nominal_macs: safeBigIntNumber(macTotal),
    assessed_nominal_macs_decimal: macTotal.toString(),
    assessed_operator_io_count: (graph?.nodes?.length || 0) - payloadRows.length,
    unassessed_operator_io_count: payloadRows.length,
    complete_operator_io_payload_bytes: payloadRows.length ? null : safeBigIntNumber(payloadTotal),
    complete_operator_io_payload_bytes_decimal: payloadRows.length ? null : payloadTotal.toString(),
    assessed_operator_io_payload_bytes: safeBigIntNumber(payloadTotal),
    assessed_operator_io_payload_bytes_decimal: payloadTotal.toString(),
    mac_residuals: computeRows,
    payload_residuals: payloadRows,
    method: "One serialized scope invocation: sum exact nominal MAC definitions for supported ai.onnx compute signatures and sum each operator's logical dense input/output payload once. Safe-number mirrors are null when the exact decimal exceeds JavaScript's integer range.",
    interpretation_boundary: "Scope rows are intrinsic and are not multiplied by FunctionProto calls, If branch selection, Loop/Scan iterations, SequenceMap cardinality, fusion, or runtime scheduling. Logical operator-I/O payload is not allocator memory, physical traffic, or peak liveness.",
  };
}

function estimateCachePayload(node, tensorMap) {
  const base = {
    schema: "deepbom.cache_payload.v1",
    status: "not_applicable",
    evidence_class: "NOT_ASSESSABLE",
    input_strip_bytes: null,
    output_row_bytes: null,
    logical_row_payload_bytes: null,
    serialized_kernel_bytes: null,
    serialized_bias_bytes: null,
    input_width: null,
    input_channels: null,
    output_width: null,
    output_channels: null,
    kernel_height: null,
    kernel_width: null,
    effective_kernel_height: null,
    input_dtype: null,
    output_dtype: null,
    method: "Logical row payload = effective-kernel-height input strip + one output row. Serialized kernel and bias payloads are reported separately and are not assumed simultaneously resident.",
    boundary: "This is a graph-semantic logical payload, not an observed cache-residency, hit-rate, or microkernel-tile measurement.",
  };
  if (!isStandardOnnxNode(node)) {
    return { ...base, status: "not_assessed", boundary: "Custom-domain operator cache semantics are not inferred." };
  }
  if (!["Conv", "QLinearConv", "ConvInteger"].includes(node.opType)) {
    return { ...base, boundary: "Logical row payload is not defined for this ONNX operator signature." };
  }
  const weightSlot = node.opType === "QLinearConv" ? 3 : 1;
  const biasSlot = node.opType === "QLinearConv" ? 8 : node.opType === "Conv" ? 2 : -1;
  const input = tensorMap.get(node.inputs[0]);
  const weight = tensorMap.get(node.inputs[weightSlot]);
  const output = tensorMap.get(node.outputs[0]);
  const bias = biasSlot >= 0 ? tensorMap.get(node.inputs[biasSlot]) : null;
  if ([input, weight, output, bias].filter(Boolean).some(tensorContractBlocksDeterministicCost)) {
    return { ...base, status: "not_assessed", boundary: `${node.opType} logical row payload is blocked by an invalid or partial conditional tensor contract.` };
  }
  if (!isDenseTensorValue(input) || !isDenseTensorValue(weight) || !isDenseTensorValue(output)) {
    return { ...base, status: "not_assessed", boundary: `${node.opType} logical row payload requires dense input, weight, and output tensors.` };
  }
  const kernel = attrInts(node, "kernel_shape");
  const kH = positive(weight?.shape?.[2]) || positive(kernel[0]);
  const kW = positive(weight?.shape?.[3]) || positive(kernel[1]);
  const dilationH = positive(attrInts(node, "dilations")[0]) || 1;
  const effectiveKH = kH ? (kH - 1) * dilationH + 1 : 0;
  const inputWidth = positive(input?.shape?.[3]) || 0;
  const inputChannels = positive(input?.shape?.[1]) || 0;
  const outputWidth = positive(output?.shape?.[3]) || 0;
  const outputChannels = positive(output?.shape?.[1]) || 0;
  const inputStripBytes = dtypePayloadBytes(input?.dtype, effectiveKH * inputWidth * inputChannels);
  const outputRowBytes = dtypePayloadBytes(output?.dtype, outputWidth * outputChannels);
  if (!kH || !kW || !inputWidth || !inputChannels || !outputWidth || !outputChannels
    || inputStripBytes == null || outputRowBytes == null) {
    return {
      ...base,
      status: "not_assessed",
      boundary: `${node.opType} logical row payload requires known NCHW input/output shapes, kernel shape, dilation, and dtype storage widths.`,
    };
  }
  return {
    ...base,
    status: "assessed",
    evidence_class: "DERIVED",
    input_strip_bytes: inputStripBytes,
    output_row_bytes: outputRowBytes,
    logical_row_payload_bytes: inputStripBytes + outputRowBytes,
    serialized_kernel_bytes: availableInitializerBytes(weight),
    serialized_bias_bytes: availableInitializerBytes(bias),
    input_width: inputWidth,
    input_channels: inputChannels,
    output_width: outputWidth,
    output_channels: outputChannels,
    kernel_height: kH,
    kernel_width: kW,
    effective_kernel_height: effectiveKH,
    input_dtype: input.dtype,
    output_dtype: output.dtype,
  };
}

function availableInitializerBytes(tensor) {
  if (!tensor || !["initializer", "sparse_initializer"].includes(tensor.role)) return null;
  if (Number(tensor.externalDataEntries || 0) > 0 && tensor.externalPayloadVerified !== true) return null;
  const value = Number(tensor.initializerAvailableBytes ?? tensor.initializerBytes);
  return Number.isFinite(value) && value >= 0 ? value : null;
}

function classifyStaticBound(opType, macsStatus, macs, intensity, macsDecimal = null) {
  if (macsStatus === "not_assessed") return "not-assessed";
  if (macsDecimal === "0" || macs === 0 || RUNTIME_REVIEW_OPS.has(opType)) return "memory-bound";
  if (!Number.isFinite(intensity)) return "not-assessed";
  if (intensity >= 16) return "compute-bound";
  if (intensity >= 4) return "mixed";
  return "memory-bound";
}

function staticAction(opType, bound) {
  if (RUNTIME_REVIEW_OPS.has(opType)) return `Runtime review: ${RUNTIME_REVIEW_OPS.get(opType)[1]}`;
  if (bound === "compute-bound") return "Profile kernel selection, vectorization, and delegate support.";
  if (bound === "mixed") return "Check both tensor traffic and arithmetic intensity on target hardware.";
  return "Likely memory/copy sensitive; inspect tensor layout, fusion, and neighboring ops.";
}

function buildStages(ops) {
  const stages = [];
  let current = null;
  for (const op of ops) {
    const key = op.standard_domain ? op.name : `${op.domain}:${op.name}`;
    if (!current || current.key !== key) {
      current = {
        index: stages.length,
        key,
        first_op: op.index,
        last_op: op.index,
        op_count: 0,
        channels: [],
        macs: 0,
        macs_decimal: "0",
        _macs_exact: 0n,
        estimated_bytes: 0,
        mac_assessed_ops: 0,
        mac_not_assessed_ops: 0,
        byte_assessed_ops: 0,
        byte_not_assessed_ops: 0,
      };
      stages.push(current);
    }
    current.last_op = op.index;
    current.op_count += 1;
    if (op.macs_status === "assessed") {
      current._macs_exact += BigInt(op.macs_decimal ?? op.macs);
      current.macs_decimal = current._macs_exact.toString();
      current.macs = safeBigIntNumber(current._macs_exact);
      current.mac_assessed_ops += 1;
    }
    if (op.macs_status === "not_assessed") current.mac_not_assessed_ops += 1;
    if (op.estimated_bytes_status === "assessed") {
      current.estimated_bytes += Number(op.estimated_bytes || 0);
      current.byte_assessed_ops += 1;
    } else {
      current.byte_not_assessed_ops += 1;
    }
    const channel = positive(op.output_shapes?.[0]?.[1]) || positive(op.output_shapes?.[0]?.at?.(-1));
    if (channel && !current.channels.includes(channel)) current.channels.push(channel);
  }
  const totalMacs = stages.reduce((total, stage) => total + stage._macs_exact, 0n);
  for (const stage of stages) {
    stage.mac_percent = totalMacs > 0n ? exactNonnegativeRatio(stage._macs_exact, totalMacs) : 0;
    delete stage._macs_exact;
  }
  return stages;
}

function buildRooflineCsv(ops) {
  const rows = ["op_index,op_domain,op_name,macs,macs_decimal,macs_status,macs_reason,estimated_bytes,intensity_ops_per_byte,static_bound_guess,quantized_path,quantized_compute_path,quantization_state,quantization_detail,static_action"];
  for (const op of ops) {
    rows.push(
      [
        op.index,
        csvCell(op.domain || "ai.onnx"),
        csvCell(op.name),
        op.macs_status === "assessed" && op.macs != null ? op.macs : "",
        op.macs_status === "assessed" ? op.macs_decimal : "",
        op.macs_status,
        csvCell(op.macs_reason || ""),
        op.estimated_bytes_status === "assessed" ? op.estimated_bytes : "",
        Number.isFinite(op.intensity_ops_per_byte) ? op.intensity_ops_per_byte.toFixed(4) : "",
        op.static_bound_guess,
        Boolean(op.quantized_path),
        Boolean(op.quantized_compute_path),
        csvCell(op.quantization_state || "none"),
        csvCell(op.quantization_detail || ""),
        csvCell(op.static_action),
      ].join(","),
    );
  }
  return `${rows.join("\n")}\n`;
}

function buildStageMermaid(stages) {
  const lines = ["flowchart LR"];
  for (const stage of stages) {
    lines.push(`  S${stage.index}["#${stage.index} ${escapeMermaid(stage.key)}<br/>ops ${stage.first_op}-${stage.last_op}"]`);
    if (stage.index > 0) lines.push(`  S${stage.index - 1} --> S${stage.index}`);
  }
  return `${lines.join("\n")}\n`;
}

function buildMarkdown(filename, model, graph, ops, tensors, inputs, totalMacs, quantizedTensors, quantizationStatus, macAssessment) {
  const lines = [
    `# ${filename} ONNX Audit`,
    "",
    "## Summary",
    "",
    `- Format: ONNX`,
    `- Graph: ${graph.name || "-"}`,
    `- Producer: ${model.producer || "-"}`,
    `- IR / opset: ${model.irVersion || "unknown"} / ${model.opsets.map((opset) => `${opset.domain || "ai.onnx"}:${opset.version || "unknown"}`).join(" / ") || "unknown"}`,
    `- Operators: ${ops.length}`,
    `- Tensors: ${tensors.length}`,
    `- Model inputs: ${inputs.map((tensor) => `${tensor.name}${shapeText(tensor.shape)}`).join(" / ") || "-"}`,
    `- Quantization status: ${quantizationStatus?.label || "Unknown"} - ${quantizationStatus?.summary || "-"}`,
    `- Quantized compute MACs: ${quantizationStatus?.quantized_compute_mac_percent == null ? "N/A; MAC coverage incomplete" : formatPercentPlain(quantizationStatus.quantized_compute_mac_percent)}`,
    `- Op quantization states: ${countItemsText(quantizationStatus?.op_state_counts, "none detected")}`,
    `- Quantized tensor signals: ${quantizedTensors}`,
    `- Assessed MAC total: ${formatNumberPlain(macAssessment.total_assessed_macs_decimal)} (${macAssessment.assessed_compute_ops}/${macAssessment.compute_ops} compute ops; status ${macAssessment.status})`,
    "",
    "## Top Static Ops",
    "",
  ];
  for (const op of [...ops].sort((a, b) => {
    const left = exactNonnegativeInteger(a.macs_decimal ?? a.macs) || 0n;
    const right = exactNonnegativeInteger(b.macs_decimal ?? b.macs) || 0n;
    return left === right ? 0 : left > right ? -1 : 1;
  }).slice(0, 12)) {
    const macText = op.macs_status === "assessed" ? `${formatNumberPlain(op.macs_decimal ?? op.macs)} MACs` : `MACs N/A (${op.macs_reason})`;
    lines.push(`- #${String(op.index).padStart(3, "0")} ${op.name}: ${macText}, ${postureLabel(op.static_bound_guess)}`);
  }
  lines.push("", "## Op Quantization Map", "");
  lines.push("| Op | State | MACs | Detail |");
  lines.push("|---|---|---:|---|");
  const quantOps = ops.filter((op) => op.quantization_state && op.quantization_state !== "none");
  for (const op of quantOps.slice(0, 30)) {
    lines.push(`| #${String(op.index).padStart(3, "0")} ${op.name} | \`${op.quantization_state}\` | ${op.macs_status === "assessed" ? formatNumberPlain(op.macs_decimal ?? op.macs) : "N/A"} | ${escapeTableCell(op.quantization_detail || "-")} |`);
  }
  if (!quantOps.length) lines.push("| - | `none` | 0 | No op-level quantization signal was detected. |");
  return `${lines.join("\n")}\n`;
}

function classifyOnnxQuantization(ops, tensors, inputs, outputs, quantizedTensors) {
  const denseTensors = tensors.filter(isDenseTensorValue);
  const nonDenseValues = tensors.length - denseTensors.length;
  const int8Tensors = denseTensors.filter((tensor) => tensor.dtype === "INT8").length;
  const uint8Tensors = denseTensors.filter((tensor) => tensor.dtype === "UINT8").length;
  const floatTensors = denseTensors.filter((tensor) => isFloatDtype(tensor.dtype)).length;
  const inputDtypes = compactDtypeCounts(inputs);
  const outputDtypes = compactDtypeCounts(outputs);
  const allInputs8Bit = inputs.length > 0 && inputs.every((tensor) => is8BitDtype(tensor.dtype));
  const allOutputs8Bit = outputs.length > 0 && outputs.every((tensor) => is8BitDtype(tensor.dtype));
  const anyFloatIo = [...inputs, ...outputs].some((tensor) => isFloatDtype(tensor.dtype));
  const quantizeOps = ops.filter((op) => op.standard_domain && op.name === "QuantizeLinear").length;
  const dequantizeOps = ops.filter((op) => op.standard_domain && op.name === "DequantizeLinear").length;
  const computeOps = ops.filter((op) => op.standard_domain && ["Conv", "Gemm", "MatMul", "QLinearConv", "QLinearMatMul", "MatMulInteger", "ConvInteger"].includes(op.name));
  const quantizedComputeOps = computeOps.filter((op) => op.quantized_compute_path).length;
  const assessedComputeOps = computeOps.filter((op) => op.macs_status === "assessed");
  const macCoverageComplete = assessedComputeOps.length === computeOps.length;
  const totalMacs = exactNonnegativeSum(assessedComputeOps.map((op) => op.macs_decimal ?? op.macs));
  const quantizedComputeMacs = exactNonnegativeSum(assessedComputeOps.filter((op) => op.quantized_compute_path).map((op) => op.macs_decimal ?? op.macs));
  const opStateCounts = countBy(ops.map((op) => op.quantization_state || "none"));
  const quantizedTensorPercent = denseTensors.length ? quantizedTensors / denseTensors.length : 0;
  const quantizedComputeMacPercent = macCoverageComplete && totalMacs != null && quantizedComputeMacs != null && totalMacs > 0n
    ? exactNonnegativeRatio(quantizedComputeMacs, totalMacs)
    : null;
  const hasIntegerSignal = quantizedTensors > 0 || int8Tensors > 0 || uint8Tensors > 0 || quantizeOps > 0 || dequantizeOps > 0;

  let classification = "not_quantized_float";
  let label = "Not quantized";
  let summary = "No INT8/UINT8 tensors, Q/DQ ops, or ONNX integer compute ops were detected.";
  let fullInteger = false;
  if (hasIntegerSignal && !macCoverageComplete) {
    classification = "quantization_signals_partial_mac_assessment";
    label = "Quantization signals; MAC coverage partial";
    summary = "Quantization signals were observed, but quantized-compute MAC share is not reported because one or more compute op shapes/signatures were not assessed.";
  } else if (hasIntegerSignal && allInputs8Bit && allOutputs8Bit && quantizedComputeMacPercent >= 0.8) {
    classification = "full_integer";
    label = "Full integer quantized";
    summary = "Model I/O is 8-bit and most compute MACs appear to use ONNX integer quantized kernels.";
    fullInteger = true;
  } else if (hasIntegerSignal && anyFloatIo && quantizedComputeMacPercent >= 0.8) {
    classification = "integer_internal_float_io";
    label = "Internal INT8 with float I/O";
    summary = "Most compute MACs appear quantized, but ONNX graph inputs or outputs remain floating point.";
  } else if (hasIntegerSignal && quantizedComputeMacPercent >= 0.2) {
    classification = "mixed_quantization";
    label = "Mixed quantization";
    summary = "Quantized and floating-point compute regions are both present.";
  } else if (hasIntegerSignal) {
    classification = quantizedTensors || int8Tensors || uint8Tensors ? "dynamic_range_or_weight_only" : "qdq_signals_only";
    label = quantizedTensors || int8Tensors || uint8Tensors ? "Weight-only or dynamic-range quantization" : "Q/DQ signals only";
    summary = quantizedTensors || int8Tensors || uint8Tensors
      ? "Quantized tensors exist, but a mostly 8-bit activation compute path was not inferred."
      : "Q/DQ operators were detected, but no clear 8-bit compute path was inferred.";
  }

  return {
    classification,
    label,
    summary,
    detail: `Quantized dense-tensor signals: ${quantizedTensors}/${denseTensors.length} (${formatPercentPlain(quantizedTensorPercent)}); ${nonDenseValues} non-dense value(s) excluded from the tensor denominator. Quantized compute MACs: ${quantizedComputeMacPercent == null ? "N/A (MAC coverage incomplete)" : formatPercentPlain(quantizedComputeMacPercent)} across ${quantizedComputeOps}/${computeOps.length} compute ops. I/O dtype/kind contract: inputs [${inputDtypes.join(" / ") || "-"}], outputs [${outputDtypes.join(" / ") || "-"}]. Q/DQ ops: QuantizeLinear=${quantizeOps} / DequantizeLinear=${dequantizeOps}.`,
    quantized_tensor_percent: quantizedTensorPercent,
    quantized_compute_mac_percent: quantizedComputeMacPercent,
    quantized_compute_ops: quantizedComputeOps,
    compute_ops: computeOps.length,
    mac_assessed_compute_ops: assessedComputeOps.length,
    mac_coverage_complete: macCoverageComplete,
    quantize_ops: quantizeOps,
    dequantize_ops: dequantizeOps,
    int8_tensors: int8Tensors,
    uint8_tensors: uint8Tensors,
    float_tensors: floatTensors,
    dense_tensor_count: denseTensors.length,
    non_dense_value_count: nonDenseValues,
    input_dtypes: inputDtypes,
    output_dtypes: outputDtypes,
    op_state_counts: opStateCounts,
    full_integer: fullInteger,
  };
}

function compactDtypeCounts(tensors) {
  const counts = new Map();
  for (const tensor of tensors || []) {
    const key = isDenseTensorValue(tensor) ? tensor.dtype || "UNKNOWN" : `NON_DENSE:${tensor.value_kind || tensor.valueKind || "UNKNOWN"}`;
    counts.set(key, (counts.get(key) || 0) + 1);
  }
  return [...counts.entries()].sort(([a], [b]) => a.localeCompare(b)).map(([dtype, count]) => `${dtype}:${count}`);
}

function countItemsText(items, fallback = "-") {
  if (!Array.isArray(items) || !items.length) return fallback;
  if (items.every((item) => (item && typeof item === "object" ? item.name : item) === "none")) return fallback;
  const rows = items
    .map((item) => {
      if (item && typeof item === "object") return `${item.name || "unknown"}:${item.count ?? 0}`;
      return String(item || "");
    })
    .filter(Boolean);
  return rows.length ? rows.join(" / ") : fallback;
}

function postureLabel(bound) {
  if (bound === "compute-bound") return "high-intensity posture";
  if (bound === "memory-bound") return "low-intensity posture";
  if (bound === "mixed") return "mixed-intensity posture";
  return bound || "not-assessed";
}

function is8BitDtype(dtype) {
  return dtype === "INT8" || dtype === "UINT8";
}

function isFloatDtype(dtype) {
  return ["FLOAT4E2M1", "FLOAT8E4M3FN", "FLOAT8E4M3FNUZ", "FLOAT8E5M2", "FLOAT8E5M2FNUZ", "FLOAT8E8M0", "FLOAT16", "BFLOAT16", "FLOAT32", "FLOAT64"].includes(dtype);
}

function selectModelInputs(graphInputs, initializerNames, tensors, tensorIdByName) {
  const filtered = graphInputs.filter((input) => input.name && !initializerNames.has(input.name));
  const source = filtered.length ? filtered : graphInputs;
  return source
    .map((input) => tensors[tensorIdByName.get(input.name)])
    .filter(Boolean);
}

function upsertTensor(tensorMap, name, patch) {
  if (!name) return;
  const existing = tensorMap.get(name) || { name, dtype: "UNKNOWN", shape: [], shapeDeclared: false };
  const patchDeclaresShape = patch.shapeDeclared === true || (Array.isArray(patch.shape) && patch.shape.length > 0);
  tensorMap.set(name, {
    ...existing,
    ...patch,
    dtype: patch.dtype && patch.dtype !== "UNKNOWN" ? patch.dtype : existing.dtype,
    shape: patchDeclaresShape ? patch.shape : existing.shape,
    shapeDeclared: patchDeclaresShape ? true : existing.shapeDeclared === true,
  });
}

function upsertSparseInitializer(tensorMap, sparse) {
  const values = sparse?.values;
  const indices = sparse?.indices;
  const name = values?.name || "";
  if (!name) return;
  const components = [values, indices].filter(Boolean);
  const externalComponents = components.filter(isExternalInitializer);
  const availableComponents = components.filter((tensor) => !isExternalInitializer(tensor) || tensor.externalPayloadVerified === true);
  const embeddedComponents = components.filter((tensor) => !isExternalInitializer(tensor));
  const verifiedExternalComponents = externalComponents.filter((tensor) => tensor.externalPayloadVerified === true);
  const valueElements = values ? onnxInitializerElementCount(values) : 0;
  const indexElements = indices ? onnxInitializerElementCount(indices) : 0;
  const componentElements = (list) => {
    const exact = exactNonnegativeSum(list.map((tensor) => onnxInitializerElementCount(tensor)));
    return exact == null ? null : safeBigIntNumber(exact);
  };
  const componentBytes = (list) => sum(list.map((tensor) => Number(tensor.storedDataBytes || 0)));
  const componentRawBytes = (list) => sum(list.map((tensor) => Number(tensor.rawDataBytes || 0)));
  const componentRawZeroBytes = (list) => sum(list.map(rawTensorZeroByteCount));
  const projectEmbedded = (targetFloatBytes) => {
    const exact = exactNonnegativeSum(embeddedComponents.map((tensor) => isFloatDtype(tensor.dtype)
      ? safeExactProduct([onnxInitializerElementCount(tensor), targetFloatBytes])
      : Number(tensor.storedDataBytes || 0)));
    return exact == null ? null : safeBigIntNumber(exact);
  };
  upsertTensor(tensorMap, name, {
    dtype: values?.dtype || "UNKNOWN",
    shape: [...(sparse.dims || [])],
    shapeDeclared: true,
    valueKind: "tensor",
    role: "sparse_initializer",
    initializerStorageKind: "sparse_tensor_proto",
    initializerBytes: componentBytes(availableComponents),
    initializerElements: shapeElementCount(sparse.dims || [], true),
    initializerStoredElements: valueElements + indexElements,
    initializerEmbeddedBytes: componentBytes(embeddedComponents),
    initializerEmbeddedStoredElements: componentElements(embeddedComponents),
    initializerVerifiedExternalBytes: componentBytes(verifiedExternalComponents),
    initializerVerifiedExternalStoredElements: componentElements(verifiedExternalComponents),
    initializerAvailableBytes: componentBytes(availableComponents),
    initializerAvailableStoredElements: componentElements(availableComponents),
    initializerEmbeddedFloatBytes: componentBytes(embeddedComponents.filter((tensor) => isFloatDtype(tensor.dtype))),
    initializerProjectedEmbeddedFp16Bytes: projectEmbedded(2),
    initializerProjectedEmbeddedInt8Bytes: projectEmbedded(1),
    initializerExternalComponentCount: externalComponents.length,
    initializerVerifiedExternalComponentCount: verifiedExternalComponents.length,
    initializerRawDataBytes: componentRawBytes(availableComponents),
    initializerRawZeroBytes: componentRawZeroBytes(availableComponents),
    initializerTypedDataBytes: sum(availableComponents.map((tensor) => Number(tensor.typedDataBytes || 0))),
    externalDataEntries: sum(components.map((tensor) => Number(tensor.externalDataEntries || 0))),
    externalData: components.flatMap((tensor) => tensor.externalData || []),
    dataLocation: externalComponents.length ? 1 : 0,
    externalPayloadStatus: !externalComponents.length ? "not_applicable"
      : externalComponents.every((tensor) => tensor.externalPayloadVerified === true) ? "verified" : "incomplete_sparse_components",
    externalPayloadVerified: externalComponents.length > 0 && externalComponents.every((tensor) => tensor.externalPayloadVerified === true),
    initializerIntegerValuesStatus: "not_assessed_sparse_tensor_requires_index_reconstruction",
    initializerIntegerValuesComplete: false,
    initializerIntegerValues: [],
    initializerIntegerValuesExactComplete: false,
    initializerIntegerValuesExactDecimals: [],
    staticValuesStatus: "not_assessed_sparse_tensor_requires_index_reconstruction",
    staticValuesComplete: false,
    staticValues: [],
    staticValuesSource: "sparse_initializer",
    sparseNnz: values ? onnxInitializerElementCount(values) : null,
    sparseValueElements: valueElements,
    sparseIndexElements: indexElements,
    sparseIndexEncoding: indices?.shape?.length === 1 ? "linear_indices" : indices?.shape?.length === 2 ? "coordinate_indices" : "unresolved",
  });
}

function rawTensorZeroByteCount(tensor) {
  if (!(tensor?.rawData instanceof Uint8Array)) return 0;
  return tensor.rawData.reduce((count, value) => count + (value === 0 ? 1 : 0), 0);
}

function countQuantizedTensors(tensors) {
  return tensors.filter((tensor) => isDenseTensorValue(tensor) && !tensor.quantization_parameter_role
    && (["INT2", "UINT2", "INT4", "UINT4", "INT8", "UINT8"].includes(tensor.dtype) || Number(tensor.quant_scales || 0) > 0)).length;
}

function countPerChannelQuantizers(tensors) {
  return tensors.filter((tensor) => isDenseTensorValue(tensor)
    && (tensor.quantization_parameterization === "per_axis" || tensor.quantization_parameterization === "blocked")).length;
}

function countBy(values) {
  const counts = new Map();
  values.forEach((value) => counts.set(value || "UNKNOWN", (counts.get(value || "UNKNOWN") || 0) + 1));
  return [...counts.entries()]
    .map(([name, count]) => ({ name, count }))
    .sort((a, b) => b.count - a.count || a.name.localeCompare(b.name));
}

function tensorByteSize(tensor) {
  if (!isDenseTensorValue(tensor)) return Number(tensor?.initializerBytes || tensor?.initializer_bytes || 0);
  const raw = Number(tensor?.initializerBytes || tensor?.initializer_bytes || 0);
  const shapeDeclared = tensor?.shapeDeclared ?? tensor?.shape_declared;
  const elements = shapeElementCount(tensor?.shape, shapeDeclared);
  const shaped = dtypePayloadBytes(tensor?.dtype, elements);
  return shaped == null ? raw : Math.max(raw, shaped);
}

function tensorPayloadBytes(tensor) {
  if (!isDenseTensorValue(tensor) || tensorContractBlocksDeterministicCost(tensor)) return null;
  const shapeDeclared = tensor?.shapeDeclared ?? tensor?.shape_declared;
  if ((tensor?.initializerStorageKind || tensor?.initializer_storage_kind) === "sparse_tensor_proto") {
    if (!tensorShapeKnown(tensor?.shape, shapeDeclared)) return null;
    return dtypePayloadBytes(tensor?.dtype, shapeElementCount(tensor.shape, shapeDeclared));
  }
  const raw = Number(tensor?.initializerBytes || tensor?.initializer_bytes || 0);
  if (raw > 0) return raw;
  if (!tensorShapeKnown(tensor?.shape, shapeDeclared)) return null;
  return dtypePayloadBytes(tensor?.dtype, shapeElementCount(tensor.shape, shapeDeclared));
}

function shapeElementCount(shape, shapeDeclared = Array.isArray(shape) && shape.length > 0) {
  if (!Array.isArray(shape) || shapeDeclared !== true || shape.some((dim) => !Number.isSafeInteger(Number(dim)) || dim < 0)) return null;
  if (!shape.length) return 1;
  let product = 1;
  for (const rawDimension of shape) {
    const dimension = Number(rawDimension);
    if (dimension === 0) return 0;
    if (product > Math.floor(Number.MAX_SAFE_INTEGER / dimension)) return null;
    product *= dimension;
  }
  return product;
}

function tensorTypeName(value) {
  return TENSOR_TYPES[value]?.name || `TYPE_${value}`;
}

function attrInts(node, name) {
  const values = node.attributes.get(name)?.ints || [];
  return values.every(Number.isSafeInteger) ? values : [];
}

function attrInt(node, name, fallback) {
  const attr = node.attributes.get(name);
  if (attr?.i != null) return attr.i;
  if (attr?.ints?.length && Number.isSafeInteger(attr.ints[0])) return attr.ints[0];
  return fallback;
}

function attrString(node, name, fallback = "") {
  const value = node.attributes.get(name)?.s;
  return typeof value === "string" ? value : fallback;
}

function exactSpatialAttribute(node, name, rank, fallback, allowZero) {
  if (!node.attributes.has(name)) return Array(rank).fill(fallback);
  const values = attrInts(node, name);
  return values.length === rank && values.every((value) => allowZero ? value >= 0 : value > 0) ? values : null;
}

function convTransposePads(node, input, kernel, output, strides, dilations, outputPadding) {
  if (![strides, dilations, outputPadding].every(Boolean)
    || outputPadding.some((value, axis) => value >= strides[axis] && value >= dilations[axis])) return null;
  const rank = input.length, autoPad = attrString(node, "auto_pad", "NOTSET"), outputShape = attrInts(node, "output_shape");
  const kernelShape = node.attributes.has("kernel_shape") ? attrInts(node, "kernel_shape") : kernel;
  if (kernelShape.length !== rank || kernelShape.some((value, axis) => value !== kernel[axis])) return null;
  if (node.attributes.has("output_shape") && (outputShape.length !== rank || outputShape.some((value, axis) => value !== output[axis]))) return null;
  if (!["NOTSET", "VALID", "SAME_UPPER", "SAME_LOWER"].includes(autoPad)) return null;
  let pads;
  if (outputShape.length || autoPad.startsWith("SAME_")) {
    if (!outputShape.length && output.some((value, axis) => BigInt(value) !== BigInt(input[axis]) * BigInt(strides[axis]))) return null;
    pads = [];
    const ends = [];
    for (let axis = 0; axis < rank; axis += 1) {
      const total = BigInt(strides[axis]) * BigInt(input[axis] - 1) + BigInt(outputPadding[axis])
        + BigInt(kernel[axis] - 1) * BigInt(dilations[axis]) + 1n - BigInt(output[axis]);
      if (total < 0n || total > BigInt(Number.MAX_SAFE_INTEGER)) return null;
      const start = autoPad === "SAME_UPPER" ? total / 2n : total - total / 2n;
      pads.push(Number(start));
      ends.push(Number(total - start));
    }
    pads.push(...ends);
  } else if (autoPad === "VALID") {
    if (node.attributes.has("pads")) return null;
    pads = Array(rank * 2).fill(0);
  } else {
    pads = exactSpatialAttribute(node, "pads", rank * 2, 0, true);
  }
  if (!pads) return null;
  for (let axis = 0; axis < rank; axis += 1) {
    const expected = BigInt(strides[axis]) * BigInt(input[axis] - 1) + BigInt(outputPadding[axis])
      + BigInt(kernel[axis] - 1) * BigInt(dilations[axis]) + 1n - BigInt(pads[axis]) - BigInt(pads[axis + rank]);
    if (expected !== BigInt(output[axis])) return null;
  }
  return pads;
}

function floorDiv(value, divisor) {
  let quotient = value / divisor;
  if (value % divisor < 0n) quotient -= 1n;
  return quotient;
}

function convTransposeAxisPairs(input, kernel, stride, dilation, padStart, output) {
  if (![input, kernel, stride, dilation, padStart, output].every(Number.isSafeInteger) || input < 0 || kernel <= 0 || stride <= 0 || dilation <= 0 || padStart < 0 || output < 0) return null;
  if (Math.min(input, kernel) > 1_000_000) return null;
  const I = BigInt(input), K = BigInt(kernel), S = BigInt(stride), D = BigInt(dilation), P = BigInt(padStart), O = BigInt(output);
  let total = 0n;
  if (kernel <= input) {
    for (let k = 0n; k < K; k += 1n) {
      const low = [-1n, floorDiv(P - k * D - 1n, S)].reduce((a, b) => a > b ? a : b) + 1n;
      const high = [I - 1n, floorDiv(O - 1n + P - k * D, S)].reduce((a, b) => a < b ? a : b);
      if (high >= low) total += high - low + 1n;
    }
  } else {
    for (let i = 0n; i < I; i += 1n) {
      const low = [-1n, floorDiv(P - i * S - 1n, D)].reduce((a, b) => a > b ? a : b) + 1n;
      const high = [K - 1n, floorDiv(O - 1n + P - i * S, D)].reduce((a, b) => a < b ? a : b);
      if (high >= low) total += high - low + 1n;
    }
  }
  return total;
}

function positive(value) {
  const number = Number(value || 0);
  return number > 0 ? number : 0;
}

function sum(values) {
  return values.reduce((acc, value) => acc + Number(value || 0), 0);
}

function decodeString(bytes) {
  return textDecoder.decode(bytes);
}

function toSafeNumber(value) {
  const max = BigInt(Number.MAX_SAFE_INTEGER);
  return Number(value > max ? max : value);
}

function safeBigIntNumber(value) {
  const max = BigInt(Number.MAX_SAFE_INTEGER);
  const min = -max;
  return value > max || value < min ? null : Number(value);
}

function toSafeSignedNumber(value, bits) {
  const signed = signedVarint(value, bits);
  const max = BigInt(Number.MAX_SAFE_INTEGER);
  const min = -max;
  return Number(signed > max ? max : signed < min ? min : signed);
}

function sameShape(left, right) {
  return Array.isArray(left) && Array.isArray(right) && left.length === right.length && left.every((value, index) => value === right[index]);
}

function csvCell(value) {
  const text = String(value ?? "");
  return /[",\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
}

function escapeTableCell(value) {
  return String(value ?? "").replaceAll("|", "\\|").replaceAll("\n", " ");
}

function escapeMermaid(value) {
  return String(value || "").replaceAll('"', "'");
}

function shapeText(shape) {
  return Array.isArray(shape) && shape.length ? `[${shape.join("x")}]` : "[]";
}

function formatNumberPlain(value) {
  const exact = exactNonnegativeInteger(value);
  return new Intl.NumberFormat("en-US", { maximumFractionDigits: 0 }).format(exact ?? Number(value || 0));
}

function formatPercentPlain(value) {
  return `${(Number(value || 0) * 100).toFixed(1)}%`;
}

import { INLINE_PARAMETER_LIMIT } from "./constants.js";
import { canonicalJson } from "../../report-utils.js";
import { sha256TextHex } from "../../sha256-sync.js";
import {
  finiteNumberArray, integerArray, list, nonNegativeInteger, normalizeSha256, optionalInteger, positiveInteger, tensorIndex,
} from "./shared.js";

export function buildQuantizationContracts(analysis, format, graph, storage, tensors) {
  const graphValues = new Map(graph.values.map((value) => [value.native_index, value.id]));
  const storageObjects = new Map(storage.objects.map((value) => [value.native_index, value.id]));
  const records = [];
  for (const [position, tensor] of tensors.entries()) {
    const index = tensorIndex(tensor, position);
    const scales = finiteNumberArray(tensor.scale_sample?.length ? tensor.scale_sample : tensor.interface_scale_values);
    const zeroPoints = integerArray(tensor.zero_point_sample?.length ? tensor.zero_point_sample : tensor.interface_zero_point_values);
    const encoded = format === "gguf" && /^Q\d|^IQ\d|^TQ\d|^MXFP|^BF16$/i.test(String(tensor.dtype || ""));
    if (!scales.length && !zeroPoints.length && !encoded) continue;
    const parameterization = encoded
      ? { kind: "per_block", axes: [], block_size: positiveInteger(tensor.block_elements) }
      : quantizationParameterization(tensor, scales);
    const subjectRef = graphValues.get(index) || storageObjects.get(index);
    if (!subjectRef) continue;
    records.push({
      id: `quantization:${subjectRef}`,
      subject_ref: subjectRef,
      mapping: encoded ? {
        family: "format_defined_block_encoding",
        scheme: String(tensor.dtype || "unknown").toUpperCase(),
        zero_point_constraint: "format_defined",
      } : {
        family: "affine",
        scheme: "affine_unspecified_symmetry",
        zero_point_constraint: zeroPointConstraint(zeroPoints),
      },
      parameterization,
      storage: {
        data_type: String(tensor.dtype || "UNKNOWN").toUpperCase(),
        ...codeDomain(tensor.dtype),
        block_bytes: positiveInteger(tensor.block_bytes),
      },
      parameters: encoded ? {
        status: "encoded_in_format_defined_blocks",
        scale: null,
        zero_point: null,
      } : {
        status: scales.length && zeroPoints.length ? "complete_affine_vectors" : "partial_affine_vectors",
        scale: parameterVector(scales),
        zero_point: parameterVector(zeroPoints),
      },
      source: {
        native_locator: nativeQuantizationLocator(format, tensor, index),
        evidence_class: "OBSERVED_SERIALIZED_ARTIFACT",
      },
      completeness: encoded || (scales.length && zeroPoints.length) ? "complete_for_serialized_contract" : "partial_serialized_contract",
    });
  }
  for (const [index, parameter] of (["coreml", "mlmodel"].includes(format) ? list(analysis?.weight_integrity?.parameters) : []).entries()) {
    const quantization = parameter?.quantization;
    if (!quantization) continue;
    const subjectRef = `storage:parameter:${index}`;
    if (!storage.objects.some((row) => row.id === subjectRef)) continue;
    const perAxis = Number(quantization.scale_count || 0) > 1;
    records.push({
      id: `quantization:${subjectRef}`,
      subject_ref: subjectRef,
      mapping: {
        family: quantization.scheme === "lookup_table" ? "lookup_table" : "affine_scale_and_additive_bias",
        scheme: String(quantization.scheme || "unknown"),
        zero_point_constraint: "not_represented_as_zero_point",
      },
      parameterization: {
        kind: perAxis ? "per_axis" : "per_tensor",
        axes: Number.isSafeInteger(Number(quantization.axis)) ? [Number(quantization.axis)] : [],
        axis_status: perAxis && !Number.isSafeInteger(Number(quantization.axis)) ? "not_exposed_by_serialized_weight_contract" : "not_applicable_or_explicit",
        block_size: null,
      },
      storage: {
        data_type: parameter.storage === "int8_dynamic" ? "INT8" : `UINT${quantization.number_of_bits || ""}`,
        code_min: parameter.storage === "int8_dynamic" ? -128 : 0,
        code_max: parameter.storage === "int8_dynamic" ? 127 : Number.isSafeInteger(Number(quantization.number_of_bits)) ? 2 ** Number(quantization.number_of_bits) - 1 : null,
        code_domain_status: "coreml_serialized_weight_encoding",
        block_bytes: null,
      },
      parameters: {
        status: "complete_serialized_parameter_digests",
        scale: digestDescriptor(quantization.scale_count, quantization.scale_payload_sha256),
        additive_bias: digestDescriptor(quantization.bias_count, quantization.bias_payload_sha256),
        lookup_table: digestDescriptor(quantization.lookup_table_count, quantization.lookup_table_payload_sha256),
      },
      source: { native_locator: { format: "coreml", path: `weight_integrity.parameters[${index}].quantization` }, evidence_class: "OBSERVED_SERIALIZED_ARTIFACT" },
      completeness: "complete_for_serialized_contract",
    });
  }
  const safeContract = analysis?.safetensors?.quantization_contract;
  const storageByName = new Map(storage.objects.map((row) => [row.name, row.id]));
  for (const [index, module] of list(safeContract?.modules).entries()) {
    const related = Object.values(module?.tensors || {}).map((row) => storageByName.get(String(row?.tensor_name || ""))).filter(Boolean);
    const subjectRef = related[0];
    if (!subjectRef) continue;
    records.push({
      id: `quantization:safetensors:module:${index}`,
      subject_ref: subjectRef,
      related_storage_refs: [...new Set(related)],
      mapping: { family: "packed_integer_weight", scheme: String(safeContract.method || "unknown"), zero_point_constraint: module.symmetric === true ? "declared_symmetric" : module.symmetric === false ? "declared_asymmetric" : "not_declared" },
      parameterization: { kind: "per_group", axes: Number.isSafeInteger(Number(module.logical_weight_axis)) ? [Number(module.logical_weight_axis)] : [], block_size: positiveInteger(module.group_size), group_count: nonNegativeInteger(module.group_count) },
      storage: { data_type: `PACKED_UINT${module.bits || safeContract.bits || ""}`, code_min: 0, code_max: Number.isSafeInteger(Number(module.bits || safeContract.bits)) ? 2 ** Number(module.bits || safeContract.bits) - 1 : null, code_domain_status: "source_pinned_packed_layout", block_bytes: null },
      parameters: { status: String(module.quantization_payload_integrity?.status || module.status || "not_assessed"), scale: { count: decimalCount(module.scale_element_count), sha256: null, inline_values: null, inline_status: "referenced_storage_object" }, zero_point: { count: decimalCount(module.zero_point_code_capacity), sha256: null, inline_values: null, inline_status: module.zero_point_storage_transform || "not_assessed" } },
      source: { native_locator: { format: "safetensors", path: `safetensors.quantization_contract.modules[${index}]` }, evidence_class: String(safeContract.evidence_class || "OBSERVED/DERIVED_FROM_PINNED_FORMAT_SOURCE") },
      completeness: module.status === "pass" ? "complete_for_serialized_contract" : "partial_serialized_contract",
    });
  }
  return {
    status: records.length ? "assessed" : "not_applicable_no_serialized_quantization_contract",
    records,
    totals: {
      record_count: records.length,
      affine_record_count: records.filter((row) => row.mapping.family === "affine").length,
      block_encoding_record_count: records.filter((row) => row.mapping.family === "format_defined_block_encoding").length,
      complete_record_count: records.filter((row) => row.completeness === "complete_for_serialized_contract").length,
      partial_record_count: records.filter((row) => row.completeness !== "complete_for_serialized_contract").length,
    },
    interpretation_boundary: "Each record is bound to one canonical value or storage object. Per-tensor, per-axis, and per-block are partition descriptions, not an ordered precision scale. A zero zero-point is preserved as observed data and is not by itself promoted to a symmetric-quantization claim.",
  };
}

function nativeQuantizationLocator(format, tensor, index) {
  if (format === "tflite") return { format, path: `SubGraph[0].tensors[${index}].quantization` };
  if (format === "onnx") return { format, path: `graph.value[${JSON.stringify(String(tensor.name || ""))}].quantization_bindings` };
  if (format === "gguf") return { format, path: `tensor_infos[${index}].ggml_type` };
  return { format, path: `tensors[${index}].quantization` };
}

function quantizationParameterization(tensor, scales) {
  const declared = String(tensor.quantization_parameterization || tensor.scale_mode || "").toLowerCase().replaceAll("-", "_");
  const blockSize = positiveInteger(tensor.quantization_block_size);
  if (blockSize || declared.includes("block")) return { kind: "per_block", axes: optionalAxis(tensor), block_size: blockSize };
  if (scales.length > 1 || declared.includes("axis") || declared.includes("channel")) return { kind: "per_axis", axes: optionalAxis(tensor), block_size: null };
  return { kind: "per_tensor", axes: [], block_size: null };
}

function optionalAxis(tensor) {
  const value = optionalInteger(tensor.quantized_dimension);
  return value == null ? [] : [value];
}

function parameterVector(values) {
  const normalized = list(values);
  return {
    count: normalized.length,
    sha256: sha256TextHex(canonicalJson(normalized)),
    inline_values: normalized.length <= INLINE_PARAMETER_LIMIT ? normalized : null,
    inline_status: normalized.length <= INLINE_PARAMETER_LIMIT ? "complete" : "digest_only_large_vector",
  };
}

function digestDescriptor(count, digest) {
  const normalizedCount = nonNegativeInteger(count) || 0;
  return {
    count: normalizedCount,
    sha256: normalizeSha256(digest),
    inline_values: null,
    inline_status: normalizedCount ? "digest_only_native_parameter_vector" : "not_applicable",
  };
}

function decimalCount(value) {
  if (typeof value === "string" && /^\d+$/.test(value)) return value;
  const number = nonNegativeInteger(value);
  return number == null ? null : String(number);
}

function zeroPointConstraint(values) {
  if (!values.length) return "not_encoded";
  return values.every((value) => value === 0) ? "observed_all_zero_not_symmetry_proof" : "observed_contains_nonzero";
}

function codeDomain(dtype) {
  const key = String(dtype || "").toUpperCase();
  const domains = { INT8: [-128, 127], UINT8: [0, 255], INT16: [-32768, 32767], UINT16: [0, 65535], INT4: [-8, 7], UINT4: [0, 15] };
  const row = domains[key];
  return row ? { code_min: row[0], code_max: row[1], code_domain_status: "declared_storage_dtype" } : { code_min: null, code_max: null, code_domain_status: "format_defined_or_not_integer" };
}

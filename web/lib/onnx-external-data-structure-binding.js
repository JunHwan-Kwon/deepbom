export const ONNX_EXTERNAL_DATA_STRUCTURE_BINDING_SCHEMA = "deepbom.onnx_external_data_structure_binding.v1";

const SHA256 = /^[a-f0-9]{64}$/;

export function buildOnnxExternalDataStructureBinding(analysis, members) {
  const tensors = Array.isArray(analysis?.onnx_external_data?.tensors) ? analysis.onnx_external_data.tensors : [];
  const locations = [...new Set(tensors.map((row) => String(row.normalized_location || "")).filter(Boolean))].sort();
  const byLocation = new Map();
  for (const member of members || []) {
    const location = safeRelativePath(member?.model_relative_path);
    if (byLocation.has(location)) throw new Error(`ONNX external-data structure binding repeats ${location}.`);
    const byteLength = exactByteLength(member?.byte_length);
    if (!SHA256.test(String(member?.sha256 || "")) || byteLength == null) {
      throw new Error(`ONNX external-data structure binding identity is invalid for ${location}.`);
    }
    byLocation.set(location, { location, sha256: member.sha256, byte_length: byteLength });
  }
  const missing = locations.filter((location) => !byLocation.has(location));
  const unexpected = [...byLocation.keys()].filter((location) => !locations.includes(location));
  if (missing.length || unexpected.length) {
    throw new Error(`ONNX external-data closure differs from serialized locations: missing ${missing.join(", ") || "none"}; unexpected ${unexpected.join(", ") || "none"}.`);
  }
  let declaredBytes = 0n;
  let declaredElements = 0n;
  const inventory = new Map();
  const intervalsByLocation = new Map();
  for (const tensor of tensors) {
    const member = byLocation.get(String(tensor.normalized_location || ""));
    const offset = exactNonNegative(tensor.offset);
    const length = exactNonNegative(tensor.expected_payload_bytes ?? tensor.length);
    const end = exactNonNegative(tensor.range_end);
    if (!member || offset == null || length == null || end == null || end !== offset + length || end > member.byte_length) {
      throw new Error(`ONNX external-data range is invalid for ${String(tensor.tensor_name || "unnamed tensor")}.`);
    }
    const elements = exactShapeElements(tensor.shape);
    if (elements == null) throw new Error(`ONNX external-data shape is invalid for ${String(tensor.tensor_name || "unnamed tensor")}.`);
    declaredBytes += length;
    declaredElements += elements;
    const dtype = String(tensor.dtype || "UNKNOWN");
    const row = inventory.get(dtype) || { dtype, tensor_count: 0, element_count: 0n, declared_payload_bytes: 0n };
    row.tensor_count += 1;
    row.element_count += elements;
    row.declared_payload_bytes += length;
    inventory.set(dtype, row);
    const intervals = intervalsByLocation.get(member.location) || [];
    intervals.push({ start: offset, end });
    intervalsByLocation.set(member.location, intervals);
  }
  let uniqueBytes = 0n;
  for (const intervals of intervalsByLocation.values()) uniqueBytes += unionLength(intervals);
  return {
    schema: ONNX_EXTERNAL_DATA_STRUCTURE_BINDING_SCHEMA,
    status: "assessed_hash_bound_ranges_payload_numerical_decode_not_assessed",
    evidence_class: "OBSERVED_ACQUISITION/DERIVED_RANGE_VALIDATION",
    tensor_count: tensors.length,
    file_count: byLocation.size,
    declared_payload_bytes: exact(declaredBytes),
    unique_payload_bytes: exact(uniqueBytes),
    overlapping_reference_bytes: exact(declaredBytes - uniqueBytes),
    declared_element_count: exact(declaredElements),
    encoding_inventory: [...inventory.values()]
      .sort((left, right) => right.tensor_count - left.tensor_count || left.dtype.localeCompare(right.dtype))
      .map((row) => ({
        dtype: row.dtype,
        tensor_count: row.tensor_count,
        element_count: exact(row.element_count),
        declared_payload_bytes: exact(row.declared_payload_bytes),
        effective_bits_per_element: ratio(row.declared_payload_bytes * 8n, row.element_count),
      })),
    files: [...byLocation.values()].sort((left, right) => left.location.localeCompare(right.location)).map((row) => ({
      location: row.location,
      sha256: row.sha256,
      byte_length: exact(row.byte_length),
    })),
    range_conservation_status: "complete",
    numerical_payload_decode: "not_assessed_scan_policy_structure",
    interpretation_boundary: "Content-addressed sidecar identity, serialized location binding, tensor cardinality, and every declared offset/length range are assessed. Declared payload bytes count tensor references; unique payload bytes merge overlapping ranges per sidecar. Initializer values, finiteness, distributions, and model execution are not assessed in structure mode.",
  };
}

export function onnxExternalInitializerElementCount(analysis) {
  const tensors = Array.isArray(analysis?.onnx_external_data?.tensors) ? analysis.onnx_external_data.tensors : [];
  let count = 0n;
  for (const tensor of tensors) {
    if (!Array.isArray(tensor.shape) || tensor.shape.some((value) => !Number.isSafeInteger(Number(value)) || Number(value) < 0)) return null;
    count += tensor.shape.reduce((product, value) => product * BigInt(Number(value)), 1n);
  }
  return count;
}

function exactByteLength(value) {
  const raw = value?.decimal ?? value?.number ?? value;
  return exactNonNegative(raw);
}

function exactNonNegative(value) {
  if (typeof value === "bigint") return value >= 0n ? value : null;
  if (typeof value === "string" && /^\d+$/.test(value)) return BigInt(value);
  if (Number.isSafeInteger(Number(value)) && Number(value) >= 0) return BigInt(Number(value));
  return null;
}

function exact(value) {
  return { decimal: String(value), number: value <= BigInt(Number.MAX_SAFE_INTEGER) ? Number(value) : null };
}

function exactShapeElements(shape) {
  if (!Array.isArray(shape) || shape.some((value) => !Number.isSafeInteger(Number(value)) || Number(value) < 0)) return null;
  return shape.reduce((product, value) => product * BigInt(Number(value)), 1n);
}

function unionLength(intervals) {
  const sorted = [...intervals].sort((left, right) => left.start < right.start ? -1 : left.start > right.start ? 1 : 0);
  let total = 0n;
  let start = null;
  let end = null;
  for (const interval of sorted) {
    if (start == null || interval.start > end) {
      if (start != null) total += end - start;
      start = interval.start;
      end = interval.end;
    } else if (interval.end > end) end = interval.end;
  }
  return start == null ? 0n : total + end - start;
}

function ratio(numerator, denominator) {
  if (denominator <= 0n) return null;
  const scaled = numerator * 1_000_000n / denominator;
  return Number(scaled) / 1_000_000;
}

function safeRelativePath(value) {
  const normalized = String(value || "").replaceAll("\\", "/");
  if (!normalized || normalized.startsWith("/") || /^[A-Za-z]:/.test(normalized)
    || normalized.split("/").some((part) => !part || part === "." || part === "..")) {
    throw new Error("ONNX external-data member location is unsafe.");
  }
  return normalized;
}

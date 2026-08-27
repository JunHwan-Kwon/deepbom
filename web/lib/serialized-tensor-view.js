const INTEGER_DTYPES = /^(?:U|I)\d+$/;
const FLOAT_DTYPES = /^(?:F|BF)\d+/;

function safeElementCount(shape) {
  if (!Array.isArray(shape) || shape.some((value) => !Number.isSafeInteger(value) || value < 0)) return null;
  let count = 1n;
  for (const value of shape) {
    count *= BigInt(value);
    if (count > BigInt(Number.MAX_SAFE_INTEGER)) return null;
  }
  return Number(count);
}

function tensorGroup(rows, keyFor) {
  const groups = new Map();
  for (const row of rows) {
    const key = keyFor(row);
    const current = groups.get(key) || { id: key, tensor_count: 0, byte_length: 0 };
    current.tensor_count += 1;
    if (Number.isSafeInteger(row.byte_length) && row.byte_length >= 0) current.byte_length += row.byte_length;
    groups.set(key, current);
  }
  return [...groups.values()].sort((left, right) =>
    right.byte_length - left.byte_length || right.tensor_count - left.tensor_count || left.id.localeCompare(right.id));
}

function ggufTile(tensor) {
  const assessed = tensor.storage_status === "assessed";
  const block = assessed && Number(tensor.block_elements) > 1;
  const elements = safeElementCount(tensor.shape);
  const integrity = tensor.numerical_integrity || {};
  const integrityRisk = Number(integrity.nan_value_count || 0) + Number(integrity.positive_infinity_value_count || 0)
    + Number(integrity.negative_infinity_value_count || 0) + Number(integrity.invalid_encoding_value_count || 0) > 0;
  const integrityWarn = integrity.status === "not_assessed" || integrity.all_zero || integrity.constant_finite;
  return {
    ...tensor,
    encoding_class: !assessed ? "unsupported_or_invalid" : block ? "block_quantized" : "scalar",
    tone: !assessed || integrityRisk ? "risk" : integrityWarn ? "warn" : block ? "good" : "neutral",
    elements,
    bits_per_element: assessed && Number(tensor.block_elements) > 0
      ? Number(tensor.block_bytes) * 8 / Number(tensor.block_elements)
      : null,
  };
}

function safeTensorsTile(tensor) {
  const dtype = String(tensor.dtype || "UNKNOWN").toUpperCase();
  const elements = safeElementCount(tensor.shape);
  const bitsPerElement = elements > 0 && Number.isSafeInteger(tensor.byte_length)
    ? tensor.byte_length * 8 / elements
    : null;
  const encodingClass = INTEGER_DTYPES.test(dtype)
    ? "integer_storage"
    : FLOAT_DTYPES.test(dtype) ? "floating_storage"
      : dtype === "BOOL" ? "boolean_storage" : dtype.startsWith("C") ? "complex_storage" : "other_storage";
  const integrity = tensor.numerical_integrity || {};
  const integrityRisk = Number(integrity.nan_value_count || 0) + Number(integrity.positive_infinity_value_count || 0)
    + Number(integrity.negative_infinity_value_count || 0) + Number(integrity.invalid_encoding_value_count || 0) > 0;
  const integrityWarn = integrity.status === "not_assessed" || integrity.all_zero || integrity.constant_finite;
  return {
    ...tensor,
    encoding_class: encodingClass,
    tone: integrityRisk ? "risk" : integrityWarn ? "warn" : encodingClass === "integer_storage" ? "mixed" : "neutral",
    elements,
    bits_per_element: bitsPerElement,
  };
}

export function serializedTensorPresentation(analysis = {}) {
  const format = String(analysis.format || "").toLowerCase();
  if (!["gguf", "safetensors"].includes(format)) return null;
  const source = Array.isArray(analysis.tensors) ? analysis.tensors : [];
  const tiles = source.map(format === "gguf" ? ggufTile : safeTensorsTile);
  const assessedBytes = tiles.reduce((sum, tensor) =>
    sum + (Number.isSafeInteger(tensor.byte_length) && tensor.byte_length >= 0 ? tensor.byte_length : 0), 0);
  const groups = tensorGroup(tiles, (tensor) => String(tensor.dtype || "UNKNOWN"));
  const integrity = analysis.tensor_numerical_integrity || {};
  const assessedPayloads = Number(integrity.assessed_tensor_count || 0);
  const unassessedPayloads = Number(integrity.unassessed_tensor_count || 0);
  const nonfinitePayloads = tiles.filter((tensor) => {
    const row = tensor.numerical_integrity || {};
    return Number(row.nan_value_count || 0) + Number(row.positive_infinity_value_count || 0) + Number(row.negative_infinity_value_count || 0) > 0;
  }).length;
  const allZeroPayloads = tiles.filter((tensor) => tensor.numerical_integrity?.all_zero).length;

  if (format === "gguf") {
    const blockCount = tiles.filter((tensor) => tensor.encoding_class === "block_quantized").length;
    const scalarCount = tiles.filter((tensor) => tensor.encoding_class === "scalar").length;
    const unsupportedCount = tiles.length - blockCount - scalarCount;
    return {
      format,
      title: "GGUF Tensor Encoding & Storage Map",
      count_label: `${tiles.length} serialized tensor${tiles.length === 1 ? "" : "s"}`,
      legend: [
        ["good", "source-pinned block decode passed"],
        ["neutral", "scalar encoding"],
        ["warn", "all-zero / constant / unassessed payload"],
        ["risk", "non-finite / invalid encoding"],
      ],
      tiles,
      groups,
      assessed_bytes: assessedBytes,
      summary_rows: [
        { label: "Full numerical payload scans", count: assessedPayloads, denominator: tiles.length, tone: assessedPayloads === tiles.length ? "good" : "warn" },
        { label: "Block-encoded tensors", count: blockCount, denominator: tiles.length, tone: "good" },
        { label: "Scalar-encoded tensors", count: scalarCount, denominator: tiles.length, tone: "neutral" },
        { label: "Unsupported or invalid encodings", count: unsupportedCount, denominator: tiles.length, tone: unsupportedCount ? "risk" : "good" },
        { label: "Non-finite decoded payloads", count: nonfinitePayloads, denominator: tiles.length, tone: nonfinitePayloads ? "risk" : "good" },
        { label: "All-zero decoded payloads", count: allZeroPayloads, denominator: tiles.length, tone: allZeroPayloads ? "warn" : "good" },
        { label: "Unassessed payload semantics", count: unassessedPayloads, denominator: tiles.length, tone: unassessedPayloads ? "warn" : "good" },
      ],
      coverage_status: `${analysis.gguf?.payload_coverage_status || "not_assessed"}; numerical scan ${integrity.status || "not_assessed"}; byte conservation ${integrity.byte_conservation_status || "not_assessed"}`,
      scope: "GGUF values are fully decoded for source-pinned scalar and block encodings. It stores no execution-operator DAG, and GGML block encoding is not affine scale/zero-point quantization.",
    };
  }

  return {
    format,
    title: "SafeTensors Dtype & Payload Map",
    count_label: `${tiles.length} serialized tensor${tiles.length === 1 ? "" : "s"}`,
    legend: [
      ["mixed", "integer storage dtype"],
      ["neutral", "floating / other storage dtype"],
      ["warn", "all-zero / constant / unassessed payload"],
      ["risk", "non-finite / invalid encoding"],
    ],
    tiles,
    groups,
    assessed_bytes: assessedBytes,
    summary_rows: [
      { label: "Full numerical payload scans", count: assessedPayloads, denominator: tiles.length, tone: assessedPayloads === tiles.length ? "good" : "warn" },
      { label: "Non-finite payloads", count: nonfinitePayloads, denominator: tiles.length, tone: nonfinitePayloads ? "risk" : "good" },
      { label: "All-zero payloads", count: allZeroPayloads, denominator: tiles.length, tone: allZeroPayloads ? "warn" : "good" },
      { label: "Unassessed payload semantics", count: unassessedPayloads, denominator: tiles.length, tone: unassessedPayloads ? "warn" : "good" },
      ...groups.map((group) => ({
        label: `${group.id} storage`,
        count: group.tensor_count,
        denominator: tiles.length,
        byte_length: group.byte_length,
        byte_denominator: assessedBytes,
        tone: INTEGER_DTYPES.test(group.id) ? "mixed" : "neutral",
      })),
    ],
    coverage_status: `${analysis.safetensors?.payload_coverage_status || "not_assessed"}; numerical scan ${integrity.status || "not_assessed"}; byte conservation ${integrity.byte_conservation_status || "not_assessed"}`,
    scope: "SafeTensors scalar payloads are scanned in full where dtype semantics are source-bound. The container does not serialize an execution graph or affine activation contract.",
  };
}

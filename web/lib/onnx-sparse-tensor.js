const SOURCE_COMMIT = "be2b5fde82d9c8874f3d19328bdfe3b6962dc67b";

export const ONNX_SPARSE_TENSOR_SOURCE = Object.freeze({
  release: "v1.21.0",
  commit: SOURCE_COMMIT,
  source_ref: `https://raw.githubusercontent.com/onnx/onnx/${SOURCE_COMMIT}/onnx/onnx.in.proto`,
  sha256: "f4cbc198df3a0f3f4519d4d38cd2262e8f84057583b7313e2d0f981b3f68c213",
});

export function buildOnnxSparseTensorContract(entries = []) {
  const rows = entries.map((entry) => assessSparseTensor(entry));
  const invalidRows = rows.filter((row) => row.status === "fail");
  const unassessedRows = rows.filter((row) => row.status === "partial");
  const externalComponents = rows.reduce((sum, row) => sum + row.external_payload_component_count, 0);
  const verifiedExternalComponents = rows.reduce((sum, row) => sum + row.verified_external_payload_component_count, 0);
  return {
    schema: "deepbom.onnx_sparse_tensor_contract.v1.2",
    status: invalidRows.length ? "fail" : unassessedRows.length ? "partial" : "assessed",
    evidence_class: "SOURCE_PINNED_AND_OBSERVED",
    source_release: ONNX_SPARSE_TENSOR_SOURCE.release,
    source_commit: ONNX_SPARSE_TENSOR_SOURCE.commit,
    source_ref: ONNX_SPARSE_TENSOR_SOURCE.source_ref,
    source_sha256: ONNX_SPARSE_TENSOR_SOURCE.sha256,
    sparse_tensor_count: rows.length,
    graph_sparse_initializer_count: rows.filter((row) => row.tensor_role === "graph_sparse_initializer").length,
    attribute_sparse_tensor_count: rows.filter((row) => row.tensor_role !== "graph_sparse_initializer").length,
    valid_sparse_tensor_count: rows.filter((row) => row.status === "pass").length,
    invalid_sparse_tensor_count: invalidRows.length,
    partially_assessed_sparse_tensor_count: unassessedRows.length,
    declared_nnz_total: rows.every((row) => row.nnz != null) ? rows.reduce((sum, row) => sum + row.nnz, 0) : null,
    dense_logical_element_total: rows.every((row) => row.dense_logical_elements != null) ? rows.reduce((sum, row) => sum + row.dense_logical_elements, 0) : null,
    embedded_payload_bytes: rows.reduce((sum, row) => sum + row.embedded_payload_bytes, 0),
    external_payload_component_count: externalComponents,
    verified_external_payload_component_count: verifiedExternalComponents,
    external_payload_coverage_status: externalComponents === 0
      ? "not_applicable_no_external_payload"
      : verifiedExternalComponents === externalComponents ? "verified" : "incomplete",
    index_content_assessed_sparse_tensor_count: rows.filter((row) => ["assessed", "fail"].includes(row.index_content_status)).length,
    index_content_failed_sparse_tensor_count: rows.filter((row) => row.index_content_status === "fail").length,
    index_content_unassessed_sparse_tensor_count: rows.filter((row) => row.index_content_status.startsWith("not_assessed")).length,
    assessed_index_count: rows.reduce((sum, row) => sum + Number(row.assessed_index_count || 0), 0),
    out_of_bounds_index_count: rows.reduce((sum, row) => sum + Number(row.out_of_bounds_index_count || 0), 0),
    duplicate_index_count: rows.reduce((sum, row) => sum + Number(row.duplicate_index_count || 0), 0),
    unsorted_index_count: rows.reduce((sum, row) => sum + Number(row.unsorted_index_count || 0), 0),
    invalid_rows: invalidRows,
    partially_assessed_rows: unassessedRows,
    rows,
    method: "Validate every parsed SparseTensorProto values/indices/dims tuple: values rank and NNZ, INT64 linear [NNZ] or coordinate [NNZ, rank] encoding, dense-shape cardinality, graph-initializer naming, independently verified TensorProto payload components, and exact ascending/unique/in-bounds index contents whenever the complete index payload is available.",
    interpretation_boundary: "SparseTensorProto storage integrity is assessed without densifying values. A graph sparse_initializer is a logical dense tensor initializer stored in sparse form, while a TypeProto sparse-tensor value remains non-dense. Sparse-kernel execution behavior is runtime evidence and is not inferred from storage encoding.",
  };
}

function assessSparseTensor(entry) {
  const sparse = entry.sparse || {};
  const values = sparse.values || null;
  const indices = sparse.indices || null;
  const reasons = [];
  const dims = Array.isArray(sparse.dims) ? sparse.dims.map(Number) : [];
  if (!values) reasons.push("sparse_values_tensor_missing");
  if (!indices) reasons.push("sparse_indices_tensor_missing");
  if (!dims.every((dim) => Number.isSafeInteger(dim) && dim >= 0)) reasons.push("sparse_dense_shape_invalid");
  if (entry.role === "graph_sparse_initializer" && !values?.name) reasons.push("sparse_initializer_values_name_missing");
  if (!validElementType(values?.dtype)) reasons.push("sparse_values_dtype_invalid");
  if (indices?.dtype !== "INT64") reasons.push("sparse_indices_dtype_not_int64");
  const valueShape = values?.shape || [];
  const indexShape = indices?.shape || [];
  const nnz = valueShape.length === 1 && validDimension(valueShape[0]) ? Number(valueShape[0]) : null;
  if (nnz == null) reasons.push("sparse_values_shape_not_rank1_nnz");
  let indexEncoding = "unresolved";
  if (nnz != null && indexShape.length === 1 && indexShape[0] === nnz) indexEncoding = "linear_indices";
  else if (nnz != null && indexShape.length === 2 && indexShape[0] === nnz && indexShape[1] === dims.length) indexEncoding = "coordinate_indices";
  else reasons.push("sparse_indices_shape_incompatible");
  const denseElements = checkedProduct(dims);
  if (denseElements == null) reasons.push("sparse_dense_shape_cardinality_unsafe");
  else if (nnz != null && nnz > denseElements) reasons.push("sparse_nnz_exceeds_dense_cardinality");
  const payloads = [values, indices].filter(Boolean);
  const externalPayloads = payloads.filter(isExternalTensor);
  const verifiedExternalPayloads = externalPayloads.filter((tensor) => tensor.externalPayloadVerified === true);
  const indexAssessment = assessIndexContents(indices, indexEncoding, nnz, dims, denseElements, reasons);
  const payloadIncomplete = externalPayloads.length !== verifiedExternalPayloads.length;
  return {
    scope: entry.scope,
    tensor_role: entry.role,
    sparse_tensor_name: values?.name || "",
    status: reasons.length ? "fail" : payloadIncomplete || indexAssessment.status.startsWith("not_assessed") ? "partial" : "pass",
    reason_codes: [...new Set(reasons)].sort(),
    values_dtype: values?.dtype || "UNKNOWN",
    values_shape: [...valueShape],
    indices_dtype: indices?.dtype || "UNKNOWN",
    indices_shape: [...indexShape],
    dense_shape: dims,
    dense_rank: dims.length,
    nnz,
    dense_logical_elements: denseElements,
    index_encoding: indexEncoding,
    index_content_status: indexAssessment.status,
    assessed_index_count: indexAssessment.assessedIndexCount,
    out_of_bounds_index_count: indexAssessment.outOfBoundsCount,
    duplicate_index_count: indexAssessment.duplicateCount,
    unsorted_index_count: indexAssessment.unsortedCount,
    embedded_payload_bytes: payloads.filter((tensor) => !isExternalTensor(tensor)).reduce((sum, tensor) => sum + Number(tensor.storedDataBytes || 0), 0),
    external_payload_component_count: externalPayloads.length,
    verified_external_payload_component_count: verifiedExternalPayloads.length,
    payload_status: externalPayloads.length === 0
      ? "embedded_or_empty"
      : verifiedExternalPayloads.length === externalPayloads.length ? "verified" : "incomplete",
  };
}

function assessIndexContents(indices, encoding, nnz, dims, denseElements, reasons) {
  const empty = { assessedIndexCount: 0, outOfBoundsCount: 0, duplicateCount: 0, unsortedCount: 0 };
  if (!indices || encoding === "unresolved" || nnz == null || denseElements == null) return { status: "not_assessed_invalid_structure", ...empty };
  if (isExternalTensor(indices) && indices.externalPayloadVerified !== true) return { status: "not_assessed_external_payload_unavailable", ...empty };
  if (indices.staticValuesComplete !== true) {
    reasons.push("sparse_indices_payload_missing_invalid_or_undecodable");
    return { status: "not_assessed_payload_invalid", ...empty };
  }
  const values = indices.staticValues || [];
  const expected = encoding === "linear_indices" ? nnz : nnz * dims.length;
  if (!Number.isSafeInteger(expected) || values.length !== expected) {
    reasons.push("sparse_indices_payload_cardinality_mismatch");
    return { status: "not_assessed_payload_invalid", ...empty };
  }
  let outOfBoundsCount = 0;
  let duplicateCount = 0;
  let unsortedCount = 0;
  if (encoding === "linear_indices") {
    for (let index = 0; index < values.length; index += 1) {
      const value = values[index];
      if (!Number.isSafeInteger(value) || value < 0 || value >= denseElements) outOfBoundsCount += 1;
      if (index > 0 && value === values[index - 1]) duplicateCount += 1;
      else if (index > 0 && value < values[index - 1]) unsortedCount += 1;
    }
  } else {
    let previous = null;
    for (let rowIndex = 0; rowIndex < nnz; rowIndex += 1) {
      const row = values.slice(rowIndex * dims.length, (rowIndex + 1) * dims.length);
      if (row.some((value, axis) => !Number.isSafeInteger(value) || value < 0 || value >= dims[axis])) outOfBoundsCount += 1;
      if (previous) {
        const order = lexicographicCompare(row, previous);
        if (order === 0) duplicateCount += 1;
        else if (order < 0) unsortedCount += 1;
      }
      previous = row;
    }
  }
  if (outOfBoundsCount) reasons.push("sparse_indices_out_of_bounds");
  if (duplicateCount) reasons.push("sparse_indices_duplicate");
  if (unsortedCount) reasons.push("sparse_indices_not_ascending");
  return {
    status: outOfBoundsCount || duplicateCount || unsortedCount ? "fail" : "assessed",
    assessedIndexCount: nnz,
    outOfBoundsCount,
    duplicateCount,
    unsortedCount,
  };
}

function lexicographicCompare(left, right) {
  for (let index = 0; index < left.length; index += 1) {
    if (left[index] < right[index]) return -1;
    if (left[index] > right[index]) return 1;
  }
  return 0;
}

function checkedProduct(dims) {
  let product = 1;
  for (const dim of dims) {
    if (!validDimension(dim) || product > Math.floor(Number.MAX_SAFE_INTEGER / Math.max(1, dim))) return null;
    product *= dim;
  }
  return product;
}

function validDimension(value) {
  return Number.isSafeInteger(Number(value)) && Number(value) >= 0;
}

function validElementType(dtype) {
  return Boolean(dtype) && dtype !== "UNKNOWN" && dtype !== "UNDEFINED" && !String(dtype).startsWith("TYPE_");
}

function isExternalTensor(tensor) {
  return Number(tensor?.dataLocation || 0) === 1 || Number(tensor?.externalDataEntries || 0) > 0 || (tensor?.externalData || []).length > 0;
}

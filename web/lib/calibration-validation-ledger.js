import { canonicalJson } from "./report-utils.js";
import { sha256TextHex } from "./sha256-sync.js";

export const REPRESENTATIVE_DATASET_CAPTURE_SCHEMA = "deepbom.representative_dataset_capture.v1";
export const CALIBRATION_VALIDATION_LEDGER_SCHEMA = "deepbom.calibration_validation_ledger.v1";

const MAX_SAMPLES = 10_000;
const MAX_VALUES = 20_000_000;
const SHA256 = /^[a-f0-9]{64}$/;
const INTEGER_RANGES = Object.freeze({
  INT8: [-128, 127],
  UINT8: [0, 255],
  INT16: [-32768, 32767],
  UINT16: [0, 65535],
  INT32: [-2147483648, 2147483647],
  UINT32: [0, 4294967295],
});
const FLOAT_DTYPES = new Set(["FLOAT16", "FLOAT32", "FLOAT64", "BFLOAT16"]);

export function buildCalibrationValidationLedger(capture, { expectedArtifactSha256 = null, expectedInterface = null } = {}) {
  validateCaptureRoot(capture, expectedArtifactSha256);
  const interfaceBinding = normalizeExpectedInterface(expectedInterface);
  let totalValues = 0;
  const samples = capture.samples.map((sample, sampleIndex) => {
    const row = buildSample(sample, sampleIndex, interfaceBinding);
    totalValues += row.captured_value_count;
    assert(totalValues <= MAX_VALUES, `Capture exceeds the ${MAX_VALUES} value safety limit.`);
    return row;
  });
  const sourceCaptureSha256 = sha256TextHex(canonicalJson(capture));
  const aggregate = aggregateSamples(samples);
  const ledger = {
    schema: CALIBRATION_VALIDATION_LEDGER_SCHEMA,
    method_version: "2026-08-16.1",
    status: "assessed",
    evidence_class: "DERIVED_FROM_HASH_BOUND_CAPTURED_DATASET",
    artifact_sha256: capture.artifact_sha256,
    source_capture_schema: capture.schema,
    source_capture_sha256: sourceCaptureSha256,
    dataset: {
      id: capture.dataset.id,
      version: capture.dataset.version,
      manifest_sha256: capture.dataset.manifest_sha256,
      preprocessing_contract_sha256: capture.dataset.preprocessing_contract_sha256 ?? null,
      representativeness_claim: capture.dataset.representativeness_claim ?? "externally_declared_not_verified_by_deepbom",
    },
    runtime: normalizeRuntime(capture.runtime),
    interface_binding: interfaceBinding ? {
      status: "matched_to_static_audit_external_interface",
      input_count: interfaceBinding.inputs.length,
      output_count: interfaceBinding.outputs.length,
      inputs: interfaceBinding.inputs,
      outputs: interfaceBinding.outputs,
      dynamic_dimension_rule: "rank and every non-negative declared dimension must match; negative or symbolic dimensions accept the captured concrete runtime dimension",
    } : {
      status: "not_assessed_no_static_interface_supplied",
      input_count: null,
      output_count: null,
      inputs: null,
      outputs: null,
      dynamic_dimension_rule: null,
    },
    sample_count: samples.length,
    captured_value_count: totalValues,
    input_endpoint_saturation: aggregate.input_endpoint_saturation,
    reference_output_drift: aggregate.reference_output_drift,
    repeat_nondeterminism: aggregate.repeat_nondeterminism,
    samples,
    hash_contract: {
      algorithm: "SHA-256",
      canonicalization: "RFC8785-JCS",
      payload: "the complete ledger object with /ledger_sha256 omitted",
      excluded_pointers: ["/ledger_sha256"],
    },
    interpretation_boundary: "Input endpoint counts are exact for the captured integer interface tensors. Reference drift compares captured numerical outputs only when a same-dtype, same-shape external reference is present. Repeat nondeterminism compares repeated outputs from the declared runtime capture. These observations do not establish dataset representativeness, calibration quality, task accuracy, clinical validity, production preprocessing identity, or device-wide determinism.",
    ledger_sha256: "",
  };
  ledger.ledger_sha256 = ledgerDigest(ledger);
  return ledger;
}

export function validateCalibrationValidationLedger(ledger, capture, { expectedArtifactSha256 = null, expectedInterface = null } = {}) {
  assert(ledger?.schema === CALIBRATION_VALIDATION_LEDGER_SCHEMA, "Calibration validation ledger schema is invalid.");
  const rebuilt = buildCalibrationValidationLedger(capture, { expectedArtifactSha256, expectedInterface });
  assert(canonicalJson(ledger) === canonicalJson(rebuilt), "Calibration validation ledger does not reconstruct from the bound capture.");
  assert(ledger.ledger_sha256 === ledgerDigest(ledger), "Calibration validation ledger SHA-256 mismatch.");
  return {
    status: "independently_reconstructed",
    sample_count: rebuilt.sample_count,
    source_capture_sha256: rebuilt.source_capture_sha256,
    ledger_sha256: rebuilt.ledger_sha256,
  };
}

export function representativeDatasetInterfaceFromAnalysis(analysis) {
  if (!analysis) return null;
  const inputs = Array.isArray(analysis.input_contracts) && analysis.input_contracts.length
    ? analysis.input_contracts
    : Array.isArray(analysis.inputs) ? analysis.inputs : [];
  const outputs = Array.isArray(analysis.outputs) ? analysis.outputs : [];
  if (!inputs.length || !outputs.length) return null;
  return {
    inputs: inputs.map((row, index) => interfaceRow(row, index)),
    outputs: outputs.map((row, index) => interfaceRow(row, index)),
  };
}

function validateCaptureRoot(capture, expectedArtifactSha256) {
  assert(isPlainObject(capture), "Representative dataset capture must be a JSON object.");
  assert(capture.schema === REPRESENTATIVE_DATASET_CAPTURE_SCHEMA, `Expected capture schema ${REPRESENTATIVE_DATASET_CAPTURE_SCHEMA}.`);
  assertSha256(capture.artifact_sha256, "artifact_sha256");
  if (expectedArtifactSha256 != null) {
    assertSha256(expectedArtifactSha256, "expected artifact SHA-256");
    assert(capture.artifact_sha256 === expectedArtifactSha256, "Representative dataset capture is bound to a different artifact SHA-256.");
  }
  assert(isPlainObject(capture.dataset), "Capture dataset identity is required.");
  assertNonempty(capture.dataset.id, "dataset.id");
  assertNonempty(capture.dataset.version, "dataset.version");
  assertSha256(capture.dataset.manifest_sha256, "dataset.manifest_sha256");
  if (capture.dataset.preprocessing_contract_sha256 != null) assertSha256(capture.dataset.preprocessing_contract_sha256, "dataset.preprocessing_contract_sha256");
  assert(isPlainObject(capture.runtime), "Capture runtime identity is required.");
  assertNonempty(capture.runtime.name, "runtime.name");
  assertNonempty(capture.runtime.version, "runtime.version");
  assertNonempty(capture.runtime.backend, "runtime.backend");
  for (const key of ["binary_sha256", "build_inventory_sha256", "device_profile_sha256"]) {
    if (capture.runtime[key] != null) assertSha256(capture.runtime[key], `runtime.${key}`);
  }
  assert(Array.isArray(capture.samples) && capture.samples.length > 0, "Capture must contain at least one sample.");
  assert(capture.samples.length <= MAX_SAMPLES, `Capture exceeds the ${MAX_SAMPLES} sample safety limit.`);
}

function buildSample(sample, sampleIndex, expectedInterface) {
  assert(isPlainObject(sample), `samples[${sampleIndex}] must be an object.`);
  assertNonempty(sample.sample_id, `samples[${sampleIndex}].sample_id`);
  const inputs = normalizeTensorList(sample.inputs, `samples[${sampleIndex}].inputs`, true);
  const runs = normalizeRuns(sample.runs, sampleIndex);
  const references = sample.reference_outputs == null
    ? null
    : normalizeTensorList(sample.reference_outputs, `samples[${sampleIndex}].reference_outputs`, false);
  const firstOutputs = runs[0].outputs;
  if (expectedInterface) {
    assertCapturedInterface(expectedInterface.inputs, inputs, `${sample.sample_id}/inputs`);
    assertCapturedInterface(expectedInterface.outputs, firstOutputs, `${sample.sample_id}/outputs`);
  }
  for (const run of runs) assertTensorListContract(firstOutputs, run.outputs, `${sample.sample_id}/run_${run.run_index}`);
  if (references) assertTensorListContract(references, firstOutputs, `${sample.sample_id}/reference`);

  const endpointRows = inputs
    .map((tensor, inputIndex) => endpointSummary(tensor, inputIndex))
    .filter(Boolean);
  const referenceComparisons = references
    ? runs.map((run) => compareTensorLists(references, run.outputs, `${sample.sample_id}/reference/run_${run.run_index}`))
    : [];
  const repeatComparisons = runs.slice(1)
    .map((run) => ({
      baseline_run_index: runs[0].run_index,
      candidate_run_index: run.run_index,
      ...compareTensorLists(firstOutputs, run.outputs, `${sample.sample_id}/run_${runs[0].run_index}/run_${run.run_index}`),
    }));
  return {
    sample_index: sampleIndex,
    sample_id: sample.sample_id,
    sample_manifest_entry_sha256: optionalSha256(sample.sample_manifest_entry_sha256, `samples[${sampleIndex}].sample_manifest_entry_sha256`),
    input_tensor_count: inputs.length,
    output_tensor_count: firstOutputs.length,
    run_count: runs.length,
    captured_value_count: inputs.reduce((sum, row) => sum + row.element_count, 0)
      + runs.reduce((sum, run) => sum + run.outputs.reduce((inner, row) => inner + row.element_count, 0), 0)
      + (references || []).reduce((sum, row) => sum + row.element_count, 0),
    inputs: inputs.map(publicTensor),
    input_endpoint_saturation: summarizeEndpointRows(endpointRows),
    reference_status: references ? "assessed" : "not_provided",
    reference_outputs: references?.map(publicTensor) || null,
    runs: runs.map((run) => ({ run_index: run.run_index, outputs: run.outputs.map(publicTensor) })),
    reference_comparisons: referenceComparisons,
    repeat_status: repeatComparisons.length ? "assessed" : "not_assessed_single_run",
    repeat_comparisons: repeatComparisons,
  };
}

function normalizeExpectedInterface(value) {
  if (value == null) return null;
  assert(isPlainObject(value), "Expected interface must be an object.");
  assert(Array.isArray(value.inputs) && value.inputs.length > 0, "Expected interface inputs are required.");
  assert(Array.isArray(value.outputs) && value.outputs.length > 0, "Expected interface outputs are required.");
  return {
    inputs: value.inputs.map((row, index) => normalizeInterfaceRow(row, `expectedInterface.inputs[${index}]`)),
    outputs: value.outputs.map((row, index) => normalizeInterfaceRow(row, `expectedInterface.outputs[${index}]`)),
  };
}

function interfaceRow(row, fallbackIndex) {
  return {
    tensor_index: Number.isSafeInteger(row?.tensor_index) ? row.tensor_index : Number.isSafeInteger(row?.index) ? row.index : null,
    name: String(row?.name ?? ""),
    dtype: String(row?.dtype ?? row?.data_type ?? "").toUpperCase(),
    shape: Array.isArray(row?.shape) ? row.shape : Array.isArray(row?.shape_signature) ? row.shape_signature : [],
    parameter_index: fallbackIndex,
  };
}

function normalizeInterfaceRow(row, path) {
  assert(isPlainObject(row), `${path} must be an object.`);
  const dtype = String(row.dtype || "").toUpperCase();
  assert(dtype, `${path}.dtype is required.`);
  assert(Array.isArray(row.shape), `${path}.shape is required.`);
  const shape = row.shape.map((dim, index) => {
    if (Number.isSafeInteger(dim)) return dim;
    if (typeof dim === "string" && dim.trim()) return dim;
    assert(false, `${path}.shape[${index}] is invalid.`);
  });
  return {
    parameter_index: Number.isSafeInteger(row.parameter_index) ? row.parameter_index : null,
    tensor_index: Number.isSafeInteger(row.tensor_index) ? row.tensor_index : null,
    name: String(row.name ?? ""),
    dtype,
    shape,
  };
}

function assertCapturedInterface(expected, captured, label) {
  assert(expected.length === captured.length, `${label} tensor count ${captured.length} does not match audited interface count ${expected.length}.`);
  expected.forEach((contract, index) => {
    const tensor = captured[index];
    assert(contract.dtype === tensor.dtype, `${label}[${index}] dtype ${tensor.dtype} does not match audited ${contract.dtype}.`);
    assert(contract.shape.length === tensor.shape.length, `${label}[${index}] rank ${tensor.shape.length} does not match audited rank ${contract.shape.length}.`);
    contract.shape.forEach((dim, axis) => {
      if (Number.isSafeInteger(dim) && dim >= 0) assert(dim === tensor.shape[axis], `${label}[${index}] shape axis ${axis}=${tensor.shape[axis]} does not match audited ${dim}.`);
    });
    if (contract.tensor_index != null && tensor.tensor_index != null) assert(contract.tensor_index === tensor.tensor_index, `${label}[${index}] tensor index mismatch.`);
    if (contract.name && tensor.name) assert(contract.name === tensor.name, `${label}[${index}] tensor name mismatch.`);
  });
}

function normalizeRuns(runs, sampleIndex) {
  assert(Array.isArray(runs) && runs.length > 0, `samples[${sampleIndex}].runs must contain at least one run.`);
  const seen = new Set();
  return runs.map((run, runIndex) => {
    assert(isPlainObject(run), `samples[${sampleIndex}].runs[${runIndex}] must be an object.`);
    assert(Number.isSafeInteger(run.run_index) && run.run_index >= 0, `samples[${sampleIndex}].runs[${runIndex}].run_index is invalid.`);
    assert(!seen.has(run.run_index), `samples[${sampleIndex}] contains duplicate run_index ${run.run_index}.`);
    seen.add(run.run_index);
    return {
      run_index: run.run_index,
      outputs: normalizeTensorList(run.outputs, `samples[${sampleIndex}].runs[${runIndex}].outputs`, false),
    };
  }).sort((left, right) => left.run_index - right.run_index);
}

function normalizeTensorList(tensors, path, allowQuantization) {
  assert(Array.isArray(tensors) && tensors.length > 0, `${path} must contain at least one tensor.`);
  return tensors.map((tensor, index) => normalizeTensor(tensor, `${path}[${index}]`, allowQuantization));
}

function normalizeTensor(tensor, path, allowQuantization) {
  assert(isPlainObject(tensor), `${path} must be an object.`);
  const dtype = String(tensor.dtype || "").toUpperCase();
  assert(INTEGER_RANGES[dtype] || FLOAT_DTYPES.has(dtype) || dtype === "BOOL", `${path}.dtype ${dtype || "<empty>"} is unsupported.`);
  assert(Array.isArray(tensor.shape) && tensor.shape.every((dim) => Number.isSafeInteger(dim) && dim >= 0), `${path}.shape must contain non-negative integer dimensions.`);
  const expected = tensor.shape.reduce((product, dim) => product * dim, 1);
  assert(Number.isSafeInteger(expected), `${path}.shape element count is not safely representable.`);
  assert(Array.isArray(tensor.values) && tensor.values.length === expected, `${path}.values cardinality ${tensor.values?.length ?? "missing"} does not equal shape product ${expected}.`);
  const values = tensor.values.map((value, valueIndex) => normalizeValue(value, dtype, `${path}.values[${valueIndex}]`));
  const tensorIndex = tensor.tensor_index == null ? null : tensor.tensor_index;
  assert(tensorIndex == null || Number.isSafeInteger(tensorIndex) && tensorIndex >= 0, `${path}.tensor_index is invalid.`);
  const name = String(tensor.name ?? "");
  const quantization = allowQuantization && tensor.quantization != null ? normalizeQuantization(tensor.quantization, path) : null;
  const digestPayload = { tensor_index: tensorIndex, name, dtype, shape: tensor.shape, values };
  return {
    tensor_index: tensorIndex,
    name,
    dtype,
    shape: [...tensor.shape],
    element_count: expected,
    values,
    quantization,
    tensor_capture_sha256: sha256TextHex(canonicalJson(digestPayload)),
  };
}

function normalizeValue(value, dtype, path) {
  assert(typeof value === "number" && Number.isFinite(value), `${path} must be a finite JSON number.`);
  if (dtype === "BOOL") {
    assert(value === 0 || value === 1, `${path} must be 0 or 1 for BOOL.`);
    return value;
  }
  const range = INTEGER_RANGES[dtype];
  if (range) assert(Number.isInteger(value) && value >= range[0] && value <= range[1], `${path} is outside ${dtype}.`);
  return value;
}

function normalizeQuantization(value, path) {
  assert(isPlainObject(value), `${path}.quantization must be an object.`);
  assert(typeof value.scale === "number" && Number.isFinite(value.scale) && value.scale > 0, `${path}.quantization.scale must be finite and greater than zero.`);
  assert(Number.isSafeInteger(value.zero_point), `${path}.quantization.zero_point must be an integer.`);
  return { scale: value.scale, zero_point: value.zero_point };
}

function endpointSummary(tensor, inputIndex) {
  const range = INTEGER_RANGES[tensor.dtype];
  if (!range) return null;
  let lower = 0;
  let upper = 0;
  for (const value of tensor.values) {
    if (value === range[0]) lower += 1;
    if (value === range[1]) upper += 1;
  }
  const endpoint = lower + upper;
  return {
    input_index: inputIndex,
    tensor_index: tensor.tensor_index,
    name: tensor.name,
    dtype: tensor.dtype,
    element_count: tensor.element_count,
    lower_endpoint: range[0],
    upper_endpoint: range[1],
    lower_endpoint_count: lower,
    upper_endpoint_count: upper,
    endpoint_count: endpoint,
    endpoint_ratio: tensor.element_count ? endpoint / tensor.element_count : 0,
    quantization_contract_status: tensor.quantization ? "declared_in_capture" : "not_declared",
    quantization: tensor.quantization,
  };
}

function summarizeEndpointRows(rows) {
  const values = rows.reduce((sum, row) => sum + row.element_count, 0);
  const lower = rows.reduce((sum, row) => sum + row.lower_endpoint_count, 0);
  const upper = rows.reduce((sum, row) => sum + row.upper_endpoint_count, 0);
  return {
    status: rows.length ? "assessed" : "not_applicable_no_bounded_integer_inputs",
    assessed_tensor_count: rows.length,
    assessed_value_count: values,
    lower_endpoint_count: lower,
    upper_endpoint_count: upper,
    endpoint_count: lower + upper,
    endpoint_ratio: values ? (lower + upper) / values : null,
    tensors: rows,
    formula: "endpoint_ratio = count(value == dtype_min or value == dtype_max) / assessed integer input values",
  };
}

function compareTensorLists(reference, candidate, label) {
  assertTensorListContract(reference, candidate, label);
  const tensors = reference.map((left, index) => compareTensor(left, candidate[index], index));
  const total = tensors.reduce((sum, row) => sum + row.element_count, 0);
  const changed = tensors.reduce((sum, row) => sum + row.changed_value_count, 0);
  const totalAbs = tensors.reduce((sum, row) => sum + row.total_absolute_difference, 0);
  const sumSq = tensors.reduce((sum, row) => sum + row.sum_squared_difference, 0);
  const referenceSq = tensors.reduce((sum, row) => sum + row.reference_sum_squares, 0);
  const dot = tensors.reduce((sum, row) => sum + row.dot_product, 0);
  const candidateSq = tensors.reduce((sum, row) => sum + row.candidate_sum_squares, 0);
  return {
    tensor_count: tensors.length,
    value_count: total,
    changed_value_count: changed,
    changed_value_ratio: total ? changed / total : 0,
    total_absolute_difference: totalAbs,
    mean_absolute_difference: total ? totalAbs / total : 0,
    root_mean_square_difference: total ? Math.sqrt(sumSq / total) : 0,
    maximum_absolute_difference: Math.max(0, ...tensors.map((row) => row.maximum_absolute_difference)),
    relative_l2_difference: referenceSq > 0 ? Math.sqrt(sumSq / referenceSq) : null,
    cosine_distance: referenceSq > 0 && candidateSq > 0 ? 1 - clamp(dot / Math.sqrt(referenceSq * candidateSq), -1, 1) : null,
    raw_argmax_flip_count: tensors.reduce((sum, row) => sum + Number(row.raw_argmax_changed), 0),
    exact_value_identity: changed === 0,
    tensors: tensors.map(({ sum_squared_difference, reference_sum_squares, dot_product, candidate_sum_squares, ...row }) => row),
  };
}

function compareTensor(reference, candidate, outputIndex) {
  let changed = 0;
  let totalAbs = 0;
  let sumSq = 0;
  let maximum = 0;
  let referenceSq = 0;
  let candidateSq = 0;
  let dot = 0;
  for (let index = 0; index < reference.values.length; index += 1) {
    const left = reference.values[index];
    const right = candidate.values[index];
    if (!Object.is(left, right)) changed += 1;
    const delta = right - left;
    const absolute = Math.abs(delta);
    totalAbs += absolute;
    sumSq += delta * delta;
    maximum = Math.max(maximum, absolute);
    referenceSq += left * left;
    candidateSq += right * right;
    dot += left * right;
  }
  const referenceArgmax = stableArgmax(reference.values);
  const candidateArgmax = stableArgmax(candidate.values);
  return {
    output_index: outputIndex,
    tensor_index: reference.tensor_index,
    name: reference.name,
    dtype: reference.dtype,
    shape: reference.shape,
    element_count: reference.element_count,
    comparison_domain: INTEGER_RANGES[reference.dtype] || reference.dtype === "BOOL" ? "exact_storage_code" : "captured_numeric_value",
    reference_tensor_sha256: reference.tensor_capture_sha256,
    candidate_tensor_sha256: candidate.tensor_capture_sha256,
    changed_value_count: changed,
    changed_value_ratio: reference.element_count ? changed / reference.element_count : 0,
    total_absolute_difference: totalAbs,
    mean_absolute_difference: reference.element_count ? totalAbs / reference.element_count : 0,
    root_mean_square_difference: reference.element_count ? Math.sqrt(sumSq / reference.element_count) : 0,
    maximum_absolute_difference: maximum,
    relative_l2_difference: referenceSq > 0 ? Math.sqrt(sumSq / referenceSq) : null,
    cosine_distance: referenceSq > 0 && candidateSq > 0 ? 1 - clamp(dot / Math.sqrt(referenceSq * candidateSq), -1, 1) : null,
    reference_raw_argmax_index: referenceArgmax,
    candidate_raw_argmax_index: candidateArgmax,
    raw_argmax_changed: referenceArgmax !== candidateArgmax,
    sum_squared_difference: sumSq,
    reference_sum_squares: referenceSq,
    candidate_sum_squares: candidateSq,
    dot_product: dot,
  };
}

function assertTensorListContract(expected, actual, label) {
  assert(expected.length === actual.length, `${label} output tensor count mismatch.`);
  expected.forEach((left, index) => {
    const right = actual[index];
    assert(left.dtype === right.dtype, `${label} tensor ${index} dtype mismatch.`);
    assert(canonicalJson(left.shape) === canonicalJson(right.shape), `${label} tensor ${index} shape mismatch.`);
    assert(left.tensor_index == null || right.tensor_index == null || left.tensor_index === right.tensor_index, `${label} tensor ${index} index mismatch.`);
    assert(!left.name || !right.name || left.name === right.name, `${label} tensor ${index} name mismatch.`);
  });
}

function aggregateSamples(samples) {
  const endpointRows = samples.map((row) => row.input_endpoint_saturation);
  const endpointValues = endpointRows.reduce((sum, row) => sum + row.assessed_value_count, 0);
  const endpointCount = endpointRows.reduce((sum, row) => sum + row.endpoint_count, 0);
  const reference = samples.flatMap((row) => row.reference_comparisons);
  const repeats = samples.flatMap((row) => row.repeat_comparisons);
  return {
    input_endpoint_saturation: {
      status: endpointValues ? "assessed" : "not_applicable_no_bounded_integer_inputs",
      assessed_sample_count: samples.filter((row) => row.input_endpoint_saturation.assessed_value_count > 0).length,
      total_sample_count: samples.length,
      assessed_tensor_count: endpointRows.reduce((sum, row) => sum + row.assessed_tensor_count, 0),
      assessed_value_count: endpointValues,
      lower_endpoint_count: endpointRows.reduce((sum, row) => sum + row.lower_endpoint_count, 0),
      upper_endpoint_count: endpointRows.reduce((sum, row) => sum + row.upper_endpoint_count, 0),
      endpoint_count: endpointCount,
      endpoint_ratio: endpointValues ? endpointCount / endpointValues : null,
      formula: "sum endpoint counts / sum assessed bounded-integer interface input values",
    },
    reference_output_drift: aggregateComparisons(reference, samples.filter((row) => row.reference_status === "assessed").length, samples.length, "not_assessed_no_reference_outputs"),
    repeat_nondeterminism: aggregateComparisons(repeats, samples.filter((row) => row.repeat_status === "assessed").length, samples.length, "not_assessed_fewer_than_two_runs"),
  };
}

function aggregateComparisons(rows, assessedSamples, totalSamples, notAssessedStatus) {
  const values = rows.reduce((sum, row) => sum + row.value_count, 0);
  const changed = rows.reduce((sum, row) => sum + row.changed_value_count, 0);
  const totalAbs = rows.reduce((sum, row) => sum + row.total_absolute_difference, 0);
  const sumSq = rows.reduce((sum, row) => sum + row.root_mean_square_difference ** 2 * row.value_count, 0);
  return {
    status: rows.length ? "assessed" : notAssessedStatus,
    assessed_sample_count: assessedSamples,
    total_sample_count: totalSamples,
    comparison_count: rows.length,
    compared_value_count: values,
    changed_value_count: changed,
    changed_value_ratio: values ? changed / values : null,
    mean_absolute_difference: values ? totalAbs / values : null,
    root_mean_square_difference: values ? Math.sqrt(sumSq / values) : null,
    maximum_absolute_difference: rows.length ? Math.max(...rows.map((row) => row.maximum_absolute_difference)) : null,
    maximum_relative_l2_difference: finiteMaximum(rows.map((row) => row.relative_l2_difference)),
    maximum_cosine_distance: finiteMaximum(rows.map((row) => row.cosine_distance)),
    raw_argmax_flip_count: rows.reduce((sum, row) => sum + row.raw_argmax_flip_count, 0),
    exact_identity_comparison_count: rows.filter((row) => row.exact_value_identity).length,
  };
}

function publicTensor(tensor) {
  return {
    tensor_index: tensor.tensor_index,
    name: tensor.name,
    dtype: tensor.dtype,
    shape: tensor.shape,
    element_count: tensor.element_count,
    quantization: tensor.quantization,
    tensor_capture_sha256: tensor.tensor_capture_sha256,
  };
}

function normalizeRuntime(runtime) {
  return {
    name: runtime.name,
    version: runtime.version,
    backend: runtime.backend,
    binary_sha256: runtime.binary_sha256 ?? null,
    build_inventory_sha256: runtime.build_inventory_sha256 ?? null,
    device_profile_sha256: runtime.device_profile_sha256 ?? null,
    execution_mode: runtime.execution_mode ?? null,
  };
}

function ledgerDigest(ledger) {
  const payload = { ...ledger };
  delete payload.ledger_sha256;
  return sha256TextHex(canonicalJson(payload));
}

function stableArgmax(values) {
  if (!values.length) return null;
  let selected = 0;
  for (let index = 1; index < values.length; index += 1) if (values[index] > values[selected]) selected = index;
  return selected;
}

function finiteMaximum(values) {
  const finite = values.filter((value) => Number.isFinite(value));
  return finite.length ? Math.max(...finite) : null;
}

function optionalSha256(value, label) {
  if (value == null) return null;
  assertSha256(value, label);
  return value;
}

function assertSha256(value, label) {
  assert(typeof value === "string" && SHA256.test(value), `${label} must be a lowercase SHA-256 hex digest.`);
}

function assertNonempty(value, label) {
  assert(typeof value === "string" && value.trim(), `${label} must be a non-empty string.`);
}

function isPlainObject(value) {
  return value != null && typeof value === "object" && !Array.isArray(value) && (Object.getPrototypeOf(value) === Object.prototype || Object.getPrototypeOf(value) === null);
}

function clamp(value, low, high) {
  return Math.min(high, Math.max(low, value));
}

function assert(condition, message) {
  if (!condition) throw new Error(message);
}

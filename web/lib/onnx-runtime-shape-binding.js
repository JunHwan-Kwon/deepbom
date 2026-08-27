import { estimateOnnxMacs } from "../onnx.js";
import { evaluateDynamicIntegerFormula } from "./dynamic-shape-cost.js";
import { isOnnxMacBearingOperation } from "./onnx-operation-cost.js";

export const ONNX_RUNTIME_SHAPE_BINDING_SCHEMA = "deepbom.onnx_runtime_shape_binding.v2";

const DTYPE_BITS = Object.freeze({
  FLOAT4E2M1: 4, INT4: 4, UINT4: 4, INT2: 2, UINT2: 2,
  FLOAT8E4M3FN: 8, FLOAT8E4M3FNUZ: 8, FLOAT8E5M2: 8, FLOAT8E5M2FNUZ: 8, FLOAT8E8M0: 8,
  INT8: 8, UINT8: 8, BOOL: 8, FLOAT16: 16, BFLOAT16: 16, INT16: 16, UINT16: 16,
  FLOAT32: 32, INT32: 32, UINT32: 32, FLOAT64: 64, INT64: 64, UINT64: 64,
  COMPLEX64: 64, COMPLEX128: 128,
});

export function buildOnnxRuntimeShapeBinding(analysis, runtimeResult = {}) {
  const artifactSha256 = String(analysis?.model_sha256 || analysis?.sha256 || "").toLowerCase();
  const contract = analysis?.dynamic_shape_cost_contract || {};
  const symbols = Array.isArray(contract.symbols) ? contract.symbols : [];
  const tensorByIndex = new Map((analysis?.tensors || []).map((tensor, index) => [Number(tensor?.index ?? index), tensor]));
  const tensorByName = new Map((analysis?.tensors || []).map((tensor) => [String(tensor?.name || ""), tensor]));
  const collected = collectObservations(runtimeResult, analysis);
  const conflicts = [...collected.conflicts];
  const exclusions = [...collected.exclusions];
  const acceptedByName = new Map();

  if (!/^[a-f0-9]{64}$/.test(artifactSha256)) conflicts.push({ reason: "artifact_sha256_missing_or_invalid" });
  for (const observation of collected.observations) {
    const tensor = tensorByName.get(observation.tensor_name);
    if (!tensor) {
      conflicts.push({ tensor_name: observation.tensor_name, reason: "runtime_tensor_not_found_in_static_analysis" });
      continue;
    }
    const declaredShape = Array.isArray(tensor.shape) ? tensor.shape : [];
    const shapeDeclared = tensor.shape_declared ?? tensor.shapeDeclared ?? declaredShape.length > 0;
    if (shapeDeclared && observation.executed_shape.length !== declaredShape.length) {
      conflicts.push({ tensor_name: tensor.name, reason: "runtime_rank_mismatch", declared_rank: declaredShape.length, observed_rank: observation.executed_shape.length });
      continue;
    }
    let compatible = true;
    for (let axis = 0; axis < observation.executed_shape.length; axis += 1) {
      const declared = Number(declaredShape[axis]);
      const observed = Number(observation.executed_shape[axis]);
      if (!Number.isSafeInteger(observed) || observed < 0) {
        conflicts.push({ tensor_name: tensor.name, axis, reason: "runtime_dimension_invalid", observed_dimension: observation.executed_shape[axis] });
        compatible = false;
      } else if (shapeDeclared && Number.isSafeInteger(declared) && declared >= 0 && declared !== observed) {
        conflicts.push({ tensor_name: tensor.name, axis, reason: "runtime_dimension_conflicts_with_static_contract", declared_dimension: declared, observed_dimension: observed });
        compatible = false;
      }
    }
    const staticDtype = normalizeDtype(tensor.dtype);
    const runtimeDtype = normalizeDtype(observation.dtype);
    if (observation.ort_type && !runtimeDtype) {
      conflicts.push({ tensor_name: tensor.name, reason: "runtime_dtype_unmapped", runtime_type: observation.ort_type });
      compatible = false;
    } else if (staticDtype && staticDtype !== "UNKNOWN" && runtimeDtype && staticDtype !== runtimeDtype) {
      conflicts.push({ tensor_name: tensor.name, reason: "runtime_dtype_conflicts_with_static_contract", declared_dtype: staticDtype, observed_dtype: runtimeDtype });
      compatible = false;
    }
    if (!compatible) continue;
    const normalized = {
      ...observation,
      tensor_index: Number(tensor.index),
      dtype: runtimeDtype || staticDtype || null,
      payload_bytes_decimal: payloadBytesDecimal(runtimeDtype || staticDtype, observation.executed_shape),
    };
    const previous = acceptedByName.get(observation.tensor_name);
    if (previous && (previous.dtype !== normalized.dtype || !sameArray(previous.executed_shape, normalized.executed_shape))) {
      conflicts.push({
        tensor_name: observation.tensor_name,
        reason: "repeated_runtime_tensor_contract_conflict",
        first_dtype: previous.dtype,
        second_dtype: normalized.dtype,
        first_shape: previous.executed_shape,
        second_shape: normalized.executed_shape,
      });
      continue;
    }
    if (previous) {
      previous.observation_sources = [...new Set([...previous.observation_sources, ...normalized.observation_sources])].sort();
    } else {
      acceptedByName.set(observation.tensor_name, normalized);
    }
  }

  const assignments = new Map();
  for (const symbol of symbols) {
    const observedValues = [];
    for (const occurrence of symbol.occurrences || []) {
      const tensor = tensorByIndex.get(Number(occurrence.tensor_index));
      const observation = tensor ? acceptedByName.get(String(tensor.name || "")) : null;
      const value = observation?.executed_shape?.[Number(occurrence.axis)];
      if (Number.isSafeInteger(Number(value)) && Number(value) >= 0) observedValues.push(Number(value));
    }
    const unique = [...new Set(observedValues)];
    if (unique.length > 1) {
      conflicts.push({ symbol_id: symbol.symbol_id, reason: "repeated_symbol_runtime_values_conflict", observed_values: unique });
    } else if (unique.length === 1) assignments.set(symbol.symbol_id, unique[0]);
  }

  const formula = contract.total_macs_formula || null;
  const requiredSymbols = formula?.symbol_ids || [];
  const unboundSymbols = requiredSymbols.filter((symbolId) => !assignments.has(symbolId));
  const evaluated = formula && !conflicts.length && !unboundSymbols.length
    ? evaluateDynamicIntegerFormula(formula, assignments)
    : null;
  const macReassessment = reassessOnnxMacs(analysis, acceptedByName);
  const accepted = [...acceptedByName.values()].sort((left, right) => left.tensor_index - right.tensor_index || left.tensor_name.localeCompare(right.tensor_name));
  const observedPayloads = accepted.map((row) => row.payload_bytes_decimal).filter((value) => value != null).map(BigInt);
  const observedPayloadTotal = observedPayloads.reduce((sum, value) => sum + value, 0n);
  const internalObserved = accepted.filter((row) => row.observation_sources.some((source) => source.startsWith("ort_internal_"))).length;
  const partial = unboundSymbols.length || macReassessment.remaining_unassessed_mac_op_count || exclusions.length;
  const status = conflicts.length ? "fail"
    : internalObserved ? partial ? "partial_runtime_internal_binding" : "assessed_runtime_internal_binding"
      : !symbols.length ? "not_applicable_no_dynamic_symbols"
        : unboundSymbols.length ? "partial_runtime_io_binding"
          : "assessed_runtime_io_binding";
  return {
    schema: ONNX_RUNTIME_SHAPE_BINDING_SCHEMA,
    evidence_class: internalObserved ? "OBSERVED_RUNTIME_INTERNAL_SHAPES_AND_DERIVED" : "OBSERVED_RUNTIME_IO_AND_DERIVED",
    status,
    artifact_sha256: artifactSha256,
    runtime_backend: String(runtimeResult?.runtime?.backend || runtimeResult.backend || "not_recorded"),
    generated_at: String(runtimeResult?.source?.collected_at || runtimeResult.generated_at || ""),
    observed_tensor_count: accepted.length,
    observed_interface_tensor_count: accepted.filter((row) => row.observation_sources.some((source) => source.startsWith("interface_"))).length,
    observed_internal_tensor_count: internalObserved,
    observed_tensor_payload_assessed_count: observedPayloads.length,
    observed_tensor_payload_bytes_decimal: observedPayloads.length === accepted.length ? observedPayloadTotal.toString() : null,
    observed_tensor_payload_bytes: observedPayloads.length === accepted.length && observedPayloadTotal <= BigInt(Number.MAX_SAFE_INTEGER) ? Number(observedPayloadTotal) : null,
    observed_tensors: accepted,
    symbol_count: symbols.length,
    bound_symbol_count: assignments.size,
    unbound_formula_symbol_count: unboundSymbols.length,
    unbound_formula_symbol_ids: unboundSymbols,
    assignments: [...assignments.entries()].map(([symbol_id, value]) => ({ symbol_id, value })),
    conflict_count: conflicts.length,
    conflicts,
    exclusion_count: exclusions.length,
    exclusions,
    evaluated_symbolic_total_macs_decimal: evaluated == null ? null : evaluated.toString(),
    evaluated_symbolic_total_macs: evaluated != null && evaluated <= BigInt(Number.MAX_SAFE_INTEGER) ? Number(evaluated) : null,
    evaluated_total_macs_decimal: evaluated == null ? null : evaluated.toString(),
    evaluated_total_macs: evaluated != null && evaluated <= BigInt(Number.MAX_SAFE_INTEGER) ? Number(evaluated) : null,
    ...macReassessment,
    method: "Bind hash-matched interface shapes and optimization-disabled, uniquely mapped ONNX Runtime node input/output type-shape observations to serialized tensor identity; require repeated observations, static dimensions, dtypes, and ORT output-size arithmetic to agree; then rerun the same source-pinned ONNX MAC definitions with observed concrete shapes.",
    interpretation_boundary: "Observed type-shape rows describe one profiled invocation. Unique logical tensor payload is not peak liveness, allocator arena, physical CPU/GPU transfer, synchronization, residency, or latency. Optimized-node slot identity, conflicting repeated shapes, unsupported dtypes, and unresolved MAC signatures remain explicit and are never replaced with zero.",
  };
}

function collectObservations(runtimeResult, analysis) {
  const observations = [];
  const conflicts = [];
  const exclusions = [];
  for (const [kind, contracts] of [["input", runtimeResult.input_contracts || runtimeResult.inputContracts], ["output", runtimeResult.output_contracts || runtimeResult.outputContracts]]) {
    for (const contract of contracts || []) {
      const name = String(contract[`${kind}_name`] || contract.name || "");
      const shape = contract.executed_shape || contract.shape || contract.dims;
      if (name && Array.isArray(shape)) observations.push(observation(name, shape, contract.dtype || contract.type, null, `interface_${kind}`));
    }
  }
  const adapter = runtimeResult?.source?.adapter;
  const native = adapter?.native_capture;
  for (const input of native?.invocation?.inputs || []) {
    if (input?.name && Array.isArray(input.shape)) observations.push(observation(input.name, input.shape, input.type, null, "interface_native_capture_input"));
  }
  for (const output of native?.output_observations || []) {
    if (output?.name && Array.isArray(output.dims)) observations.push(observation(output.name, output.dims, output.type, null, "interface_native_capture_output"));
  }
  if (adapter?.schema !== "deepbom.ort_profile_adapter.v2.2") return { observations, conflicts, exclusions };
  if (runtimeResult?.runtime?.graph_optimization_level !== "disabled") {
    for (const row of adapter.runtime_tensor_observations || []) {
      exclusions.push({ op_index: row.op_index, op_name: row.op_name, reason: "optimized_runtime_node_tensor_slots_not_bound_to_original_graph" });
    }
    return { observations, conflicts, exclusions };
  }
  const opByIndex = new Map((analysis?.ops || []).map((op) => [Number(op.index), op]));
  for (const row of adapter.runtime_tensor_observations || []) {
    if (row.status === "conflict_repeated_events") {
      conflicts.push({ op_index: row.op_index, op_name: row.op_name, reason: "repeated_profile_event_tensor_contract_conflict", observed_contract_variant_count: row.observed_contract_variant_count });
      continue;
    }
    const op = opByIndex.get(Number(row.op_index));
    if (!op) {
      conflicts.push({ op_index: row.op_index, reason: "runtime_shape_op_not_found" });
      continue;
    }
    bindOpSlots(observations, conflicts, row.input_type_shapes, op.input_names, row, "input", analysis);
    bindOpSlots(observations, conflicts, row.output_type_shapes, op.output_names, row, "output", analysis);
    const outputPayloads = (row.output_type_shapes || []).map((shape) => payloadBytesDecimal(shape.dtype, shape.shape));
    if (row.output_size_bytes_decimal != null && outputPayloads.every((value) => value != null)) {
      const derived = outputPayloads.map(BigInt).reduce((sum, value) => sum + value, 0n).toString();
      if (derived !== row.output_size_bytes_decimal) {
        conflicts.push({ op_index: row.op_index, op_name: row.op_name, reason: "runtime_output_size_conflicts_with_type_shape", observed_output_size_bytes_decimal: row.output_size_bytes_decimal, derived_output_size_bytes_decimal: derived });
      }
    }
  }
  return { observations, conflicts, exclusions };
}

function bindOpSlots(observations, conflicts, typeShapes, tensorNames, row, direction, analysis) {
  const shapes = typeShapes || [];
  const names = tensorNames || [];
  const tensorByName = new Map((analysis?.tensors || []).map((tensor) => [String(tensor.name || ""), tensor]));
  const serializedSlots = names.map((name, slot) => ({ name: String(name || ""), slot, tensor: tensorByName.get(String(name || "")) }));
  const runtimeSlots = shapes.length === serializedSlots.length && serializedSlots.every((item) => item.name)
    ? serializedSlots
    : direction === "input"
      ? uniqueOrderedRuntimeSlotMapping(shapes, serializedSlots)
      : [];
  if (runtimeSlots.length !== shapes.length) {
    conflicts.push({
      op_index: row.op_index,
      op_name: row.op_name,
      reason: `runtime_${direction}_slot_identity_not_exact`,
      runtime_slot_count: shapes.length,
      serialized_slot_count: names.length,
      uniquely_mapped_serialized_slot_count: direction === "input" ? runtimeSlots.length : null,
    });
    return;
  }
  for (const shape of shapes) {
    const mapped = runtimeSlots[shape.slot];
    observations.push(observation(mapped.name, shape.shape, shape.dtype, shape.ort_type, `ort_internal_${direction}_op_${row.op_index}_runtime_slot_${shape.slot}_serialized_slot_${mapped.slot}`));
  }
}

function uniqueOrderedRuntimeSlotMapping(runtimeShapes, serializedSlots) {
  const solutions = [];
  let visits = 0;
  const search = (runtimeIndex, serializedStart, selected) => {
    visits += 1;
    if (visits > 10_000 || solutions.length > 1) return;
    if (runtimeIndex === runtimeShapes.length) {
      solutions.push([...selected]);
      return;
    }
    const remaining = runtimeShapes.length - runtimeIndex;
    for (let index = serializedStart; index <= serializedSlots.length - remaining; index += 1) {
      const candidate = serializedSlots[index];
      if (!runtimeShapeMatchesSerializedTensor(runtimeShapes[runtimeIndex], candidate)) continue;
      selected.push(candidate);
      search(runtimeIndex + 1, index + 1, selected);
      selected.pop();
    }
  };
  search(0, 0, []);
  return solutions.length === 1 ? solutions[0] : [];
}

function runtimeShapeMatchesSerializedTensor(runtimeShape, candidate) {
  if (!candidate?.name || !candidate.tensor) return false;
  const runtimeDtype = normalizeDtype(runtimeShape?.dtype);
  const staticDtype = normalizeDtype(candidate.tensor.dtype);
  if (runtimeDtype && staticDtype && staticDtype !== "UNKNOWN" && runtimeDtype !== staticDtype) return false;
  const staticShape = candidate.tensor.shape;
  const shapeDeclared = candidate.tensor.shape_declared ?? candidate.tensor.shapeDeclared ?? (Array.isArray(staticShape) && staticShape.length > 0);
  if (!shapeDeclared || !Array.isArray(staticShape) || staticShape.length !== runtimeShape.shape.length) return false;
  return staticShape.every((dimension, axis) => {
    const declared = Number(dimension);
    return !Number.isSafeInteger(declared) || declared < 0 || declared === Number(runtimeShape.shape[axis]);
  });
}

function observation(name, shape, dtype, ortType, source) {
  return {
    tensor_name: String(name),
    executed_shape: shape.map(Number),
    dtype: normalizeRuntimeType(dtype),
    ort_type: ortType || null,
    observation_sources: [source],
  };
}

function reassessOnnxMacs(analysis, acceptedByName) {
  const tensorMap = new Map((analysis?.tensors || []).map((tensor) => [String(tensor.name || ""), {
    ...tensor,
    valueKind: tensor.value_kind || tensor.valueKind || "tensor",
    shapeDeclared: tensor.shape_declared ?? tensor.shapeDeclared ?? (Array.isArray(tensor.shape) && tensor.shape.length > 0),
  }]));
  for (const [name, observationRow] of acceptedByName) {
    const tensor = tensorMap.get(name);
    if (tensor) tensorMap.set(name, { ...tensor, shape: [...observationRow.executed_shape], shapeDeclared: true, shape_declared: true, dtype: observationRow.dtype || tensor.dtype });
  }
  let staticSubtotal = 0n;
  let runtimeAdded = 0n;
  let staticAssessedCount = 0;
  const closed = [];
  const residuals = [];
  for (const op of (analysis?.ops || []).filter((row) => isOnnxMacBearingOperation(row.name, row.standard_domain !== false))) {
    if (op.macs_status === "assessed" && Number.isSafeInteger(Number(op.macs)) && Number(op.macs) >= 0) {
      staticSubtotal += BigInt(op.macs);
      staticAssessedCount += 1;
      continue;
    }
    const assessment = estimateOnnxMacs(reconstructOnnxNode(op), tensorMap);
    if (assessment.status === "assessed") {
      runtimeAdded += BigInt(assessment.value);
      closed.push({ op_index: op.index, op_name: op.name, macs: assessment.value, macs_decimal: String(assessment.value), method: assessment.reason });
    } else {
      residuals.push({ op_index: op.index, op_name: op.name, static_reason: op.macs_reason || null, runtime_reason: assessment.reason });
    }
  }
  const assessedSubtotal = staticSubtotal + runtimeAdded;
  return {
    static_assessed_mac_op_count: staticAssessedCount,
    runtime_closed_mac_op_count: closed.length,
    remaining_unassessed_mac_op_count: residuals.length,
    runtime_closed_macs_decimal: runtimeAdded.toString(),
    runtime_closed_macs: runtimeAdded <= BigInt(Number.MAX_SAFE_INTEGER) ? Number(runtimeAdded) : null,
    runtime_bound_assessed_macs_decimal: assessedSubtotal.toString(),
    runtime_bound_assessed_macs: assessedSubtotal <= BigInt(Number.MAX_SAFE_INTEGER) ? Number(assessedSubtotal) : null,
    runtime_bound_complete_macs_decimal: residuals.length ? null : assessedSubtotal.toString(),
    runtime_bound_complete_macs: residuals.length || assessedSubtotal > BigInt(Number.MAX_SAFE_INTEGER) ? null : Number(assessedSubtotal),
    runtime_closed_mac_ops: closed,
    remaining_mac_residuals: residuals,
  };
}

function reconstructOnnxNode(op) {
  const attributes = new Map((op.onnx_attributes || []).map((attribute) => [attribute.name, {
    i: Number.isSafeInteger(attribute.int_value) ? attribute.int_value : null,
    ints: Array.isArray(attribute.int_values) ? attribute.int_values.filter(Number.isSafeInteger) : [],
  }]));
  return {
    opType: op.name,
    domain: op.domain,
    inputs: [...(op.input_names || [])],
    outputs: [...(op.output_names || [])],
    attributes,
  };
}

function payloadBytesDecimal(dtype, shape) {
  const bits = DTYPE_BITS[normalizeDtype(dtype)];
  if (!bits || !Array.isArray(shape) || shape.some((dimension) => !Number.isSafeInteger(Number(dimension)) || Number(dimension) < 0)) return null;
  const elements = shape.reduce((product, dimension) => product * BigInt(dimension), 1n);
  return ((elements * BigInt(bits) + 7n) / 8n).toString();
}

function normalizeRuntimeType(value) {
  const text = String(value || "").trim().toLowerCase();
  const map = {
    float: "FLOAT32", float32: "FLOAT32", double: "FLOAT64", float64: "FLOAT64",
    float16: "FLOAT16", bfloat16: "BFLOAT16", bool: "BOOL", string: "STRING",
    int8: "INT8", int8_t: "INT8", uint8: "UINT8", uint8_t: "UINT8",
    int16: "INT16", int16_t: "INT16", uint16: "UINT16", uint16_t: "UINT16",
    int32: "INT32", int32_t: "INT32", uint32: "UINT32", uint32_t: "UINT32",
    int64: "INT64", int64_t: "INT64", uint64: "UINT64", uint64_t: "UINT64",
    complex64: "COMPLEX64", complex128: "COMPLEX128",
  };
  return map[text] || normalizeDtype(value);
}

function normalizeDtype(value) {
  const text = String(value || "").trim().toUpperCase();
  return text && text !== "UNDEFINED" ? text : null;
}

function sameArray(left, right) {
  return left.length === right.length && left.every((value, index) => value === right[index]);
}

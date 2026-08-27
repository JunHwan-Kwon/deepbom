export const BACKEND_PLACEMENT_PROJECTION_SCHEMA = "deepbom.backend_placement_projection.v1";
export const BACKEND_WORKLOAD_ENVELOPE_SCHEMA = "deepbom.backend_workload_envelope.v1";

export const BACKEND_PLACEMENT_STATES = Object.freeze({
  CONDITIONALLY_ELIGIBLE: "CONDITIONALLY_ELIGIBLE",
  DEFINITE_EXCLUSION: "DEFINITE_EXCLUSION",
  UNRESOLVED: "UNRESOLVED",
});

const VALID_STATES = new Set(Object.values(BACKEND_PLACEMENT_STATES));
const DTYPE_BITS = Object.freeze({
  BOOL: 8,
  UINT1: 1,
  UINT2: 2, INT2: 2,
  UINT3: 3,
  UINT4: 4, INT4: 4, FLOAT4E2M1: 4,
  UINT6: 6,
  INT8: 8, UINT8: 8,
  FLOAT8E4M3FN: 8, FLOAT8E4M3FNUZ: 8, FLOAT8E5M2: 8, FLOAT8E5M2FNUZ: 8, FLOAT8E8M0: 8,
  FLOAT16: 16, BFLOAT16: 16, INT16: 16, UINT16: 16,
  FLOAT32: 32, INT32: 32, UINT32: 32,
  FLOAT64: 64, INT64: 64, UINT64: 64, COMPLEX64: 64,
  COMPLEX128: 128,
});

/**
 * Derive an independent backend eligibility projection from one canonical graph ledger.
 * The result is not a runtime assignment and logical edge payload is not a copy claim.
 */
export function buildBackendPlacementProjection({
  analysis,
  profileId,
  label,
  evidenceClass,
  rows,
  scopeId = "main_graph",
  source = null,
  interpretationBoundary = null,
}) {
  if (!analysis || !Array.isArray(analysis.ops) || !Array.isArray(analysis.tensors)) {
    throw new Error("Backend placement projection requires operator and tensor ledgers.");
  }
  if (!String(profileId || "").trim() || !String(label || "").trim() || !String(evidenceClass || "").trim()) {
    throw new Error("Backend placement projection identity is incomplete.");
  }
  const ops = canonicalOps(analysis.ops);
  const states = canonicalStates(ops, rows);
  const graph = buildCanonicalGraphLedger(analysis, ops, scopeId);
  const segments = buildSegments(ops, states);
  const stateCounts = countStates(states);
  const edges = graph.edges.map((edge) => {
    const producerState = states.get(edge.producer_op_index).state;
    const consumerState = states.get(edge.consumer_op_index).state;
    return {
      ...edge,
      producer_state: producerState,
      consumer_state: consumerState,
      transition: `${producerState}->${consumerState}`,
      crosses_state_boundary: producerState !== consumerState,
    };
  });
  const boundaryEdges = edges.filter((edge) => edge.crosses_state_boundary);
  const payload = summarizePayload(boundaryEdges);
  const result = {
    schema: BACKEND_PLACEMENT_PROJECTION_SCHEMA,
    method_version: "1.0.0",
    profile_id: String(profileId),
    label: String(label),
    evidence_class: String(evidenceClass),
    scope_id: String(scopeId),
    source,
    assessment_status: "assessed_independent_static_projection",
    artifact_sha256: String(analysis.model_sha256 || ""),
    op_count: ops.length,
    state_counts: stateCounts,
    rows: ops.map((op) => ({
      scope_id: String(scopeId),
      op_index: op.index,
      op_name: String(op.value?.name || ""),
      state: states.get(op.index).state,
      reason_codes: states.get(op.index).reason_codes,
      unresolved_predicates: states.get(op.index).unresolved_predicates,
    })),
    segment_count: segments.length,
    segments,
    graph_edge_count: edges.length,
    graph_edges: edges,
    boundary_edge_count: boundaryEdges.length,
    boundary_edges: boundaryEdges,
    boundary_payload: payload,
    conservation: {
      op_state_count: Object.values(stateCounts).reduce((sum, value) => sum + value, 0),
      op_count_status: Object.values(stateCounts).reduce((sum, value) => sum + value, 0) === ops.length ? "complete" : "failed",
      segment_op_count: segments.reduce((sum, segment) => sum + segment.op_count, 0),
      segment_count_status: segments.reduce((sum, segment) => sum + segment.op_count, 0) === ops.length ? "complete" : "failed",
      edge_classified_count: edges.length,
      edge_count_status: edges.length === graph.edges.length ? "complete" : "failed",
    },
    interpretation_boundary: interpretationBoundary
      || "Independent source-backed eligibility projection. Segments do not establish selected-build acceptance or runtime assignment. Boundary payload is serialized-shape logical exposure, not observed transfer, copy, or latency.",
  };
  result.workload_envelope = buildBackendWorkloadEnvelope(analysis, result);
  validateBackendPlacementProjection(result, { analysis });
  return result;
}

export function buildBackendWorkloadEnvelope(analysis, projection) {
  const ops = canonicalOps(list(analysis?.ops));
  const projectionRows = new Map(list(projection?.rows).map((row) => [Number(row.op_index), row]));
  if (ops.length !== projectionRows.size || ops.some((op) => !projectionRows.has(op.index))) {
    throw new Error("Backend workload envelope requires complete placement state coverage.");
  }
  const tensorByIndex = new Map(list(analysis?.tensors).map((tensor, position) => [
    Number.isSafeInteger(Number(tensor?.index)) ? Number(tensor.index) : position, tensor,
  ]));
  const byState = Object.fromEntries([...VALID_STATES].map((state) => [state, []]));
  for (const op of ops) byState[projectionRows.get(op.index).state].push(op);
  const stateWorkload = Object.fromEntries([...VALID_STATES].map((state) => [
    state, workloadSummary(byState[state], tensorByIndex),
  ]));
  const total = workloadSummary(ops, tensorByIndex);
  const candidate = stateWorkload[BACKEND_PLACEMENT_STATES.CONDITIONALLY_ELIGIBLE];
  return {
    schema: BACKEND_WORKLOAD_ENVELOPE_SCHEMA,
    method_version: "1.0.0",
    profile_id: projection.profile_id,
    scope_id: projection.scope_id,
    evidence_class: "DERIVED_FROM_ARTIFACT_AND_STATIC_ELIGIBILITY",
    status: "assessed_artifact_workload_without_backend_cost_model",
    total,
    by_state: stateWorkload,
    conditionally_eligible_mac_share: safeRatio(candidate.complete_macs_decimal, total.complete_macs_decimal),
    conditionally_eligible_mac_share_decimal: decimalRatio(candidate.complete_macs_decimal, total.complete_macs_decimal),
    conditionally_eligible_logical_byte_share: safeRatio(candidate.complete_logical_bytes_decimal, total.complete_logical_bytes_decimal),
    conditionally_eligible_logical_byte_share_decimal: decimalRatio(candidate.complete_logical_bytes_decimal, total.complete_logical_bytes_decimal),
    boundary_payload: structuredCloneSafe(projection.boundary_payload),
    backend_cost_model: {
      status: "not_assessed",
      peak_compute: null,
      memory_bandwidth: null,
      occupancy: null,
      shader_or_kernel_selection: null,
      latency: null,
      reason: "The artifact and source-eligibility ledger do not bind a selected device, generated shader/kernel, occupancy, backend memory plan, measured bandwidth, or runtime timing.",
    },
    method: "Partition exact serialized-graph MAC and logical-op-byte ledgers by independent static eligibility state; derive a MAC-equivalent arithmetic-intensity descriptor only when every op in that state has assessed MAC and logical-byte values. Preserve output dtype counts from serialized tensor contracts.",
    interpretation_boundary: "This envelope describes artifact workload associated with a source-backed eligibility state. It is not a GPU roofline, accepted partition, executed workload, transfer volume, occupancy estimate, kernel choice, or latency prediction.",
  };
}

export function tfliteDelegateProjectionRows(profile) {
  return list(profile?.rows).map((row) => ({
    op_index: row.op_index,
    state: row.artifact_precheck_status === "source_candidate_partial"
      ? BACKEND_PLACEMENT_STATES.CONDITIONALLY_ELIGIBLE
      : BACKEND_PLACEMENT_STATES.DEFINITE_EXCLUSION,
    reason_codes: row.artifact_precheck_status === "source_candidate_partial"
      ? [] : list(row.definite_exclusion_reasons).map(reasonCode),
    unresolved_predicates: row.artifact_precheck_status === "source_candidate_partial"
      ? list(row.unresolved_predicates).map(reasonCode) : [],
  }));
}

export function tfliteXnnpackProjectionRows(analysis) {
  return list(analysis?.ops).map((op, position) => {
    const opIndex = Number.isSafeInteger(Number(op?.index)) ? Number(op.index) : position;
    const candidate = Number(op?.xnnpack_chain_id) >= 0 && op?.xnnpack_supported !== false;
    const reason = reasonCode(op?.xnnpack_reason || op?.xnnpack_break_class || "xnnpack_not_conditionally_delegatable");
    const buildRequirement = reasonCode(op?.xnnpack_build_requirement || "");
    return {
      op_index: opIndex,
      state: candidate
        ? BACKEND_PLACEMENT_STATES.CONDITIONALLY_ELIGIBLE
        : BACKEND_PLACEMENT_STATES.DEFINITE_EXCLUSION,
      reason_codes: candidate ? [] : [reason],
      unresolved_predicates: candidate && buildRequirement ? [buildRequirement] : [],
    };
  });
}

export function ortProviderProjectionRows(provider) {
  return list(provider?.ops).map((row) => {
    const unresolvedResolution = new Set([
      "MODEL_LOCAL_FUNCTION_REQUIRES_RUNTIME_RESOLUTION",
      "EXTERNAL_CUSTOM_REGISTRY_REQUIRED",
      "OP_SCHEMA_VERSION_NOT_RESOLVED",
    ]).has(String(row.status || ""));
    const unresolvedPrecheck = row.artifact_precheck_status === "ARTIFACT_PRECHECK_UNRESOLVED";
    const conditional = row.source_candidate_after_artifact_precheck === true && !unresolvedPrecheck;
    const state = conditional
      ? BACKEND_PLACEMENT_STATES.CONDITIONALLY_ELIGIBLE
      : unresolvedPrecheck || unresolvedResolution
        ? BACKEND_PLACEMENT_STATES.UNRESOLVED
        : BACKEND_PLACEMENT_STATES.DEFINITE_EXCLUSION;
    return {
      op_index: row.op_index,
      state,
      reason_codes: state === BACKEND_PLACEMENT_STATES.DEFINITE_EXCLUSION
        ? [reasonCode(row.artifact_precheck_status || row.status || "source_exclusion")] : [],
      unresolved_predicates: state === BACKEND_PLACEMENT_STATES.UNRESOLVED
        ? [reasonCode(row.artifact_precheck_status || row.status || "unresolved")]
        : list(row.unresolved_source_condition_fragments).map(reasonCode),
    };
  });
}

export function validateBackendPlacementProjection(value, { analysis = null } = {}) {
  const issues = [];
  if (!value || value.schema !== BACKEND_PLACEMENT_PROJECTION_SCHEMA) issues.push("schema mismatch");
  if (!value?.profile_id || !value?.label || !value?.evidence_class || !value?.scope_id) issues.push("identity incomplete");
  if (!Number.isSafeInteger(value?.op_count) || value.op_count < 0) issues.push("op count invalid");
  const rows = list(value?.rows);
  if (rows.length !== value?.op_count || new Set(rows.map((row) => `${row.scope_id}:${row.op_index}`)).size !== rows.length
    || rows.some((row) => row.scope_id !== value.scope_id || !Number.isSafeInteger(row.op_index) || !VALID_STATES.has(row.state))) {
    issues.push("op-state ledger invalid");
  }
  const counts = countStates(new Map(rows.map((row) => [row.op_index, row])));
  for (const state of VALID_STATES) if (Number(value?.state_counts?.[state]) !== counts[state]) issues.push(`state count mismatch: ${state}`);
  const segments = list(value?.segments);
  let nextPosition = 0;
  for (const segment of segments) {
    if (!VALID_STATES.has(segment.state) || segment.start_position !== nextPosition
      || segment.end_position !== segment.start_position + segment.op_count - 1 || segment.op_count <= 0) {
      issues.push("segment ledger invalid");
      break;
    }
    nextPosition = segment.end_position + 1;
  }
  if (nextPosition !== value?.op_count || Number(value?.segment_count) !== segments.length) issues.push("segments do not conserve ops");
  const edges = list(value?.graph_edges);
  const edgeKeys = new Set();
  for (const edge of edges) {
    const key = `${edge.scope_id}:${edge.producer_op_index}:${edge.consumer_op_index}:${edge.tensor_index}`;
    if (edgeKeys.has(key) || edge.scope_id !== value.scope_id || !VALID_STATES.has(edge.producer_state)
      || !VALID_STATES.has(edge.consumer_state) || edge.crosses_state_boundary !== (edge.producer_state !== edge.consumer_state)
      || edge.transition !== `${edge.producer_state}->${edge.consumer_state}`
      || (edge.logical_payload_bytes == null) === !edge.logical_payload_reason) {
      issues.push("graph edge ledger invalid");
      break;
    }
    edgeKeys.add(key);
  }
  const boundary = edges.filter((edge) => edge.crosses_state_boundary);
  if (Number(value?.graph_edge_count) !== edges.length || Number(value?.boundary_edge_count) !== boundary.length
    || list(value?.boundary_edges).length !== boundary.length) issues.push("edge counts do not conserve graph edges");
  const payload = summarizePayload(boundary);
  if (JSON.stringify(value?.boundary_payload) !== JSON.stringify(payload)) issues.push("boundary payload does not reproduce");
  if (analysis && JSON.stringify(value?.workload_envelope) !== JSON.stringify(buildBackendWorkloadEnvelope(analysis, value))) {
    issues.push("workload envelope does not reproduce");
  }
  if (value?.conservation?.op_count_status !== "complete" || value?.conservation?.segment_count_status !== "complete"
    || value?.conservation?.edge_count_status !== "complete") issues.push("conservation status incomplete");
  if (analysis) {
    if (String(value.artifact_sha256 || "") !== String(analysis.model_sha256 || "")) issues.push("artifact binding mismatch");
    if (value.op_count !== list(analysis.ops).length) issues.push("analysis op count mismatch");
  }
  if (issues.length) throw new Error(`Backend placement projection is invalid: ${issues.join("; ")}`);
  return true;
}

function canonicalOps(values) {
  const seen = new Set();
  return values.map((value, position) => {
    const index = Number.isSafeInteger(Number(value?.index)) ? Number(value.index) : position;
    if (seen.has(index)) throw new Error(`Operator identity duplicates op #${index}.`);
    seen.add(index);
    return { index, position, value };
  });
}

function canonicalStates(ops, values) {
  if (!Array.isArray(values) || values.length !== ops.length) throw new Error("Backend state rows must cover every operator exactly once.");
  const opIds = new Set(ops.map((op) => op.index));
  const result = new Map();
  for (const row of values) {
    const index = Number(row?.op_index);
    if (!Number.isSafeInteger(index) || !opIds.has(index) || result.has(index) || !VALID_STATES.has(row?.state)) {
      throw new Error(`Backend state identity is invalid at op #${String(row?.op_index)}.`);
    }
    result.set(index, {
      state: row.state,
      reason_codes: uniqueStrings(row.reason_codes),
      unresolved_predicates: uniqueStrings(row.unresolved_predicates),
    });
  }
  return result;
}

function buildCanonicalGraphLedger(analysis, ops, scopeId) {
  const opIds = new Set(ops.map((op) => op.index));
  const tensors = new Map();
  for (const [position, tensor] of analysis.tensors.entries()) {
    const index = Number.isSafeInteger(Number(tensor?.index)) ? Number(tensor.index) : position;
    if (tensors.has(index)) throw new Error(`Tensor identity duplicates tensor #${index}.`);
    tensors.set(index, tensor);
  }
  const producerByTensor = new Map();
  const consumersByTensor = new Map();
  for (const op of ops) {
    for (const tensorIndex of tensorIds(op.value?.outputs)) {
      if (producerByTensor.has(tensorIndex) && producerByTensor.get(tensorIndex) !== op.index) {
        throw new Error(`Tensor #${tensorIndex} has multiple producers in ${scopeId}.`);
      }
      producerByTensor.set(tensorIndex, op.index);
    }
    for (const tensorIndex of tensorIds(op.value?.inputs)) {
      if (!consumersByTensor.has(tensorIndex)) consumersByTensor.set(tensorIndex, new Set());
      consumersByTensor.get(tensorIndex).add(op.index);
    }
  }
  const edges = [];
  for (const [tensorIndex, producer] of producerByTensor) {
    if (!opIds.has(producer)) continue;
    for (const consumer of consumersByTensor.get(tensorIndex) || []) {
      if (consumer === producer || !opIds.has(consumer)) continue;
      const tensor = tensors.get(tensorIndex);
      const payload = tensorPayload(tensor);
      edges.push({
        scope_id: String(scopeId),
        producer_op_index: producer,
        consumer_op_index: consumer,
        tensor_index: tensorIndex,
        tensor_name: tensor ? String(tensor.name || "") : null,
        tensor_dtype: tensor ? String(tensor.dtype || "") : null,
        tensor_shape: tensor && Array.isArray(tensor.shape) ? tensor.shape.map(Number) : null,
        logical_payload_bytes: payload.bytes,
        logical_payload_status: payload.status,
        logical_payload_reason: payload.reason,
      });
    }
  }
  edges.sort((left, right) => left.producer_op_index - right.producer_op_index
    || left.consumer_op_index - right.consumer_op_index || left.tensor_index - right.tensor_index);
  return { edges };
}

function tensorPayload(tensor) {
  if (!tensor) return { bytes: null, status: "not_assessed", reason: "tensor_descriptor_missing" };
  if (!Array.isArray(tensor.shape)) return { bytes: null, status: "not_assessed", reason: "shape_not_declared" };
  if (tensor.shape.some((dim) => !Number.isSafeInteger(Number(dim)) || Number(dim) < 0)) {
    return { bytes: null, status: "not_assessed", reason: "shape_dynamic_or_invalid" };
  }
  if (Array.isArray(tensor.shape_signature) && tensor.shape_signature.length
    && (tensor.shape_signature.length !== tensor.shape.length
      || tensor.shape_signature.some((dim, index) => !Number.isSafeInteger(Number(dim))
        || Number(dim) < 0 || Number(dim) !== Number(tensor.shape[index])))) {
    return { bytes: null, status: "not_assessed", reason: "shape_signature_not_statically_bound" };
  }
  const bits = DTYPE_BITS[String(tensor.dtype || "").toUpperCase()];
  if (!bits) return { bytes: null, status: "not_assessed", reason: "dtype_width_unknown" };
  let elements = 1n;
  for (const dim of tensor.shape) elements *= BigInt(Number(dim));
  const bytes = (elements * BigInt(bits) + 7n) / 8n;
  if (bytes > BigInt(Number.MAX_SAFE_INTEGER)) return { bytes: null, status: "not_assessed", reason: "payload_exceeds_safe_integer" };
  return { bytes: Number(bytes), status: "assessed_serialized_static_shape", reason: null };
}

function buildSegments(ops, states) {
  const result = [];
  for (const op of ops) {
    const state = states.get(op.index).state;
    const previous = result.at(-1);
    if (previous?.state === state) {
      previous.end_position = op.position;
      previous.end_op_index = op.index;
      previous.op_count += 1;
    } else {
      result.push({
        segment_index: result.length,
        state,
        start_position: op.position,
        end_position: op.position,
        start_op_index: op.index,
        end_op_index: op.index,
        op_count: 1,
      });
    }
  }
  return result;
}

function summarizePayload(edges) {
  const assessed = edges.filter((edge) => Number.isSafeInteger(edge.logical_payload_bytes));
  const assessedBytes = safeSum(assessed.map((edge) => edge.logical_payload_bytes));
  const unique = new Map();
  for (const edge of edges) if (!unique.has(edge.tensor_index)) unique.set(edge.tensor_index, edge.logical_payload_bytes);
  const assessedUnique = [...unique.values()].filter(Number.isSafeInteger);
  return {
    semantics: "summed_logical_edge_payload_not_observed_transfer",
    edge_count: edges.length,
    assessed_edge_count: assessed.length,
    unassessed_edge_count: edges.length - assessed.length,
    assessed_edge_payload_bytes: assessedBytes,
    summed_edge_payload_bytes: assessed.length === edges.length ? assessedBytes : null,
    unique_tensor_count: unique.size,
    assessed_unique_tensor_count: assessedUnique.length,
    assessed_unique_tensor_payload_bytes: safeSum(assessedUnique),
    unique_tensor_payload_bytes: assessedUnique.length === unique.size ? safeSum(assessedUnique) : null,
  };
}

function workloadSummary(ops, tensorByIndex) {
  const macRows = ops.map((op) => assessedOpValue(op.value, "macs"));
  const byteRows = ops.map((op) => assessedOpValue(op.value, "estimated_bytes"));
  const assessedMacs = macRows.filter((value) => value != null);
  const assessedBytes = byteRows.filter((value) => value != null);
  const macs = bigSum(assessedMacs);
  const bytes = bigSum(assessedBytes);
  const completeMacs = assessedMacs.length === ops.length ? macs : null;
  const completeBytes = assessedBytes.length === ops.length ? bytes : null;
  const outputDtypes = {};
  let outputTensorReferenceCount = 0;
  for (const op of ops) {
    for (const tensorIndex of tensorIds(op.value?.outputs)) {
      outputTensorReferenceCount += 1;
      const dtype = String(tensorByIndex.get(tensorIndex)?.dtype || "UNKNOWN").toUpperCase();
      outputDtypes[dtype] = (outputDtypes[dtype] || 0) + 1;
    }
  }
  return {
    op_count: ops.length,
    assessed_mac_op_count: assessedMacs.length,
    unassessed_mac_op_count: ops.length - assessedMacs.length,
    assessed_macs_decimal: macs.toString(),
    assessed_macs: safeMirror(macs),
    complete_macs_decimal: completeMacs?.toString() ?? null,
    complete_macs: completeMacs == null ? null : safeMirror(completeMacs),
    assessed_logical_byte_op_count: assessedBytes.length,
    unassessed_logical_byte_op_count: ops.length - assessedBytes.length,
    assessed_logical_bytes_decimal: bytes.toString(),
    assessed_logical_bytes: safeMirror(bytes),
    complete_logical_bytes_decimal: completeBytes?.toString() ?? null,
    complete_logical_bytes: completeBytes == null ? null : safeMirror(completeBytes),
    mac_equivalent_ops_per_logical_byte: completeMacs != null && completeBytes != null && completeBytes > 0n
      ? safeRatio((completeMacs * 2n).toString(), completeBytes.toString()) : null,
    mac_equivalent_ops_per_logical_byte_decimal: completeMacs != null && completeBytes != null && completeBytes > 0n
      ? decimalRatio((completeMacs * 2n).toString(), completeBytes.toString()) : null,
    intensity_status: completeMacs == null || completeBytes == null
      ? "not_assessed_incomplete_mac_or_logical_byte_ledger"
      : completeBytes === 0n ? "not_applicable_zero_logical_bytes" : "derived_complete_artifact_ledger",
    output_tensor_reference_count: outputTensorReferenceCount,
    output_dtype_reference_counts: Object.fromEntries(Object.entries(outputDtypes).sort(([left], [right]) => left.localeCompare(right))),
  };
}

function assessedOpValue(op, field) {
  if (!op || op[field] == null || String(op[`${field}_status`] || "") === "not_assessed") return null;
  const value = Number(op[field]);
  return Number.isSafeInteger(value) && value >= 0 ? value : null;
}

function safeRatio(numerator, denominator) {
  if (numerator == null || denominator == null) return null;
  const num = BigInt(numerator);
  const den = BigInt(denominator);
  return den > 0n && num <= BigInt(Number.MAX_SAFE_INTEGER) && den <= BigInt(Number.MAX_SAFE_INTEGER)
    ? Number(num) / Number(den) : null;
}

function decimalRatio(numerator, denominator, digits = 12) {
  if (numerator == null || denominator == null) return null;
  const num = BigInt(numerator);
  const den = BigInt(denominator);
  if (den <= 0n) return null;
  const scale = 10n ** BigInt(digits);
  const scaled = (num * scale + den / 2n) / den;
  const whole = scaled / scale;
  const fraction = String(scaled % scale).padStart(digits, "0").replace(/0+$/, "");
  return fraction ? `${whole}.${fraction}` : whole.toString();
}

function bigSum(values) { return values.reduce((sum, value) => sum + BigInt(value), 0n); }
function safeMirror(value) { return value <= BigInt(Number.MAX_SAFE_INTEGER) ? Number(value) : null; }
function structuredCloneSafe(value) { return value == null ? value : JSON.parse(JSON.stringify(value)); }

function countStates(values) {
  const result = Object.fromEntries([...VALID_STATES].map((state) => [state, 0]));
  for (const row of values.values()) if (VALID_STATES.has(row.state)) result[row.state] += 1;
  return result;
}

function tensorIds(values) {
  return list(values).map(Number).filter((value) => Number.isSafeInteger(value) && value >= 0);
}
function uniqueStrings(values) { return [...new Set(list(values).map(reasonCode).filter(Boolean))].sort(); }
function reasonCode(value) { return String(value?.id || value?.reason_code || value || "").trim(); }
function safeSum(values) {
  const sum = values.reduce((total, value) => total + value, 0);
  if (!Number.isSafeInteger(sum)) throw new Error("Backend placement payload sum exceeds the safe integer range.");
  return sum;
}
function list(value) { return Array.isArray(value) ? value : []; }

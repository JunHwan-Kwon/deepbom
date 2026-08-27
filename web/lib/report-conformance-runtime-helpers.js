import { ANALYZER_METADATA } from "./report-metadata.js";
import { benchmarkNoise, latencyStats } from "./format.js";
import { BROWSER_BENCHMARK_NOISE_METHOD, BROWSER_BENCHMARK_STATISTICS_METHOD } from "./runtime.js";
import { sha256TextHex } from "./sha256-sync.js";
export function deriveRuntimeOpComparisons(analysis, runtimeAssignment, predictionApplicable) {
  const runtimeByOp = new Map((runtimeAssignment?.assignments || []).map((item) => [Number(item.op_index), item]));
  return (analysis?.ops || []).map((op) => {
    const runtime = runtimeByOp.get(Number(op.index));
    const predicted = predictionApplicable ? Number(op.xnnpack_chain_id) >= 0 : null;
    const observed = typeof runtime?.delegated === "boolean" ? runtime.delegated : null;
    const matches = observed == null || predicted == null ? null : predicted === observed;
    return {
      op_index: Number(op.index),
      op_name: op.name || "",
      predicted_delegated: predicted,
      predicted_domain: predicted == null ? "NOT_APPLICABLE" : predicted ? `XNNPACK:C${op.xnnpack_chain_id}` : "TFLITE_CPU",
      observed_delegated: observed,
      observed_provider: runtime?.provider || null,
      matches_prediction: matches,
      classification: observed == null
        ? "not_assessed"
        : predicted == null
          ? "observed_provider_no_static_prediction"
          : matches
            ? predicted ? "matched_delegated" : "matched_cpu"
            : predicted ? "overpredicted_delegation" : "underpredicted_delegation",
    };
  });
}

export function summarizeRuntimePlacement(rows, graphOpCount) {
  const observed = rows.filter((item) => item.observed_delegated != null);
  const assessed = observed.filter((item) => item.predicted_delegated != null);
  const matches = assessed.filter((item) => item.matches_prediction).length;
  return {
    observed: observed.length,
    observedCoverageRatio: graphOpCount > 0 ? observed.length / graphOpCount : null,
    assessed: assessed.length,
    unassessed: graphOpCount - assessed.length,
    matches,
    mismatches: assessed.length - matches,
    overpredicted: assessed.filter((item) => item.classification === "overpredicted_delegation").length,
    underpredicted: assessed.filter((item) => item.classification === "underpredicted_delegation").length,
    matchRatio: assessed.length ? matches / assessed.length : null,
  };
}

export function deriveRuntimeBoundaryComparisonEdges(analysis, runtimeAssignment, predictionApplicable) {
  const runtimeByOp = new Map((runtimeAssignment?.assignments || []).map((item) => [Number(item.op_index), item]));
  const tensors = new Map((analysis?.tensors || []).map((tensor) => [Number(tensor.index), tensor]));
  const producers = new Map();
  for (const op of analysis?.ops || []) for (const tensorIndex of op.outputs || []) if (Number(tensorIndex) >= 0) producers.set(Number(tensorIndex), op);
  const edges = [];
  for (const consumer of analysis?.ops || []) {
    for (const tensorIndex of new Set((consumer.inputs || []).map(Number).filter((index) => index >= 0))) {
      const producer = producers.get(tensorIndex);
      const tensor = tensors.get(tensorIndex);
      if (!producer || !tensor || tensor.constant_buffer) continue;
      const predicted = predictionApplicable ? predictedDomain(producer) !== predictedDomain(consumer) : null;
      const observed = independentlyObservedBoundary(runtimeByOp.get(Number(producer.index)), runtimeByOp.get(Number(consumer.index)));
      edges.push({
        tensor_index: tensorIndex,
        producer_op_index: Number(producer.index),
        consumer_op_index: Number(consumer.index),
        predicted_producer_domain: predictionApplicable ? predictedDomain(producer) : "NOT_APPLICABLE",
        predicted_consumer_domain: predictionApplicable ? predictedDomain(consumer) : "NOT_APPLICABLE",
        predicted_boundary: predicted,
        observed_boundary: observed,
        classification: observed == null
          ? "not_assessed"
          : predicted == null
            ? observed ? "observed_boundary_no_static_prediction" : "observed_same_domain_no_static_prediction"
            : observed === predicted
              ? predicted ? "matched_boundary" : "matched_no_boundary"
              : predicted ? "overpredicted_boundary" : "underpredicted_boundary",
        ...deterministicTensorPayloadAssessment(tensor),
      });
    }
  }
  return edges.sort((left, right) => runtimeBoundaryKey(left).localeCompare(runtimeBoundaryKey(right)));
}

export function independentlyObservedBoundary(producer, consumer) {
  const producerDelegated = typeof producer?.delegated === "boolean" ? producer.delegated : null;
  const consumerDelegated = typeof consumer?.delegated === "boolean" ? consumer.delegated : null;
  if (producerDelegated == null || consumerDelegated == null) return null;
  if (producerDelegated !== consumerDelegated) return true;
  if (!producerDelegated) return false;
  if (producer.provider !== consumer.provider) return true;
  if (producer.partition_id == null || consumer.partition_id == null) return null;
  return String(producer.partition_id) !== String(consumer.partition_id);
}

export function summarizeRuntimeBoundaryComparison(edges) {
  const observedRelations = edges.filter((edge) => edge.observed_boundary != null);
  const assessed = observedRelations.filter((edge) => edge.predicted_boundary != null);
  const matches = assessed.filter((edge) => edge.predicted_boundary === edge.observed_boundary).length;
  return {
    assessed: assessed.length,
    observedRelations: observedRelations.length,
    unassessed: edges.length - assessed.length,
    matches,
    mismatches: assessed.length - matches,
    predicted: edges.filter((edge) => edge.predicted_boundary === true),
    observed: edges.filter((edge) => edge.observed_boundary === true),
  };
}

export function summarizeRuntimeBoundaryPayload(edges) {
  const assessed = edges.filter((edge) => edge.payload_bytes != null);
  const unique = new Map();
  for (const edge of edges) if (!unique.has(edge.tensor_index)) unique.set(edge.tensor_index, edge.payload_bytes);
  const uniqueAssessed = [...unique.values()].filter((value) => value != null);
  return {
    assessedBytes: safeIntegerTotal(assessed.map((edge) => edge.payload_bytes)),
    totalBytes: assessed.length === edges.length ? safeIntegerTotal(assessed.map((edge) => edge.payload_bytes)) : null,
    uniqueCount: unique.size,
    uniqueBytes: uniqueAssessed.length === unique.size ? safeIntegerTotal(uniqueAssessed) : null,
  };
}

export function safeIntegerTotal(values) {
  let total = 0;
  for (const value of values) {
    total += Number(value);
    if (!Number.isSafeInteger(total)) return null;
  }
  return total;
}

export function runtimeBoundaryKey(edge) {
  return `${edge?.producer_op_index}:${edge?.consumer_op_index}:${edge?.tensor_index}`;
}

export function sameRuntimeBoundarySet(actual, expected) {
  const actualByKey = new Map(actual.map((edge) => [runtimeBoundaryKey(edge), edge]));
  return actual.length === expected.length && expected.every((edge) => {
    const item = actualByKey.get(runtimeBoundaryKey(edge));
    return item?.predicted_boundary === edge.predicted_boundary
      && item?.observed_boundary === edge.observed_boundary
      && item?.classification === edge.classification
      && item?.payload_bytes === edge.payload_bytes;
  });
}

export function deriveRuntimePartitionSummary(analysis, runtimeAssignment, allowImplicit) {
  const assignments = new Map((runtimeAssignment?.assignments || []).map((item) => [Number(item.op_index), item]));
  const ops = [...(analysis?.ops || [])].sort((left, right) => Number(left.index) - Number(right.index));
  const segments = [];
  let active = null;
  const flush = () => { if (active) segments.push(active); active = null; };
  for (const op of ops) {
    const row = assignments.get(Number(op.index));
    if (row?.delegated !== true) { flush(); continue; }
    const explicit = row.partition_id != null;
    const key = explicit ? `${row.provider}\u0000${row.partition_id}` : `implicit\u0000${row.provider}`;
    if (!active || active.key !== key) {
      flush();
      active = { key, provider: row.provider, partition_id: row.partition_id ?? null, explicit_partition_id: explicit, first_op: Number(op.index), last_op: Number(op.index), op_indices: [] };
    }
    active.last_op = Number(op.index);
    active.op_indices.push(Number(op.index));
  }
  flush();
  const explicit = new Map();
  for (const segment of segments.filter((item) => item.explicit_partition_id)) {
    let partition = explicit.get(segment.key);
    if (!partition) {
      partition = { ...segment, op_indices: [], contiguous_segment_count: 0 };
      explicit.set(segment.key, partition);
    }
    partition.first_op = Math.min(partition.first_op, segment.first_op);
    partition.last_op = Math.max(partition.last_op, segment.last_op);
    partition.op_indices.push(...segment.op_indices);
    partition.contiguous_segment_count += 1;
  }
  const explicitPartitions = [...explicit.values()].map((partition) => ({ ...partition, op_indices: partition.op_indices.sort((left, right) => left - right) }));
  const implicitPartitions = allowImplicit
    ? segments.filter((item) => !item.explicit_partition_id).map((segment) => ({ ...segment, contiguous_segment_count: 1 }))
    : [];
  const partitions = [...explicitPartitions, ...implicitPartitions];
  let providerSegmentCount = 0;
  let previousProviderKey = null;
  for (const op of ops) {
    const row = assignments.get(Number(op.index));
    const key = typeof row?.delegated === "boolean" ? `${row.provider}\u0000${row.delegated}` : null;
    if (key != null && key !== previousProviderKey) providerSegmentCount += 1;
    previousProviderKey = key;
  }
  return {
    status: partitions.length ? "assessed_from_imported_rows" : providerSegmentCount && !allowImplicit ? "not_assessed_no_partition_ids" : "not_assessed",
    partitionCount: partitions.length,
    explicitCount: explicitPartitions.length,
    implicitCount: implicitPartitions.length,
    noncontiguousCount: explicitPartitions.filter((partition) => partition.contiguous_segment_count > 1).length,
    providerSegmentCount,
    partitions,
  };
}

export function sameRuntimePartitionInventory(actual, expected) {
  const keyFor = (item) => item.explicit_partition_id
    ? `explicit\u0000${item.provider}\u0000${item.partition_id}`
    : `implicit\u0000${item.provider}\u0000${item.first_op}\u0000${item.last_op}`;
  const actualByKey = new Map(actual.map((item) => [keyFor(item), item]));
  return actual.length === expected.length && expected.every((item) => {
    const emitted = actualByKey.get(keyFor(item));
    return emitted?.first_op === item.first_op
      && emitted?.last_op === item.last_op
      && emitted?.contiguous_segment_count === item.contiguous_segment_count
      && sameArray(emitted?.op_indices, item.op_indices);
  });
}

export function nullableClose(actual, expected) {
  return actual == null || expected == null ? actual == null && expected == null : Math.abs(Number(actual) - Number(expected)) < 1e-15;
}

export function validIsoTimestamp(value) {
  const parsed = Date.parse(value);
  return Number.isFinite(parsed) && new Date(parsed).toISOString() === value;
}

export function validBenchmarkInputContract(input) {
  return validBenchmarkTensorContract(input, "input_name", false);
}

export function validBenchmarkStatistics(row) {
  const samples = row?.measured_samples_ms;
  if (row?.statistics_method !== BROWSER_BENCHMARK_STATISTICS_METHOD || !Array.isArray(samples)
    || samples.length !== row.runs || samples.some((value) => !Number.isFinite(value) || value < 0)) return false;
  const expected = latencyStats(samples);
  const expectedNoise = benchmarkNoise(samples);
  return row.noise_method === BROWSER_BENCHMARK_NOISE_METHOD
    && ["min", "max", "p50", "p90", "p95", "p99", "mean", "stddev", "cv"]
      .every((key) => Number(row.stats?.[key]) === expected[key] && Number(row.steady_stats?.[key]) === expected[key])
    && ["outlierCount", "gcSpikeCount", "trendSlope", "trimmedP50", "trimmedMean"]
      .every((key) => Number(row.noise_diagnostics?.[key]) === expectedNoise[key])
    && row.noise_diagnostics?.trendLabel === expectedNoise.trendLabel;
}

export function validBenchmarkOutputContract(output) {
  return output?.basis === "observed_runtime_output" && validBenchmarkTensorContract(output, "output_name", true);
}

export function validOnnxBenchmarkExternalDataBinding(binding, staticAnalysis) {
  const external = staticAnalysis?.onnx_external_data || {};
  const tensorCount = Number(external.tensor_count || 0);
  if (!binding || binding.schema !== ANALYZER_METADATA.schemas.onnxExternalRuntimeBinding) return false;
  if (!tensorCount) {
    return binding.status === "not_applicable" && binding.tensor_count === 0
      && binding.file_count === 0 && binding.total_file_bytes === 0
      && Array.isArray(binding.files) && binding.files.length === 0;
  }
  const files = Array.isArray(binding.files) ? binding.files : [];
  const paths = new Set(files.map((file) => file.path));
  const referencedPaths = new Set((external.tensors || []).map((row) => row.sidecar_path));
  return Number(external.verified_payload_count || 0) === tensorCount
    && binding.status === "bound_verified_external_data"
    && binding.evidence_class === "OBSERVED/DERIVED"
    && binding.tensor_count === tensorCount
    && binding.file_count === files.length
    && paths.size === files.length && paths.size === referencedPaths.size
    && [...referencedPaths].every((path) => paths.has(path))
    && binding.total_file_bytes === files.reduce((total, file) => total + Number(file.byte_length || 0), 0)
    && files.every((file) => externalLocationStatus(file.path) === "safe_relative_path"
      && Number.isSafeInteger(file.byte_length) && file.byte_length >= 0
      && /^[a-f0-9]{64}$/.test(file.sha256 || "")
      && (external.tensors || []).filter((row) => row.sidecar_path === file.path)
        .every((row) => row.sidecar_sha256 === file.sha256 && row.sidecar_bytes === file.byte_length));
}

export function validBenchmarkTensorContract(input, nameKey, allowMissingDeclaredShape) {
  if (!input || !String(input.artifact_dtype || "").trim()
    || !String(input.runtime_dtype || "").trim() || !String(input.basis || "").trim()) return false;
  if (!String(input[nameKey] || "").trim()
    || !Object.prototype.hasOwnProperty.call(input, "artifact_shape_signature")
    || !Object.prototype.hasOwnProperty.call(input, "runtime_declared_shape")
    || (!Array.isArray(input.declared_shape) && !(allowMissingDeclaredShape && input.declared_shape == null))
    || !Array.isArray(input.executed_shape)
    || (input.artifact_shape_signature != null && !Array.isArray(input.artifact_shape_signature))
    || (input.runtime_declared_shape != null && !Array.isArray(input.runtime_declared_shape))) return false;
  const executed = input.executed_shape;
  if ((input.declared_shape && input.declared_shape.length !== executed.length)
    || (input.artifact_shape_signature && input.artifact_shape_signature.length !== executed.length)
    || (input.runtime_declared_shape && input.runtime_declared_shape.length !== executed.length)
    || executed.some((dim) => !Number.isSafeInteger(dim) || dim <= 0)) return false;
  const matchesKnown = (shape) => !shape || shape.every((dim, axis) => Number(dim) <= 0 || Number(dim) === executed[axis]);
  const elements = executed.reduce((total, dim) => total * dim, 1);
  const artifactContractShape = input.artifact_shape_signature || input.declared_shape;
  const artifactType = String(input.artifact_dtype).toLowerCase();
  return Number.isSafeInteger(elements) && Number(input.element_count) === elements
    && artifactType !== "unknown" && artifactType === String(input.runtime_dtype).toLowerCase()
    && matchesKnown(artifactContractShape) && matchesKnown(input.runtime_declared_shape);
}

export function independentCommonRunSubtotal(rows) {
  if (!rows.length || rows.some((item) => !Number.isSafeInteger(item.run_count) || item.run_count <= 0)) return null;
  if (new Set(rows.map((item) => item.run_count)).size !== 1) return null;
  return rows.reduce((sum, item) => sum + Number(item.sum_us) / item.run_count, 0);
}

export function collectAssessmentMetricFailures(value, path = "") {
  const failures = [];
  if (!value || typeof value !== "object") return failures;
  if (!Array.isArray(value) && ["assessed", "not_assessed", "not_applicable"].includes(value.status) && Object.prototype.hasOwnProperty.call(value, "value")) {
    if ((value.status === "not_assessed" || value.status === "not_applicable") && value.value !== null) failures.push(`${path || "/"}: ${value.status} metric must have value=null.`);
    if (value.status === "assessed" && value.value === null) failures.push(`${path || "/"}: assessed metric must have a concrete value.`);
  }
  for (const [key, child] of Object.entries(value)) failures.push(...collectAssessmentMetricFailures(child, `${path}/${escapePointer(key)}`));
  return failures;
}

export function escapePointer(value) {
  return String(value).replaceAll("~", "~0").replaceAll("/", "~1");
}

export function sameArray(left, right) {
  return Array.isArray(left) && Array.isArray(right)
    && left.length === right.length
    && left.every((value, index) => Number(value) === Number(right[index]));
}

export function tensorElementCount(shape) {
  return Array.isArray(shape)
    ? shape.reduce((product, dim) => product * Math.max(0, Number(dim) || 0), 1)
    : 0;
}

export function derivePredictedBoundaryEdges(analysis) {
  const tensors = new Map((analysis?.tensors || []).map((tensor) => [Number(tensor.index), tensor]));
  const producers = new Map();
  for (const op of analysis?.ops || []) {
    for (const tensorIndex of op.outputs || []) if (Number(tensorIndex) >= 0) producers.set(Number(tensorIndex), op);
  }
  const edges = [];
  for (const consumer of analysis?.ops || []) {
    for (const tensorIndex of new Set((consumer.inputs || []).map(Number).filter((index) => index >= 0))) {
      const producer = producers.get(tensorIndex);
      const tensor = tensors.get(tensorIndex);
      if (!producer || !tensor || tensor.constant_buffer) continue;
      const producerDomain = predictedDomain(producer);
      const consumerDomain = predictedDomain(consumer);
      if (producerDomain === consumerDomain) continue;
      const payload = deterministicTensorPayloadAssessment(tensor);
      edges.push({
        tensor_index: tensorIndex,
        tensor_shape: tensor.shape || [],
        tensor_dtype: tensor.dtype || "",
        payload_bytes: payload.payload_bytes,
        payload_status: payload.payload_status,
        payload_binding: payload.payload_binding,
        producer_op_index: producer.index,
        producer_domain: producerDomain,
        consumer_op_index: consumer.index,
        consumer_domain: consumerDomain,
        direction: producer.xnnpack_chain_id >= 0
          ? consumer.xnnpack_chain_id >= 0 ? "delegate_partition_to_delegate_partition" : "delegate_to_cpu"
          : "cpu_to_delegate",
      });
    }
  }
  return edges.sort((left, right) => boundaryEdgeKey(left).localeCompare(boundaryEdgeKey(right)));
}

export function predictedDomain(op) {
  return Number(op?.xnnpack_chain_id) >= 0 ? `XNNPACK:C${op.xnnpack_chain_id}` : "TFLITE_CPU";
}

export function boundaryEdgeKey(edge) {
  return `${edge.producer_op_index}:${edge.consumer_op_index}:${edge.tensor_index}`;
}

export function deterministicTensorPayloadBytes(tensor) {
  return deterministicTensorPayloadAssessment(tensor).payload_bytes;
}

export function deterministicTensorPayloadAssessment(tensor) {
  const shape = tensor?.shape || [];
  const signature = tensor?.shape_signature || [];
  if (shape.some((dim) => !Number.isInteger(Number(dim)) || Number(dim) < 0)) {
    return { payload_bytes: null, payload_status: "not_assessed", payload_binding: "unbound" };
  }
  const staticSignature = !signature.length
    || (signature.length === shape.length && signature.every((dim, index) => Number.isInteger(Number(dim)) && Number(dim) >= 0 && Number(dim) === Number(shape[index])));
  const serializedBatchOne = signature.length === shape.length
    && Number(signature[0]) === -1
    && Number(shape[0]) === 1
    && signature.slice(1).every((dim, index) => Number.isInteger(Number(dim)) && Number(dim) >= 0 && Number(dim) === Number(shape[index + 1]));
  if (!staticSignature && !serializedBatchOne) {
    return { payload_bytes: null, payload_status: "not_assessed", payload_binding: "unbound" };
  }
  const elements = shape.reduce((product, dim) => product * Number(dim), 1);
  if (!Number.isSafeInteger(elements)) {
    return { payload_bytes: null, payload_status: "not_assessed", payload_binding: "unbound" };
  }
  const dtype = String(tensor?.dtype || "").toUpperCase();
  const payload = onnxTensorPayloadBytes(dtype, elements);
  if (payload == null) {
    return { payload_bytes: null, payload_status: "not_assessed", payload_binding: "unbound" };
  }
  return {
    payload_bytes: payload,
    payload_status: serializedBatchOne ? "assessed_serialized_batch1" : "assessed_static",
    payload_binding: serializedBatchOne ? "serialized_batch1_projection" : "static",
  };
}

export function deriveDelegationRepairGraph(analysis) {
  const ops = analysis?.ops || [];
  const tensors = new Map((analysis?.tensors || []).map((tensor) => [Number(tensor.index), tensor]));
  const producers = new Map();
  ops.forEach((op, position) => {
    for (const tensorIndex of op.outputs || []) if (Number(tensorIndex) >= 0) producers.set(Number(tensorIndex), position);
  });
  const edges = [];
  ops.forEach((consumer, consumerPosition) => {
    for (const tensorIndex of new Set((consumer.inputs || []).map(Number).filter((index) => index >= 0))) {
      const producerPosition = producers.get(tensorIndex);
      const tensor = tensors.get(tensorIndex);
      if (producerPosition == null || !tensor || tensor.constant_buffer) continue;
      const producer = ops[producerPosition];
      edges.push({
        key: `${producer.index}:${consumer.index}:${tensorIndex}`,
        producerPosition,
        consumerPosition,
        producerIndex: producer.index,
        consumerIndex: consumer.index,
        tensorIndex,
        payloadBytes: deterministicTensorPayloadBytes(tensor),
      });
    }
  });
  return {
    edges: edges.sort((left, right) => left.producerIndex - right.producerIndex || left.consumerIndex - right.consumerIndex || left.tensorIndex - right.tensorIndex),
    baselineAssignments: ops.map((op) => Number(op.xnnpack_chain_id) >= 0),
  };
}

export function deriveDelegationState(graph, assignments) {
  const segments = Array(assignments.length).fill(null);
  let activeSegment = null;
  let nextSegment = 0;
  assignments.forEach((delegated, position) => {
    if (!delegated) {
      activeSegment = null;
      return;
    }
    if (activeSegment == null) activeSegment = nextSegment++;
    segments[position] = activeSegment;
  });
  const boundaries = new Map();
  for (const edge of graph.edges) {
    const producer = assignments[edge.producerPosition];
    const consumer = assignments[edge.consumerPosition];
    const boundary = producer && consumer
      ? segments[edge.producerPosition] !== segments[edge.consumerPosition]
      : producer !== consumer;
    if (!boundary) continue;
    boundaries.set(edge.key, {
      ...edge,
      direction: producer && !consumer ? "delegate_to_cpu"
        : !producer && consumer ? "cpu_to_delegate"
          : "delegate_partition_to_delegate_partition",
    });
  }
  const payloads = [...boundaries.values()].map((edge) => edge.payloadBytes);
  const assessed = payloads.filter((value) => value != null);
  return {
    boundaries,
    summary: {
      delegate_segment_count: new Set(segments.filter((value) => value != null)).size,
      cpu_segment_count: deriveCpuIslandRanges(assignments).length,
      boundary_edge_count: boundaries.size,
      assessed_payload_edge_count: assessed.length,
      unassessed_payload_edge_count: payloads.length - assessed.length,
      assessed_edge_payload_bytes: assessed.reduce((sum, value) => sum + value, 0),
      summed_edge_payload_bytes: assessed.length === payloads.length ? assessed.reduce((sum, value) => sum + value, 0) : null,
    },
  };
}

export function deriveCpuIslandRanges(assignments) {
  const ranges = [];
  let start = null;
  assignments.forEach((delegated, position) => {
    if (!delegated && start == null) start = position;
    if (delegated && start != null) {
      ranges.push([start, position - 1]);
      start = null;
    }
  });
  if (start != null) ranges.push([start, assignments.length - 1]);
  return ranges;
}

export function deriveDelegationChanges(baseline, counterfactual) {
  const keys = [...new Set([...baseline.keys(), ...counterfactual.keys()])].sort();
  const changes = [];
  for (const key of keys) {
    const left = baseline.get(key);
    const right = counterfactual.get(key);
    if (left && !right) changes.push({ transition: "removed", ...left, baselineDirection: left.direction, counterfactualDirection: null });
    else if (!left && right) changes.push({ transition: "added", ...right, baselineDirection: null, counterfactualDirection: right.direction });
    else if (left?.direction !== right?.direction) changes.push({ transition: "reclassified", ...left, baselineDirection: left.direction, counterfactualDirection: right.direction });
  }
  const transitionRank = (value) => value === "removed" ? 0 : value === "added" ? 1 : 2;
  return changes.sort((left, right) => transitionRank(left.transition) - transitionRank(right.transition) || left.producerIndex - right.producerIndex || left.consumerIndex - right.consumerIndex || left.tensorIndex - right.tensorIndex);
}

export function sameDelegationSummary(actual, expected) {
  return ["delegate_segment_count", "cpu_segment_count", "boundary_edge_count", "assessed_payload_edge_count", "unassessed_payload_edge_count", "assessed_edge_payload_bytes"]
    .every((field) => actual?.[field] === expected[field])
    && (actual?.summed_edge_payload_bytes ?? null) === expected.summed_edge_payload_bytes;
}

export function sameDelegationChanges(actual, expected) {
  return actual.length === expected.length && expected.every((item, index) => {
    const row = actual[index];
    return row?.transition === item.transition
      && row?.producer_op_index === item.producerIndex
      && row?.consumer_op_index === item.consumerIndex
      && row?.tensor_index === item.tensorIndex
      && (row?.payload_bytes ?? null) === item.payloadBytes
      && (row?.baseline_direction ?? null) === item.baselineDirection
      && (row?.counterfactual_direction ?? null) === item.counterfactualDirection;
  });
}

const ARENA_IN_PLACE_REGISTRATIONS = Object.freeze({
  RESHAPE: { slots: [0], dataUnmodified: true },
  SQUEEZE: { slots: [0], dataUnmodified: true },
  BITCAST: { slots: [0], dataUnmodified: true },
  EXPAND_DIMS: { slots: [0], dataUnmodified: true },
  SOFTMAX: { slots: [0], dataUnmodified: false },
  DYNAMIC_UPDATE_SLICE: { slots: [0], dataUnmodified: false },
  ADD: { slots: [0, 1], dataUnmodified: false },
  SUB: { slots: [0, 1], dataUnmodified: false },
  MUL: { slots: [0, 1], dataUnmodified: false },
  DIV: { slots: [0, 1], dataUnmodified: false },
});

const MOVEMENT_OPERATOR_NAMES = new Set([
  "TRANSPOSE", "PAD", "PAD_V2", "MIRROR_PAD", "CONCATENATION",
  "RESIZE_BILINEAR", "RESIZE_NEAREST_NEIGHBOR", "QUANTIZE", "DEQUANTIZE",
  "SLICE", "STRIDED_SLICE", "GATHER", "GATHER_ND", "SPLIT", "SPLIT_V", "TILE",
]);

export function deriveDeclaredLiveness(analysis) {
  const tensors = new Map((analysis?.tensors || []).map((tensor) => [Number(tensor.index), tensor]));
  const ops = analysis?.ops || [];
  const inputs = (analysis?.input_tensor_indices || []).map(Number).filter((index) => index >= 0);
  const outputs = new Set((analysis?.output_tensor_indices || []).map(Number).filter((index) => index >= 0));
  const relevant = new Set(inputs);
  for (const op of ops) for (const rawIndex of op.outputs || []) if (Number(rawIndex) >= 0) relevant.add(Number(rawIndex));
  const sizes = new Map();
  for (const index of relevant) {
    const tensor = tensors.get(index);
    if (!tensor) return { assessed: false, peakBytes: null, peakAtOp: null, peakAtOpName: null };
    if (tensor.constant_buffer) continue;
    const bytes = declaredArenaPayloadBytes(tensor);
    if (bytes == null) return { assessed: false, peakBytes: null, peakAtOp: null, peakAtOpName: null };
    sizes.set(index, bytes);
  }
  const lastConsumer = new Map();
  for (const op of ops) {
    for (const rawIndex of op.inputs || []) {
      const index = Number(rawIndex);
      if (index >= 0) lastConsumer.set(index, Math.max(lastConsumer.get(index) || 0, Number(op.index)));
    }
  }
  for (const index of outputs) lastConsumer.set(index, Number.POSITIVE_INFINITY);
  const live = new Set();
  let current = 0;
  const add = (index) => {
    if (!live.has(index)) {
      live.add(index);
      current += sizes.get(index) || 0;
    }
    return Number.isSafeInteger(current);
  };
  for (const index of inputs) if (!add(index)) return { assessed: false, peakBytes: null, peakAtOp: null, peakAtOpName: null };
  let peakBytes = current;
  let peakAtOp = null;
  let peakAtOpName = null;
  for (const op of ops) {
    for (const rawIndex of op.outputs || []) {
      const index = Number(rawIndex);
      if (index >= 0 && !add(index)) return { assessed: false, peakBytes: null, peakAtOp: null, peakAtOpName: null };
    }
    if (current > peakBytes) {
      peakBytes = current;
      peakAtOp = Number(op.index);
      peakAtOpName = String(op.name || "");
    }
    for (const rawIndex of [...(op.inputs || []), ...(op.outputs || [])]) {
      const index = Number(rawIndex);
      if (index < 0 || outputs.has(index)) continue;
      if (lastConsumer.get(index) === Number(op.index) && live.delete(index)) current -= sizes.get(index) || 0;
    }
  }
  return { assessed: true, peakBytes, peakAtOp, peakAtOpName };
}

export function deriveMovementPayload(analysis) {
  const tensors = new Map((analysis?.tensors || []).map((tensor) => [Number(tensor.index), tensor]));
  const ops = analysis?.ops || [];
  let assessedBytes = 0;
  let assessedBreakBytes = 0;
  let issueCount = 0;
  let opCount = 0;
  for (const op of ops) {
    if (!MOVEMENT_OPERATOR_NAMES.has(String(op.name || ""))) continue;
    opCount += 1;
    let opBytes = 0;
    for (const rawIndex of op.outputs || []) {
      const index = Number(rawIndex);
      if (index < 0) continue;
      const bytes = declaredArenaPayloadBytes(tensors.get(index));
      if (bytes == null || !Number.isSafeInteger(opBytes + bytes)) {
        issueCount += 1;
        continue;
      }
      opBytes += bytes;
    }
    if (!Number.isSafeInteger(assessedBytes + opBytes)) issueCount += 1;
    else assessedBytes += opBytes;
    if (op.xnnpack_chain_break) {
      if (!Number.isSafeInteger(assessedBreakBytes + opBytes)) issueCount += 1;
      else assessedBreakBytes += opBytes;
    }
  }
  return {
    status: issueCount ? "partial" : "assessed",
    totalBytes: issueCount ? null : assessedBytes,
    breakBytes: issueCount ? null : assessedBreakBytes,
    assessedBytes,
    assessedBreakBytes,
    opCount,
    opRatio: ops.length ? opCount / ops.length : 0,
  };
}

export function declaredArenaPayloadBytes(tensor) {
  const shape = tensor?.shape || [];
  if (shape.some((dim) => !Number.isInteger(Number(dim)) || Number(dim) < 0)) return null;
  const elements = shape.reduce((product, dim) => product * Number(dim), 1);
  if (!Number.isSafeInteger(elements)) return null;
  const dtype = String(tensor?.dtype || "").toUpperCase();
  return onnxTensorPayloadBytes(dtype, elements);
}

export function deriveArenaProjection(analysis) {
  const tensors = new Map((analysis?.tensors || []).map((tensor) => [Number(tensor.index), tensor]));
  const ops = analysis?.ops || [];
  const inputs = new Set((analysis?.input_tensor_indices || []).map(Number).filter((index) => index >= 0));
  const outputs = new Set((analysis?.output_tensor_indices || []).map(Number).filter((index) => index >= 0));
  const variables = new Set([...tensors.values()].filter((tensor) => tensor.is_variable).map((tensor) => Number(tensor.index)));
  const producer = new Map();
  const allocationNode = new Map();
  for (const index of inputs) allocationNode.set(index, 0);
  for (const index of variables) allocationNode.set(index, 0);
  for (const op of ops) {
    for (const rawIndex of op.outputs || []) {
      const tensorIndex = Number(rawIndex);
      if (tensorIndex < 0) continue;
      if (!producer.has(tensorIndex)) producer.set(tensorIndex, Number(op.index));
      if (!allocationNode.has(tensorIndex)) allocationNode.set(tensorIndex, Number(op.index));
    }
  }
  const relevant = new Set([...inputs, ...outputs, ...variables, ...producer.keys()]);
  for (const [index, tensor] of tensors) if (tensor.constant_buffer) relevant.delete(index);

  const increment = (map, key) => map.set(key, (map.get(key) || 0) + 1);
  const originalRefs = new Map();
  for (const index of [...outputs, ...inputs, ...variables]) increment(originalRefs, index);
  for (const op of ops) for (const rawIndex of op.inputs || []) if (Number(rawIndex) >= 0) increment(originalRefs, Number(rawIndex));

  const arenaClass = (tensor) => tensor?.constant_buffer ? null : tensor?.is_variable ? "kTfLiteArenaRwPersistent" : "kTfLiteArenaRw";
  const sizeByIndex = new Map([...tensors].map(([index, tensor]) => [index, declaredArenaPayloadBytes(tensor)]));
  const sharedRoots = new Map();
  const rootFor = (index) => sharedRoots.get(index) ?? index;
  const aliases = [];
  for (const op of ops) {
    const registration = ARENA_IN_PLACE_REGISTRATIONS[String(op.name || "")];
    const outputIndex = Number(op.outputs?.[0]);
    const outputTensor = tensors.get(outputIndex);
    if (!registration || outputIndex < 0 || !outputTensor || outputs.has(outputIndex) || outputTensor.constant_buffer) continue;
    for (const inputSlot of registration.slots) {
      const inputIndex = Number(op.inputs?.[inputSlot]);
      const inputTensor = tensors.get(inputIndex);
      if (inputIndex < 0 || !inputTensor || inputs.has(inputIndex) || inputTensor.constant_buffer) continue;
      const inputArena = arenaClass(inputTensor);
      const outputArena = arenaClass(outputTensor);
      if (inputArena !== "kTfLiteArenaRw" || outputArena !== "kTfLiteArenaRw") continue;
      const inputBytes = sizeByIndex.get(inputIndex);
      const outputBytes = sizeByIndex.get(outputIndex);
      if (inputBytes == null || outputBytes == null || inputBytes !== outputBytes) continue;
      if (!registration.dataUnmodified) {
        if (inputBytes <= 4 || (originalRefs.get(inputIndex) || 0) > 1) continue;
      }
      const rootTensorIndex = rootFor(inputIndex);
      if (!registration.dataUnmodified && (originalRefs.get(rootTensorIndex) || 0) > 1) continue;
      sharedRoots.set(outputIndex, rootTensorIndex);
      aliases.push({
        tensorIndex: outputIndex,
        rootTensorIndex,
        opIndex: Number(op.index),
        dataUnmodified: registration.dataUnmodified,
      });
      break;
    }
  }

  const runtimeRefs = new Map();
  for (const index of [...outputs, ...inputs, ...variables]) increment(runtimeRefs, rootFor(index));
  for (const op of ops) for (const rawIndex of op.inputs || []) if (Number(rawIndex) >= 0) increment(runtimeRefs, rootFor(Number(rawIndex)));
  const deallocationNode = new Map();
  for (const op of ops) {
    for (const rawIndex of op.inputs || []) {
      const tensorIndex = Number(rawIndex);
      if (tensorIndex < 0) continue;
      const root = rootFor(tensorIndex);
      const next = Math.max(0, (runtimeRefs.get(root) || 0) - 1);
      runtimeRefs.set(root, next);
      if (next === 0) deallocationNode.set(root, Number(op.index));
    }
  }

  const candidates = [];
  const issues = [];
  for (const tensorIndex of [...relevant].sort((left, right) => left - right)) {
    if (sharedRoots.has(tensorIndex)) continue;
    const tensor = tensors.get(tensorIndex);
    const firstNode = allocationNode.get(tensorIndex);
    const arena = arenaClass(tensor);
    if (firstNode == null || !arena) continue;
    const sizeBytes = sizeByIndex.get(tensorIndex);
    if (sizeBytes == null) {
      issues.push(tensorIndex);
      continue;
    }
    candidates.push({
      tensorIndex,
      sizeBytes,
      firstNode,
      lastNode: inputs.has(tensorIndex) || outputs.has(tensorIndex) || variables.has(tensorIndex)
        ? null
        : (deallocationNode.get(tensorIndex) ?? null),
      arena,
      offsetBytes: null,
    });
  }
  const comparatorTieCounts = new Map();
  for (const candidate of candidates) {
    if (candidate.firstNode === 0 && candidate.lastNode == null) continue;
    const key = `${candidate.sizeBytes}:${candidate.firstNode}`;
    comparatorTieCounts.set(key, (comparatorTieCounts.get(key) || 0) + 1);
  }
  const sourceComparatorTieGroups = [...comparatorTieCounts.values()].filter((count) => count > 1).length;
  const sourceComparatorTiedTensors = [...comparatorTieCounts.values()].filter((count) => count > 1).reduce((sum, count) => sum + count, 0);
  candidates.sort((left, right) => {
    const leftFull = left.firstNode === 0 && left.lastNode == null;
    const rightFull = right.firstNode === 0 && right.lastNode == null;
    if (leftFull && rightFull) return left.tensorIndex - right.tensorIndex;
    if (leftFull !== rightFull) return leftFull ? -1 : 1;
    return right.sizeBytes - left.sizeBytes || left.firstNode - right.firstNode || left.tensorIndex - right.tensorIndex;
  });

  const states = new Map([
    ["kTfLiteArenaRw", { highWater: 0, allocations: [] }],
    ["kTfLiteArenaRwPersistent", { highWater: 0, allocations: [] }],
  ]);
  for (const candidate of candidates) {
    const state = states.get(candidate.arena);
    const offset = placeArenaCandidate(state, candidate, 64);
    if (offset == null) {
      issues.push(candidate.tensorIndex);
      continue;
    }
    candidate.offsetBytes = offset;
  }
  const assessed = issues.length === 0;
  const nonPersistent = states.get("kTfLiteArenaRw").highWater;
  const persistent = states.get("kTfLiteArenaRwPersistent").highWater;
  const combined = nonPersistent + persistent;
  return {
    status: assessed ? "assessed" : candidates.length ? "partial" : "not_assessed",
    nonPersistentBytes: assessed ? nonPersistent : null,
    persistentBytes: assessed ? persistent : null,
    combinedBytes: assessed && Number.isSafeInteger(combined) ? combined : null,
    candidates,
    aliases,
    sourceComparatorTieGroups,
    sourceComparatorTiedTensors,
  };
}

export function placeArenaCandidate(state, candidate, alignment) {
  if (!state || !Number.isSafeInteger(candidate.sizeBytes)) return null;
  if (candidate.sizeBytes === 0) return 0;
  let currentOffset = 0;
  let bestOffset = null;
  let bestOffsetFit = Number.POSITIVE_INFINITY;
  for (const allocation of state.allocations) {
    if (!lifetimesOverlap(allocation.firstNode, allocation.lastNode, candidate.firstNode, candidate.lastNode)) continue;
    const alignedCurrentOffset = alignArenaOffset(currentOffset, alignment);
    const alignedCurrentEnd = alignedCurrentOffset + candidate.sizeBytes;
    if (!Number.isSafeInteger(alignedCurrentEnd)) return null;
    if (alignedCurrentEnd <= allocation.offsetBytes && allocation.offsetBytes - alignedCurrentOffset < bestOffsetFit) {
      bestOffset = alignedCurrentOffset;
      bestOffsetFit = allocation.offsetBytes - currentOffset;
    }
    currentOffset = Math.max(currentOffset, allocation.offsetBytes + allocation.sizeBytes);
    if (!Number.isSafeInteger(currentOffset)) return null;
    if (bestOffsetFit === 0) break;
  }
  const offsetBytes = bestOffset ?? alignArenaOffset(currentOffset, alignment);
  const required = offsetBytes + candidate.sizeBytes;
  if (!Number.isSafeInteger(offsetBytes) || !Number.isSafeInteger(required)) return null;
  state.highWater = Math.max(state.highWater, required);
  const placed = { ...candidate, offsetBytes };
  const insertion = state.allocations.findIndex((allocation) => allocation.offsetBytes > offsetBytes);
  if (insertion < 0) state.allocations.push(placed);
  else state.allocations.splice(insertion, 0, placed);
  return offsetBytes;
}

export function alignArenaOffset(value, alignment) {
  const aligned = Math.ceil(value / alignment) * alignment;
  return Number.isSafeInteger(aligned) ? aligned : Number.NaN;
}

export function lifetimesOverlap(leftFirst, leftLast, rightFirst, rightLast) {
  const leftEnd = leftLast == null ? Number.POSITIVE_INFINITY : Number(leftLast);
  const rightEnd = rightLast == null ? Number.POSITIVE_INFINITY : Number(rightLast);
  return !(leftEnd < Number(rightFirst) || Number(leftFirst) > rightEnd);
}

export function arenaAllocationsDoNotConflict(allocations) {
  for (let leftIndex = 0; leftIndex < allocations.length; leftIndex += 1) {
    const left = allocations[leftIndex];
    for (let rightIndex = leftIndex + 1; rightIndex < allocations.length; rightIndex += 1) {
      const right = allocations[rightIndex];
      if (left.arena !== right.arena || !lifetimesOverlap(left.first_node, left.last_node, right.first_node, right.last_node)) continue;
      const leftStart = Number(left.offset_bytes);
      const leftEnd = leftStart + Number(left.size_bytes);
      const rightStart = Number(right.offset_bytes);
      const rightEnd = rightStart + Number(right.size_bytes);
      if (leftStart < rightEnd && rightStart < leftEnd) return false;
    }
  }
  return true;
}

export function reconstructRuntimeArenaEvidence(staticAnalysis, runtimeAssignment) {
  const memory = runtimeAssignment?.runtime_memory || null;
  const reconciliation = runtimeAssignment?.arena_reconciliation || null;
  const collector = runtimeAssignment?.source?.collector || null;
  const instrumentationDeclared = collector?.instrumentation?.arena_allocations === true;
  if (!memory) {
    return {
      present: false,
      valid: !instrumentationDeclared && reconciliation == null,
      errors: instrumentationDeclared ? ["declared arena instrumentation has no runtime memory evidence"] : reconciliation == null ? [] : ["reconciliation exists without runtime memory evidence"],
      reconciliation: null,
      hasDifference: false,
    };
  }
  const errors = [];
  const requireCondition = (condition, message) => {
    if (!condition) errors.push(message);
  };
  const integer = (value, label, { positive = false } = {}) => {
    const valid = typeof value === "number" && Number.isSafeInteger(value) && value >= (positive ? 1 : 0);
    requireCondition(valid, `${label} is not a ${positive ? "positive" : "non-negative"} safe integer`);
    return valid ? value : 0;
  };
  const commit = "87bbf65b8d23d3f06912b1b2183587e1884bc45c";
  requireCondition(runtimeAssignment?.schema === "deepbom.runtime_assignment.v1.9", "runtime assignment schema does not carry the arena contract");
  requireCondition(collector?.schema === "deepbom.native_runtime_collector.v1.1" && instrumentationDeclared, "collector did not declare v1.1 arena instrumentation");
  requireCondition(memory.schema === "deepbom.runtime_memory.v1" && memory.status === "assessed" && memory.evidence_class === "OBSERVED_RUNTIME", "runtime memory schema/status/evidence class is invalid");
  requireCondition(memory.tensorflow_source_commit === commit, "runtime memory TensorFlow commit is not pinned");
  requireCondition(/^[a-f0-9]{64}$/.test(String(memory.allocation_ledger_sha256 || "")), "runtime memory ledger digest is invalid");
  const snapshots = Array.isArray(memory.snapshots) ? memory.snapshots : [];
  requireCondition(snapshots.length >= 1 && snapshots.length <= 4096, "runtime memory snapshot inventory is empty or unbounded");
  requireCondition(integer(memory.snapshot_count, "snapshot_count", { positive: true }) === snapshots.length, "snapshot_count does not conserve snapshots");
  const artifactTensorCount = Array.isArray(staticAnalysis?.tensors) ? staticAnalysis.tensors.length : 0;
  const normalizedSnapshots = snapshots.map((snapshot, snapshotIndex) => {
    const snapshotId = integer(snapshot?.memory_snapshot_id, `snapshot ${snapshotIndex} id`);
    requireCondition(snapshotId === snapshotIndex, `snapshot ${snapshotIndex} ID is not contiguous`);
    const nonPersistent = integer(snapshot?.non_persistent_arena_bytes, `snapshot ${snapshotIndex} nonpersistent bytes`);
    const persistent = integer(snapshot?.persistent_arena_bytes, `snapshot ${snapshotIndex} persistent bytes`);
    const combined = integer(snapshot?.combined_arena_bytes, `snapshot ${snapshotIndex} combined bytes`);
    requireCondition(Number.isSafeInteger(nonPersistent + persistent) && combined === nonPersistent + persistent, `snapshot ${snapshotIndex} does not conserve arena bytes`);
    const tensorCount = integer(snapshot?.tensor_count, `snapshot ${snapshotIndex} tensor count`, { positive: true });
    const executionNodeCount = integer(snapshot?.execution_node_count, `snapshot ${snapshotIndex} execution-node count`, { positive: true });
    requireCondition(tensorCount >= artifactTensorCount, `snapshot ${snapshotIndex} omits artifact tensors`);
    const allocations = Array.isArray(snapshot?.allocations) ? snapshot.allocations : [];
    const aliases = Array.isArray(snapshot?.aliases) ? snapshot.aliases : [];
    requireCondition(allocations.length <= 100000 && aliases.length <= 100000, `snapshot ${snapshotIndex} row inventory is unbounded`);
    requireCondition(integer(snapshot?.allocation_count, `snapshot ${snapshotIndex} allocation count`) === allocations.length, `snapshot ${snapshotIndex} allocation count differs`);
    requireCondition(integer(snapshot?.alias_count, `snapshot ${snapshotIndex} alias count`) === aliases.length, `snapshot ${snapshotIndex} alias count differs`);
    const normalizedAllocations = [];
    let previousTensor = -1;
    const owning = new Set();
    let intervalBytes = 0;
    for (const [allocationIndex, allocation] of allocations.entries()) {
      const tensorIndex = integer(allocation?.tensor_index, `snapshot ${snapshotIndex} allocation ${allocationIndex} tensor`);
      requireCondition(tensorIndex > previousTensor && tensorIndex < tensorCount, `snapshot ${snapshotIndex} allocations are not canonical or in range`);
      previousTensor = tensorIndex;
      owning.add(tensorIndex);
      const arena = String(allocation?.arena || "");
      requireCondition(arena === "kTfLiteArenaRw" || arena === "kTfLiteArenaRwPersistent", `snapshot ${snapshotIndex} allocation ${allocationIndex} arena is invalid`);
      const offset = integer(allocation?.offset_bytes, `snapshot ${snapshotIndex} allocation ${allocationIndex} offset`);
      const size = integer(allocation?.size_bytes, `snapshot ${snapshotIndex} allocation ${allocationIndex} size`, { positive: true });
      const first = integer(allocation?.first_node, `snapshot ${snapshotIndex} allocation ${allocationIndex} first node`);
      const last = allocation?.last_node == null ? null : integer(allocation.last_node, `snapshot ${snapshotIndex} allocation ${allocationIndex} last node`);
      requireCondition(first < executionNodeCount && (last == null || (last >= first && last < executionNodeCount)), `snapshot ${snapshotIndex} allocation ${allocationIndex} lifetime is invalid`);
      const limit = arena === "kTfLiteArenaRw" ? nonPersistent : persistent;
      requireCondition(Number.isSafeInteger(offset + size) && offset + size <= limit, `snapshot ${snapshotIndex} allocation ${allocationIndex} exceeds its arena`);
      requireCondition(Number.isSafeInteger(intervalBytes + size), `snapshot ${snapshotIndex} interval byte sum overflows`);
      intervalBytes += size;
      normalizedAllocations.push({ tensor_index: tensorIndex, arena, offset_bytes: offset, size_bytes: size, first_node: first, last_node: last });
    }
    requireCondition(arenaAllocationsDoNotConflict(normalizedAllocations), `snapshot ${snapshotIndex} has overlapping live allocations`);
    requireCondition(integer(snapshot?.allocated_interval_bytes, `snapshot ${snapshotIndex} allocated interval bytes`) === intervalBytes, `snapshot ${snapshotIndex} interval bytes do not conserve rows`);
    let previousAlias = -1;
    const aliasIndices = new Set();
    const normalizedAliases = [];
    for (const [aliasIndex, alias] of aliases.entries()) {
      const tensorIndex = integer(alias?.tensor_index, `snapshot ${snapshotIndex} alias ${aliasIndex} tensor`);
      const root = integer(alias?.shared_with_tensor_index, `snapshot ${snapshotIndex} alias ${aliasIndex} root`);
      requireCondition(tensorIndex > previousAlias && tensorIndex < tensorCount, `snapshot ${snapshotIndex} aliases are not canonical or in range`);
      previousAlias = tensorIndex;
      requireCondition(tensorIndex !== root && root < tensorCount && owning.has(root), `snapshot ${snapshotIndex} alias ${aliasIndex} is not rooted in an owning allocation`);
      requireCondition(!owning.has(tensorIndex) && !aliasIndices.has(root), `snapshot ${snapshotIndex} alias ${aliasIndex} owns storage or forms an alias chain`);
      aliasIndices.add(tensorIndex);
      normalizedAliases.push({ tensor_index: tensorIndex, shared_with_tensor_index: root });
    }
    requireCondition(normalizedAliases.every((alias) => !aliasIndices.has(alias.shared_with_tensor_index)), `snapshot ${snapshotIndex} contains an alias chain`);
    return {
      memory_snapshot_id: snapshotId,
      non_persistent_arena_bytes: nonPersistent,
      persistent_arena_bytes: persistent,
      combined_arena_bytes: combined,
      tensor_count: tensorCount,
      execution_node_count: executionNodeCount,
      allocation_count: allocations.length,
      alias_count: aliases.length,
      allocated_interval_bytes: intervalBytes,
      allocations: normalizedAllocations,
      aliases: normalizedAliases,
    };
  });
  const finalSnapshot = normalizedSnapshots.at(-1) || { allocations: [], aliases: [], combined_arena_bytes: 0 };
  requireCondition(sha256TextHex(JSON.stringify(normalizedSnapshots)) === memory.allocation_ledger_sha256, "runtime memory ledger digest differs from canonical snapshots");
  const expectedPeakNonPersistent = Math.max(0, ...normalizedSnapshots.map((item) => item.non_persistent_arena_bytes));
  const expectedPeakPersistent = Math.max(0, ...normalizedSnapshots.map((item) => item.persistent_arena_bytes));
  const expectedPeakCombined = Math.max(0, ...normalizedSnapshots.map((item) => item.combined_arena_bytes));
  requireCondition(memory.peak_non_persistent_arena_bytes === expectedPeakNonPersistent, "peak nonpersistent bytes differ from snapshots");
  requireCondition(memory.peak_persistent_arena_bytes === expectedPeakPersistent, "peak persistent bytes differ from snapshots");
  requireCondition(memory.peak_combined_arena_bytes === expectedPeakCombined, "peak combined bytes differ from snapshots");
  requireCondition(memory.final_non_persistent_arena_bytes === finalSnapshot.non_persistent_arena_bytes, "final nonpersistent bytes differ from the final snapshot");
  requireCondition(memory.final_persistent_arena_bytes === finalSnapshot.persistent_arena_bytes, "final persistent bytes differ from the final snapshot");
  requireCondition(memory.final_combined_arena_bytes === finalSnapshot.combined_arena_bytes, "final combined bytes differ from the final snapshot");

  const plan = staticAnalysis?.tensor_arena_plan || null;
  const projectedAllocations = new Map((plan?.allocations || []).filter((item) => item?.allocation_status === "allocated").map((item) => [Number(item.tensor_index), item]));
  const observedAllocations = new Map(finalSnapshot.allocations.map((item) => [item.tensor_index, item]));
  const projectedAliases = new Map((plan?.aliases || []).map((item) => [Number(item.tensor_index), Number(item.shared_with_tensor_index)]));
  const observedAliases = new Map(finalSnapshot.aliases.map((item) => [item.tensor_index, item.shared_with_tensor_index]));
  const allocationRows = [...new Set([...projectedAllocations.keys(), ...observedAllocations.keys()])].sort((left, right) => left - right).map((tensorIndex) => {
    const projected = projectedAllocations.get(tensorIndex) || null;
    const observed = observedAllocations.get(tensorIndex) || null;
    const artifactTensor = tensorIndex < artifactTensorCount;
    return {
      tensor_index: tensorIndex,
      tensor_name: staticAnalysis?.tensors?.[tensorIndex]?.name || (artifactTensor ? `tensor_${tensorIndex}` : `runtime_temporary_${tensorIndex}`),
      artifact_tensor: artifactTensor,
      projected_present: projected != null,
      observed_present: observed != null,
      projected_arena: projected?.arena || null,
      observed_arena: observed?.arena || null,
      projected_size_bytes: projected?.size_bytes ?? null,
      observed_size_bytes: observed?.size_bytes ?? null,
      size_delta_bytes: projected == null || observed == null ? null : observed.size_bytes - Number(projected.size_bytes),
      projected_offset_bytes: projected?.offset_bytes ?? null,
      observed_offset_bytes: observed?.offset_bytes ?? null,
      offset_delta_bytes: projected == null || observed == null ? null : observed.offset_bytes - Number(projected.offset_bytes),
      size_match: projected == null || observed == null ? null : Number(projected.size_bytes) === observed.size_bytes,
      arena_match: projected == null || observed == null ? null : projected.arena === observed.arena,
      offset_match: projected == null || observed == null ? null : Number(projected.offset_bytes) === observed.offset_bytes,
    };
  });
  const aliasRows = [...new Set([...projectedAliases.keys(), ...observedAliases.keys()])].sort((left, right) => left - right).map((tensorIndex) => ({
    tensor_index: tensorIndex,
    projected_root_tensor_index: projectedAliases.get(tensorIndex) ?? null,
    observed_root_tensor_index: observedAliases.get(tensorIndex) ?? null,
    root_match: projectedAliases.has(tensorIndex) && observedAliases.has(tensorIndex) ? projectedAliases.get(tensorIndex) === observedAliases.get(tensorIndex) : null,
  }));
  const runtimeOnly = allocationRows.filter((item) => item.observed_present && !item.projected_present);
  const missingObserved = allocationRows.filter((item) => item.projected_present && !item.observed_present);
  const sizeMismatch = allocationRows.filter((item) => item.size_match === false);
  const offsetMismatch = allocationRows.filter((item) => item.offset_match === false);
  const aliasMismatch = aliasRows.filter((item) => item.root_match === false || item.projected_root_tensor_index == null || item.observed_root_tensor_index == null);
  const runtimeTemporaries = runtimeOnly.filter((item) => !item.artifact_tensor);
  const projectedCombined = plan?.combined_arena_bytes == null ? null : Number(plan.combined_arena_bytes);
  const expectedReconciliation = {
    schema: "deepbom.arena_runtime_reconciliation.v1",
    status: plan?.status === "assessed" ? "assessed" : "partial_static_projection_unavailable",
    evidence_class: "DERIVED_FROM_OBSERVED_RUNTIME",
    static_projection_schema: plan?.schema || null,
    runtime_memory_schema: memory.schema,
    tensorflow_source_commit: memory.tensorflow_source_commit,
    runtime_snapshot_id: finalSnapshot.memory_snapshot_id,
    projected_combined_arena_bytes: projectedCombined,
    observed_peak_combined_arena_bytes: expectedPeakCombined,
    observed_final_combined_arena_bytes: finalSnapshot.combined_arena_bytes,
    peak_delta_bytes: projectedCombined == null ? null : expectedPeakCombined - projectedCombined,
    peak_to_projection_ratio: projectedCombined > 0 ? expectedPeakCombined / projectedCombined : null,
    projected_root_allocation_count: projectedAllocations.size,
    observed_root_allocation_count: finalSnapshot.allocations.length,
    matched_allocation_count: allocationRows.filter((item) => item.projected_present && item.observed_present).length,
    runtime_only_allocation_count: runtimeOnly.length,
    missing_observed_allocation_count: missingObserved.length,
    size_mismatch_count: sizeMismatch.length,
    offset_mismatch_count: offsetMismatch.length,
    projected_alias_count: projectedAliases.size,
    observed_alias_count: observedAliases.size,
    alias_mismatch_count: aliasMismatch.length,
    runtime_temporary_allocation_count: runtimeTemporaries.length,
    runtime_temporary_interval_bytes: runtimeTemporaries.reduce((sum, item) => sum + Number(item.observed_size_bytes || 0), 0),
    allocation_rows: allocationRows,
    alias_rows: aliasRows,
  };
  const scalarKeys = Object.keys(expectedReconciliation).filter((key) => !["allocation_rows", "alias_rows"].includes(key));
  requireCondition(reconciliation != null, "runtime arena reconciliation is missing");
  for (const key of scalarKeys) {
    const close = typeof expectedReconciliation[key] === "number" && !Number.isInteger(expectedReconciliation[key]);
    requireCondition(close ? nullableClose(reconciliation?.[key], expectedReconciliation[key]) : reconciliation?.[key] === expectedReconciliation[key], `arena reconciliation ${key} differs`);
  }
  const allocationRowsMatch = Array.isArray(reconciliation?.allocation_rows) && reconciliation.allocation_rows.length === allocationRows.length
    && allocationRows.every((expected, index) => Object.entries(expected).every(([key, value]) => reconciliation.allocation_rows[index]?.[key] === value));
  const aliasRowsMatch = Array.isArray(reconciliation?.alias_rows) && reconciliation.alias_rows.length === aliasRows.length
    && aliasRows.every((expected, index) => Object.entries(expected).every(([key, value]) => reconciliation.alias_rows[index]?.[key] === value));
  requireCondition(allocationRowsMatch, "arena reconciliation allocation rows differ");
  requireCondition(aliasRowsMatch, "arena reconciliation alias rows differ");
  const hasDifference = expectedReconciliation.peak_delta_bytes !== 0
    || runtimeOnly.length > 0 || missingObserved.length > 0 || sizeMismatch.length > 0 || offsetMismatch.length > 0 || aliasMismatch.length > 0;
  return { present: true, valid: errors.length === 0, errors, reconciliation: expectedReconciliation, hasDifference };
}

export function sourceLedgerProblems(sources, expectedByRole, requiredPrefix = "") {
  const problems = [];
  const roles = sources.map((source) => source.role);
  if (sources.length !== expectedByRole.size) problems.push(`source count ${sources.length} does not equal expected ${expectedByRole.size}`);
  if (new Set(roles).size !== roles.length) problems.push("source roles are not unique");
  for (const role of expectedByRole.keys()) {
    if (!roles.includes(role)) problems.push(`missing source role ${role}`);
  }
  for (const source of sources) {
    if (!expectedByRole.has(source.role)) problems.push(`unexpected source role ${source.role || "<empty>"}`);
    else if (expectedByRole.get(source.role) !== source.sha256) problems.push(`source hash mismatch for ${source.role}`);
    if (requiredPrefix && !String(source.source_ref || "").startsWith(requiredPrefix)) problems.push(`source URL is outside the pinned commit for ${source.role || "<empty>"}`);
  }
  return problems;
}

export function parseExternalDataDecimal(value, absentValue) {
  if (value == null || value === "") return { valid: true, value: absentValue, status: "absent" };
  const text = String(value);
  if (!/^(0|[1-9][0-9]*)$/.test(text)) return { valid: false, value: null, status: "invalid" };
  const parsed = BigInt(text);
  if (parsed > BigInt(Number.MAX_SAFE_INTEGER)) return { valid: false, value: null, status: "invalid" };
  return { valid: true, value: Number(parsed), status: "declared" };
}

export function onnxTensorPayloadBytes(dtype, elements) {
  const bits = ({
    UINT2: 2, INT2: 2,
    UINT4: 4, INT4: 4, FLOAT4E2M1: 4,
    BOOL: 8, INT8: 8, UINT8: 8, FLOAT8E4M3FN: 8, FLOAT8E4M3FNUZ: 8,
    FLOAT8E5M2: 8, FLOAT8E5M2FNUZ: 8, FLOAT8E8M0: 8,
    FLOAT16: 16, BFLOAT16: 16, INT16: 16, UINT16: 16,
    FLOAT32: 32, INT32: 32, UINT32: 32,
    FLOAT64: 64, INT64: 64, UINT64: 64, COMPLEX64: 64, COMPLEX128: 128,
  })[String(dtype || "").toUpperCase()] || 0;
  const count = Number(elements);
  if (!(bits > 0) || !Number.isSafeInteger(count) || count < 0 || count > Math.floor(Number.MAX_SAFE_INTEGER / bits)) return null;
  const bytes = Math.ceil(count * bits / 8);
  return Number.isSafeInteger(bytes) ? bytes : null;
}

export function externalLocationStatus(location) {
  const value = String(location || "");
  if (!value) return "missing";
  if (value.includes("\0")) return "unsafe_nul";
  const normalized = normalizeExternalLocation(value);
  if (/^[a-z][a-z0-9+.-]*:/i.test(normalized)) return "unsafe_absolute_or_uri";
  if (normalized.startsWith("/")) return "unsafe_path_escape";
  if (normalized.split("/").some((part) => !part || part === "." || part === "..")) return "unsafe_noncanonical_segment";
  return "safe_relative_path";
}

export function normalizeExternalLocation(location) {
  return String(location || "").replace(/\\/g, "/").replace(/^(?:\.\/)+/, "");
}

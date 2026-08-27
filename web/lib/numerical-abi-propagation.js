import { sha256Hex } from "./hash.js";

export const NUMERICAL_ABI_PROPAGATION_SCHEMA = "deepbom.numerical_abi_propagation.v1.1";
const METHOD_VERSION = "2026-07-28.1";
const SOURCE_EVIDENCE_SCHEMA = "deepbom.rounding_equivalence.v1 + deepbom.accumulator_reachability.v1";
const GRAPH_LEDGER_PREFIX = new TextEncoder().encode("deepbom.numerical_abi_propagation.graph.v1\0");
const SOURCE_LEDGER_PREFIX = new TextEncoder().encode("deepbom.numerical_abi_propagation.source.v1.1\0");
const U128_MAX = (1n << 128n) - 1n;
const U64_MAX = (1n << 64n) - 1n;
const ROUTE_DEFINITION = "A graph route is a distinct producer-tensor-consumer edge sequence from one divergent source op output tensor to a declared model output tensor. Route multiplicity is counted exactly only for an acyclic operator graph.";
const RANKING_DEFINITION = "Lexicographic, not a composite score: output reachability, predicted-boundary edge count, exact output-route multiplicity, reconvergence count, reachable-op count, divergent interval-state ratio, then ascending op index.";
const METHOD = "Join every rounding-equivalence source op and accumulator-reachability partition to the artifact producer/tensor/consumer graph. Traverse all downstream nonconstant graph edges, retain every corridor edge index, classify branch reconvergence versus single-branch merge, count exact acyclic routes, and bind each source corridor to the graph, equivalence, and reachability ledgers. Logical payload uses the serialized concrete tensor shape; a dynamic shape signature does not suppress a payload when that concrete shape is fully bound.";
const INTERPRETATION_BOUNDARY = "Exact-local qualification proves at least one bounded-sum reachable kernel-local accumulator where the pinned rounding paths differ. Downstream corridors remain tensor-level structural potential: they do not prove a full-model input realizes the local assignment, that a declared output changes, or that cancellation, requantization, pooling, activation clamp, or task semantics preserve the difference. Predicted execution-domain crossings and logical payloads are static evidence, not observed copies, latency, runtime assignment, or executed microkernels.";

export function validateNumericalAbiPropagationShape(evidence) {
  assert(evidence?.schema === NUMERICAL_ABI_PROPAGATION_SCHEMA, "Numerical ABI propagation schema mismatch.");
  assert(evidence.method_version === METHOD_VERSION, "Numerical ABI propagation method mismatch.");
  assert(evidence.source_evidence_schema === SOURCE_EVIDENCE_SCHEMA, "Numerical ABI source evidence mismatch.");
  assert(Array.isArray(evidence.graph_edges) && evidence.graph_edges.length === Number(evidence.graph_edge_count), "Numerical ABI graph edge count mismatch.");
  assert(Array.isArray(evidence.sources) && evidence.sources.length === Number(evidence.candidate_source_op_count), "Numerical ABI source count mismatch.");
  assert(Array.isArray(evidence.propagation_ranking_op_indices), "Numerical ABI ranking is missing.");
  assert(/^[a-f0-9]{64}$/.test(evidence.graph_ledger_sha256 || ""), "Numerical ABI graph ledger digest is invalid.");
  evidence.graph_edges.forEach((edge, index) => {
    assert(edge.edge_index === index, `Numerical ABI graph edge order mismatch at ${index}.`);
    assert(Number.isInteger(edge.producer_op_index) && Number.isInteger(edge.consumer_op_index) && Number.isInteger(edge.tensor_index), `Numerical ABI edge identity is invalid at ${index}.`);
    assert(Array.isArray(edge.consumer_input_slots) && edge.consumer_input_slots.length > 0, `Numerical ABI edge input slots are missing at ${index}.`);
  });
  for (const source of evidence.sources) {
    for (const key of ["source_output_tensor_indices", "corridor_edge_indices", "reachable_op_indices", "reachable_tensor_indices", "merge_points", "model_output_paths"]) {
      assert(Array.isArray(source[key]), `Numerical ABI ${key} is missing at op #${source.op_index}.`);
    }
    assert(source.corridor_edge_count === source.corridor_edge_indices.length, `Numerical ABI corridor edge conservation failed at op #${source.op_index}.`);
    assert(source.reachable_op_count === source.reachable_op_indices.length, `Numerical ABI reachable-op conservation failed at op #${source.op_index}.`);
    assert(source.reachable_tensor_count === source.reachable_tensor_indices.length, `Numerical ABI reachable-tensor conservation failed at op #${source.op_index}.`);
    assert(source.reachable_model_output_tensor_count === source.model_output_paths.length, `Numerical ABI output-path conservation failed at op #${source.op_index}.`);
    if (source.assessment_status === "propagates_structurally") {
      assert(source.divergent_channel_count > 0 && /^[a-f0-9]{64}$/.test(source.propagation_ledger_sha256 || ""), `Numerical ABI propagating source is incomplete at op #${source.op_index}.`);
      const interval = BigInt(source.divergent_state_count_decimal || "0");
      const exact = BigInt(source.exact_reachable_divergent_state_count_decimal || "0");
      const excluded = BigInt(source.provably_unreachable_divergent_state_count_decimal || "0");
      const unresolved = BigInt(source.unresolved_divergent_state_count_decimal || "0");
      assert(exact + excluded + unresolved === interval, `Numerical ABI local reachability does not conserve divergent states at op #${source.op_index}.`);
      assert(["exact_local_counterexample", "unresolved_interval_divergence", "residue_excluded_interval_divergence", "not_assessed"].includes(source.local_reachability_status), `Numerical ABI local reachability status is invalid at op #${source.op_index}.`);
      if (source.local_reachability_status !== "not_assessed") assert(/^[a-f0-9]{64}$/.test(source.source_reachability_ledger_sha256 || ""), `Numerical ABI reachability ledger is invalid at op #${source.op_index}.`);
    }
  }
  const interval = BigInt(evidence.interval_divergent_state_count_decimal || "0");
  const exact = BigInt(evidence.exact_local_divergent_state_count_decimal || "0");
  const excluded = BigInt(evidence.residue_excluded_divergent_state_count_decimal || "0");
  const unresolved = BigInt(evidence.unresolved_divergent_state_count_decimal || "0");
  assert(exact + excluded + unresolved === interval, "Numerical ABI portfolio local reachability does not conserve divergent states.");
  for (const [countKey, indicesKey] of [["exact_unique_reachable_op_count", "exact_unique_reachable_op_indices"], ["exact_unique_reachable_tensor_count", "exact_unique_reachable_tensor_indices"], ["exact_unique_predicted_boundary_edge_count", "exact_unique_predicted_boundary_edge_indices"]]) {
    assert(Array.isArray(evidence[indicesKey]) && evidence[indicesKey].length === Number(evidence[countKey]), `Numerical ABI ${indicesKey} conservation failed.`);
  }
  return evidence;
}

export function reconstructNumericalAbiPropagation(analysis) {
  const ops = Array.isArray(analysis?.ops) ? analysis.ops : [];
  const tensors = Array.isArray(analysis?.tensors) ? analysis.tensors : [];
  const equivalenceRows = Array.isArray(analysis?.rounding_equivalence?.ops) ? analysis.rounding_equivalence.ops : [];
  const reachabilityByOp = new Map((analysis?.accumulator_reachability?.ops || []).map((row) => [row.op_index, row]));
  const outputSet = new Set((analysis?.output_tensor_indices || []).filter((index) => Number.isInteger(index) && index >= 0));
  const graphEdges = buildGraphEdges(ops, tensors);
  const topologicalOrder = topologicalOpOrder(ops, graphEdges);
  const graphCycle = topologicalOrder == null && ops.length > 0;
  const byTensor = edgesByTensor(graphEdges);
  const producers = producerByTensor(ops);
  const sources = equivalenceRows.map((row) => buildSource({ row, reachabilityRow: reachabilityByOp.get(row.op_index), ops, tensors, outputSet, graphEdges, byTensor, producers, topologicalOrder }));
  const divergent = sources.filter((source) => source.assessment_status === "propagates_structurally");
  const exactSources = divergent.filter((source) => source.local_reachability_status === "exact_local_counterexample");
  const uniqueReachableOps = sortedUnique(divergent.flatMap((source) => source.reachable_op_indices));
  const uniqueReachableTensors = sortedUnique(divergent.flatMap((source) => source.reachable_tensor_indices));
  const uniqueOutputs = sortedUnique(divergent.flatMap((source) => source.model_output_paths.map((path) => path.output_tensor_index)));
  const uniqueBoundaryEdges = sortedUnique(divergent.flatMap((source) => source.corridor_edge_indices.filter((index) => graphEdges[index]?.predicted_boundary)));
  const uniqueBoundaryPayload = sumOptional(uniqueBoundaryEdges.map((index) => graphEdges[index]?.logical_payload_bytes));
  const exactUniqueReachableOps = sortedUnique(exactSources.flatMap((source) => source.reachable_op_indices));
  const exactUniqueReachableTensors = sortedUnique(exactSources.flatMap((source) => source.reachable_tensor_indices));
  const exactUniqueBoundaryEdges = sortedUnique(exactSources.flatMap((source) => source.corridor_edge_indices.filter((index) => graphEdges[index]?.predicted_boundary)));
  const exactUniqueBoundaryPayload = sumOptional(exactUniqueBoundaryEdges.map((index) => graphEdges[index]?.logical_payload_bytes));
  const ranking = divergent.map((source) => source.op_index).sort((left, right) => compareSources(sources.find((source) => source.op_index === left), sources.find((source) => source.op_index === right)));
  const routeCounts = divergent.map((source) => optionalBigInt(source.exact_model_output_graph_route_count_decimal)).filter((value) => value != null);
  const unassessed = sources.filter((source) => source.assessment_status === "not_assessed").length;
  const localUnassessed = divergent.filter((source) => source.local_reachability_status === "not_assessed").length;
  return {
    schema: NUMERICAL_ABI_PROPAGATION_SCHEMA,
    method_version: METHOD_VERSION,
    evidence_class: "DERIVED",
    status: unassessed > 0 || localUnassessed > 0 || graphCycle ? "partial" : "assessed",
    source_evidence_schema: SOURCE_EVIDENCE_SCHEMA,
    candidate_source_op_count: sources.length,
    divergent_source_op_count: divergent.length,
    equivalent_source_op_count: sources.filter((source) => source.assessment_status === "complete_interval_equivalent").length,
    unassessed_source_op_count: unassessed,
    local_reachability_unassessed_source_op_count: localUnassessed,
    output_reachable_source_op_count: divergent.filter((source) => source.reachable_model_output_tensor_count > 0).length,
    output_isolated_source_op_count: divergent.filter((source) => source.reachable_model_output_tensor_count === 0).length,
    exact_local_counterexample_source_op_count: exactSources.length,
    residue_excluded_divergence_source_op_count: divergent.filter((source) => BigInt(source.provably_unreachable_divergent_state_count_decimal || "0") > 0n).length,
    unresolved_divergence_source_op_count: divergent.filter((source) => BigInt(source.unresolved_divergent_state_count_decimal || "0") > 0n).length,
    exact_output_reachable_source_op_count: exactSources.filter((source) => source.reachable_model_output_tensor_count > 0).length,
    interval_divergent_state_count_decimal: sumDecimalFields(divergent, "divergent_state_count_decimal"),
    exact_local_divergent_state_count_decimal: sumDecimalFields(divergent, "exact_reachable_divergent_state_count_decimal"),
    residue_excluded_divergent_state_count_decimal: sumDecimalFields(divergent, "provably_unreachable_divergent_state_count_decimal"),
    unresolved_divergent_state_count_decimal: sumDecimalFields(divergent, "unresolved_divergent_state_count_decimal"),
    graph_edge_count: graphEdges.length,
    graph_ledger_sha256: "",
    source_corridor_edge_instance_count: sumNumbers(divergent.map((source) => source.corridor_edge_count)),
    source_boundary_edge_instance_count: sumNumbers(divergent.map((source) => source.predicted_boundary_edge_count)),
    assessed_source_boundary_edge_instance_payload_bytes_decimal: divergent.reduce((total, source) => total + BigInt(source.assessed_boundary_logical_payload_bytes), 0n).toString(),
    unassessed_source_boundary_edge_instance_payload_count: sumNumbers(divergent.map((source) => source.unassessed_boundary_payload_edge_count)),
    unique_reachable_op_count: uniqueReachableOps.length,
    unique_reachable_op_indices: uniqueReachableOps,
    unique_reachable_tensor_count: uniqueReachableTensors.length,
    unique_reachable_tensor_indices: uniqueReachableTensors,
    unique_model_output_tensor_count: uniqueOutputs.length,
    unique_model_output_tensor_indices: uniqueOutputs,
    unique_predicted_boundary_edge_count: uniqueBoundaryEdges.length,
    unique_predicted_boundary_edge_indices: uniqueBoundaryEdges,
    unique_predicted_boundary_logical_payload_bytes: uniqueBoundaryPayload,
    exact_source_corridor_edge_instance_count: sumNumbers(exactSources.map((source) => source.corridor_edge_count)),
    exact_source_boundary_edge_instance_count: sumNumbers(exactSources.map((source) => source.predicted_boundary_edge_count)),
    exact_unique_reachable_op_count: exactUniqueReachableOps.length,
    exact_unique_reachable_op_indices: exactUniqueReachableOps,
    exact_unique_reachable_tensor_count: exactUniqueReachableTensors.length,
    exact_unique_reachable_tensor_indices: exactUniqueReachableTensors,
    exact_unique_predicted_boundary_edge_count: exactUniqueBoundaryEdges.length,
    exact_unique_predicted_boundary_edge_indices: exactUniqueBoundaryEdges,
    exact_unique_predicted_boundary_logical_payload_bytes: exactUniqueBoundaryPayload,
    reconvergence_source_op_instance_count: sumNumbers(divergent.map((source) => source.reconvergence_op_count)),
    single_branch_merge_source_op_instance_count: sumNumbers(divergent.map((source) => source.single_branch_merge_op_count)),
    maximum_model_output_op_hops: maximumOptional(divergent.map((source) => source.minimum_model_output_op_hops)),
    maximum_model_output_graph_route_count_decimal: routeCounts.length ? routeCounts.reduce((maximum, value) => value > maximum ? value : maximum, 0n).toString() : null,
    graph_cycle_status: graphCycle ? "cycle_detected_route_count_not_assessed" : "acyclic",
    propagation_ranking_op_indices: ranking,
    graph_edges: graphEdges,
    sources,
    route_definition: ROUTE_DEFINITION,
    ranking_definition: RANKING_DEFINITION,
    method: METHOD,
    interpretation_boundary: INTERPRETATION_BOUNDARY,
  };
}

export function validateNumericalAbiPropagationAgainstReconstruction(evidence, reconstructed) {
  validateNumericalAbiPropagationShape(evidence);
  const actual = normalizeOptionalFields(withoutDigests(evidence));
  const expected = normalizeOptionalFields(withoutDigests(reconstructed));
  assert(canonicalJson(actual) === canonicalJson(expected), "Numerical ABI propagation differs from independent graph reconstruction.");
  return reconstructed;
}

export async function validateNumericalAbiPropagationAnalysis(analysis) {
  const reconstructed = reconstructNumericalAbiPropagation(analysis);
  validateNumericalAbiPropagationAgainstReconstruction(analysis?.numerical_abi_propagation, reconstructed);
  await attachNumericalAbiPropagationDigests(reconstructed);
  validateNumericalAbiPropagationDigestsAgainstReconstruction(analysis.numerical_abi_propagation, reconstructed);
  return reconstructed;
}

export async function attachNumericalAbiPropagationDigests(reconstructed) {
  reconstructed.graph_ledger_sha256 = await graphLedgerSha256(reconstructed.graph_edges);
  for (const source of reconstructed.sources) {
    source.propagation_ledger_sha256 = source.assessment_status === "propagates_structurally"
      ? await sourceLedgerSha256(source, reconstructed.graph_ledger_sha256)
      : "";
  }
  return reconstructed;
}

export function validateNumericalAbiPropagationDigestsAgainstReconstruction(evidence, reconstructed) {
  assert(evidence.graph_ledger_sha256 === reconstructed.graph_ledger_sha256, "Numerical ABI graph ledger SHA-256 mismatch.");
  assert(evidence.sources.length === reconstructed.sources.length, "Numerical ABI source digest cardinality mismatch.");
  evidence.sources.forEach((source, index) => {
    assert(source.propagation_ledger_sha256 === reconstructed.sources[index].propagation_ledger_sha256, `Numerical ABI source ledger SHA-256 mismatch at op #${source.op_index}.`);
  });
  return reconstructed;
}

function buildGraphEdges(ops, tensors) {
  const producer = new Map();
  for (const op of ops) for (const output of op.outputs || []) if (output >= 0) producer.set(output, op);
  const pending = [];
  for (const consumer of ops) {
    const slotsByTensor = new Map();
    (consumer.inputs || []).forEach((tensorIndex, slot) => {
      if (tensorIndex < 0) return;
      const slots = slotsByTensor.get(tensorIndex) || [];
      slots.push(slot);
      slotsByTensor.set(tensorIndex, slots);
    });
    for (const [tensorIndex, consumerInputSlots] of [...slotsByTensor].sort((left, right) => left[0] - right[0])) {
      const parent = producer.get(tensorIndex);
      const tensor = tensors[tensorIndex];
      if (!parent || !tensor || tensor.constant_buffer) continue;
      const producerDomain = predictedDomain(parent);
      const consumerDomain = predictedDomain(consumer);
      const payload = deterministicPayload(tensor);
      pending.push({
        edge_index: 0,
        producer_op_index: parent.index,
        producer_op_name: parent.name,
        tensor_index: tensor.index,
        tensor_name: tensor.name,
        tensor_shape: [...(tensor.shape || [])],
        tensor_dtype: tensor.dtype,
        logical_payload_bytes: payload,
        payload_status: payload == null ? "not_assessed" : "assessed",
        consumer_op_index: consumer.index,
        consumer_op_name: consumer.name,
        consumer_input_slots: consumerInputSlots,
        producer_domain: producerDomain,
        consumer_domain: consumerDomain,
        predicted_boundary: producerDomain !== consumerDomain,
        predicted_boundary_direction: producerDomain === consumerDomain ? "same_predicted_domain" : boundaryDirection(parent, consumer),
      });
    }
  }
  pending.sort((left, right) => left.producer_op_index - right.producer_op_index || left.consumer_op_index - right.consumer_op_index || left.tensor_index - right.tensor_index);
  pending.forEach((edge, index) => { edge.edge_index = index; });
  return pending;
}

function qualifyLocalReachability(row, reachabilityRow) {
  const empty = () => ({
    local_reachability_status: "not_assessed",
    exact_reachable_divergent_channel_count: 0,
    unresolved_divergent_channel_count: 0,
    interval_only_divergent_channel_count: 0,
    exact_reachable_divergent_state_count_decimal: "0",
    provably_unreachable_divergent_state_count_decimal: "0",
    unresolved_divergent_state_count_decimal: "0",
    source_reachability_ledger_sha256: "",
  });
  if (row?.assessment_status !== "assessed" || reachabilityRow?.assessment_status !== "assessed") return empty();
  let interval;
  let exact;
  let excluded;
  let unresolved;
  try {
    interval = BigInt(reachabilityRow.interval_divergent_state_count_decimal);
    exact = BigInt(reachabilityRow.exact_reachable_divergent_state_count_decimal);
    excluded = BigInt(reachabilityRow.provably_unreachable_divergent_state_count_decimal);
    unresolved = BigInt(reachabilityRow.unresolved_divergent_state_count_decimal);
  } catch {
    return empty();
  }
  if (reachabilityRow.interval_divergent_state_count_decimal !== row.divergent_state_count_decimal
    || exact + excluded + unresolved !== interval
    || !/^[a-f0-9]{64}$/.test(reachabilityRow.reachability_ledger_sha256 || "")) return empty();
  const status = interval === 0n
    ? "complete_interval_equivalent"
    : exact > 0n
      ? "exact_local_counterexample"
      : unresolved > 0n
        ? "unresolved_interval_divergence"
        : "residue_excluded_interval_divergence";
  return {
    local_reachability_status: status,
    exact_reachable_divergent_channel_count: reachabilityRow.exact_reachable_divergent_channel_count,
    unresolved_divergent_channel_count: reachabilityRow.unresolved_divergent_channel_count,
    interval_only_divergent_channel_count: reachabilityRow.interval_only_divergent_channel_count,
    exact_reachable_divergent_state_count_decimal: reachabilityRow.exact_reachable_divergent_state_count_decimal,
    provably_unreachable_divergent_state_count_decimal: reachabilityRow.provably_unreachable_divergent_state_count_decimal,
    unresolved_divergent_state_count_decimal: reachabilityRow.unresolved_divergent_state_count_decimal,
    source_reachability_ledger_sha256: reachabilityRow.reachability_ledger_sha256,
  };
}

function buildSource({ row, reachabilityRow, ops, tensors, outputSet, graphEdges, byTensor, producers, topologicalOrder }) {
  const local = qualifyLocalReachability(row, reachabilityRow);
  const sourceOutputs = (ops[row.op_index]?.outputs || []).filter((index) => index >= 0 && tensors[index] && !tensors[index].constant_buffer);
  const empty = (status, reason) => ({
    op_index: row.op_index,
    op_name: row.op_name,
    assessment_status: status,
    not_assessed_reason: reason,
    source_output_tensor_indices: sourceOutputs,
    assessed_channel_count: row.assessed_channel_count,
    divergent_channel_count: row.divergent_channel_count,
    interval_state_count_decimal: row.interval_state_count_decimal,
    divergent_state_count_decimal: row.divergent_state_count_decimal,
    divergent_state_ratio: row.divergent_state_ratio,
    maximum_absolute_output_delta: row.maximum_absolute_output_delta,
    source_equivalence_ledger_sha256: row.equivalence_ledger_sha256,
    ...local,
    direct_consumer_edge_count: 0,
    corridor_edge_count: 0,
    corridor_edge_indices: [],
    reachable_op_count: 0,
    reachable_op_indices: [],
    reachable_tensor_count: sourceOutputs.length,
    reachable_tensor_indices: [...sourceOutputs].sort((a, b) => a - b),
    reachable_model_output_tensor_count: sourceOutputs.filter((index) => outputSet.has(index)).length,
    minimum_model_output_op_hops: sourceOutputs.some((index) => outputSet.has(index)) ? 0 : null,
    maximum_reachable_op_hops: null,
    exact_model_output_graph_route_count_decimal: null,
    route_count_status: "not_applicable",
    predicted_boundary_edge_count: 0,
    assessed_boundary_logical_payload_bytes: 0,
    unassessed_boundary_payload_edge_count: 0,
    reconvergence_op_count: 0,
    single_branch_merge_op_count: 0,
    merge_points: [],
    model_output_paths: [],
    propagation_ledger_sha256: "",
  });
  if (row.assessment_status !== "assessed") return empty("not_assessed", "Source rounding-equivalence row was not assessed.");
  if (row.divergent_channel_count === 0) return empty("complete_interval_equivalent", "Both pinned rounding paths are equivalent over every assessed channel interval hull.");
  if (!sourceOutputs.length) return empty("not_assessed", "Divergent source op has no nonconstant output tensor.");

  const tensorDepths = new Map(sourceOutputs.map((index) => [index, 0]));
  const opDepths = new Map();
  const queue = [...sourceOutputs];
  for (let cursor = 0; cursor < queue.length; cursor += 1) {
    const tensorIndex = queue[cursor];
    const tensorDepth = tensorDepths.get(tensorIndex);
    for (const edgeIndex of byTensor.get(tensorIndex) || []) {
      const edge = graphEdges[edgeIndex];
      const opDepth = tensorDepth + 1;
      if (!opDepths.has(edge.consumer_op_index) || opDepth < opDepths.get(edge.consumer_op_index)) opDepths.set(edge.consumer_op_index, opDepth);
      for (const output of (ops[edge.consumer_op_index]?.outputs || []).filter((index) => index >= 0)) {
        if (!tensorDepths.has(output) || opDepth < tensorDepths.get(output)) {
          tensorDepths.set(output, opDepth);
          queue.push(output);
        }
      }
    }
  }
  const reachableOps = [...opDepths.keys()].sort((a, b) => a - b);
  const reachableTensors = [...tensorDepths.keys()].sort((a, b) => a - b);
  const corridorEdges = graphEdges.filter((edge) => tensorDepths.has(edge.tensor_index) && opDepths.has(edge.consumer_op_index)).map((edge) => edge.edge_index);
  const mergePoints = buildMergePoints(ops, graphEdges, corridorEdges, tensorDepths, opDepths);
  const routeCounts = topologicalOrder ? exactRouteCounts(row.op_index, topologicalOrder, graphEdges, corridorEdges) : null;
  const outputPaths = buildOutputPaths(row.op_index, tensors, outputSet, graphEdges, corridorEdges, tensorDepths, producers, routeCounts);
  const boundaryEdges = corridorEdges.map((index) => graphEdges[index]).filter((edge) => edge.predicted_boundary);
  const outputRouteCount = outputPaths.every((path) => path.exact_graph_route_count_decimal != null)
    ? outputPaths.reduce((total, path) => checkedU128Add(total, BigInt(path.exact_graph_route_count_decimal)), 0n)
    : null;
  return {
    op_index: row.op_index,
    op_name: row.op_name,
    assessment_status: "propagates_structurally",
    not_assessed_reason: "",
    source_output_tensor_indices: sourceOutputs,
    assessed_channel_count: row.assessed_channel_count,
    divergent_channel_count: row.divergent_channel_count,
    interval_state_count_decimal: row.interval_state_count_decimal,
    divergent_state_count_decimal: row.divergent_state_count_decimal,
    divergent_state_ratio: row.divergent_state_ratio,
    maximum_absolute_output_delta: row.maximum_absolute_output_delta,
    source_equivalence_ledger_sha256: row.equivalence_ledger_sha256,
    ...local,
    direct_consumer_edge_count: corridorEdges.filter((index) => graphEdges[index].producer_op_index === row.op_index).length,
    corridor_edge_count: corridorEdges.length,
    corridor_edge_indices: corridorEdges,
    reachable_op_count: reachableOps.length,
    reachable_op_indices: reachableOps,
    reachable_tensor_count: reachableTensors.length,
    reachable_tensor_indices: reachableTensors,
    reachable_model_output_tensor_count: outputPaths.length,
    minimum_model_output_op_hops: minimumOptional(outputPaths.map((path) => path.shortest_op_hops)),
    maximum_reachable_op_hops: maximumOptional([...opDepths.values()]),
    exact_model_output_graph_route_count_decimal: outputRouteCount?.toString() ?? null,
    route_count_status: routeCounts && outputPaths.every((path) => path.exact_graph_route_count_decimal != null) ? "assessed_acyclic" : "not_assessed_cycle_or_overflow",
    predicted_boundary_edge_count: boundaryEdges.length,
    assessed_boundary_logical_payload_bytes: boundaryEdges.reduce((total, edge) => total + (edge.logical_payload_bytes ?? 0), 0),
    unassessed_boundary_payload_edge_count: boundaryEdges.filter((edge) => edge.logical_payload_bytes == null).length,
    reconvergence_op_count: mergePoints.filter((point) => point.merge_class === "reconvergence").length,
    single_branch_merge_op_count: mergePoints.filter((point) => point.merge_class === "single_branch_merge").length,
    merge_points: mergePoints,
    model_output_paths: outputPaths,
    propagation_ledger_sha256: "",
  };
}

function buildMergePoints(ops, graphEdges, corridorEdgeIndices, tensorDepths, opDepths) {
  const incoming = new Map();
  for (const edge of graphEdges) {
    const edges = incoming.get(edge.consumer_op_index) || [];
    edges.push(edge);
    incoming.set(edge.consumer_op_index, edges);
  }
  const corridor = new Set(corridorEdgeIndices);
  return [...opDepths.keys()].flatMap((opIndex) => {
    const graphInputs = incoming.get(opIndex) || [];
    if (graphInputs.length < 2) return [];
    const influenced = sortedUnique(graphInputs.filter((edge) => corridor.has(edge.edge_index)).map((edge) => edge.tensor_index));
    if (!influenced.length) return [];
    const uninfluenced = sortedUnique(graphInputs.filter((edge) => !tensorDepths.has(edge.tensor_index)).map((edge) => edge.tensor_index));
    return [{
      op_index: opIndex,
      op_name: ops[opIndex].name,
      minimum_op_hops: opDepths.get(opIndex),
      merge_class: influenced.length >= 2 ? "reconvergence" : "single_branch_merge",
      graph_input_tensor_count: graphInputs.length,
      influenced_input_tensor_indices: influenced,
      uninfluenced_input_tensor_indices: uninfluenced,
      predicted_execution_domain: predictedDomain(ops[opIndex]),
    }];
  }).sort((left, right) => left.minimum_op_hops - right.minimum_op_hops || left.op_index - right.op_index);
}

function buildOutputPaths(sourceOpIndex, tensors, outputSet, graphEdges, corridorEdgeIndices, tensorDepths, producers, routeCounts) {
  const corridor = new Set(corridorEdgeIndices);
  const adjacency = new Map();
  for (const edge of graphEdges.filter((edge) => corridor.has(edge.edge_index))) {
    const edges = adjacency.get(edge.producer_op_index) || [];
    edges.push(edge);
    adjacency.set(edge.producer_op_index, edges);
  }
  for (const edges of adjacency.values()) edges.sort((left, right) => left.edge_index - right.edge_index);
  const distances = new Map([[sourceOpIndex, 0]]);
  const predecessor = new Map();
  const queue = [sourceOpIndex];
  for (let cursor = 0; cursor < queue.length; cursor += 1) {
    const opIndex = queue[cursor];
    for (const edge of adjacency.get(opIndex) || []) {
      if (!distances.has(edge.consumer_op_index)) {
        distances.set(edge.consumer_op_index, distances.get(opIndex) + 1);
        predecessor.set(edge.consumer_op_index, edge.edge_index);
        queue.push(edge.consumer_op_index);
      }
    }
  }
  return [...outputSet].sort((a, b) => a - b).filter((index) => tensorDepths.has(index)).map((outputTensorIndex) => {
    const producer = producers.get(outputTensorIndex);
    const pathEdges = [];
    let current = producer ?? sourceOpIndex;
    while (current !== sourceOpIndex) {
      const edgeIndex = predecessor.get(current);
      assert(edgeIndex != null, `Numerical ABI shortest path is incomplete for output tensor ${outputTensorIndex}.`);
      pathEdges.push(edgeIndex);
      current = graphEdges[edgeIndex].producer_op_index;
    }
    pathEdges.reverse();
    const pathOps = [sourceOpIndex, ...pathEdges.map((index) => graphEdges[index].consumer_op_index)];
    const boundaryEdges = pathEdges.map((index) => graphEdges[index]).filter((edge) => edge.predicted_boundary);
    const routeCount = producer == null || producer === sourceOpIndex ? 1n : routeCounts?.get(producer) ?? null;
    return {
      output_tensor_index: outputTensorIndex,
      output_tensor_name: tensors[outputTensorIndex]?.name || "",
      shortest_op_hops: producer == null ? 0 : distances.get(producer) ?? 0,
      shortest_path_op_indices: pathOps,
      shortest_path_edge_indices: pathEdges,
      shortest_path_predicted_boundary_count: boundaryEdges.length,
      shortest_path_boundary_logical_payload_bytes: sumOptional(boundaryEdges.map((edge) => edge.logical_payload_bytes)),
      exact_graph_route_count_decimal: routeCount?.toString() ?? null,
      route_count_status: routeCount == null ? "not_assessed_cycle_or_overflow" : "assessed_acyclic",
    };
  });
}

function exactRouteCounts(sourceOpIndex, topologicalOrder, graphEdges, corridorEdgeIndices) {
  const corridor = new Set(corridorEdgeIndices);
  const incoming = new Map();
  for (const edge of graphEdges.filter((edge) => corridor.has(edge.edge_index))) {
    const edges = incoming.get(edge.consumer_op_index) || [];
    edges.push(edge);
    incoming.set(edge.consumer_op_index, edges);
  }
  const routes = new Map([[sourceOpIndex, 1n]]);
  try {
    for (const opIndex of topologicalOrder) {
      if (opIndex === sourceOpIndex || !incoming.has(opIndex)) continue;
      const count = incoming.get(opIndex).reduce((total, edge) => checkedU128Add(total, routes.get(edge.producer_op_index) || 0n), 0n);
      if (count > 0n) routes.set(opIndex, count);
    }
    return routes;
  } catch {
    return null;
  }
}

function topologicalOpOrder(ops, graphEdges) {
  const indegrees = Array(ops.length).fill(0);
  const adjacency = Array.from({ length: ops.length }, () => new Set());
  for (const edge of graphEdges) if (!adjacency[edge.producer_op_index].has(edge.consumer_op_index)) {
    adjacency[edge.producer_op_index].add(edge.consumer_op_index);
    indegrees[edge.consumer_op_index] += 1;
  }
  const ready = indegrees.map((degree, index) => degree === 0 ? index : null).filter((index) => index != null).sort((a, b) => a - b);
  const order = [];
  while (ready.length) {
    const index = ready.shift();
    order.push(index);
    for (const consumer of [...adjacency[index]].sort((a, b) => a - b)) {
      indegrees[consumer] -= 1;
      if (indegrees[consumer] === 0) insertSorted(ready, consumer);
    }
  }
  return order.length === ops.length ? order : null;
}

async function graphLedgerSha256(edges) {
  const writer = new BinaryWriter(GRAPH_LEDGER_PREFIX);
  for (const edge of edges) {
    writer.u64(edge.edge_index).u64(edge.producer_op_index).u64(edge.tensor_index).u64(edge.consumer_op_index)
      .optionalU64(edge.logical_payload_bytes).bool(edge.predicted_boundary).i32Slice(edge.tensor_shape)
      .u64Slice(edge.consumer_input_slots).string(edge.tensor_dtype).string(edge.producer_domain)
      .string(edge.consumer_domain).string(edge.predicted_boundary_direction);
  }
  return sha256Hex(writer.bytes());
}

async function sourceLedgerSha256(source, graphLedger) {
  const writer = new BinaryWriter(SOURCE_LEDGER_PREFIX);
  writer.string(graphLedger).string(source.source_equivalence_ledger_sha256)
    .string(source.source_reachability_ledger_sha256).string(source.local_reachability_status).u64(source.op_index)
    .u64(source.assessed_channel_count).u64(source.divergent_channel_count)
    .string(source.interval_state_count_decimal).string(source.divergent_state_count_decimal)
    .u64(source.exact_reachable_divergent_channel_count).u64(source.unresolved_divergent_channel_count)
    .u64(source.interval_only_divergent_channel_count)
    .string(source.exact_reachable_divergent_state_count_decimal)
    .string(source.provably_unreachable_divergent_state_count_decimal)
    .string(source.unresolved_divergent_state_count_decimal)
    .u64Slice(source.source_output_tensor_indices).u64Slice(source.corridor_edge_indices)
    .u64Slice(source.reachable_op_indices).u64Slice(source.reachable_tensor_indices);
  for (const path of source.model_output_paths) {
    writer.u64(path.output_tensor_index).u64(path.shortest_op_hops).u64Slice(path.shortest_path_op_indices)
      .u64Slice(path.shortest_path_edge_indices).string(path.exact_graph_route_count_decimal || "");
  }
  for (const point of source.merge_points) {
    writer.u64(point.op_index).u64(point.minimum_op_hops).string(point.merge_class)
      .u64Slice(point.influenced_input_tensor_indices).u64Slice(point.uninfluenced_input_tensor_indices);
  }
  return sha256Hex(writer.bytes());
}

class BinaryWriter {
  constructor(prefix) { this.parts = [prefix]; this.length = prefix.byteLength; }
  append(bytes) { this.parts.push(bytes); this.length += bytes.byteLength; return this; }
  u64(value) { const bytes = new Uint8Array(8); new DataView(bytes.buffer).setBigUint64(0, BigInt(value), true); return this.append(bytes); }
  optionalU64(value) { return this.u64(value == null ? U64_MAX : BigInt(value)); }
  bool(value) { return this.append(Uint8Array.of(value ? 1 : 0)); }
  i32Slice(values) { this.u64(values.length); for (const value of values) { const bytes = new Uint8Array(4); new DataView(bytes.buffer).setInt32(0, value, true); this.append(bytes); } return this; }
  u64Slice(values) { this.u64(values.length); for (const value of values) this.u64(value); return this; }
  string(value) { const bytes = new TextEncoder().encode(String(value)); return this.u64(bytes.byteLength).append(bytes); }
  bytes() { const packed = new Uint8Array(this.length); let offset = 0; for (const part of this.parts) { packed.set(part, offset); offset += part.byteLength; } return packed; }
}

function deterministicPayload(tensor) {
  if (!Array.isArray(tensor.shape) || tensor.shape.some((dim) => !Number.isInteger(dim) || dim < 0)) return null;
  let elements = 1;
  for (const dim of tensor.shape || []) {
    elements *= dim;
    if (!Number.isSafeInteger(elements)) return null;
  }
  if (tensor.dtype === "INT4") return Math.floor((elements + 1) / 2);
  const width = ({ COMPLEX128: 16, FLOAT64: 8, INT64: 8, UINT64: 8, COMPLEX64: 8, FLOAT32: 4, INT32: 4, UINT32: 4, FLOAT16: 2, BFLOAT16: 2, INT16: 2, UINT16: 2, INT8: 1, UINT8: 1, BOOL: 1 })[tensor.dtype];
  if (!width || !Number.isSafeInteger(elements * width)) return null;
  return elements * width;
}

function predictedDomain(op) { return Number(op?.xnnpack_chain_id) >= 0 ? `XNNPACK:C${op.xnnpack_chain_id}` : "TFLITE_CPU"; }
function boundaryDirection(parent, consumer) {
  const left = Number(parent.xnnpack_chain_id) >= 0;
  const right = Number(consumer.xnnpack_chain_id) >= 0;
  if (left && !right) return "delegate_to_cpu";
  if (!left && right) return "cpu_to_delegate";
  if (left && right) return "delegate_partition_to_delegate_partition";
  return "cpu_to_cpu";
}
function edgesByTensor(edges) { const map = new Map(); for (const edge of edges) { const list = map.get(edge.tensor_index) || []; list.push(edge.edge_index); map.set(edge.tensor_index, list); } return map; }
function producerByTensor(ops) { const map = new Map(); for (const op of ops) for (const output of op.outputs || []) if (output >= 0) map.set(output, op.index); return map; }
function sortedUnique(values) { return [...new Set(values)].sort((a, b) => a - b); }
function sumNumbers(values) { return values.reduce((total, value) => total + Number(value || 0), 0); }
function sumDecimalFields(rows, key) { return rows.reduce((total, row) => total + BigInt(row[key] || "0"), 0n).toString(); }
function sumOptional(values) { return values.some((value) => value == null) ? null : values.reduce((total, value) => total + value, 0); }
function minimumOptional(values) { const present = values.filter((value) => value != null); return present.length ? Math.min(...present) : null; }
function maximumOptional(values) { const present = values.filter((value) => value != null); return present.length ? Math.max(...present) : null; }
function optionalBigInt(value) { try { return value == null ? null : BigInt(value); } catch { return null; } }
function checkedU128Add(left, right) { const value = left + right; if (value > U128_MAX) throw new Error("u128 route count overflow"); return value; }
function insertSorted(values, value) { let index = 0; while (index < values.length && values[index] < value) index += 1; values.splice(index, 0, value); }
function compareSources(left, right) {
  for (const [a, b] of [[left.reachable_model_output_tensor_count, right.reachable_model_output_tensor_count], [left.predicted_boundary_edge_count, right.predicted_boundary_edge_count]]) if (a !== b) return b - a;
  const leftRoutes = optionalBigInt(left.exact_model_output_graph_route_count_decimal) || 0n;
  const rightRoutes = optionalBigInt(right.exact_model_output_graph_route_count_decimal) || 0n;
  if (leftRoutes !== rightRoutes) return leftRoutes > rightRoutes ? -1 : 1;
  for (const [a, b] of [[left.reconvergence_op_count, right.reconvergence_op_count], [left.reachable_op_count, right.reachable_op_count]]) if (a !== b) return b - a;
  const ratioOrder = BigInt(right.divergent_state_count_decimal || "0") * BigInt(left.interval_state_count_decimal || "1") - BigInt(left.divergent_state_count_decimal || "0") * BigInt(right.interval_state_count_decimal || "1");
  if (ratioOrder !== 0n) return ratioOrder > 0n ? 1 : -1;
  return left.op_index - right.op_index;
}
function withoutDigests(value) { const clone = structuredClone(value); clone.graph_ledger_sha256 = ""; for (const source of clone.sources || []) source.propagation_ledger_sha256 = ""; return clone; }
function normalizeOptionalFields(value) {
  if (Array.isArray(value)) return value.map(normalizeOptionalFields);
  if (!value || typeof value !== "object") return value;
  return Object.fromEntries(Object.entries(value).filter(([, item]) => item != null).map(([key, item]) => [key, normalizeOptionalFields(item)]));
}
function canonicalJson(value) { if (Array.isArray(value)) return `[${value.map(canonicalJson).join(",")}]`; if (value && typeof value === "object") return `{${Object.keys(value).sort().map((key) => `${JSON.stringify(key)}:${canonicalJson(value[key])}`).join(",")}}`; return JSON.stringify(value); }
function assert(condition, message) { if (!condition) throw new Error(message); }

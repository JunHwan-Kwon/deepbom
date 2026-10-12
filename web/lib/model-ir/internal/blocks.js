import { compareCanonicalText } from "../../report-utils.js";
import { sha256TextHex } from "../../sha256-sync.js";
import { partitionConnectedOperations } from "../../model-program-boundary.js";

const MAX_PATTERN_LENGTH = 64;

export function buildStructuralBlocks(operations, regions, { ports = [], values = [], relationships = [] } = {}) {
  const context = buildSignatureContext(ports, values, relationships);
  const blocks = [];
  const edgesByOperation = new Map();
  for (const edge of relationships) {
    if (!edgesByOperation.has(edge.from_ref)) edgesByOperation.set(edge.from_ref, []);
    edgesByOperation.get(edge.from_ref).push(edge);
  }
  for (const region of regions) {
    const rows = operations.filter((row) => row.region_ref === region.id).sort(displayOrder);
    if (!rows.length) continue;
    const regionEdges = rows.flatMap(row => edgesByOperation.get(row.id) || []);
    const spans = partitionConnectedOperations(rows, regionEdges).flatMap(members => {
      const memberIds = new Set(members.map(row => row.id));
      const edges = members.flatMap(row => edgesByOperation.get(row.id) || []).filter(edge => memberIds.has(edge.to_ref));
      return partitionRepeated(members, context).flatMap(span => {
        const parts = partitionConnectedOperations(span.members, edges);
        return parts.map(part => ({ ...span, members: part, ...(parts.length > 1 ? { kind: "rank_partition", repeatIndex: null } : {}) }));
      });
    });
    for (const [spanIndex, span] of spans.entries()) {
      const members = span.members;
      const signature = members.map((row) => operationSignature(row, context, false));
      const representationSignature = members.map((row) => operationSignature(row, context, true));
      const digest = sha256TextHex(JSON.stringify({ schema: "deepbom.structural_block_signature.v1", signature }));
      const representationDigest = sha256TextHex(JSON.stringify({ schema: "deepbom.structural_block_representation_signature.v1", signature: representationSignature }));
      blocks.push({
        id: `block:${sha256TextHex(region.id).slice(0, 8)}:${digest.slice(0, 12)}:${String(spanIndex).padStart(4, "0")}`,
        region_ref: region.id,
        kind: span.kind,
        signature_sha256: digest,
        representation_signature_sha256: representationDigest,
        repeat_index: span.repeatIndex,
        member_refs: members.map((row) => row.id),
        source_order_range: [Math.min(...members.map((row) => row.source_order)), Math.max(...members.map((row) => row.source_order))],
        display_order_range: [Math.min(...members.map((row) => row.display_order)), Math.max(...members.map((row) => row.display_order))],
        boundary_evidence: span.kind === "repeated_structure"
          ? ["maximal_contiguous_repeated_operation_signature"]
          : ["connected_data_component", "bounded_display_order_partition_max_50"],
        grouping_rule_id: "deepbom.structural_block_partition.v1",
        grouping_rule_version: "1.1.0",
        evidence_class: "DERIVED",
      });
    }
  }
  return blocks;
}

// Search the remainder as well as the best run. Each search is bounded by the
// common group limit; no unbounded head/tail catch-all is emitted.
function partitionRepeated(members, context) {
  const repeated = bestRepeatedRun(members.map(row => operationSignature(row, context, false)));
  if (!repeated) return [{ members, kind: "rank_partition", repeatIndex: null }];
  const end = repeated.start + repeated.length * repeated.repetitions;
  return [
    ...(repeated.start ? partitionRepeated(members.slice(0, repeated.start), context) : []),
    ...Array.from({ length: repeated.repetitions }, (_, repeatIndex) => ({
      members: members.slice(repeated.start + repeatIndex * repeated.length, repeated.start + (repeatIndex + 1) * repeated.length),
      kind: "repeated_structure", repeatIndex,
    })),
    ...(end < members.length ? partitionRepeated(members.slice(end), context) : []),
  ];
}

function bestRepeatedRun(signatures) {
  let best = null;
  for (let start = 0; start < signatures.length - 1; start += 1) {
    const maximum = Math.min(MAX_PATTERN_LENGTH, Math.floor((signatures.length - start) / 2));
    for (let length = 1; length <= maximum; length += 1) {
      let repetitions = 1;
      while (start + (repetitions + 1) * length <= signatures.length
        && equalRange(signatures, start, start + repetitions * length, length)) repetitions += 1;
      if (repetitions < 2) continue;
      const covered = length * repetitions;
      const score = covered * 1000 + length;
      if (!best || score > best.score || (score === best.score && start < best.start)) best = { start, length, repetitions, score };
    }
  }
  return best;
}

function equalRange(rows, left, right, length) {
  for (let index = 0; index < length; index += 1) if (rows[left + index] !== rows[right + index]) return false;
  return true;
}

function operationSignature(row, context, includeRepresentation) {
  const inputPorts = (row.input_port_refs || []).map((id) => context.portById.get(id)).filter(Boolean);
  const outputPorts = (row.output_port_refs || []).map((id) => context.portById.get(id)).filter(Boolean);
  const parameterValues = inputPorts.map((port) => context.valueById.get(port.value_ref)).filter((value) => value?.storage_refs?.length);
  const signature = {
    native_op: {
      domain: String(row.native_op?.domain || ""),
      name: String(row.native_op?.name || ""),
      version: row.native_op?.version ?? null,
    },
    input_count: inputPorts.length,
    output_count: outputPorts.length,
    parameter_shape_patterns: parameterValues.map((value) => shapePattern(value.shape)),
    predecessor_count: context.predecessorCount.get(row.id) || 0,
    successor_count: context.successorCount.get(row.id) || 0,
  };
  if (includeRepresentation) signature.representation = {
    input_dtypes: inputPorts.map((port) => context.valueById.get(port.value_ref)?.dtype || "UNKNOWN"),
    output_dtypes: outputPorts.map((port) => context.valueById.get(port.value_ref)?.dtype || "UNKNOWN"),
    quantization: normalizeObject(row.quantization_summary),
  };
  return JSON.stringify({
    ...signature,
  });
}

function buildSignatureContext(ports, values, relationships) {
  const portById = new Map(ports.map((row) => [row.id, row]));
  const valueById = new Map(values.map((row) => [row.id, row]));
  const predecessorCount = new Map();
  const successorCount = new Map();
  for (const row of relationships.filter((item) => ["data_dependency", "control_dependency", "state_dependency"].includes(item.kind))) {
    predecessorCount.set(row.to_ref, (predecessorCount.get(row.to_ref) || 0) + 1);
    successorCount.set(row.from_ref, (successorCount.get(row.from_ref) || 0) + 1);
  }
  return { portById, valueById, predecessorCount, successorCount };
}

function shapePattern(shape) {
  if (!Array.isArray(shape)) return { rank: null, ratios: [] };
  const dimensions = shape.map((value) => Number.isSafeInteger(Number(value)) && Number(value) > 0 ? Number(value) : null);
  const known = dimensions.filter((value) => value != null);
  const divisor = known.length ? known.reduce(gcd) : 1;
  return { rank: shape.length, ratios: dimensions.map((value) => value == null ? "?" : String(value / divisor)) };
}

function gcd(left, right) {
  let a = Math.abs(left);
  let b = Math.abs(right);
  while (b) [a, b] = [b, a % b];
  return a || 1;
}

function normalizeObject(value) {
  if (Array.isArray(value)) return value.map(normalizeObject);
  if (!value || typeof value !== "object") return value ?? null;
  return Object.fromEntries(Object.keys(value).sort().map((key) => [key, normalizeObject(value[key])]));
}

function displayOrder(left, right) { return (left.display_order ?? Number.MAX_SAFE_INTEGER) - (right.display_order ?? Number.MAX_SAFE_INTEGER) || compareCanonicalText(left.id, right.id); }

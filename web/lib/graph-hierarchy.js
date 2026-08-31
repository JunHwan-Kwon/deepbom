export const GRAPH_HIERARCHY_SCHEMA = "deepbom.graph_hierarchy.v1";

export function buildHierarchicalGraphProjection(graph, { maximumGroups = 256, chunkSize = 64 } = {}) {
  if (!Array.isArray(graph?.nodes) || !Array.isArray(graph?.edges)) throw new Error("Hierarchical projection requires canonical graph nodes and edges.");
  const grouping = chooseGrouping(graph.nodes, { maximumGroups, chunkSize });
  const groups = new Map();
  const groupByNode = new Map();
  for (const node of graph.nodes) {
    const descriptor = grouping.describe(node);
    if (!groups.has(descriptor.key)) groups.set(descriptor.key, { ...descriptor, members: [] });
    groups.get(descriptor.key).members.push(node);
  }
  const ordered = [...groups.values()].sort((left, right) => firstPosition(left.members) - firstPosition(right.members)
    || left.label.localeCompare(right.label));
  const nodes = ordered.map((group, ordinal) => {
    const id = `group:${grouping.level}:${ordinal}`;
    for (const member of group.members) groupByNode.set(member.id, id);
    return groupNode(group, id, ordinal);
  });

  const edgeGroups = new Map();
  let internalEdgeCount = 0;
  let externalEdgeCount = 0;
  for (const edge of graph.edges) {
    const from = edge.from ? groupByNode.get(edge.from) : null;
    const to = edge.to ? groupByNode.get(edge.to) : null;
    if (!from || !to) {
      externalEdgeCount += 1;
      continue;
    }
    if (from === to) {
      internalEdgeCount += 1;
      continue;
    }
    const key = `${from}->${to}`;
    if (!edgeGroups.has(key)) edgeGroups.set(key, {
      id: `contracted:${key}`,
      from,
      to,
      relation: "exact_inter_group_contraction",
      edge_count: 0,
      tensor_indices: [],
      tensor_names: [],
      assessed_payload_bytes: 0n,
      assessed_payload_edge_count: 0,
      unassessed_payload_edge_count: 0,
    });
    const aggregate = edgeGroups.get(key);
    aggregate.edge_count += 1;
    if (Number.isSafeInteger(Number(edge.tensor_index))) aggregate.tensor_indices.push(Number(edge.tensor_index));
    if (edge.tensor_name) aggregate.tensor_names.push(String(edge.tensor_name));
    const bytes = exactValue(edge.byte_length);
    if (bytes == null) aggregate.unassessed_payload_edge_count += 1;
    else {
      aggregate.assessed_payload_bytes += bytes;
      aggregate.assessed_payload_edge_count += 1;
    }
  }
  const edges = [...edgeGroups.values()].sort((left, right) => left.from.localeCompare(right.from) || left.to.localeCompare(right.to))
    .map((edge) => ({
      ...edge,
      tensor_indices: [...new Set(edge.tensor_indices)].sort((left, right) => left - right),
      tensor_names: [...new Set(edge.tensor_names)].sort(),
      tensor_name: `${edge.edge_count} contracted tensor edge${edge.edge_count === 1 ? "" : "s"}`,
      dtype: null,
      shape: [],
      byte_length: edge.unassessed_payload_edge_count === 0 ? exact(edge.assessed_payload_bytes) : null,
      assessed_payload_bytes: exact(edge.assessed_payload_bytes),
    }));
  const conservation = {
    original_node_count: graph.nodes.length,
    projected_group_count: nodes.length,
    covered_node_count: nodes.reduce((sum, node) => sum + node.member_count, 0),
    original_edge_count: graph.edges.length,
    internal_group_edge_count: internalEdgeCount,
    inter_group_edge_count: edges.reduce((sum, edge) => sum + edge.edge_count, 0),
    external_edge_count: externalEdgeCount,
  };
  validateConservation(conservation);
  return Object.freeze({
    schema: GRAPH_HIERARCHY_SCHEMA,
    level: grouping.level,
    grouping_basis: grouping.basis,
    evidence_class: "DERIVED_PRESENTATION_WITH_EXACT_EDGE_CONTRACTION",
    nodes,
    edges,
    conservation,
    interpretation_boundary: "Groups are presentation-level contractions of canonical graph nodes. Every rendered inter-group relationship comes from one or more serialized tensor edges; no sequential or runtime relationship is invented.",
  });
}

function chooseGrouping(nodes, { maximumGroups, chunkSize }) {
  const candidates = [
    candidate("stage", "serialized_or_derived_stage_key", (node) => clean(node.stage)),
    candidate("domain", "serialized_operator_domain", (node) => clean(node.domain)),
  ];
  for (const item of candidates) {
    const keys = new Set(nodes.map(item.value).filter(Boolean));
    if (keys.size > 0 && keys.size <= maximumGroups && keys.size <= nodes.length / 2) {
      return {
        level: item.level,
        basis: item.basis,
        describe: (node) => {
          const value = item.value(node) || "unclassified";
          return { key: `${item.level}:${value}`, label: value === "unclassified" ? "Unclassified" : value };
        },
      };
    }
  }
  const width = Math.max(8, Number.isSafeInteger(chunkSize) ? chunkSize : 64);
  const topologicalOrder = [...nodes].sort((left, right) => Number(left.topo_depth ?? left.index) - Number(right.topo_depth ?? right.index)
    || Number(left.index) - Number(right.index));
  const positionByNode = new Map(topologicalOrder.map((node, position) => [node.id, position]));
  return {
    level: "topology_chunk",
    basis: `stable_topological_chunks_of_${width}`,
    describe: (node) => {
      const position = positionByNode.get(node.id) ?? 0;
      const chunk = Math.floor(position / width);
      const first = chunk * width;
      const last = Math.min(topologicalOrder.length - 1, first + width - 1);
      return { key: `chunk:${chunk}`, label: `Topological positions ${first}-${last}` };
    },
  };
}

function candidate(level, basis, value) { return { level, basis, value }; }

function groupNode(group, id, index) {
  const members = [...group.members].sort((left, right) => Number(left.index) - Number(right.index));
  const placements = [...new Set(members.map((node) => String(node.placement?.status || "NOT_ASSESSABLE")))];
  const quantStates = [...new Set(members.map((node) => String(node.quantization?.state || "none")))];
  return {
    id,
    index,
    kind: "collapsed_group",
    label: group.label,
    secondary_label: `${members.length} operators`,
    domain: "hierarchy",
    stage: group.label,
    topo_depth: minimum(members.map((node) => node.topo_depth ?? node.index)),
    member_count: members.length,
    member_node_ids: members.map((node) => node.id),
    member_op_indices: members.map((node) => Number(node.index)).filter(Number.isSafeInteger),
    first_op_index: minimum(members.map((node) => Number(node.index))),
    last_op_index: maximum(members.map((node) => Number(node.index))),
    output_shapes: [],
    macs: sumComplete(members.map((node) => node.macs)),
    assessed_macs: sumAssessed(members.map((node) => node.macs)),
    estimated_bytes: sumComplete(members.map((node) => node.estimated_bytes)),
    assessed_estimated_bytes: sumAssessed(members.map((node) => node.estimated_bytes)),
    quantization: { state: quantStates.length === 1 ? quantStates[0] : "mixed", risk: quantStates.length === 1 ? "none" : "inspect_group" },
    placement: { status: placements.length === 1 ? placements[0] : "MIXED", backend: null, evidence_class: "DERIVED_GROUPING" },
  };
}

function validateConservation(value) {
  if (value.covered_node_count !== value.original_node_count) throw new Error("Hierarchical node conservation failed.");
  if (value.internal_group_edge_count + value.inter_group_edge_count + value.external_edge_count !== value.original_edge_count) {
    throw new Error("Hierarchical edge conservation failed.");
  }
}

function sumComplete(values) {
  return values.every((value) => exactValue(value) != null) ? exact(values.reduce((sum, value) => sum + exactValue(value), 0n)) : null;
}
function sumAssessed(values) { return exact(values.reduce((sum, value) => sum + (exactValue(value) ?? 0n), 0n)); }
function exactValue(value) {
  const raw = value?.decimal ?? value?.number ?? value;
  if (raw == null || raw === "") return null;
  if (typeof raw === "bigint" && raw >= 0n) return raw;
  if (typeof raw === "string" && /^\d+$/.test(raw)) return BigInt(raw);
  if (Number.isSafeInteger(Number(raw)) && Number(raw) >= 0) return BigInt(Number(raw));
  return null;
}
function exact(value) { return { decimal: value.toString(), number: value <= BigInt(Number.MAX_SAFE_INTEGER) ? Number(value) : null }; }
function clean(value) { const text = String(value ?? "").trim(); return text && text !== "?" ? text : ""; }
function firstPosition(members) { return minimum(members.map((node) => Number(node.topo_depth ?? node.index))); }
function minimum(values) { const finite = values.map(Number).filter(Number.isFinite); return finite.length ? Math.min(...finite) : 0; }
function maximum(values) { const finite = values.map(Number).filter(Number.isFinite); return finite.length ? Math.max(...finite) : 0; }

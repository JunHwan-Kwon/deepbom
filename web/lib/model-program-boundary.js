import { portableContract } from "./portable-contracts.js";
// Port-level topology is shared by summaries, source hierarchy and selected
// exports. Module containment and display order never create data edges.
export const MODEL_STRUCTURAL_GROUP_LIMIT = 50;

// Connected, bounded presentation units. The caller supplies deterministic
// display order; order alone never connects otherwise disconnected operations.
export function partitionConnectedOperations(operations, relationships, limit = MODEL_STRUCTURAL_GROUP_LIMIT) {
  if (!Number.isSafeInteger(limit) || limit < 1) throw new Error("Invalid structural group limit.");
  const byId = new Map(operations.map(row => [row.id, row]));
  if (byId.size !== operations.length) throw new Error("Duplicate partition operation.");
  return portableContract("graph_partition", { members: [...byId.keys()], edges: relationships, limit })
    .map(group => group.map(ref => byId.get(ref)));
}

export function indexModelProgram(model) {
  const program = model.program;
  const operations = new Map(program.operations.map(row => [row.id, row]));
  const values = new Map(program.values.map(row => [row.id, row]));
  const ports = new Map(program.ports.map(row => [row.id, row]));
  const producers = new Map(), consumers = new Map();
  for (const port of ports.values()) {
    if (port.direction === "output") {
      if (producers.has(port.value_ref)) throw new Error("Model value has multiple serialized producers.");
      producers.set(port.value_ref, port);
    } else {
      if (!consumers.has(port.value_ref)) consumers.set(port.value_ref, []);
      consumers.get(port.value_ref).push(port);
    }
  }
  const graphOutputs = new Set(program.programs.flatMap(row => row.output_value_refs || []));
  const edges = portableContract("graph_edges", {
    ports: [...ports.values()], programs: program.programs,
    values: [...values.values()].map(({ id, storage_refs }) => ({ id, storage_refs })),
  });
  const incident = new Map([...operations.keys()].map(ref => [ref, new Set()]));
  edges.forEach((edge, position) => {
    incident.get(edge.from_ref)?.add(position);
    incident.get(edge.to_ref)?.add(position);
  });
  return { operations, values, ports, producers, consumers, graphOutputs, edges, incident };
}

export function modelGroupBoundary(index, memberRefs) {
  // Send only incident source edges, retaining their original source order.
  const positions = new Set(memberRefs.flatMap(ref => [...(index.incident.get(ref) || [])]));
  return portableContract("graph_boundary", {
    members: memberRefs, operations: [...index.operations.keys()],
    edges: [...positions].sort((a, b) => a - b).map(position => index.edges[position]),
  });
}

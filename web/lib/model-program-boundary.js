// Port-level topology is shared by summaries, source hierarchy and selected
// exports. Module containment and display order never create data edges.
export const MODEL_STRUCTURAL_GROUP_LIMIT = 50;

// Connected, bounded presentation units. The caller supplies deterministic
// display order; order alone never connects otherwise disconnected operations.
export function partitionConnectedOperations(operations, relationships, limit = MODEL_STRUCTURAL_GROUP_LIMIT) {
  if (!Number.isSafeInteger(limit) || limit < 1) throw new Error("Invalid structural group limit.");
  const byId = new Map(operations.map(row => [row.id, row]));
  if (byId.size !== operations.length) throw new Error("Duplicate partition operation.");
  const adjacent = new Map(operations.map(row => [row.id, new Set()]));
  for (const edge of relationships) if (edge.kind === "data_dependency"
    && byId.has(edge.from_ref) && byId.has(edge.to_ref)) {
    adjacent.get(edge.from_ref).add(edge.to_ref); adjacent.get(edge.to_ref).add(edge.from_ref);
  }
  const components = rows => {
    const included = new Set(rows.map(row => row.id)), componentById = new Map(), output = [];
    for (const row of rows) {
      if (componentById.has(row.id)) continue;
      const componentIndex = output.length, queue = [row.id];
      output.push([]); componentById.set(row.id, componentIndex);
      for (let i = 0; i < queue.length; i++) for (const next of adjacent.get(queue[i])) {
        if (included.has(next) && !componentById.has(next)) { componentById.set(next, componentIndex); queue.push(next); }
      }
    }
    for (const row of rows) output[componentById.get(row.id)].push(row);
    return output;
  };
  const output = [];
  for (const component of components(operations)) {
    for (let start = 0; start < component.length; start += limit) {
      output.push(...components(component.slice(start, start + limit)));
    }
  }
  const order = new Map(operations.map((row, index) => [row.id, index]));
  return output.sort((a, b) => order.get(a[0].id) - order.get(b[0].id));
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
  const outputPorts = new Map();
  for (const region of program.programs) (region.output_value_refs || []).forEach((ref, position) => {
    if (!outputPorts.has(ref)) outputPorts.set(ref, []);
    outputPorts.get(ref).push({ program_ref: region.id, position });
  });
  const edges = [];
  for (const value of values.values()) {
    const producer = producers.get(value.id);
    for (const consumer of consumers.get(value.id) || []) edges.push({
      id: `port-edge:${consumer.id}`, value_ref: value.id,
      from_ref: producer?.operation_ref || null, to_ref: consumer.operation_ref,
      from_port_ref: producer?.id || null, to_port_ref: consumer.id,
      from_port_position: producer?.position ?? null, to_port_position: consumer.position,
      boundary_program_ref: null, boundary_position: null,
      kind: producer ? "data_dependency" : value.storage_refs.length ? "storage_input" : "external_input",
    });
    for (const port of outputPorts.get(value.id) || []) edges.push({
      id: `output-edge:${port.program_ref}:${port.position}:${value.id}`, value_ref: value.id,
      from_ref: producer?.operation_ref || null, to_ref: null,
      from_port_ref: producer?.id || null, to_port_ref: null, kind: "model_output",
      from_port_position: producer?.position ?? null, to_port_position: null,
      boundary_program_ref: port.program_ref, boundary_position: port.position,
    });
  }
  const incident = new Map([...operations.keys()].map(ref => [ref, new Set()]));
  edges.forEach((edge, position) => {
    incident.get(edge.from_ref)?.add(position);
    incident.get(edge.to_ref)?.add(position);
  });
  return { operations, values, ports, producers, consumers, graphOutputs, edges, incident };
}

export function modelGroupBoundary(index, memberRefs) {
  const members = new Set(memberRefs);
  if (members.size !== memberRefs.length || memberRefs.some(ref => !index.operations.has(ref))) throw new Error("Group membership is duplicated or unresolved.");
  const inputs = new Set(), outputs = new Set();
  const internal = [], incoming = [], outgoing = [];
  const positions = new Set(memberRefs.flatMap(ref => [...index.incident.get(ref)]));
  for (const position of [...positions].sort((a, b) => a - b)) {
    const edge = index.edges[position];
    const from = members.has(edge.from_ref), to = members.has(edge.to_ref);
    if (from && to) internal.push(edge.id);
    else if (to) { incoming.push(edge.id); inputs.add(edge.value_ref); }
    else if (from) { outgoing.push(edge.id); outputs.add(edge.value_ref); }
  }
  return { input_value_refs: [...inputs], output_value_refs: [...outputs],
    internal_edge_refs: internal, incoming_edge_refs: incoming, outgoing_edge_refs: outgoing };
}

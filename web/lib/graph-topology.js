const QUANT_BOUNDARY_OPS = new Set([
  "QUANTIZE",
  "DEQUANTIZE",
  "QUANTIZELINEAR",
  "DEQUANTIZELINEAR",
  "DYNAMICQUANTIZELINEAR",
]);

function normalizedName(value) {
  return String(value || "").replace(/[^A-Za-z0-9]+/g, "").toUpperCase();
}

function validTensorIndices(values) {
  return (Array.isArray(values) ? values : [])
    .map(Number)
    .filter((value) => Number.isInteger(value) && value >= 0);
}

export function deriveGraphTopology(ops = []) {
  const rows = Array.isArray(ops) ? ops : [];
  const byIndex = new Map(rows.map((op, position) => [
    Number.isInteger(op?.index) ? Number(op.index) : position,
    op,
  ]));
  const producerByTensor = new Map();
  const consumerCountByTensor = new Map();
  for (const [opIndex, op] of byIndex) {
    for (const tensorIndex of validTensorIndices(op?.outputs)) {
      if (!producerByTensor.has(tensorIndex)) producerByTensor.set(tensorIndex, opIndex);
    }
    for (const tensorIndex of validTensorIndices(op?.inputs)) {
      consumerCountByTensor.set(tensorIndex, (consumerCountByTensor.get(tensorIndex) || 0) + 1);
    }
  }

  const predecessors = new Map();
  for (const [opIndex, op] of byIndex) {
    const incoming = new Set();
    for (const tensorIndex of validTensorIndices(op?.inputs)) {
      const producer = producerByTensor.get(tensorIndex);
      if (producer != null && producer !== opIndex) incoming.add(producer);
    }
    predecessors.set(opIndex, incoming);
  }

  const depths = new Map();
  const visiting = new Set();
  let cycleDetected = false;
  const depthOf = (opIndex) => {
    if (depths.has(opIndex)) return depths.get(opIndex);
    if (visiting.has(opIndex)) {
      cycleDetected = true;
      return 0;
    }
    visiting.add(opIndex);
    const incoming = predecessors.get(opIndex) || new Set();
    const depth = incoming.size
      ? 1 + Math.max(...[...incoming].map(depthOf))
      : 0;
    visiting.delete(opIndex);
    depths.set(opIndex, depth);
    return depth;
  };

  const annotations = [];
  for (const [opIndex, op] of byIndex) {
    const fanOutMax = validTensorIndices(op?.outputs)
      .reduce((maximum, tensorIndex) => Math.max(maximum, consumerCountByTensor.get(tensorIndex) || 0), 0);
    const predecessorCount = (predecessors.get(opIndex) || new Set()).size;
    const name = normalizedName(op?.name);
    const role = QUANT_BOUNDARY_OPS.has(name)
      ? "quant-boundary"
      : predecessorCount > 1
        ? "branch-merge"
        : fanOutMax > 1 ? "branch-split" : "through";
    annotations.push({
      op_index: opIndex,
      role,
      depth: depthOf(opIndex),
      fan_out_max: fanOutMax,
      predecessor_count: predecessorCount,
    });
  }
  annotations.sort((left, right) => left.op_index - right.op_index);
  return {
    schema: "deepbom.graph_topology.v1",
    status: cycleDetected ? "invalid_cycle_detected" : "assessed",
    evidence_class: "DERIVED",
    method: "Derive producer/consumer edges from serialized tensor indices; compute longest predecessor depth, maximum output-tensor fan-out, and branch/quant-boundary roles.",
    cycle_detected: cycleDetected,
    annotations,
  };
}

export function applyGraphTopology(ops = []) {
  const topology = deriveGraphTopology(ops);
  const byIndex = new Map(topology.annotations.map((row) => [row.op_index, row]));
  for (const [position, op] of (Array.isArray(ops) ? ops : []).entries()) {
    const opIndex = Number.isInteger(op?.index) ? Number(op.index) : position;
    const row = byIndex.get(opIndex);
    if (!row) continue;
    if (!op.topo_role) op.topo_role = row.role;
    if (!Number.isFinite(Number(op.topo_depth))) op.topo_depth = row.depth;
    if (!Number.isFinite(Number(op.topo_fan_out_max))) op.topo_fan_out_max = row.fan_out_max;
    if (!Number.isFinite(Number(op.topo_predecessor_count))) {
      op.topo_predecessor_count = row.predecessor_count;
    }
  }
  return topology;
}

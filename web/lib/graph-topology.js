import { portableContract } from "./portable-contracts.js";

function normalizedName(value) {
  return String(value || "").replace(/[^A-Za-z0-9]+/g, "").toUpperCase();
}

function validTensorIndices(values) {
  return (Array.isArray(values) ? values : [])
    .map(Number)
    .filter((value) => Number.isInteger(value) && value >= 0);
}

export function deriveGraphTopology(ops = []) {
  return portableContract("graph_topology", {
    operations: (Array.isArray(ops) ? ops : []).map((op, position) => ({
      index: Number.isInteger(op?.index) ? Number(op.index) : position,
      name: normalizedName(op?.name),
      inputs: validTensorIndices(op?.inputs), outputs: validTensorIndices(op?.outputs),
    })),
  });
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

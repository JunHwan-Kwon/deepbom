import {
  applyGraphTopology,
  deriveGraphTopology,
} from "../web/lib/graph-topology.js";
import { createCheck } from "./check-assert.mjs";

const { done, expect, expectEqual } = createCheck("Graph topology check");

const ops = [
  { index: 0, name: "Conv", inputs: [0, 10], outputs: [1] },
  { index: 1, name: "Relu", inputs: [1], outputs: [2] },
  { index: 2, name: "Sigmoid", inputs: [1], outputs: [3] },
  { index: 3, name: "Add", inputs: [2, 3], outputs: [4] },
  { index: 4, name: "QuantizeLinear", inputs: [4, 11, 12], outputs: [5] },
];

const topology = deriveGraphTopology(ops);
const byIndex = new Map(topology.annotations.map((row) => [row.op_index, row]));

expectEqual(topology.status, "assessed", "A valid acyclic graph should be fully assessed.");
expectEqual(byIndex.get(0)?.role, "branch-split", "A tensor consumed by two operators should mark its producer as a branch split.");
expectEqual(byIndex.get(3)?.role, "branch-merge", "An operator with two producer predecessors should be a branch merge.");
expectEqual(byIndex.get(4)?.role, "quant-boundary", "QuantizeLinear should take precedence as a quantization boundary.");
expectEqual(byIndex.get(4)?.depth, 3, "Longest producer-to-consumer depth should be exact.");
expectEqual(byIndex.get(0)?.fan_out_max, 2, "Maximum output-tensor fan-out should be exact.");

const applied = applyGraphTopology(ops);
expectEqual(applied.schema, "deepbom.graph_topology.v1", "Topology schema should remain versioned.");
expectEqual(ops[0].topo_role, "branch-split", "Applying topology should populate operator roles.");
expectEqual(ops[4].topo_depth, 3, "Applying topology should populate operator depths.");
expectEqual(ops[0].topo_fan_out_max, 2, "Applying topology should populate operator fan-out.");
expectEqual(ops[3].topo_predecessor_count, 2, "Applying topology should retain the producer-predecessor count used to distinguish graph merges from parameter ADDs.");
expectEqual(ops[4].topo_predecessor_count, 1, "Constant/configuration inputs must not be miscounted as producer branches.");
expect(!topology.cycle_detected, "A valid model graph must not be reported as cyclic.");

done("Graph depth, branch roles, fan-out, and quantization boundaries are deterministic.");

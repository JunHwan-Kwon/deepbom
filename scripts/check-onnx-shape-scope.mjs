import { buildOnnxDomainAnalysis } from "../web/lib/onnx-domain-analysis.js";
import { inferOnnxShapes } from "../web/lib/onnx-shape-inference.js";

const attrs = (...entries) => new Map(entries);
const graphAttr = (graph) => ({ type: 5, graph, graphs: [] });
const node = (opType, domain = "", attributes = attrs(), inputs = ["x"], outputs = ["y"]) => ({
  name: "", opType, domain, overload: "", attributes, inputs, outputs,
});
const graph = (nodes = []) => ({ name: "", nodes, inputs: [], outputs: [], valueInfo: [], initializers: [] });
const fn = (name, nodes) => ({ name, domain: "local.test", overload: "", inputs: ["x"], outputs: ["y"], attributes: [], nodes, opsets: [] });

const nestedThen = graph([node("Identity")]);
const nestedElse = graph([]);
const foo = fn("Foo", [
  node("Relu"),
  node("If", "", attrs(["then_branch", graphAttr(nestedThen)], ["else_branch", graphAttr(nestedElse)]), ["cond"], ["branch"]),
  node("Bar", "local.test"),
]);
const bar = fn("Bar", [node("Identity")]);
const unused = fn("Unused", [node("Relu"), node("Relu")]);
const model = {
  graph: graph([node("Foo", "local.test")]),
  opsets: [{ domain: "", version: 13 }, { domain: "local.test", version: 1 }],
  functions: [foo, bar, unused],
};
const domain = buildOnnxDomainAnalysis(model);
const evidence = inferOnnxShapes(model.graph, knownMainTensors(), model.opsets, () => "UNKNOWN", model.functions, domain);
const scope = evidence.shape_scope;
expectEqual(scope.status, "partial", "reachable extended scopes are partial");
expectEqual(scope.registry_status, "pass", "acyclic unique function registry");
expectEqual(scope.nested_graph_count, 2, "empty and non-empty nested GraphProto inventory");
expectEqual(scope.nested_graph_node_count, 1, "nested graph node count");
expectEqual(scope.reachable_nested_graph_count, 2, "reachable nested graph count");
expectEqual(scope.local_function_definition_count, 3, "all local function definitions");
expectEqual(scope.local_function_body_node_count, 6, "all direct function-body nodes");
expectEqual(scope.reachable_local_function_definition_count, 2, "reachable function closure excludes unused definition");
expectEqual(scope.reachable_local_function_body_node_count, 4, "reachable direct function-body nodes");
expectEqual(scope.local_function_call_count, 2, "main and function-body local calls");
expectEqual(scope.reachable_local_function_call_count, 2, "reachable local calls");
expectEqual(scope.unassessed_reachable_node_count, 5, "function and nested-node exclusion conservation");
expectEqual(scope.reachable_exclusion_count, 4, "two function bodies and two nested graphs");
expectEqual(scope.exclusions.reduce((sum, row) => sum + row.node_count, 0), 5, "exclusion node-count conservation");

const recursive = fn("Recursive", [node("Recursive", "local.test")]);
const recursiveModel = {
  graph: graph([node("Recursive", "local.test")]),
  opsets: model.opsets,
  functions: [recursive],
};
const recursiveEvidence = inferOnnxShapes(
  recursiveModel.graph,
  knownMainTensors(),
  recursiveModel.opsets,
  () => "UNKNOWN",
  recursiveModel.functions,
  buildOnnxDomainAnalysis(recursiveModel),
);
expectEqual(recursiveEvidence.shape_scope.registry_status, "fail", "reachable recursive function registry");
expectEqual(recursiveEvidence.shape_scope.reachable_recursive_function_cycle_count, 1, "reachable cycle count");
expectEqual(recursiveEvidence.status, "fail", "reachable recursive function fails shape evidence");

const unusedCycleModel = {
  graph: graph([node("Relu")]),
  opsets: model.opsets,
  functions: [recursive],
};
const unusedCycleEvidence = inferOnnxShapes(
  unusedCycleModel.graph,
  new Map([
    ["x", { name: "x", dtype: "FLOAT32", shape: [1], shapeDeclared: true }],
    ["y", { name: "y", dtype: "FLOAT32", shape: [], shapeDeclared: false }],
  ]),
  unusedCycleModel.opsets,
  () => "UNKNOWN",
  unusedCycleModel.functions,
  buildOnnxDomainAnalysis(unusedCycleModel),
);
expectEqual(unusedCycleEvidence.shape_scope.registry_status, "pass", "unused recursive definition does not contaminate main-graph shape scope");
expectEqual(unusedCycleEvidence.shape_scope.reachable_local_function_definition_count, 0, "unused function reachability");
expectEqual(unusedCycleEvidence.status, "assessed", "supported main graph remains assessed with unused function inventory");

console.log("ONNX reachable control-flow and local-function shape exclusion ledger");

function knownMainTensors() {
  return new Map([
    ["x", { name: "x", dtype: "FLOAT32", shape: [1], shapeDeclared: true }],
    ["y", { name: "y", dtype: "FLOAT32", shape: [1], shapeDeclared: true }],
  ]);
}

function expectEqual(actual, expected, label) {
  if (actual !== expected) throw new Error(`${label}: expected ${JSON.stringify(expected)}, received ${JSON.stringify(actual)}`);
}

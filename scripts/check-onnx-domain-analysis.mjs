import { strict as assert } from "node:assert";
import { buildOnnxDomainAnalysis } from "../web/lib/onnx-domain-analysis.js";

const imports = [
  { domain: "", version: 13 },
  { domain: "ai.onnx.ml", version: 3 },
  { domain: "com.microsoft", version: 1 },
  { domain: "com.deepbom.local", version: 1 },
  { domain: "com.acme", version: 2 },
];
const localFunction = {
  name: "FusedBlock",
  domain: "com.deepbom.local",
  overload: "fp32",
  inputs: ["X"],
  outputs: ["Y"],
  attributes: [],
  opsets: [{ domain: "", version: 13 }],
  nodes: [{ opType: "Relu", domain: "", overload: "", attributes: new Map() }],
};
const nestedGraph = {
  nodes: [{ opType: "NestedCustom", domain: "com.acme", overload: "", attributes: new Map() }],
};
const graph = {
  nodes: [
    { opType: "Conv", domain: "", overload: "", attributes: new Map() },
    { opType: "TreeEnsembleClassifier", domain: "ai.onnx.ml", overload: "", attributes: new Map() },
    { opType: "Attention", domain: "com.microsoft", overload: "", attributes: new Map() },
    { opType: "FusedBlock", domain: "com.deepbom.local", overload: "fp32", attributes: new Map() },
    { opType: "ExternalKernel", domain: "com.acme", overload: "", attributes: new Map([["body", { graph: nestedGraph, graphs: [] }]]) },
  ],
};

const evidence = buildOnnxDomainAnalysis({ graph, opsets: imports, functions: [localFunction] });
assert.equal(evidence.status, "assessed");
assert.equal(evidence.standard_node_count, 3, "main standard, ML, and local-function body standard nodes must be inventoried");
assert.equal(evidence.ort_contrib_node_count, 1);
assert.equal(evidence.model_local_function_call_count, 1);
assert.equal(evidence.external_custom_node_count, 2, "main and nested external custom nodes must both be counted");
assert.deepEqual(evidence.external_custom_domains, ["com.acme"]);
assert.equal(evidence.functions[0].id, "com.deepbom.local::FusedBlock::fp32");
assert.deepEqual(evidence.functions[0].local_function_dependencies, []);
assert.equal(evidence.nodes.find((row) => row.op_name === "NestedCustom").scope_class, "nested_graph");

const ortStandardDomainExtensions = buildOnnxDomainAnalysis({
  graph: { nodes: [
    { opType: "SimplifiedLayerNormalization", domain: "", overload: "", attributes: new Map() },
    { opType: "LayerNormalization", domain: "", overload: "", attributes: new Map() },
  ] },
  opsets: [{ domain: "", version: 16 }],
  functions: [],
});
assert.equal(ortStandardDomainExtensions.ort_contrib_node_count, 2, "ORT schemas registered in ai.onnx must remain contrib evidence at pre-17 opsets");
assert.equal(ortStandardDomainExtensions.standard_node_count, 0, "ORT ai.onnx extensions must not inflate the ONNX-standard denominator");
const onnx17LayerNormalization = buildOnnxDomainAnalysis({
  graph: { nodes: [{ opType: "LayerNormalization", domain: "", overload: "", attributes: new Map() }] },
  opsets: [{ domain: "", version: 17 }],
  functions: [],
});
assert.equal(onnx17LayerNormalization.standard_node_count, 1, "LayerNormalization-17 must use the ONNX-standard schema");
assert.equal(onnx17LayerNormalization.ort_contrib_node_count, 0, "LayerNormalization-17 must not retain the legacy ORT identity");

const recursive = buildOnnxDomainAnalysis({
  graph: { nodes: [] },
  opsets: imports,
  functions: [
    { ...localFunction, nodes: [{ opType: "Second", domain: "com.deepbom.local", overload: "", attributes: new Map() }] },
    { ...localFunction, name: "Second", overload: "", nodes: [{ opType: "FusedBlock", domain: "com.deepbom.local", overload: "fp32", attributes: new Map() }] },
    localFunction,
  ],
});
assert.equal(recursive.status, "invalid_or_ambiguous_function_registry");
assert.deepEqual(recursive.duplicate_function_ids, ["com.deepbom.local::FusedBlock::fp32"]);
const cycleOnly = buildOnnxDomainAnalysis({
  graph: { nodes: [] },
  opsets: imports,
  functions: [
    { ...localFunction, nodes: [{ opType: "Second", domain: "com.deepbom.local", overload: "", attributes: new Map() }] },
    { ...localFunction, name: "Second", overload: "", nodes: [{ opType: "FusedBlock", domain: "com.deepbom.local", overload: "fp32", attributes: new Map() }] },
  ],
});
assert.equal(cycleOnly.recursive_function_cycles.length, 1);

console.log("ONNX domain analysis check passed (standard ML, contrib, local functions, nested graphs, external registries, duplicates, and cycles). ");

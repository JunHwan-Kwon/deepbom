import { isOnnxOrtStandardDomainExtension } from "./onnx-ort-extension-shape-inference.js";
import { effectiveOnnxOpsetMap } from "./onnx-opset-imports.js";

const STANDARD_DOMAINS = new Set(["ai.onnx", "ai.onnx.ml"]);

export function buildOnnxDomainAnalysis(model) {
  const modelImports = importsMap(model?.opsets);
  const functions = (model?.functions || []).map((fn, index) => normalizeFunction(fn, index));
  const functionById = new Map();
  const duplicateFunctionIds = [];
  for (const fn of functions) {
    if (functionById.has(fn.id)) duplicateFunctionIds.push(fn.id);
    else functionById.set(fn.id, fn);
  }

  const nodeRows = [];
  visitGraph(model?.graph, "main_graph", modelImports, nodeRows, functionById, true);
  for (const fn of functions) {
    const imports = fn.opsets.length ? importsMap(fn.opsets) : modelImports;
    fn.dependencies = fn.nodes.map((node) => nodeIdentity(node, imports, functionById));
    fn.local_function_dependencies = fn.dependencies.filter((item) => item.resolution_class === "model_local_function").map((item) => item.function_id);
    visitNodes(fn.nodes, `function:${fn.id}`, imports, nodeRows, functionById, false, "function_body");
  }

  const cycles = findFunctionCycles(functions);
  const cycleMembers = new Set(cycles.flat());
  functions.forEach((fn) => { fn.recursive_cycle = cycleMembers.has(fn.id); });
  const domains = buildDomains(nodeRows, modelImports, functions);
  const externalRows = nodeRows.filter((row) => row.resolution_class === "external_custom_registry");
  const contribRows = nodeRows.filter((row) => row.resolution_class === "ort_contrib_schema");
  const localRows = nodeRows.filter((row) => row.resolution_class === "model_local_function");
  return {
    schema: "deepbom.onnx_domain_analysis.v1",
    status: duplicateFunctionIds.length || cycles.length ? "invalid_or_ambiguous_function_registry" : "assessed",
    scope: "main graph, nested GraphProto attributes, and model-local FunctionProto bodies",
    imported_domains: [...modelImports].map(([domain, version]) => ({ domain, version })),
    domains,
    nodes: nodeRows,
    functions: functions.map(({ nodes: _nodes, ...fn }) => fn),
    duplicate_function_ids: [...new Set(duplicateFunctionIds)].sort(),
    recursive_function_cycles: cycles,
    external_custom_node_count: externalRows.length,
    external_custom_domains: [...new Set(externalRows.map((row) => row.domain))].sort(),
    ort_contrib_node_count: contribRows.length,
    model_local_function_call_count: localRows.length,
    standard_node_count: nodeRows.filter((row) => row.resolution_class.startsWith("onnx_standard")).length,
    interpretation_boundary: "Model-local functions are artifact-defined compositions, ai.onnx.ml is a standard ONNX domain, com.microsoft is an ORT contrib domain, and only unresolved remaining domains require an external custom-op registry. Schema/kernel support and actual EP assignment are assessed separately.",
  };
}

function visitGraph(graph, scope, imports, rows, functionById, topLevel = false) {
  if (!graph) return;
  visitNodes(graph.nodes || [], scope, imports, rows, functionById, topLevel, topLevel ? "main_graph" : "nested_graph");
}

function visitNodes(nodes, scope, imports, rows, functionById, topLevel, scopeClass) {
  for (const [index, node] of (nodes || []).entries()) {
    const identity = nodeIdentity(node, imports, functionById);
    rows.push({
      scope,
      scope_class: scopeClass,
      scope_node_index: index,
      top_level_op_index: topLevel ? index : null,
      node_name: node.name || "",
      op_name: node.opType || "",
      overload: node.overload || "",
      ...identity,
    });
    for (const [attributeName, attribute] of node.attributes || []) {
      if (attribute.graph) visitGraph(attribute.graph, `${scope}/node:${index}/attribute:${attributeName}`, imports, rows, functionById, false);
      for (const [graphIndex, graph] of (attribute.graphs || []).entries()) {
        visitGraph(graph, `${scope}/node:${index}/attribute:${attributeName}[${graphIndex}]`, imports, rows, functionById, false);
      }
    }
  }
}

function nodeIdentity(node, imports, functionById) {
  const domain = normalizeDomain(node?.domain);
  const overload = String(node?.overload || "");
  const functionId = idFor(domain, node?.opType, overload);
  let resolutionClass = "external_custom_registry";
  if (domain === "ai.onnx" && isOnnxOrtStandardDomainExtension(node, imports.get(domain))) resolutionClass = "ort_contrib_schema";
  else if (domain === "ai.onnx") resolutionClass = "onnx_standard";
  else if (domain === "ai.onnx.ml") resolutionClass = "onnx_standard_ml";
  else if (domain === "com.microsoft") resolutionClass = "ort_contrib_schema";
  else if (functionById.has(functionId)) resolutionClass = "model_local_function";
  return {
    domain,
    imported_opset: imports.get(domain) ?? null,
    import_status: imports.has(domain) ? "present" : "missing",
    resolution_class: resolutionClass,
    function_id: resolutionClass === "model_local_function" ? functionId : null,
  };
}

function buildDomains(rows, imports, functions) {
  const functionCounts = new Map();
  functions.forEach((fn) => functionCounts.set(fn.domain, (functionCounts.get(fn.domain) || 0) + 1));
  const domains = new Map([...imports].map(([domain, version]) => [domain, baseDomain(domain, version, functionCounts.get(domain) || 0)]));
  for (const row of rows) {
    const domain = domains.get(row.domain) || baseDomain(row.domain, null, functionCounts.get(row.domain) || 0);
    domain.node_count += 1;
    if (row.scope_class === "main_graph") domain.main_graph_node_count += 1;
    else if (row.scope_class === "nested_graph") domain.nested_graph_node_count += 1;
    else domain.function_body_node_count += 1;
    domain.op_types.add(row.op_name);
    domain.resolution_classes.add(row.resolution_class);
    domains.set(row.domain, domain);
  }
  return [...domains.values()].map((item) => ({
    ...item,
    op_types: [...item.op_types].sort(),
    resolution_classes: [...item.resolution_classes].sort(),
    unused_import: item.node_count === 0,
  })).sort((left, right) => left.domain.localeCompare(right.domain));
}

function baseDomain(domain, importedOpset, localFunctionDefinitionCount) {
  return {
    domain,
    imported_opset: importedOpset,
    node_count: 0,
    main_graph_node_count: 0,
    nested_graph_node_count: 0,
    function_body_node_count: 0,
    local_function_definition_count: localFunctionDefinitionCount,
    standard_domain: STANDARD_DOMAINS.has(domain),
    op_types: new Set(),
    resolution_classes: new Set(),
  };
}

function normalizeFunction(fn, index) {
  const domain = normalizeDomain(fn.domain);
  const name = String(fn.name || "");
  const overload = String(fn.overload || "");
  return {
    index,
    id: idFor(domain, name, overload),
    domain,
    name,
    overload,
    inputs: [...(fn.inputs || [])],
    outputs: [...(fn.outputs || [])],
    attributes: [...(fn.attributes || [])].sort(),
    opsets: (fn.opsets || []).map((item) => ({ domain: normalizeDomain(item.domain), version: Number(item.version || 0) })).sort((a, b) => a.domain.localeCompare(b.domain)),
    nodes: fn.nodes || [],
    body_node_count: (fn.nodes || []).length,
  };
}

function findFunctionCycles(functions) {
  const dependencies = new Map(functions.map((fn) => [fn.id, fn.local_function_dependencies || []]));
  const state = new Map();
  const stack = [];
  const cycles = new Map();
  const visit = (id) => {
    if (state.get(id) === 2) return;
    if (state.get(id) === 1) {
      const start = stack.indexOf(id);
      const cycle = [...stack.slice(start), id];
      cycles.set([...new Set(cycle)].sort().join("\0"), cycle);
      return;
    }
    state.set(id, 1);
    stack.push(id);
    for (const dependency of dependencies.get(id) || []) visit(dependency);
    stack.pop();
    state.set(id, 2);
  };
  [...dependencies.keys()].sort().forEach(visit);
  return [...cycles.values()].sort((a, b) => a.join("\0").localeCompare(b.join("\0")));
}

function importsMap(values) {
  return new Map([...effectiveOnnxOpsetMap(values)].sort(([left], [right]) => left.localeCompare(right)));
}

function idFor(domain, name, overload) { return `${domain}::${String(name || "")}::${String(overload || "")}`; }
function normalizeDomain(value) { const domain = String(value || "").trim(); return domain ? domain : "ai.onnx"; }

import { canonicalJson } from "./report-utils.js";
import { effectiveOnnxOpsetMap } from "./onnx-opset-imports.js";

export const ORT_REDUCED_OPERATOR_CONFIG_SCHEMA = "deepbom.ort_reduced_operator_config.v1";
export const ORT_REDUCED_OPERATOR_ASSESSMENT_SCHEMA = "deepbom.ort_reduced_operator_assessment.v1";
export const ORT_REDUCED_OPERATOR_SOURCES = Object.freeze([
  Object.freeze({
    role: "official_config_parser",
    source_ref: "https://raw.githubusercontent.com/microsoft/onnxruntime/8c546c37b43caaca1fa25db430dab94b901cf277/tools/python/util/reduced_build_config_parser.py",
    sha256: "7bf7afd69cd4af1958a4c6ea710f632a224850274ea25be2df3c8b9cdf645a58",
  }),
  Object.freeze({
    role: "official_reduced_build_contract",
    source_ref: "https://raw.githubusercontent.com/microsoft/onnxruntime/8c546c37b43caaca1fa25db430dab94b901cf277/docs/Reduced_Operator_Kernel_build.md",
    sha256: "5912efb6ebb16c1388b69cc48f43129853c6dda5d283b5e8838a660fd8435257",
  }),
]);

const OP_NAME = /^[A-Za-z][A-Za-z0-9_]*$/;
const MAX_CONFIG_CHARS = 4 * 1024 * 1024;

export function parseOrtReducedOperatorConfig(source) {
  if (typeof source !== "string" || source.length > MAX_CONFIG_CHARS) throw new Error("ORT reduced-operator config is missing or exceeds 4 MiB.");
  const entries = new Map();
  let globallyAllowedTypes = null;
  let noOpsMeansAll = false;
  let typeReductionEntryCount = 0;
  let operatorLineCount = 0;
  for (const [lineIndex, original] of source.split(/\r?\n/).entries()) {
    const line = original.trim();
    if (!line || line.startsWith("#")) continue;
    if (line.startsWith("!globally_allowed_types;")) {
      if (globallyAllowedTypes) throw new Error("ORT reduced-operator config repeats globally_allowed_types.");
      globallyAllowedTypes = normalizedList(line.slice(line.indexOf(";") + 1), "globally allowed type", lineIndex);
      continue;
    }
    if (line === "!no_ops_specified_means_all_ops_are_required") {
      if (noOpsMeansAll) throw new Error("ORT reduced-operator config repeats the no-ops directive.");
      noOpsMeansAll = true;
      continue;
    }
    const segments = line.split(";");
    if (segments.length !== 3) throw new Error(`ORT reduced-operator config line ${lineIndex + 1} must contain exactly three semicolon-delimited fields.`);
    const domain = normalizeDomain(segments[0]);
    const opsets = normalizedOpsets(segments[1], lineIndex);
    const operators = parseOperatorList(segments[2], lineIndex);
    operatorLineCount += 1;
    for (const opset of opsets) {
      const key = `${domain}\0${opset}`;
      const row = entries.get(key) || { domain, opset, operators: new Map() };
      for (const operator of operators) {
        const existing = row.operators.get(operator.name);
        if (existing && canonicalJson(existing.type_reduction) !== canonicalJson(operator.type_reduction)) {
          throw new Error(`ORT reduced-operator config provides conflicting type reductions for ${domain}:${opset}:${operator.name}.`);
        }
        if (operator.type_reduction && !existing) typeReductionEntryCount += 1;
        row.operators.set(operator.name, operator);
      }
      entries.set(key, row);
    }
  }
  if (globallyAllowedTypes && typeReductionEntryCount) {
    throw new Error("ORT reduced-operator config cannot combine globally allowed types with per-operator type reduction.");
  }
  const rows = [...entries.values()]
    .map((row) => ({ ...row, operators: [...row.operators.values()].sort((a, b) => a.name.localeCompare(b.name)) }))
    .sort((a, b) => a.domain.localeCompare(b.domain) || a.opset - b.opset);
  return Object.freeze({
    schema: ORT_REDUCED_OPERATOR_CONFIG_SCHEMA,
    status: rows.length ? "explicit_operator_inventory" : noOpsMeansAll ? "all_operators_required" : "no_operators_required",
    no_ops_specified_means_all_ops_are_required: noOpsMeansAll,
    globally_allowed_types: globallyAllowedTypes,
    operator_line_count: operatorLineCount,
    domain_opset_row_count: rows.length,
    operator_identity_count: rows.reduce((sum, row) => sum + row.operators.length, 0),
    type_reduction_entry_count: typeReductionEntryCount,
    entries: rows,
    source_documents: ORT_REDUCED_OPERATOR_SOURCES,
    interpretation_boundary: "This is the normalized content of an ORT reduced-operator config. It establishes selected-binary inclusion only when a build attestation binds this exact config digest to the captured binary inventory.",
  });
}

export function assessOrtReducedOperatorConfig(analysis, config) {
  if (String(analysis?.format || "").toLowerCase() !== "onnx" || config?.schema !== ORT_REDUCED_OPERATOR_CONFIG_SCHEMA) {
    throw new Error("ORT reduced-operator assessment requires an ONNX analysis and normalized config.");
  }
  const allRequired = config.status === "all_operators_required";
  const entryMap = new Map((config.entries || []).map((row) => [`${row.domain}\0${row.opset}`, row]));
  const domainRows = new Map();
  for (const row of config.entries || []) {
    const values = domainRows.get(row.domain) || [];
    values.push(row);
    domainRows.set(row.domain, values);
  }
  const nodes = artifactNodes(analysis);
  const rows = nodes.map((node) => {
    const exact = entryMap.get(`${node.domain}\0${node.imported_opset}`);
    const operator = exact?.operators?.find((candidate) => candidate.name === node.op_name) || null;
    const alternate = (domainRows.get(node.domain) || []).some((row) => row.operators.some((candidate) => candidate.name === node.op_name));
    const status = allRequired ? "included_all_operators_directive"
      : operator ? "included_exact_imported_opset"
        : alternate ? "unresolved_operator_present_at_other_opset"
          : "excluded_operator_not_listed";
    return {
      ...node,
      status,
      type_reduction_status: config.globally_allowed_types ? "global_type_reduction_present_not_artifact_assessed"
        : operator?.type_reduction ? "per_operator_type_reduction_present_not_artifact_assessed" : "not_present",
    };
  });
  const included = rows.filter((row) => row.status.startsWith("included_")).length;
  const excluded = rows.filter((row) => row.status.startsWith("excluded_")).length;
  const unresolved = rows.length - included - excluded;
  const typeUnresolved = rows.filter((row) => row.type_reduction_status !== "not_present").length;
  return Object.freeze({
    schema: ORT_REDUCED_OPERATOR_ASSESSMENT_SCHEMA,
    status: excluded ? "incompatible_missing_operator_identity" : unresolved || typeUnresolved ? "partial" : "compatible_operator_identity",
    evidence_class: "DERIVED_FROM_IMPORTED_ORT_BUILD_CONFIG",
    assessed_node_count: rows.length,
    included_node_count: included,
    excluded_node_count: excluded,
    unresolved_node_count: unresolved,
    type_reduction_unresolved_node_count: typeUnresolved,
    rows,
    interpretation_boundary: "Operator identity compatibility is evaluated against exact imported domain/opset rows. A row at another opset and every type-reduction clause remain unresolved until the generated registration/type inventory is bound. Config compatibility is not selected-binary proof without build attestation.",
  });
}

function parseOperatorList(source, lineIndex) {
  const output = [];
  let cursor = 0;
  while (cursor < source.length) {
    while (source[cursor] === " " || source[cursor] === "\t" || source[cursor] === ",") cursor += 1;
    const start = cursor;
    while (cursor < source.length && source[cursor] !== "{" && source[cursor] !== ",") cursor += 1;
    const name = source.slice(start, cursor).trim();
    if (!OP_NAME.test(name)) throw new Error(`ORT reduced-operator config line ${lineIndex + 1} contains invalid operator ${name || "(empty)"}.`);
    let typeReduction = null;
    if (source[cursor] === "{") {
      const end = matchingJsonObjectEnd(source, cursor, lineIndex);
      try { typeReduction = JSON.parse(source.slice(cursor, end)); } catch (error) { throw new Error(`ORT reduced-operator config line ${lineIndex + 1} has invalid type JSON: ${error.message}`); }
      if (!typeReduction || typeof typeReduction !== "object" || Array.isArray(typeReduction)) throw new Error(`ORT reduced-operator config line ${lineIndex + 1} type reduction must be an object.`);
      cursor = end;
    }
    output.push({ name, type_reduction: typeReduction });
    if (cursor < source.length && source[cursor] !== ",") throw new Error(`ORT reduced-operator config line ${lineIndex + 1} has trailing operator content.`);
    cursor += 1;
  }
  if (!output.length || new Set(output.map((row) => row.name)).size !== output.length) throw new Error(`ORT reduced-operator config line ${lineIndex + 1} has no operators or duplicates one.`);
  return output;
}

function matchingJsonObjectEnd(source, start, lineIndex) {
  let depth = 0;
  let quoted = false;
  let escaped = false;
  for (let index = start; index < source.length; index += 1) {
    const value = source[index];
    if (quoted) {
      if (escaped) escaped = false;
      else if (value === "\\") escaped = true;
      else if (value === '"') quoted = false;
    } else if (value === '"') quoted = true;
    else if (value === "{") depth += 1;
    else if (value === "}" && --depth === 0) return index + 1;
  }
  throw new Error(`ORT reduced-operator config line ${lineIndex + 1} has unmatched type JSON braces.`);
}

function normalizedOpsets(value, lineIndex) {
  const output = normalizedList(value, "opset", lineIndex).map(Number);
  if (output.some((item) => !Number.isSafeInteger(item) || item <= 0)) throw new Error(`ORT reduced-operator config line ${lineIndex + 1} has an invalid opset.`);
  return [...new Set(output)].sort((a, b) => a - b);
}

function normalizedList(value, label, lineIndex) {
  const output = String(value).split(",").map((item) => item.trim());
  if (!output.length || output.some((item) => !item)) throw new Error(`ORT reduced-operator config line ${lineIndex + 1} has an empty ${label}.`);
  return [...new Set(output)].sort();
}

function artifactNodes(analysis) {
  const inventory = analysis?.onnx_domain_analysis?.nodes;
  if (Array.isArray(inventory)) return inventory.map((row) => ({
    scope: row.scope,
    top_level_op_index: row.top_level_op_index,
    domain: normalizeDomain(row.domain),
    imported_opset: row.imported_opset,
    op_name: String(row.op_name || ""),
  }));
  const imports = effectiveOnnxOpsetMap(analysis.opsets || []);
  return (analysis.ops || []).map((row, index) => ({
    scope: "main_graph",
    top_level_op_index: Number.isSafeInteger(row.index) ? row.index : index,
    domain: normalizeDomain(row.domain),
    imported_opset: imports.get(normalizeDomain(row.domain)) ?? null,
    op_name: String(row.name || ""),
  }));
}

function normalizeDomain(value) {
  const domain = String(value || "").trim();
  return domain || "ai.onnx";
}

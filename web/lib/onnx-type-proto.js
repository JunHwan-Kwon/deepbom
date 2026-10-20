import { portableContract } from "./portable-contracts.js";
import { ONNX_TENSOR_TYPE_IDS as TENSOR_TYPE_IDS } from "./onnx-tensor-types.js";
const SOURCE_COMMIT = "be2b5fde82d9c8874f3d19328bdfe3b6962dc67b";

export function exactSequenceLength(value) {
  return value?.sequenceLengthStatus === "assessed_exact" && Number.isSafeInteger(value.sequenceLength)
    && value.sequenceLength >= 0 ? value.sequenceLength : null;
}
export function exactSequenceInventory(value) {
  return value?.sequenceElementInventoryStatus === "assessed_exact" && Array.isArray(value.sequenceElementTypes)
    ? value.sequenceElementTypes.map(cloneOnnxTypeProto) : null;
}
export function exactOptionalPresence(value) {
  return value?.optionalPresenceStatus === "assessed_exact" && typeof value.optionalPresence === "boolean" ? value.optionalPresence : null;
}
export function knownDtype(tensor) {
  const value = String(tensor?.dtype || "");
  return value && value !== "UNKNOWN" ? value : "";
}
const MAX_TYPE_PROTO_NODES = 1_000_000;
const MAX_TYPE_PROTO_TEXT_BYTES = 4_194_304;
const VALUE_ROLES = new Set(["graph_input", "graph_output", "graph_value_info", "function_value_info"]);

export const ONNX_TYPE_PROTO_SOURCE = Object.freeze({
  release: "v1.21.0",
  commit: SOURCE_COMMIT,
  source_ref: `https://raw.githubusercontent.com/onnx/onnx/${SOURCE_COMMIT}/onnx/onnx.in.proto`,
  sha256: "f4cbc198df3a0f3f4519d4d38cd2262e8f84057583b7313e2d0f981b3f68c213",
});

export function buildOnnxTypeProtoContract(graph, functions = []) {
  const declarations = collectDeclarations(graph, functions);
  const rows = declarations.map((declaration) => assessDeclaration(declaration));
  const typeNodeCount = rows.reduce((sum, row) => sum + row.type_node_count, 0);
  const textBytes = rows.reduce((sum, row) => sum + row.type_text_bytes, 0);
  if (typeNodeCount > MAX_TYPE_PROTO_NODES) throw new Error(`ONNX TypeProto node count ${typeNodeCount} exceeds safety limit ${MAX_TYPE_PROTO_NODES}.`);
  if (textBytes > MAX_TYPE_PROTO_TEXT_BYTES) throw new Error(`ONNX TypeProto text bytes ${textBytes} exceed safety limit ${MAX_TYPE_PROTO_TEXT_BYTES}.`);
  const invalidRows = rows.filter((row) => row.status === "fail");
  const declaredRows = rows.filter((row) => row.status !== "not_declared");
  const valueRows = rows.filter((row) => VALUE_ROLES.has(row.role));
  const kindCounts = countBy(declaredRows.map((row) => row.kind));
  return {
    schema: "deepbom.onnx_type_proto_contract.v1",
    status: invalidRows.length ? "fail" : "assessed",
    evidence_class: "SOURCE_PINNED_AND_OBSERVED",
    source_release: ONNX_TYPE_PROTO_SOURCE.release,
    source_commit: ONNX_TYPE_PROTO_SOURCE.commit,
    source_ref: ONNX_TYPE_PROTO_SOURCE.source_ref,
    source_sha256: ONNX_TYPE_PROTO_SOURCE.sha256,
    declaration_count: rows.length,
    declared_type_count: declaredRows.length,
    undeclared_optional_type_count: rows.filter((row) => row.status === "not_declared").length,
    valid_type_count: rows.filter((row) => row.status === "pass").length,
    invalid_type_count: invalidRows.length,
    type_node_count: typeNodeCount,
    maximum_type_depth: rows.reduce((maximum, row) => Math.max(maximum, row.maximum_type_depth), 0),
    type_text_bytes: textBytes,
    tensor_value_count: valueRows.filter((row) => row.kind === "tensor").length,
    sparse_tensor_value_count: valueRows.filter((row) => row.kind === "sparse_tensor").length,
    non_dense_value_count: valueRows.filter((row) => row.status !== "not_declared" && row.kind !== "tensor").length,
    sequence_value_count: valueRows.filter((row) => row.kind === "sequence").length,
    map_value_count: valueRows.filter((row) => row.kind === "map").length,
    optional_value_count: valueRows.filter((row) => row.kind === "optional").length,
    opaque_value_count: valueRows.filter((row) => row.kind === "opaque").length,
    symbolic_dimension_count: rows.reduce((sum, row) => sum + row.symbolic_dimension_count, 0),
    unknown_dimension_count: rows.reduce((sum, row) => sum + row.unknown_dimension_count, 0),
    kind_counts: kindCounts,
    invalid_rows: invalidRows,
    rows,
    method: "Parse the pinned ONNX TypeProto oneof recursively, preserve tensor, sequence, map, optional, sparse-tensor, and opaque branches, validate required child fields and map key types, and retain symbolic/unknown dimension declarations without coercing them to dense tensor payloads.",
    interpretation_boundary: "This contract validates serialized type declarations. Only kind=tensor is eligible for dense tensor shape, MAC, payload, liveness, and browser-input calculations. Sparse tensors preserve their logical shape but require sparse-aware storage/compute analysis; sequence, map, optional, and opaque values have no dense byte projection.",
  };
}

export function isDenseTensorValue(value) {
  const kind = value?.valueKind || value?.value_kind || "";
  return !kind || kind === "tensor" || kind === "unresolved" || kind === "undefined";
}

export function cloneOnnxTypeProto(type) {
  if (!type || typeof type !== "object") return null;
  const clone = { ...type };
  if (Array.isArray(type.valueFieldsPresent)) clone.valueFieldsPresent = [...type.valueFieldsPresent];
  if (Array.isArray(type.shape)) clone.shape = [...type.shape];
  if (Array.isArray(type.shapeDimensions)) clone.shapeDimensions = type.shapeDimensions.map((dimension) => ({ ...dimension }));
  if (type.elementType) clone.elementType = cloneOnnxTypeProto(type.elementType);
  if (type.valueType) clone.valueType = cloneOnnxTypeProto(type.valueType);
  return clone;
}

// Host representation decoding only. Unknown/malformed values must not become
// concrete zero through Number(null), Number(false), or Number("").
export function onnxDimensionInteger(value) {
  if (!["number", "string", "bigint"].includes(typeof value)
    || typeof value === "string" && !value.trim()) return null;
  const number = Number(value);
  return Number.isSafeInteger(number) ? number : null;
}

export function makeOnnxTensorType(dtype, shape = [], shapeDeclared = false, kind = "tensor") {
  const normalizedDtype = String(dtype || "UNKNOWN");
  const dimensions = shapeDeclared ? (shape || []).map((dimension) => onnxDimensionInteger(dimension) != null && onnxDimensionInteger(dimension) >= 0
    ? { kind: "value", value: Number(dimension), parameter: "", denotation: "", valueFieldCount: 1 }
    : { kind: "unknown", value: null, parameter: "", denotation: "", valueFieldCount: 0 }) : [];
  return {
    kind,
    dtype: normalizedDtype,
    shape: shapeDeclared ? dimensions.map((dimension) => dimension.kind === "value" ? dimension.value : -1) : [],
    shapeDeclared: shapeDeclared === true,
    shapeDimensions: dimensions,
    shapeFieldCount: shapeDeclared ? 1 : 0,
    elementTypeId: TENSOR_TYPE_IDS.get(normalizedDtype) || 0,
    elementTypeName: normalizedDtype,
    elementTypeFieldCount: normalizedDtype !== "UNKNOWN" ? 1 : 0,
    valueFieldsPresent: [kind === "sparse_tensor" ? 8 : 1],
    denotation: "",
  };
}

export function makeOnnxTensorTypeFromDimensions(dtype, dimensions = [], shapeDeclared = false, kind = "tensor") {
  if (!shapeDeclared) return makeOnnxTensorType(dtype, [], false, kind);
  const normalized = (dimensions || []).map((dimension) => normalizeDimension(dimension));
  const normalizedDtype = String(dtype || "UNKNOWN");
  return {
    kind,
    dtype: normalizedDtype,
    shape: normalized.map((dimension) => dimension.kind === "value" ? dimension.value : -1),
    shapeDeclared: true,
    shapeDimensions: normalized,
    shapeFieldCount: 1,
    elementTypeId: TENSOR_TYPE_IDS.get(normalizedDtype) || 0,
    elementTypeName: normalizedDtype,
    elementTypeFieldCount: normalizedDtype !== "UNKNOWN" ? 1 : 0,
    valueFieldsPresent: [kind === "sparse_tensor" ? 8 : 1],
    denotation: "",
  };
}

export function onnxShapeDimensionsFromValue(value) {
  const type = value?.typeProto || value?.type_proto || null;
  if (Array.isArray(type?.shapeDimensions) && type.shapeDimensions.length === (type.shape || []).length) {
    return type.shapeDimensions.map((dimension) => normalizeDimension(dimension));
  }
  if (value?.shapeDeclared !== true && value?.shape_declared !== true) return null;
  return (value?.shape || []).map((dimension) => normalizeDimension(dimension));
}

export function makeOnnxSequenceType(elementType) {
  return {
    kind: "sequence", valueFieldsPresent: [4], elementType: cloneOnnxTypeProto(elementType), childFieldCount: 1,
    dtype: "UNKNOWN", shape: [], shapeDeclared: false, shapeDimensions: [], denotation: "",
  };
}

export function makeOnnxOptionalType(elementType) {
  return {
    kind: "optional", valueFieldsPresent: [9], elementType: cloneOnnxTypeProto(elementType), childFieldCount: 1,
    dtype: "UNKNOWN", shape: [], shapeDeclared: false, shapeDimensions: [], denotation: "",
  };
}

export function makeOnnxMapType(keyTypeName, valueType) {
  const normalizedKeyType = String(keyTypeName || "UNDEFINED");
  return {
    kind: "map",
    valueFieldsPresent: [5],
    keyTypeId: TENSOR_TYPE_IDS.get(normalizedKeyType) || 0,
    keyTypeName: normalizedKeyType,
    keyFieldCount: normalizedKeyType !== "UNDEFINED" ? 1 : 0,
    valueType: cloneOnnxTypeProto(valueType),
    childFieldCount: valueType ? 1 : 0,
    dtype: "UNKNOWN",
    shape: [],
    shapeDeclared: false,
    shapeDimensions: [],
    denotation: "",
  };
}

export function onnxTypeProtoFromValue(value) {
  if (value?.typeProto || value?.type_proto) return cloneOnnxTypeProto(value.typeProto || value.type_proto);
  if (!isDenseTensorValue(value)) return null;
  return makeOnnxTensorType(
    value?.dtype || "UNKNOWN",
    Array.isArray(value?.shape) ? value.shape : [],
    value?.shapeDeclared === true || value?.shape_declared === true,
  );
}

export function onnxValueDescriptorFromType(type, state = {}) {
  const cloned = cloneOnnxTypeProto(type);
  const tensorLike = cloned?.kind === "tensor" || cloned?.kind === "sparse_tensor";
  return {
    valueKind: cloned?.kind || "undefined",
    typeProto: cloned,
    dtype: tensorLike ? cloned.dtype || cloned.elementTypeName || "UNKNOWN" : "UNKNOWN",
    shape: tensorLike && Array.isArray(cloned.shape) ? [...cloned.shape] : [],
    shapeDeclared: tensorLike && cloned.shapeDeclared === true,
    ...state,
  };
}

export function onnxTypeProtoKnown(type) {
  return portableContract("onnx_type", {operation:"known", value:type ?? null});
}

export function unionOnnxTypeProtos(types) {
  const items = (types || []).filter(Boolean).map(cloneOnnxTypeProto);
  if (!items.length) return { status: "unresolved", type: null, reason: "type_proto_missing" };
  let current = items[0];
  for (const next of items.slice(1)) {
    const merged = unionTypePair(current, next);
    if (merged.status !== "pass") return merged;
    current = merged.type;
  }
  return { status: "pass", type: current, reason: "" };
}

export function unifyOnnxTypeProtos(declared, inferred) {
  if (!declared) return { status: inferred ? "pass" : "unresolved", type: cloneOnnxTypeProto(inferred), reason: inferred ? "" : "type_proto_missing" };
  if (!inferred) return { status: "pass", type: cloneOnnxTypeProto(declared), reason: "" };
  return unifyTypePair(cloneOnnxTypeProto(declared), cloneOnnxTypeProto(inferred));
}

function collectDeclarations(root, functions) {
  const rows = [];
  const visitGraph = (graph, scope) => {
    appendValueRows(rows, graph?.inputs, scope, "graph_input", scope === "main_graph");
    appendValueRows(rows, graph?.outputs, scope, "graph_output", scope === "main_graph");
    appendValueRows(rows, graph?.valueInfo, scope, "graph_value_info", false);
    visitNodes(graph?.nodes || [], scope);
  };
  const visitNodes = (nodes, scope) => {
    for (const [nodeIndex, node] of (nodes || []).entries()) {
      for (const [attributeName, attribute] of node.attributes || []) {
        appendAttributeTypes(rows, attribute, `${scope}/node:${nodeIndex}/attribute:${attributeName}`, "node_attribute_type");
        const graphs = [attribute.graph, ...(attribute.graphs || [])].filter(Boolean);
        for (const [graphIndex, nested] of graphs.entries()) {
          const suffix = graphs.length === 1 ? attributeName : `${attributeName}[${graphIndex}]`;
          visitGraph(nested, `${scope}/node:${nodeIndex}/attribute:${suffix}`);
        }
      }
    }
  };
  visitGraph(root, "main_graph");
  for (const [functionIndex, fn] of (functions || []).entries()) {
    const id = `${String(fn.domain || "ai.onnx") || "ai.onnx"}::${String(fn.name || "")}::${String(fn.overload || "")}`;
    const scope = `function:${functionIndex}:${id}`;
    appendValueRows(rows, fn.valueInfo, scope, "function_value_info", false);
    visitNodes(fn.nodes || [], scope);
    for (const [attributeIndex, attribute] of (fn.attributeProtos || []).entries()) {
      appendAttributeTypes(rows, attribute, `${scope}/default_attribute:${attribute.name || attributeIndex}`, "function_default_attribute_type");
      const graphs = [attribute.graph, ...(attribute.graphs || [])].filter(Boolean);
      for (const [graphIndex, nested] of graphs.entries()) {
        const suffix = graphs.length === 1 ? attribute.name || attributeIndex : `${attribute.name || attributeIndex}[${graphIndex}]`;
        visitGraph(nested, `${scope}/default_attribute:${suffix}`);
      }
    }
  }
  return rows;
}

function appendValueRows(output, values, scope, role, required) {
  for (const [index, value] of (values || []).entries()) {
    output.push({ scope, role, name: value.name || "", ordinal: index, required, type: value.typeProto || null });
  }
}

function appendAttributeTypes(output, attribute, scope, role) {
  if (attribute.typeProto) output.push({ scope, role, name: attribute.name || "attribute", required: true, type: attribute.typeProto });
  for (const [index, type] of (attribute.typeProtos || []).entries()) {
    output.push({ scope, role: `${role}_list`, name: `${attribute.name || "attribute"}[${index}]`, required: true, type });
  }
}

function assessDeclaration(declaration) {
  const declarationReasons = [];
  if (VALUE_ROLES.has(declaration.role) && !declaration.name) declarationReasons.push("value_info_name_missing");
  if (!declaration.type) {
    return {
      scope: declaration.scope,
      role: declaration.role,
      value_name: declaration.name || `(missing value ${declaration.ordinal ?? 0})`,
      kind: "undeclared",
      canonical_type: "not declared",
      status: declaration.required || declarationReasons.length ? "fail" : "not_declared",
      reason_codes: [...declarationReasons, ...(declaration.required ? ["required_type_proto_missing"] : [])],
      type_node_count: 0,
      maximum_type_depth: 0,
      type_text_bytes: 0,
      symbolic_dimension_count: 0,
      unknown_dimension_count: 0,
    };
  }
  const assessment = assessType(declaration.type, 1);
  if (declaration.required && ["tensor", "sparse_tensor"].includes(declaration.type.kind) && declaration.type.shapeDeclared !== true) {
    assessment.reasons.push("top_level_tensor_shape_missing");
  }
  return {
    scope: declaration.scope,
    role: declaration.role,
    value_name: declaration.name || `(missing value ${declaration.ordinal ?? 0})`,
    kind: declaration.type.kind || "undefined",
    canonical_type: canonicalType(declaration.type),
    status: declarationReasons.length || assessment.reasons.length ? "fail" : "pass",
    reason_codes: [...new Set([...declarationReasons, ...assessment.reasons])].sort(),
    type_node_count: assessment.nodeCount,
    maximum_type_depth: assessment.maximumDepth,
    type_text_bytes: assessment.textBytes,
    symbolic_dimension_count: assessment.symbolicDimensions,
    unknown_dimension_count: assessment.unknownDimensions,
  };
}

function assessType(type, depth) {
  return portableContract("onnx_type", {operation:"assess", value:type, depth});
}

function childTypes(type) {
  if ((type.kind === "sequence" || type.kind === "optional") && type.elementType) return [type.elementType];
  if (type.kind === "map" && type.valueType) return [type.valueType];
  return [];
}

export function canonicalOnnxTypeProto(type) {
  if (!type) return "not declared";
  const shape = type.shapeDeclared
    ? `[${(type.shapeDimensions || []).map(canonicalDimension).join(",")}]`
    : "[rank?]";
  if (type.kind === "tensor") return `tensor<${type.elementTypeName || "UNDEFINED"}${shape}>`;
  if (type.kind === "sparse_tensor") return `sparse_tensor<${type.elementTypeName || "UNDEFINED"}${shape}>`;
  if (type.kind === "sequence") return `sequence<${canonicalOnnxTypeProto(type.elementType)}>`;
  if (type.kind === "optional") return `optional<${canonicalOnnxTypeProto(type.elementType)}>`;
  if (type.kind === "map") return `map<${type.keyTypeName || "UNDEFINED"},${canonicalOnnxTypeProto(type.valueType)}>`;
  if (type.kind === "opaque") return `opaque<${type.domain || ""}:${type.name || ""}>`;
  return "undefined";
}

function canonicalType(type) {
  return canonicalOnnxTypeProto(type);
}

// Transport representation normalization; the join/unification decision lives in WASM.
function portableType(type) {
  const result = cloneOnnxTypeProto(type) || {};
  if (result.kind === "tensor" || result.kind === "sparse_tensor") result.shapeDimensions = normalizedShapeDimensions(result);
  if (result.elementType) result.elementType = portableType(result.elementType);
  if (result.valueType) result.valueType = portableType(result.valueType);
  return result;
}
function mergeTypes(operation, left, right) {
  return portableContract("onnx_type", {operation, left:portableType(left), right:portableType(right), tensor_type_ids:Object.fromEntries(TENSOR_TYPE_IDS)});
}
function unionTypePair(left, right) { return mergeTypes("union", left, right); }
function unifyTypePair(left, right) { return mergeTypes("unify", left, right); }

function normalizedShapeDimensions(type) {
  if (Array.isArray(type?.shapeDimensions) && type.shapeDimensions.length === (type.shape || []).length) {
    return type.shapeDimensions.map((dimension) => ({ ...dimension }));
  }
  return (type?.shape || []).map((dimension) => onnxDimensionInteger(dimension) != null && onnxDimensionInteger(dimension) >= 0
    ? { kind: "value", value: Number(dimension), parameter: "", denotation: "", valueFieldCount: 1 }
    : { kind: "unknown", value: null, parameter: "", denotation: "", valueFieldCount: 0 });
}

function normalizeDimension(dimension) {
  if (dimension?.kind === "value" && onnxDimensionInteger(dimension.value) != null && onnxDimensionInteger(dimension.value) >= 0) {
    return { kind: "value", value: Number(dimension.value), parameter: "", denotation: String(dimension.denotation || ""), valueFieldCount: 1 };
  }
  if (dimension?.kind === "symbolic" && String(dimension.parameter || "")) {
    return { kind: "symbolic", value: null, parameter: String(dimension.parameter), denotation: String(dimension.denotation || ""), valueFieldCount: 1 };
  }
  if (onnxDimensionInteger(dimension) != null && onnxDimensionInteger(dimension) >= 0) {
    return { kind: "value", value: Number(dimension), parameter: "", denotation: "", valueFieldCount: 1 };
  }
  return { kind: "unknown", value: null, parameter: "", denotation: String(dimension?.denotation || ""), valueFieldCount: 0 };
}

function canonicalDimension(dimension) {
  if (dimension.kind === "value") return String(dimension.value);
  if (dimension.kind === "symbolic") return dimension.parameter || "symbol?";
  return "?";
}

function countBy(values) {
  const counts = new Map();
  for (const value of values) counts.set(value || "undefined", (counts.get(value || "undefined") || 0) + 1);
  return [...counts.entries()].map(([name, count]) => ({ name, count })).sort((left, right) => right.count - left.count || left.name.localeCompare(right.name));
}

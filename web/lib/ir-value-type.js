import { portableContract } from "./portable-contracts.js";
import { logicalBytesForElements, tensorRankKnown } from "./tensor-size.js";
export { scalarDtypeBits } from "./tensor-size.js";
import { canonicalJson, canonicalJsonInput } from "./report-utils.js";
import { exactInteger } from "./exact-integer.js";

// Types describe logical values, independently of whether bytes or a graph exist.
export function buildValueType(value, format, scope) {
  if (format === "onnx") scope = "symbol-scope:onnx:model"; // ONNX IR: dimension variables are shared by main and nested graphs.
  const native = value.type_proto || value.typeProto;
  const root = native ? nativeType(native, scope, 0) : tensorType(value, format, scope);
  return { schema: "deepbom.value_type.v1", root,
    native_type: native ? JSON.parse(canonicalJson(native)) : null,
    evidence_basis: native ? "native_type_tree" : "adapter_tensor_descriptor" };
}

function nativeType(type, scope, depth) {
  if (depth > 64) throw new Error("IR value type nesting exceeds 64 levels.");
  const kind = type.kind;
  if (["tensor", "sparse_tensor"].includes(kind)) return tensorType({
    ...type, shape_declared: type.shapeDeclared === true, value_kind: kind,
  }, "onnx", scope);
  if (["sequence", "optional"].includes(kind)) return { kind, element_type: type.elementType ? nativeType(type.elementType, scope, depth + 1) : { kind: "unknown" } };
  if (kind === "map") return { kind, key_dtype: String(type.keyTypeName || "UNKNOWN"), value_type: type.valueType ? nativeType(type.valueType, scope, depth + 1) : { kind: "unknown" } };
  if (kind === "opaque") return { kind, domain: String(type.domain || ""), name: String(type.name || "") };
  return { kind: "unknown" };
}

function tensorType(value, format, scope) {
  const kind = value.value_kind || value.valueKind || "tensor";
  if (!["tensor", "dense_tensor", "sparse_tensor", "unresolved"].includes(kind)) return { kind: "unknown" };
  const ranked = tensorRankKnown(value, format);
  const dims = ranked ? value.shape.map((dimension, index) => dimensionType(dimension, value.shapeDimensions?.[index], scope)) : null;
  return { kind: kind === "sparse_tensor" ? kind : "tensor", dtype: String(value.dtype || value.elementTypeName || "UNKNOWN").toUpperCase(),
    rank_status: ranked ? "ranked" : "unknown_rank", dimensions: dims };
}

function dimensionType(value, native, scope) {
  // A dim_param that happens to contain digits is still a symbol.
  if (["symbol", "param", "symbolic"].includes(native?.kind)) return { kind: "symbol", name: String(native.parameter ?? native.symbol ?? native.param ?? native.value ?? value), scope_ref: scope };
  const integer = exactInteger(value);
  if (integer) return { kind: "constant", value: integer };
  if (typeof value === "string" && value && value !== "?" && !/^-\d+$/.test(value)) return { kind: "symbol", name: value, scope_ref: scope };
  return { kind: "unknown" };
}

export function valueElementCount(contract) {
  return portableContract("ir_semantics", {operation:"elements",value:canonicalJsonInput(contract)});
}

// Fixed-width scalar encodings only. GGUF block encodings need their native block
// size/overhead contract and must never be interpreted as a scalar bit width.

export function valueLogicalBytes(contract) {
  const elements = valueElementCount(contract);
  return elements ? logicalBytesForElements(contract.root.dtype, elements.decimal) : null;
}

export function validateValueType(contract) {
  portableContract("ir_semantics", {operation:"value_type",value:canonicalJsonInput(contract)});
  return contract;
}

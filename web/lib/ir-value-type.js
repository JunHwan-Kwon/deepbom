import { logicalBytesForElements, tensorRankKnown } from "./tensor-size.js";
export { scalarDtypeBits } from "./tensor-size.js";
import { canonicalJson } from "./report-utils.js";
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
  const type = contract?.root;
  if (!["tensor", "sparse_tensor"].includes(type?.kind) || type.rank_status !== "ranked") return null;
  const dims = type.dimensions;
  // A known zero makes the product zero even when another dimension is symbolic.
  if (dims.some(d => d.kind === "constant" && d.value.decimal === "0")) return exactInteger(0);
  if (dims.some(d => d.kind !== "constant")) return null;
  return exactInteger(dims.reduce((product, d) => product * BigInt(d.value.decimal), 1n));
}

// Fixed-width scalar encodings only. GGUF block encodings need their native block
// size/overhead contract and must never be interpreted as a scalar bit width.

export function valueLogicalBytes(contract) {
  const elements = valueElementCount(contract);
  return elements ? logicalBytesForElements(contract.root.dtype, elements.decimal) : null;
}

export function validateValueType(contract) {
  if (contract?.schema !== "deepbom.value_type.v1") throw new Error("IR value type schema is invalid.");
  const visit = (type, depth = 0) => {
    if (depth > 64 || !type || typeof type !== "object") throw new Error("IR value type tree is invalid.");
    if (["tensor", "sparse_tensor"].includes(type.kind)) {
      if (typeof type.dtype !== "string" || !type.dtype || !["ranked", "unknown_rank"].includes(type.rank_status)
        || (type.rank_status === "unknown_rank" ? type.dimensions !== null : !Array.isArray(type.dimensions))) throw new Error("IR tensor rank contract is invalid.");
      for (const dim of type.dimensions || []) {
        if (!dim || typeof dim !== "object") throw new Error("IR dimension contract is invalid.");
        if (dim.kind === "constant") {
          const count = exactInteger(dim.value?.decimal);
          if (!count || canonicalJson(count) !== canonicalJson(dim.value)) throw new Error("IR dimension exact count is inconsistent.");
        } else if (dim.kind === "symbol") {
          if (typeof dim.name !== "string" || !dim.name || typeof dim.scope_ref !== "string" || !dim.scope_ref) throw new Error("IR dimension symbol scope is invalid.");
        } else if (dim.kind !== "unknown") throw new Error("IR dimension kind is invalid.");
      }
    } else if (["sequence", "optional"].includes(type.kind)) visit(type.element_type, depth + 1);
    else if (type.kind === "map") { if (!type.key_dtype) throw new Error("IR map key type is missing."); visit(type.value_type, depth + 1); }
    else if (!["unknown", "opaque"].includes(type.kind)) throw new Error("IR value kind is invalid.");
  };
  canonicalJson(contract); visit(contract.root); return contract;
}

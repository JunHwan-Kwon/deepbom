import { exactInteger } from "./exact-integer.js";
import { SCALAR_TYPES } from "./scalar-types.generated.js";

// Fixed-width logical scalar encodings. A known width does not imply decoder,
// operator or backend support. Block encodings and variable-length values have
// no scalar width here; their native storage contracts remain authoritative.
export function scalarDtypeBits(dtype) {
  return SCALAR_TYPES[String(dtype || "").toUpperCase()]?.bits ?? null;
}

export function scalarDtypeBytes(dtype) {
  const bits = scalarDtypeBits(dtype);
  return bits != null && bits % 8 === 0 ? bits / 8 : null;
}

export function integerCodeRange(dtype) {
  const type = SCALAR_TYPES[String(dtype || "").toUpperCase()];
  if (type?.kind !== "integer") return null;
  const width = BigInt(type.bits), high = 1n << (type.signed ? width - 1n : width);
  return { minimum: type.signed ? -high : 0n, maximum: high - 1n };
}

export function safeIntegerCodeRange(dtype) {
  const range = integerCodeRange(dtype);
  if (!range || range.minimum < BigInt(Number.MIN_SAFE_INTEGER) || range.maximum > BigInt(Number.MAX_SAFE_INTEGER)) return null;
  return [Number(range.minimum), Number(range.maximum)];
}

// Strict numeric shapes only. Scalar [] = 1; unknown rank is not []. Validate
// every dimension before multiplying so zero never hides an invalid dimension.
export function shapeElementCount(shape) {
  if (!Array.isArray(shape)) return null;
  // Array.every/reduce skip holes; a missing extent is unknown, not one.
  for (const dimension of shape) if (!Number.isSafeInteger(dimension) || dimension < 0) return null;
  if (shape.includes(0)) return exactInteger(0);
  return exactInteger(shape.reduce((product, d) => product * BigInt(d), 1n));
}

export function safeShapeElementCount(shape) {
  return shapeElementCount(shape)?.number ?? null;
}

// Some native contracts require positive extents; rank-zero support is explicit.
export function safePositiveShapeElementCount(shape, { allowScalar = false } = {}) {
  if (!Array.isArray(shape) || !allowScalar && !shape.length || shape.some(d => d <= 0)) return null;
  return safeShapeElementCount(shape);
}

export function safeCountProduct(...values) {
  return safeShapeElementCount(values);
}

export function safeCountSum(...values) {
  if (!values.every(v => Number.isSafeInteger(v) && v >= 0)) return null;
  return exactInteger(values.reduce((sum, value) => sum + BigInt(value), 0n)).number;
}

export function broadcastStaticShapes(left, right) {
  if (!shapeElementCount(left) || !shapeElementCount(right)) return null;
  const output = [];
  for (let offset = Math.max(left.length, right.length); offset > 0; offset -= 1) {
    const a = left[left.length - offset] ?? 1, b = right[right.length - offset] ?? 1;
    if (a !== b && a !== 1 && b !== 1) return null;
    // max(0, 1) is 1, but broadcasting an empty dimension with 1 stays empty.
    output.push(a === 1 ? b : a);
  }
  return output;
}

export function logicalBytesForElements(dtype, elements) {
  return packedBytesForElements(elements, scalarDtypeBits(dtype));
}

export function packedBytesForElements(elements, bits) {
  const count = exactInteger(elements);
  return count && Number.isSafeInteger(bits) && bits > 0
    ? exactInteger((BigInt(count.decimal) * BigInt(bits) + 7n) / 8n) : null;
}

export function logicalBytesForShape(dtype, shape) {
  const count = shapeElementCount(shape);
  return count ? logicalBytesForElements(dtype, count.decimal) : null;
}

export function tensorRankKnown(tensor, format = "") {
  return Array.isArray(tensor?.shape) && tensor.shape_declared !== false && tensor.shapeDeclared !== false
    && tensor.shape_rank_status !== "unknown_rank"
    && (tensor.shape.length > 0 || tensor.shape_declared === true || tensor.shapeDeclared === true
      || tensor.shape_rank_status === "ranked" || tensor.has_rank === true || tensor.rank === 0
      || ["safetensors", "gguf"].includes(format));
}

// Projection is a caller decision, never an implicit replacement of a dynamic
// batch with one. The binding is retained in every assessed result.
export function tensorPayloadAssessment(tensor, { allowSerializedBatchOne = false } = {}) {
  const unassessed = reason => ({ bytes: null, status: "not_assessed", binding: "unbound", reason });
  if (!tensor) return unassessed("tensor_descriptor_missing");
  if (!tensorRankKnown(tensor)) return unassessed("shape_not_declared");
  const shape = tensor.shape, count = shapeElementCount(shape);
  if (!Array.isArray(shape)) return unassessed("shape_not_declared");
  if (!count) return unassessed("shape_dynamic_or_invalid");
  const signature = tensor.shape_signature ?? tensor.shapeSignature;
  const staticSignature = signature == null || Array.isArray(signature) && (!signature.length
    || signature.length === shape.length && signature.every((d, i) => d === shape[i]));
  const batchOne = !staticSignature && allowSerializedBatchOne && Array.isArray(signature)
    && signature.length === shape.length && signature[0] === -1 && shape[0] === 1
    && signature.slice(1).every((d, i) => d === shape[i + 1]);
  if (!staticSignature && !batchOne) return unassessed("shape_signature_not_statically_bound");
  const bytes = logicalBytesForElements(tensor.dtype, count.decimal);
  if (!bytes) return unassessed("dtype_width_unknown");
  if (bytes.number == null) return unassessed("payload_exceeds_safe_integer");
  return { bytes: bytes.number, status: batchOne ? "assessed_serialized_batch1" : "assessed_static",
    binding: batchOne ? "serialized_batch1_projection" : "static", reason: null };
}

export function staticTensorPayloadBytes(tensor) {
  return tensorPayloadAssessment(tensor).bytes;
}

// Host views only; wire framing, uint64 arithmetic and malformed bounds are
// owned by the same WASM decoder for ONNX, TensorFlow and Core ML adapters.
import { numericalWasm } from "./numerical-wasm.js";
function decode(label, run) {
  try { return run(); }
  catch (error) { throw new Error(`${label}: ${error?.message || error}`); }
}
export function protobufWireFields(bytes, label = "Protobuf") {
  return decode(label, () => numericalWasm().protobuf_fields(bytes));
}
export function protobufVarint(bytes, offset = 0, label = "Protobuf") {
  if (!Number.isSafeInteger(offset) || offset < 0 || offset > bytes.length) throw new Error(`${label}: invalid varint offset`);
  const result = decode(label, () => numericalWasm().protobuf_varint(bytes.subarray(offset, offset + 10)));
  return { value: BigInt(result.value), next: offset + result.length };
}
export function protobufPackedFloats(bytes, width, label = "Protobuf") {
  return decode(label, () => Array.from(numericalWasm().protobuf_packed_floats(bytes, width)));
}

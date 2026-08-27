export function model(nodes, initializers, inputs, outputs, opset, annotations = [], irVersion = 8, graphName = "deepbom_quant_fixture") {
  const graph = graphProto(nodes, initializers, inputs, outputs, graphName, annotations);
  const opsetImport = message([stringField(1, ""), varintField(2, opset)]);
  return message([varintField(1, irVersion), stringField(2, graphName), bytesField(7, graph), bytesField(8, opsetImport)]);
}

export function graphProto(nodes, initializers, inputs, outputs, name, annotations = []) {
  return message([
    ...nodes.map((value) => bytesField(1, value)),
    stringField(2, name),
    ...initializers.map((value) => bytesField(5, value)),
    ...inputs.map((value) => bytesField(11, value)),
    ...outputs.map((value) => bytesField(12, value)),
    ...annotations.map((value) => bytesField(14, value)),
  ]);
}

export function tensorAnnotation(tensorName, entries) {
  return message([
    stringField(1, tensorName),
    ...entries.map(([key, value]) => bytesField(2, message([stringField(1, key), stringField(2, value)]))),
  ]);
}

export function node(opType, name, inputs, outputs, domain = "") {
  return message([
    ...inputs.map((value) => stringField(1, value)),
    ...outputs.map((value) => stringField(2, value)),
    stringField(3, name), stringField(4, opType),
    ...(domain ? [stringField(7, domain)] : []),
  ]);
}

export function nodeWithIntegerAttributes(opType, name, inputs, outputs, attributes, domain = "") {
  const encodedAttributes = Object.entries(attributes).map(([attributeName, value]) => bytesField(5, message([
    stringField(1, attributeName),
    varintField(3, value),
    varintField(20, 2),
  ])));
  return message([
    ...inputs.map((value) => stringField(1, value)),
    ...outputs.map((value) => stringField(2, value)),
    stringField(3, name), stringField(4, opType),
    ...encodedAttributes,
    ...(domain ? [stringField(7, domain)] : []),
  ]);
}

export function nodeWithGraphAttribute(opType, name, attributeName, graph, domain = "") {
  const attribute = message([stringField(1, attributeName), bytesField(6, graph), varintField(20, 5)]);
  return message([
    stringField(3, name), stringField(4, opType), bytesField(5, attribute),
    ...(domain ? [stringField(7, domain)] : []),
  ]);
}

export function tensor(name, dtype, dims, raw) {
  return message([...dims.map((dim) => varintField(1, dim)), varintField(2, dtype), stringField(8, name), bytesField(9, raw)]);
}

export function externalTensor(name, dtype, dims, entries) {
  return message([
    ...dims.map((dim) => varintField(1, dim)),
    varintField(2, dtype),
    stringField(8, name),
    ...entries.map(([key, value]) => bytesField(13, message([stringField(1, key), stringField(2, value)]))),
    varintField(14, 1),
  ]);
}

export function valueInfo(name, dtype, dims) {
  const shape = message(dims.map((dim) => bytesField(1, message([varintField(1, dim)]))));
  const tensorType = message([varintField(1, dtype), bytesField(2, shape)]);
  return message([stringField(1, name), bytesField(2, message([bytesField(1, tensorType)]))]);
}

export function valueInfoWithoutShape(name, dtype) {
  const tensorType = message([varintField(1, dtype)]);
  return message([stringField(1, name), bytesField(2, message([bytesField(1, tensorType)]))]);
}

export function float32(values) {
  const out = new Uint8Array(values.length * 4);
  const view = new DataView(out.buffer);
  values.forEach((value, index) => view.setFloat32(index * 4, value, true));
  return out;
}

export function int32(values) {
  const out = new Uint8Array(values.length * 4);
  const view = new DataView(out.buffer);
  values.forEach((value, index) => view.setInt32(index * 4, value, true));
  return out;
}

export function stringField(field, value) {
  return bytesField(field, new TextEncoder().encode(value));
}

export function bytesField(field, value) {
  const bytes = value instanceof Uint8Array ? value : new Uint8Array(value);
  return message([varint((field << 3) | 2), varint(bytes.length), bytes]);
}

export function varintField(field, value) {
  return message([varint(field << 3), varint(value)]);
}

export function varint(value) {
  let remaining = BigInt(value);
  const bytes = [];
  do {
    let byte = Number(remaining & 0x7fn);
    remaining >>= 7n;
    if (remaining) byte |= 0x80;
    bytes.push(byte);
  } while (remaining);
  return new Uint8Array(bytes);
}

export function message(parts) {
  const length = parts.reduce((sum, part) => sum + part.length, 0);
  const out = new Uint8Array(length);
  let offset = 0;
  for (const part of parts) {
    out.set(part, offset);
    offset += part.length;
  }
  return out;
}

const DEFAULT_LIMITS = Object.freeze({
  maxVectorElements: 2_000_000,
  maxStringBytes: 16 * 1024 * 1024,
});

export class BoundedFlatBufferReader {
  constructor(bytes, limits = {}) {
    if (!(bytes instanceof Uint8Array)) throw new TypeError("FlatBuffer payload must be a Uint8Array.");
    this.bytes = bytes;
    this.view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    this.limits = { ...DEFAULT_LIMITS, ...limits };
    this.decoder = new TextDecoder("utf-8", { fatal: true });
  }

  identifier() {
    this.requireRange(4, 4, "file identifier");
    return String.fromCharCode(...this.bytes.subarray(4, 8));
  }

  root(expectedIdentifier = "") {
    this.requireRange(0, 8, "FlatBuffer root");
    if (expectedIdentifier && this.identifier() !== expectedIdentifier) {
      throw new Error(`FlatBuffer identifier ${JSON.stringify(this.identifier())} does not match ${expectedIdentifier}.`);
    }
    const root = this.u32(0, "root offset");
    if (root < 8) throw new Error(`FlatBuffer root offset ${root} overlaps the header.`);
    return this.table(root, "root table");
  }

  table(position, label = "table") {
    this.requireRange(position, 4, label);
    const backward = this.i32(position, `${label} vtable offset`);
    if (backward === 0) throw new Error(`${label} has a zero vtable offset.`);
    const vtable = position - backward;
    if (!Number.isSafeInteger(vtable) || vtable < 0) throw new Error(`${label} has invalid vtable offset ${backward}.`);
    this.requireRange(vtable, 4, `${label} vtable`);
    const vtableLength = this.u16(vtable, `${label} vtable length`);
    const objectLength = this.u16(vtable + 2, `${label} object length`);
    if (vtableLength < 4 || vtableLength % 2 !== 0) throw new Error(`${label} has invalid vtable length ${vtableLength}.`);
    if (objectLength < 4) throw new Error(`${label} has invalid object length ${objectLength}.`);
    this.requireRange(vtable, vtableLength, `${label} vtable`);
    this.requireRange(position, objectLength, label);
    return Object.freeze({ position, vtable, vtableLength, objectLength, label });
  }

  field(table, index, width = 1) {
    const entry = table.vtable + 4 + index * 2;
    if (entry + 2 > table.vtable + table.vtableLength) return null;
    const relative = this.u16(entry, `${table.label} field ${index} entry`);
    if (relative === 0) return null;
    if (relative < 4 || relative + width > table.objectLength) {
      throw new Error(`${table.label} field ${index} lies outside its table object.`);
    }
    return table.position + relative;
  }

  scalar(table, index, kind, fallback = 0) {
    const widths = { u8: 1, i8: 1, bool: 1, u16: 2, i16: 2, u32: 4, i32: 4, f32: 4, u64: 8, i64: 8, f64: 8 };
    const width = widths[kind];
    if (!width) throw new Error(`Unsupported FlatBuffer scalar kind ${kind}.`);
    const position = this.field(table, index, width);
    return position == null ? fallback : this[kind](position, `${table.label} field ${index}`);
  }

  offsetObject(table, index, label = "offset object") {
    const field = this.field(table, index, 4);
    if (field == null) return null;
    const relative = this.u32(field, `${table.label} field ${index} offset`);
    if (relative === 0) throw new Error(`${table.label} field ${index} has a null offset.`);
    const position = field + relative;
    this.requireRange(position, 4, label);
    return position;
  }

  tableField(table, index, label = "table") {
    const position = this.offsetObject(table, index, label);
    return position == null ? null : this.table(position, label);
  }

  stringField(table, index, label = "string") {
    const position = this.offsetObject(table, index, label);
    if (position == null) return "";
    const length = this.u32(position, `${label} length`);
    if (length > this.limits.maxStringBytes) throw new Error(`${label} length ${length} exceeds the configured limit.`);
    this.requireRange(position + 4, length + 1, label);
    if (this.bytes[position + 4 + length] !== 0) throw new Error(`${label} is not NUL terminated.`);
    try {
      return this.decoder.decode(this.bytes.subarray(position + 4, position + 4 + length));
    } catch {
      throw new Error(`${label} is not valid UTF-8.`);
    }
  }

  vector(table, index, elementWidth, label = "vector") {
    const position = this.offsetObject(table, index, label);
    if (position == null) return null;
    const length = this.u32(position, `${label} length`);
    if (length > this.limits.maxVectorElements) throw new Error(`${label} length ${length} exceeds the configured limit.`);
    const data = position + 4;
    const byteLength = this.checkedProduct(length, elementWidth, `${label} byte length`);
    this.requireRange(data, byteLength, label);
    return Object.freeze({ position, data, length, elementWidth, byteLength, label });
  }

  tableVector(table, index, label = "table vector") {
    const vector = this.vector(table, index, 4, label);
    if (!vector) return [];
    const result = [];
    for (let i = 0; i < vector.length; i += 1) {
      const slot = vector.data + i * 4;
      const relative = this.u32(slot, `${label}[${i}] offset`);
      if (relative === 0) throw new Error(`${label}[${i}] has a null table offset.`);
      result.push(this.table(slot + relative, `${label}[${i}]`));
    }
    return result;
  }

  scalarVector(table, index, kind, label = "scalar vector") {
    const widths = { u8: 1, i8: 1, bool: 1, u16: 2, i16: 2, u32: 4, i32: 4, f32: 4, u64: 8, i64: 8, f64: 8 };
    const width = widths[kind];
    if (!width) throw new Error(`Unsupported FlatBuffer vector scalar kind ${kind}.`);
    const vector = this.vector(table, index, width, label);
    if (!vector) return [];
    const values = new Array(vector.length);
    for (let i = 0; i < vector.length; i += 1) values[i] = this[kind](vector.data + i * width, `${label}[${i}]`);
    return values;
  }

  byteVector(table, index, label = "byte vector") {
    const vector = this.vector(table, index, 1, label);
    return vector ? this.bytes.subarray(vector.data, vector.data + vector.length) : new Uint8Array();
  }

  u8(position, label = "u8") { this.requireRange(position, 1, label); return this.view.getUint8(position); }
  i8(position, label = "i8") { this.requireRange(position, 1, label); return this.view.getInt8(position); }
  bool(position, label = "bool") { const value = this.u8(position, label); if (value > 1) throw new Error(`${label} has invalid boolean value ${value}.`); return value === 1; }
  u16(position, label = "u16") { this.requireRange(position, 2, label); return this.view.getUint16(position, true); }
  i16(position, label = "i16") { this.requireRange(position, 2, label); return this.view.getInt16(position, true); }
  u32(position, label = "u32") { this.requireRange(position, 4, label); return this.view.getUint32(position, true); }
  i32(position, label = "i32") { this.requireRange(position, 4, label); return this.view.getInt32(position, true); }
  f32(position, label = "f32") { this.requireRange(position, 4, label); return this.view.getFloat32(position, true); }
  u64(position, label = "u64") { this.requireRange(position, 8, label); return this.view.getBigUint64(position, true); }
  i64(position, label = "i64") { this.requireRange(position, 8, label); return this.view.getBigInt64(position, true); }
  f64(position, label = "f64") { this.requireRange(position, 8, label); return this.view.getFloat64(position, true); }

  checkedProduct(left, right, label = "product") {
    const value = Number(left) * Number(right);
    if (!Number.isSafeInteger(value) || value < 0) throw new Error(`${label} is outside the safe byte-address range.`);
    return value;
  }

  requireRange(position, length, label = "range") {
    if (!Number.isSafeInteger(position) || !Number.isSafeInteger(length) || position < 0 || length < 0 || position > this.bytes.length - length) {
      throw new Error(`${label} [${position}, ${position + length}) exceeds payload length ${this.bytes.length}.`);
    }
  }
}

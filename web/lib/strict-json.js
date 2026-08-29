export function parseStrictJson(source, label = "JSON document") {
  if (typeof source !== "string") throw new Error(`${label} must be UTF-8 JSON text`);
  let offset = 0;
  let nodes = 0;
  const whitespace = () => { while (/\s/.test(source[offset] || "")) offset += 1; };
  const string = () => {
    if (source[offset] !== '"') throw new Error(`invalid JSON string at ${offset}`);
    const start = offset++;
    while (offset < source.length) {
      if (source[offset] === "\\") { offset += 2; continue; }
      if (source[offset++] === '"') return JSON.parse(source.slice(start, offset));
    }
    throw new Error("unterminated JSON string");
  };
  const value = (depth = 0) => {
    if (depth > 128) throw new Error(`${label} exceeds the JSON nesting limit`);
    if (++nodes > 2_000_000) throw new Error(`${label} exceeds the JSON value-count limit`);
    whitespace();
    if (source[offset] === "{") return object(depth);
    if (source[offset] === "[") return array(depth);
    if (source[offset] === '"') { string(); return; }
    const match = source.slice(offset).match(/^(?:-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?|true|false|null)/);
    if (!match) throw new Error(`invalid JSON value at ${offset}`);
    offset += match[0].length;
  };
  const object = (depth) => {
    offset += 1;
    whitespace();
    const keys = new Set();
    if (source[offset] === "}") { offset += 1; return; }
    while (offset < source.length) {
      const key = string();
      if (keys.has(key)) throw new Error(`duplicate JSON key ${key}`);
      keys.add(key);
      whitespace();
      if (source[offset++] !== ":") throw new Error(`missing JSON colon at ${offset - 1}`);
      value(depth + 1);
      whitespace();
      const separator = source[offset++];
      if (separator === "}") return;
      if (separator !== ",") throw new Error(`invalid JSON object separator at ${offset - 1}`);
      whitespace();
    }
    throw new Error("unterminated JSON object");
  };
  const array = (depth) => {
    offset += 1;
    whitespace();
    if (source[offset] === "]") { offset += 1; return; }
    while (offset < source.length) {
      value(depth + 1);
      whitespace();
      const separator = source[offset++];
      if (separator === "]") return;
      if (separator !== ",") throw new Error(`invalid JSON array separator at ${offset - 1}`);
    }
    throw new Error("unterminated JSON array");
  };
  value();
  whitespace();
  if (offset !== source.length) throw new Error(`trailing JSON content at ${offset}`);
  try {
    return JSON.parse(source);
  } catch (error) {
    throw new Error(`${label} is invalid: ${error?.message || error}`);
  }
}

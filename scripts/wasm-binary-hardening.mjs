import { readFileSync, writeFileSync } from "node:fs";

const WASM_HEADER = Uint8Array.from([0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00]);
const REDACTION_LABEL = "<internal-source>";

const SENSITIVE_PATH_PATTERNS = [
  ["absolute Windows Rust source path", /[A-Za-z]:\\(?:[A-Za-z0-9_.@~ -]+\\)+[A-Za-z0-9_.@~-]+\.rs/g],
  ["remapped Cargo Rust source path", /~[\\/](?:[A-Za-z0-9_.@~-]+[\\/])+[A-Za-z0-9_.@~-]+\.rs/g],
  ["Rust toolchain source path", /\/(?:rustc\/[a-f0-9]+\/|rust\/deps\/)(?:[A-Za-z0-9_.@~-]+\/)+[A-Za-z0-9_.@~-]+\.rs/g],
  ["Rust library source path", /library\/(?:[A-Za-z0-9_.@~-]+\/)+[A-Za-z0-9_.@~-]+\.rs/g],
  ["crate-relative Rust source path", /src[\\/](?:[A-Za-z0-9_.@~-]+[\\/])*[A-Za-z0-9_.@~-]+\.rs/g],
];

export function inspectWasmBytes(input) {
  const bytes = asUint8Array(input);
  const sections = parseWasmSections(bytes);
  const module = new WebAssembly.Module(bytes);
  return {
    byteLength: bytes.byteLength,
    sections,
    customSections: sections.filter((section) => section.id === 0).map((section) => section.name || "(unnamed)"),
    sensitivePaths: findSensitivePaths(bytes, sections),
    exports: WebAssembly.Module.exports(module),
    imports: WebAssembly.Module.imports(module),
  };
}

export function inspectWasmFile(filePath) {
  return inspectWasmBytes(readFileSync(filePath));
}

export function hardenWasmBytes(input) {
  const source = asUint8Array(input);
  const before = inspectWasmBytes(source);
  const redacted = new Uint8Array(source);
  const redactions = redactSensitivePaths(redacted, before.sections);
  const retainedParts = [redacted.subarray(0, WASM_HEADER.length)];
  for (const section of before.sections) {
    if (section.id !== 0) retainedParts.push(redacted.subarray(section.sectionStart, section.sectionEnd));
  }
  const hardened = concatBytes(retainedParts);
  const after = inspectWasmBytes(hardened);
  assertRuntimeContractPreserved(before, after);
  if (after.customSections.length) {
    throw new Error(`WASM hardening retained custom sections: ${after.customSections.join(", ")}`);
  }
  if (after.sensitivePaths.length) {
    throw new Error(`WASM hardening retained internal source paths: ${after.sensitivePaths.map((item) => item.value).join(", ")}`);
  }
  return {
    bytes: hardened,
    report: {
      beforeBytes: source.byteLength,
      afterBytes: hardened.byteLength,
      strippedCustomSections: before.customSections,
      redactedPathCount: redactions.length,
      exports: after.exports.length,
      imports: after.imports.length,
    },
  };
}

export function hardenWasmFile(filePath) {
  const result = hardenWasmBytes(readFileSync(filePath));
  writeFileSync(filePath, result.bytes);
  return result.report;
}

function parseWasmSections(bytes) {
  assertWasmHeader(bytes);
  const sections = [];
  let cursor = WASM_HEADER.length;
  while (cursor < bytes.length) {
    const sectionStart = cursor;
    const id = bytes[cursor];
    cursor += 1;
    const sizeField = readUleb(bytes, cursor);
    cursor = sizeField.cursor;
    const payloadStart = cursor;
    const sectionEnd = payloadStart + sizeField.value;
    if (sectionEnd > bytes.length) throw new Error(`WASM section ${id} exceeds the binary boundary.`);
    let name = null;
    if (id === 0) {
      const nameLength = readUleb(bytes, payloadStart);
      const nameEnd = nameLength.cursor + nameLength.value;
      if (nameEnd > sectionEnd) throw new Error("WASM custom-section name exceeds its section boundary.");
      name = new TextDecoder().decode(bytes.subarray(nameLength.cursor, nameEnd));
    }
    sections.push({ id, name, sectionStart, payloadStart, sectionEnd, payloadBytes: sizeField.value });
    cursor = sectionEnd;
  }
  if (cursor !== bytes.length) throw new Error("WASM section parsing did not consume the complete binary.");
  return sections;
}

function findSensitivePaths(bytes, sections) {
  const matches = [];
  for (const section of sections.filter((item) => item.id === 11)) {
    const payloadText = Buffer.from(bytes.subarray(section.payloadStart, section.sectionEnd)).toString("latin1");
    for (const [label, pattern] of SENSITIVE_PATH_PATTERNS) {
      pattern.lastIndex = 0;
      for (const match of payloadText.matchAll(pattern)) {
        matches.push({
          label,
          value: match[0],
          start: section.payloadStart + match.index,
          end: section.payloadStart + match.index + match[0].length,
        });
      }
    }
  }
  return mergeMatches(matches);
}

function redactSensitivePaths(bytes, sections) {
  const matches = findSensitivePaths(bytes, sections);
  for (const match of matches) {
    const replacement = `${REDACTION_LABEL}${"_".repeat(match.end - match.start)}`.slice(0, match.end - match.start);
    for (let offset = 0; offset < replacement.length; offset += 1) {
      bytes[match.start + offset] = replacement.charCodeAt(offset);
    }
  }
  return matches;
}

function mergeMatches(matches) {
  const sorted = matches.sort((left, right) => left.start - right.start || right.end - left.end);
  const merged = [];
  for (const match of sorted) {
    const previous = merged.at(-1);
    if (previous && match.start < previous.end) {
      if (match.end > previous.end) previous.end = match.end;
      previous.label = `${previous.label}; ${match.label}`;
      previous.value = `${previous.value}; ${match.value}`;
    } else {
      merged.push({ ...match });
    }
  }
  return merged;
}

function assertRuntimeContractPreserved(before, after) {
  for (const field of ["exports", "imports"]) {
    if (JSON.stringify(before[field]) !== JSON.stringify(after[field])) {
      throw new Error(`WASM hardening changed the ${field} contract.`);
    }
  }
}

function readUleb(bytes, start) {
  let cursor = start;
  let value = 0;
  let shift = 0;
  while (cursor < bytes.length && shift <= 35) {
    const byte = bytes[cursor];
    cursor += 1;
    value += (byte & 0x7f) * (2 ** shift);
    if ((byte & 0x80) === 0) return { value, cursor };
    shift += 7;
  }
  throw new Error("Invalid WASM unsigned LEB128 value.");
}

function concatBytes(parts) {
  const total = parts.reduce((sum, part) => sum + part.byteLength, 0);
  const output = new Uint8Array(total);
  let offset = 0;
  for (const part of parts) {
    output.set(part, offset);
    offset += part.byteLength;
  }
  return output;
}

function assertWasmHeader(bytes) {
  if (bytes.byteLength < WASM_HEADER.length || WASM_HEADER.some((value, index) => bytes[index] !== value)) {
    throw new Error("Input is not a WebAssembly v1 binary.");
  }
}

function asUint8Array(input) {
  return input instanceof Uint8Array ? input : new Uint8Array(input);
}

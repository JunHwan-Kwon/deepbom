import { createHash } from "node:crypto";
import { readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { ONNX_SHAPE_INFERENCE_OPS } from "../web/lib/onnx-shape-ops.js";
import { fetchPinnedText } from "./fetch-pinned-source.mjs";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const COMMIT = "be2b5fde82d9c8874f3d19328bdfe3b6962dc67b";
const SOURCE_REF = `https://raw.githubusercontent.com/onnx/onnx/${COMMIT}/docs/Changelog.md`;
const SOURCE_SHA256 = "315293e31dd0f415efc7dd821380b53418845f3a719a4930c8df87a30023b6e3";
const OUTPUT = path.join(ROOT, "web", "lib", "onnx-shape-schema-generated.js");

const text = await fetchText(SOURCE_REF);
const digest = sha256(text);
if (digest !== SOURCE_SHA256) throw new Error(`ONNX schema source digest mismatch: ${digest}`);

const histories = parseSchemaHistory(text);
const selected = [...ONNX_SHAPE_INFERENCE_OPS].sort().map((op) => {
  const forms = histories.get(op);
  if (!forms?.length) throw new Error(`Pinned ONNX schema history has no standard-domain ${op} schema.`);
  return [op, forms];
});
const attributeNames = [...new Set(selected.flatMap(([, forms]) => forms.flatMap((form) => form.attributes.map((attr) => attr.name))))].sort();
const attributeIndex = new Map(attributeNames.map((name, index) => [name, index]));
const rows = selected.map(([op, forms]) => [op, forms.map((form) => [
  form.sinceVersion,
  form.inputs.min,
  form.inputs.max,
  form.inputs.options,
  form.outputs.min,
  form.outputs.max,
  form.outputs.options,
  form.attributes.map((attr) => [attributeIndex.get(attr.name), attr.type, attr.required ? 1 : 0]),
])]);
const schemaCount = rows.reduce((sum, [, forms]) => sum + forms.length, 0);
const body = `${header()}
export const ONNX_SHAPE_SCHEMA_SOURCE = Object.freeze(${JSON.stringify({
  release: "v1.21.0",
  commit: COMMIT,
  source_ref: SOURCE_REF,
  sha256: SOURCE_SHA256,
  extraction: "Generated ONNX Changelog OpSchema formal input/output and attribute contracts for the public bounded shape-rule set.",
  op_count: rows.length,
  schema_version_count: schemaCount,
})});

export const ONNX_SHAPE_SCHEMA_ATTRIBUTE_NAMES = Object.freeze(${JSON.stringify(attributeNames)});

export const ONNX_SHAPE_SCHEMA_FORMS = new Map([
${rows.map((row) => `  ${JSON.stringify(row)},`).join("\n")}
]);
`;
if (process.argv.includes("--check")) {
  const current = await readFile(OUTPUT, "utf8");
  if (current !== body) throw new Error(`${path.relative(ROOT, OUTPUT)} is stale; regenerate it from the pinned source.`);
  console.log(`Verified ${path.relative(ROOT, OUTPUT)} with ${rows.length} ops / ${schemaCount} schema versions from ${digest}.`);
} else {
  await writeFile(OUTPUT, body, "utf8");
  console.log(`Generated ${path.relative(ROOT, OUTPUT)} with ${rows.length} ops / ${schemaCount} schema versions from ${digest}.`);
}

function parseSchemaHistory(markdown) {
  const heading = /^### <a name="([^"]+)-(\d+)"><\/a>\*\*[^\n]+\*\*<\/a>\s*$/gm;
  const matches = [...markdown.matchAll(heading)];
  const histories = new Map();
  for (let index = 0; index < matches.length; index += 1) {
    const op = matches[index][1];
    if (!ONNX_SHAPE_INFERENCE_OPS.has(op)) continue;
    const sinceVersion = Number(matches[index][2]);
    const start = matches[index].index;
    const end = index + 1 < matches.length ? matches[index + 1].index : markdown.length;
    const section = markdown.slice(start, end);
    const form = {
      sinceVersion,
      inputs: parseFormalSection(section, "Inputs"),
      outputs: parseFormalSection(section, "Outputs"),
      attributes: parseAttributes(section),
    };
    const forms = histories.get(op) || [];
    forms.push(form);
    histories.set(op, forms);
  }
  for (const [op, forms] of histories) {
    forms.sort((left, right) => left.sinceVersion - right.sinceVersion);
    if (new Set(forms.map((form) => form.sinceVersion)).size !== forms.length) throw new Error(`Duplicate ${op} schema version.`);
  }
  return histories;
}

function parseFormalSection(section, label) {
  const header = new RegExp(`^#### ${label}(?: \\((\\d+) - (\\d+|&#8734;)\\))?\\s*$`, "m").exec(section);
  if (!header) return { min: 0, max: 0, options: "" };
  const body = section.slice(header.index + header[0].length).split(/^#### /m, 1)[0];
  const entries = [...body.matchAll(/^<dt><tt>[^<]+<\/tt>([^:]*)\s*:[^\n]+<\/dt>\s*$/gm)];
  const options = entries.map((match) => {
    const modifiers = match[1].toLowerCase();
    if (modifiers.includes("variadic")) return "V";
    if (modifiers.includes("optional")) return "O";
    return "R";
  }).join("");
  const derivedMin = [...options].filter((option) => option === "R").length;
  const derivedMax = options.includes("V") ? 0 : options.length;
  const min = header[1] == null ? derivedMin : Number(header[1]);
  const max = header[2] == null ? derivedMax : header[2] === "&#8734;" ? 0 : Number(header[2]);
  if (!Number.isSafeInteger(min) || min < 0 || !Number.isSafeInteger(max) || max < 0 || max > 0 && max < min) {
    throw new Error(`Invalid ${label} cardinality ${header[0]}.`);
  }
  if (max > 0 && options.length > max) throw new Error(`${label} formal list exceeds declared maximum.`);
  return { min, max, options };
}

function parseAttributes(section) {
  const header = /^#### Attributes\s*$/m.exec(section);
  if (!header) return [];
  const body = section.slice(header.index + header[0].length).split(/^#### /m, 1)[0];
  return [...body.matchAll(/^<dt><tt>([^<]+)<\/tt>\s*:\s*([^\n]+)<\/dt>\s*$/gm)].map((match) => {
    const descriptor = match[2].trim();
    return {
      name: match[1],
      type: attributeType(descriptor),
      required: /\(required\)\s*$/.test(descriptor),
    };
  }).sort((left, right) => left.name.localeCompare(right.name));
}

function attributeType(descriptor) {
  const normalized = descriptor.toLowerCase().trim();
  const types = new Map([
    ["float", 1], ["int", 2], ["string", 3], ["tensor", 4], ["graph", 5],
    ["list of floats", 6], ["list of ints", 7], ["list of strings", 8],
    ["list of tensors", 9], ["list of graphs", 10], ["sparse_tensor", 11],
    ["list of sparse_tensors", 12], ["type_proto", 13], ["list of type_protos", 14],
  ]);
  const type = [...types].find(([name]) => normalized === name || normalized.startsWith(`${name} (`))?.[1];
  if (!type) throw new Error(`Unsupported generated AttributeProto type: ${descriptor}`);
  return type;
}

async function fetchText(url) {
  return fetchPinnedText(url, { label: "ONNX schema" });
}

function sha256(value) {
  return createHash("sha256").update(value).digest("hex");
}

function header() {
  return "// @generated by scripts/generate-onnx-shape-schema.mjs. Do not edit manually.";
}

import { createHash } from "node:crypto";
import { readFile, writeFile } from "node:fs/promises";

const OUTPUT = "web/lib/executorch-operator-signatures.generated.js";
const CHECK = process.argv.includes("--check");
const SOURCES = Object.freeze({
  executorch: Object.freeze({
    repository: "pytorch/executorch",
    release: "v1.4.1",
    commit: "e4d02f41f7909e8ed5bf4a14ffc520d733453d9f",
    portable_functions_path: "kernels/portable/functions.yaml",
    portable_functions_sha256: "34e3d94cedf2bf54bf9872a076dbc0556e0ff20c0ae390ad510a7f89eb261bad",
    emitter_path: "exir/emit/_emitter.py",
    emitter_sha256: "34bb93bff45fb5c666463c9ab5b95ca222a01dbb2542f25b5713d469e8fa44f5",
    kernel_contract_path: "kernels/README.md",
    kernel_contract_sha256: "6f94b158c3fc9266ad96a0f9153a126a7b15d5f1b0d6f2d4b78a25f3cc226b99",
  }),
  pytorch: Object.freeze({
    repository: "pytorch/pytorch",
    release: "v2.13.0",
    commit: "cf30153c4c131c8164ee7798e5022d810682e2cb",
    native_functions_path: "aten/src/ATen/native/native_functions.yaml",
    native_functions_sha256: "63bd04646fa7ee7a496a8f3095204ae0c8cfcde5b1282789ea92f9faa713491c",
  }),
});

const [portable, emitter, contract, native] = await Promise.all([
  sourceText(SOURCES.executorch, "portable_functions"),
  sourceText(SOURCES.executorch, "emitter"),
  sourceText(SOURCES.executorch, "kernel_contract"),
  sourceText(SOURCES.pytorch, "native_functions"),
]);
requireMarkers(emitter, [
  "for i, schema_arg in enumerate(target._schema.arguments):",
  "kernel_args.append(self._emit_argument(kernel_arg, schema_arg.type).id)",
  "kernel_args.append(cast(_AbstractValue, elem).id)",
  "Instruction(KernelCall(op_index=op_index, args=kernel_args))",
], "ExecuTorch emitter");
requireMarkers(contract, ["**Out variants only**", "final position", "name `out`"], "ExecuTorch kernel contract");

const nativeSchemas = new Map(schemaLines(native).map((schema) => [schemaName(schema), schema]));
const rows = [];
for (const match of portable.matchAll(/^\s*- op:\s*(.+?)\s*$/gm)) {
  const operator = match[1];
  const direct = nativeSchemas.get(operator);
  const functionalName = functionalOperatorName(operator);
  const functional = direct ? null : nativeSchemas.get(functionalName);
  if (!direct && !functional) throw new Error(`${operator}: no pinned PyTorch schema was found.`);
  rows.push(buildRow(operator, direct || functional, direct ? "pytorch_exact_out_schema" : "pytorch_functional_schema_executorch_out_derivation"));
}
for (const schema of schemaLines(portable, "func")) {
  rows.push(buildRow(schemaName(schema), schema, "executorch_custom_schema"));
}
rows.sort((left, right) => left.key.localeCompare(right.key));
if (new Set(rows.map((row) => row.key)).size !== rows.length) throw new Error("ExecuTorch signature keys are not unique.");
if (rows.length !== 209) throw new Error(`Expected 209 ExecuTorch portable signatures; found ${rows.length}.`);

const source = `${header()}export const EXECUTORCH_OPERATOR_SIGNATURE_SOURCE = Object.freeze(${JSON.stringify({
  schema: "deepbom.executorch_operator_signature_source.v1",
  ...SOURCES,
  portable_operator_count: rows.length,
  exact_pytorch_out_schema_count: rows.filter((row) => row.basis === "pytorch_exact_out_schema").length,
  functional_out_derivation_count: rows.filter((row) => row.basis === "pytorch_functional_schema_executorch_out_derivation").length,
  executorch_custom_schema_count: rows.filter((row) => row.basis === "executorch_custom_schema").length,
  argument_encoding: "Pinned ExecuTorch emitter serializes schema arguments in order and then appends flattened return EValues. Portable operators are out-style by the pinned kernel contract.",
}, null, 2)});

export const EXECUTORCH_PORTABLE_OPERATOR_SIGNATURES = Object.freeze(${JSON.stringify(Object.fromEntries(rows.map((row) => [row.key, Object.freeze(row)])), null, 2)});
`;

if (CHECK) {
  const current = await readFile(OUTPUT, "utf8");
  if (current !== source) throw new Error(`${OUTPUT} is stale; run node scripts/generate-executorch-operator-signatures.mjs.`);
  console.log(`Verified ${rows.length} pinned ExecuTorch portable operator signatures.`);
} else {
  await writeFile(OUTPUT, source);
  console.log(`Wrote ${OUTPUT}: ${rows.length} pinned ExecuTorch portable operator signatures.`);
}

function buildRow(operator, sourceSchema, basis) {
  const parsed = parseSchema(sourceSchema);
  const derived = basis === "pytorch_functional_schema_executorch_out_derivation";
  const explicitOutputArguments = parsed.arguments.filter((argument) => argument.outputOnly).map((argument) => argument.index);
  const returnCount = derived ? parsed.returns.length : explicitOutputArguments.length || parsed.returns.length;
  if (!returnCount) throw new Error(`${operator}: portable operator has no serialized output contract.`);
  const outputArguments = derived
    ? Array.from({ length: returnCount }, (_, index) => parsed.arguments.length + index)
    : explicitOutputArguments;
  const inoutArguments = derived ? [] : parsed.arguments.filter((argument) => argument.inout).map((argument) => argument.index);
  const argumentCount = parsed.arguments.length + (derived ? returnCount : 0);
  const tensorInputs = parsed.arguments.filter((argument) => argument.tensorLike && !argument.outputOnly).map((argument) => argument.index);
  if (!outputArguments.length && !inoutArguments.length) {
    throw new Error(`${operator}: output or in/out tensor argument cannot be reconstructed.`);
  }
  return {
    key: serializedKey(operator),
    operator,
    basis,
    source_schema: sourceSchema,
    argument_count: argumentCount,
    appended_return_count: returnCount,
    tensor_input_argument_positions: tensorInputs,
    tensor_output_argument_positions: outputArguments,
    tensor_inout_argument_positions: inoutArguments,
    nominal_mac_rule: nominalMacRule(operator),
  };
}

function parseSchema(schema) {
  const open = schema.indexOf("(");
  const close = schema.lastIndexOf(") ->");
  if (open <= 0 || close <= open) throw new Error(`Invalid operator schema: ${schema}`);
  const argumentTokens = splitTopLevel(schema.slice(open + 1, close));
  let keywordOnly = false;
  const arguments_ = [];
  for (const token of argumentTokens) {
    if (token === "*") { keywordOnly = true; continue; }
    const declaration = stripDefault(token);
    const match = declaration.match(/^(.*?)\s+([A-Za-z_]\w*)$/);
    if (!match) throw new Error(`Cannot parse schema argument ${JSON.stringify(token)} in ${schema}.`);
    const type = match[1].trim();
    const tensorLike = /^Tensor(?:\?|\[|\()/.test(type) || type === "Tensor";
    const mutable = tensorLike && /Tensor\([^)]*!/.test(type);
    const outputOnly = mutable && keywordOnly;
    arguments_.push({ index: arguments_.length, tensorLike, outputOnly, inout: mutable && !outputOnly });
  }
  const returnText = schema.slice(close + 4).trim();
  const returns = returnText === "()" ? [] : returnText.startsWith("(") && returnText.endsWith(")")
    ? splitTopLevel(returnText.slice(1, -1)) : [returnText];
  return { arguments: arguments_, returns };
}

function splitTopLevel(value) {
  const output = [];
  let depth = 0;
  let quote = "";
  let start = 0;
  for (let index = 0; index < value.length; index += 1) {
    const character = value[index];
    if (quote) {
      if (character === quote && value[index - 1] !== "\\") quote = "";
    } else if (character === "\"" || character === "'") quote = character;
    else if ("([{<".includes(character)) depth += 1;
    else if (")]}>".includes(character)) depth -= 1;
    else if (character === "," && depth === 0) {
      output.push(value.slice(start, index).trim());
      start = index + 1;
    }
  }
  output.push(value.slice(start).trim());
  return output.filter(Boolean);
}

function stripDefault(value) {
  let depth = 0;
  for (let index = 0; index < value.length; index += 1) {
    const character = value[index];
    if ("([{<".includes(character)) depth += 1;
    else if (")]}>".includes(character)) depth -= 1;
    else if (character === "=" && depth === 0) return value.slice(0, index).trim();
  }
  return value.trim();
}

function schemaLines(source, field = "func") {
  return [...source.matchAll(new RegExp(`^\\s*- ${field}:\\s*(.+?)\\s*$`, "gm"))].map((match) => match[1]);
}

function schemaName(schema) { return schema.slice(0, schema.indexOf("(")); }
function functionalOperatorName(operator) {
  if (operator.endsWith(".out")) return operator.slice(0, -4);
  if (operator.endsWith("_out")) return operator.slice(0, -4);
  return operator;
}
function serializedKey(operator) {
  const dot = operator.indexOf(".");
  const name = dot < 0 ? operator : operator.slice(0, dot);
  const overload = dot < 0 ? "" : operator.slice(dot + 1);
  return `${name.includes("::") ? name : `aten::${name}`}${overload ? `.${overload}` : ""}`;
}
function nominalMacRule(operator) {
  if (["mm.out", "bmm.out", "addmm.out", "convolution.out"].includes(operator)) return operator.split(".")[0];
  if (operator === "convolution_backward.out") return "convolution_backward";
  return "zero_nominal_tensor_contraction_macs";
}
async function sourceText(source, name) {
  const path = source[`${name}_path`];
  const expected = source[`${name}_sha256`];
  const url = `https://raw.githubusercontent.com/${source.repository}/${source.commit}/${path}`;
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${url}: HTTP ${response.status}`);
  const bytes = Buffer.from(await response.arrayBuffer());
  const observed = createHash("sha256").update(bytes).digest("hex");
  if (observed !== expected) throw new Error(`${path}: SHA-256 ${observed} != ${expected}.`);
  return bytes.toString("utf8");
}
function requireMarkers(source, markers, label) {
  for (const marker of markers) if (!source.includes(marker)) throw new Error(`${label} lacks pinned marker ${JSON.stringify(marker)}.`);
}
function header() { return "// Generated by scripts/generate-executorch-operator-signatures.mjs. Do not edit.\n"; }

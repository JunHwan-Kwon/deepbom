import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { readFile } from "node:fs/promises";

import Ajv2020 from "ajv/dist/2020.js";

import { analyzeExecuTorchModel } from "../web/executorch.js";
import { getArtifactIrContext } from "../web/lib/artifact-ir-context.js";
import {
  buildModelSummary,
  compactModelSummaryForConversation,
  renderModelSummaryMarkdown,
  renderModelSummaryTable,
  validateModelSummary,
} from "../web/lib/model-summary.js";
import { sha256BytesHex } from "../web/lib/sha256-sync.js";
import { decodeFixtureBase64, EXECUTORCH_ADD_PTE_BASE64 } from "./fixtures/executorch-fixtures.mjs";

const schema = JSON.parse(await readFile("docs/schemas/deepbom-model-summary-v1.schema.json", "utf8"));
const validateSchema = new Ajv2020({ allErrors: true, strict: true }).compile(schema);

const cases = [
  { format: "onnx", file: "web/samples/gpu_partition_probe.onnx", level: "operation" },
  { format: "tflite", file: "web/samples/mobilenet_v2_1.0_224_quant.tflite", level: "operation" },
  { format: "coreml", file: "web/samples/MNISTClassifier.mlmodel", level: "operation" },
  { format: "gguf", file: "web/samples/tinymqa1m.Q4_0.gguf", level: "storage" },
  { format: "safetensors", file: "web/samples/nanofable-1m-fp16.safetensors", level: "storage" },
];

for (const entry of cases) {
  const modelIr = graph(entry.file).model_ir;
  const first = buildModelSummary(modelIr);
  const second = buildModelSummary(modelIr);
  assert.deepEqual(first, second, `${entry.format}: deterministic projection`);
  assert.deepEqual(validateModelSummary(first), first, `${entry.format}: semantic and digest validation`);
  assert.equal(validateSchema(first), true, `${entry.format}: JSON Schema validation: ${JSON.stringify(validateSchema.errors)}`);
  assert.equal(first.schema, "deepbom.model_summary.v1");
  assert.equal(first.projection.selected_level, entry.level, `${entry.format}: auto projection`);
  assert.equal(first.source_contract.model_ir_sha256, modelIr.model_ir_sha256, `${entry.format}: Model IR binding`);
  assert.equal(first.trainability.trainable_parameter_count, null, `${entry.format}: trainability boundary`);
  assert.equal(first.projection.ordering.runtime_order_claim, false, `${entry.format}: runtime order boundary`);
  assert.equal(first.coverage.unknown_is_zero, false, `${entry.format}: unknown boundary`);
  assert.equal(first.totals.operation_count, modelIr.program.operations.length, `${entry.format}: operation conservation`);
  assert.equal(first.totals.storage_object_count, modelIr.tensors_and_storage.storage_objects.length, `${entry.format}: storage conservation`);
  assert.equal(new Set(first.rows.map((row) => row.subject_ref)).size, first.rows.length, `${entry.format}: row identity uniqueness`);
  if (entry.level === "operation") {
    assert.equal(first.rows.length, modelIr.program.operations.length, `${entry.format}: operation row conservation`);
    assert(first.rows.every((row) => row.kind === "operation" && row.order.runtime === null), `${entry.format}: operation row contract`);
  } else {
    assert.equal(first.rows.length, modelIr.tensors_and_storage.storage_objects.length, `${entry.format}: storage row conservation`);
    assert(first.rows.every((row) => row.kind === "serialized_storage" && row.predecessor_relationships.length === 0), `${entry.format}: graphless rows must not fabricate execution edges`);
  }
  const compact = compactModelSummaryForConversation(first, { maximumRows: 2 });
  assert.equal(compact.row_count, first.rows.length);
  assert.equal(compact.rows.length, Math.min(2, first.rows.length));
  assert.equal(compact.truncated, first.rows.length > 2);
  assert.match(renderModelSummaryTable(first), /Trainability: not_assessable/);
  assert.match(renderModelSummaryMarkdown(first), /Evidence boundary:/);
}

const executorchBytes = decodeFixtureBase64(EXECUTORCH_ADD_PTE_BASE64);
const executorchSha256 = sha256BytesHex(executorchBytes);
const executorchAnalysis = analyzeExecuTorchModel(executorchBytes, "add.pte");
const executorchSummary = getArtifactIrContext(executorchAnalysis, {
  filename: "add.pte", format: "executorch", sha256: executorchSha256, size: executorchBytes.length,
}).model_summary;
assert.equal(executorchSummary.projection.selected_level, "operation", "ExecuTorch auto summary level");
assert.equal(executorchSummary.rows.length, 1, "ExecuTorch operation row conservation");
assert.equal(validateSchema(executorchSummary), true, `ExecuTorch JSON Schema validation: ${JSON.stringify(validateSchema.errors)}`);

const onnx = graph(cases[0].file).model_ir;
const block = buildModelSummary(onnx, { level: "block" });
assert.equal(block.rows.flatMap((row) => row.member_refs).length, onnx.program.operations.length, "block projection member conservation");
assert.equal(new Set(block.rows.flatMap((row) => row.member_refs)).size, onnx.program.operations.length, "block projection member uniqueness");

const cliJson = run(["model-summary", cases[0].file, "--format", "json-compact"]);
const cliDocument = JSON.parse(cliJson.stdout);
assert.equal(cliDocument.schema, "deepbom.model_summary.v1");
assert.equal(cliDocument.artifact.sha256, cases.length ? buildModelSummary(onnx).artifact.sha256 : null);
assert.match(run(["model-summary", cases[0].file, "--format", "table"]).stdout, /ORDER\tNAME\tNATIVE TYPE/);
assert.match(run(["model-summary", cases[0].file, "--format", "markdown"]).stdout, /\| Order \| Name \| Native type \|/);
assert.equal(run(["audit", cases[0].file, "--section", "model_summary", "--compact"]).status, 0);

const tampered = structuredClone(buildModelSummary(onnx));
tampered.model_summary_sha256 = "0".repeat(64);
assert.throws(() => validateModelSummary(tampered), /SHA-256 is invalid/);

console.log("Format-neutral Model Summary checks passed (graph, graphless storage, block, digest, CLI table/Markdown/JSON, and bounded MCP projection).\n");

function graph(file) {
  const result = run(["graph", file, "--format", "json", "--compact"]);
  return JSON.parse(result.stdout);
}

function run(args) {
  const result = spawnSync(process.execPath, ["bin/deepbom.mjs", ...args], { encoding: "utf8", maxBuffer: 512 * 1024 * 1024, timeout: 180_000 });
  assert.equal(result.status, 0, `${args.join(" ")} failed: ${result.stderr || result.stdout}`);
  return result;
}

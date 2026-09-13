import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { readFile } from "node:fs/promises";
import path from "node:path";

import Ajv2020 from "ajv/dist/2020.js";

import { validateModelIr } from "../web/lib/model-ir.js";
import { buildStructuralBlocks } from "../web/lib/model-ir/internal/blocks.js";

const root = path.resolve(".");
const schema = JSON.parse(await readFile(path.join(root, "docs/schemas/deepbom-model-ir-v1.schema.json"), "utf8"));
const validateSchema = new Ajv2020({ allErrors: true, strict: true }).compile(schema);
const cases = [
  { format: "onnx", file: "web/samples/sample_cnn_float.onnx", graph: true },
  { format: "tflite", file: "web/samples/mobilenet_v2_1.0_224_quant.tflite", graph: true },
  { format: "gguf", file: "web/samples/tinymqa1m.Q4_0.gguf", graph: false },
  { format: "safetensors", file: "web/samples/nanofable-1m-fp16.safetensors", graph: false },
  { format: "coreml", file: "web/samples/MNISTClassifier.mlmodel", graph: true },
  { format: "onnx-recursive", file: "scripts/fixtures/onnx_recursive_scope.onnx", graph: true },
];

const outputs = new Map();
for (const entry of cases) {
  const output = runGraph(entry.file);
  const modelIr = output.model_ir;
  outputs.set(entry.format, modelIr);
  assert.equal(modelIr.schema, "deepbom.model_ir.v1", `${entry.format} schema`);
  assert.equal(modelIr.method_version, "1.0.0", `${entry.format} method version`);
  assert.equal(modelIr.source_contract.schema, "deepbom.artifact_ir.v2", `${entry.format} source schema`);
  assert.equal(modelIr.source_contract.sha256, output.artifact_ir.artifact_ir_sha256, `${entry.format} source digest binding`);
  assert.equal(validateSchema(modelIr), true, `${entry.format} JSON Schema: ${errors(validateSchema.errors)}`);
  assert.deepEqual(validateModelIr(modelIr), modelIr, `${entry.format} semantic and digest validation`);
  assert.equal(modelIr.program.status, entry.graph ? "serialized" : "not_serialized", `${entry.format} graph applicability`);
  assert.equal(modelIr.evidence_vocabulary.unknown_is_zero, false, `${entry.format} unknown boundary`);
  assert.equal(modelIr.evidence_vocabulary.static_prediction_is_runtime_observation, false, `${entry.format} static/runtime boundary`);
  assert.equal(modelIr.completeness.loss_count, modelIr.loss_ledger.entries.length, `${entry.format} loss conservation`);
  assert.equal(modelIr.generic_analysis.pass_contract.format_specific_branch_count, 0, `${entry.format} generic passes have no format branch`);
  assert.equal(modelIr.generic_analysis.pass_contract.native_ledger_access, false, `${entry.format} generic passes do not read native ledgers`);
  assert.equal(modelIr.generic_analysis.pass_count, 5, `${entry.format} generic pass count`);
  assert(modelIr.profiles.every((row) => row.applicability && row.status), `${entry.format} explicit profile applicability`);
  for (const projection of modelIr.static_runtime.projections) {
    assert(projection.segments.every((row) => row.actual_runtime_assignment_claim === false), `${entry.format} static projection claim boundary`);
  }

  if (entry.graph) {
    assert.equal(modelIr.program.operations.length, output.artifact_ir.graph.operators.length, `${entry.format} operation conservation`);
    assert.equal(modelIr.program.values.length, output.artifact_ir.graph.values.length, `${entry.format} value conservation`);
    assert.equal(modelIr.program.blocks.flatMap((row) => row.member_refs).length, modelIr.program.operations.length, `${entry.format} structural block member conservation`);
    assert.equal(new Set(modelIr.program.blocks.flatMap((row) => row.member_refs)).size, modelIr.program.operations.length, `${entry.format} structural block member uniqueness`);
    assert(modelIr.program.operations.every((row) => Number.isSafeInteger(row.source_order)), `${entry.format} source order`);
    assert(modelIr.program.operations.every((row) => Number.isSafeInteger(row.display_order)), `${entry.format} deterministic display order`);
    const operationIds = new Set(modelIr.program.operations.map((row) => row.id));
    for (const relation of modelIr.program.relationships.filter((row) => row.kind === "data_dependency")) {
      assert(operationIds.has(relation.from_ref) && operationIds.has(relation.to_ref), `${entry.format} data dependency references`);
    }
  } else {
    assert.equal(modelIr.program.programs.length, 0, `${entry.format} no fabricated program`);
    assert.equal(modelIr.program.operations.length, 0, `${entry.format} no fabricated operations`);
    assert.equal(modelIr.program.relationships.length, 0, `${entry.format} no fabricated execution relationships`);
    assert(modelIr.loss_ledger.entries.some((row) => row.subject === "program_graph"), `${entry.format} explicit graph loss`);
  }
  if (entry.format === "gguf") {
    const transformer = modelIr.architecture.model_profiles.find((row) => row.id === "deepbom.transformer_profile.v1");
    assert(transformer, "GGUF transformer storage profile must be recognized");
    assert.equal(transformer.evidence_class, "DERIVED", "GGUF transformer profile must remain derived from storage names");
    assert.equal(modelIr.program.relationships.length, 0, "GGUF model-profile recognition must not create execution edges");
  }
  if (entry.format === "onnx") {
    const cnn = modelIr.architecture.model_profiles.find((row) => row.id === "deepbom.cnn_profile.v1");
    assert(cnn, "ONNX convolution profile must be recognized from serialized operations");
    assert.equal(cnn.evidence_class, "OBSERVED_SERIALIZED_ARTIFACT");
  }
}

const deterministic = runGraph(cases[0].file).model_ir;
assert.equal(JSON.stringify(deterministic), JSON.stringify(outputs.get("onnx")), "Model IR must be deterministic");

const recursive = outputs.get("onnx-recursive");
assert.equal(recursive.program.regions.length, 6, "nested region conservation");
assert(recursive.program.relationships.some((row) => row.native_relationship === "serialized_nested_graph_ownership"), "nested region ownership conservation");

const region = [{ id: "region:test" }];
const ports = [{ id: "port:op:input:0", value_ref: "value:w" }];
const baseOperation = {
  id: "op", region_ref: "region:test", native_index: 0, source_order: 0, display_order: 0,
  native_op: { domain: "test", name: "MatMul", version: 1 }, input_port_refs: [ports[0].id], output_port_refs: [],
};
const structureQ4 = buildStructuralBlocks([{ ...baseOperation, quantization_summary: { encoding: "Q4" } }], region,
  { ports, values: [{ id: "value:w", dtype: "INT4", shape: [4, 8], storage_refs: ["storage:w"] }], relationships: [] })[0];
const structureQ5 = buildStructuralBlocks([{ ...baseOperation, quantization_summary: { encoding: "Q5" } }], region,
  { ports, values: [{ id: "value:w", dtype: "INT5", shape: [4, 8], storage_refs: ["storage:w"] }], relationships: [] })[0];
const differentShape = buildStructuralBlocks([{ ...baseOperation, quantization_summary: { encoding: "Q4" } }], region,
  { ports, values: [{ id: "value:w", dtype: "INT4", shape: [4, 16], storage_refs: ["storage:w"] }], relationships: [] })[0];
assert.equal(structureQ4.id, structureQ5.id, "representation-only changes must retain the structural block id");
assert.equal(structureQ4.signature_sha256, structureQ5.signature_sha256);
assert.notEqual(structureQ4.representation_signature_sha256, structureQ5.representation_signature_sha256,
  "representation-only changes must remain visible in a separate digest");
assert.notEqual(structureQ4.signature_sha256, differentShape.signature_sha256,
  "a changed parameter shape pattern must change the structural signature");

const tamperedDigest = structuredClone(outputs.get("onnx"));
tamperedDigest.model_ir_sha256 = "0".repeat(64);
assert.throws(() => validateModelIr(tamperedDigest), /SHA-256 is invalid/, "tampered digest must fail closed");

const fabricatedGraph = structuredClone(outputs.get("gguf"));
fabricatedGraph.program.operations.push({ id: "fabricated" });
assert.equal(validateSchema(fabricatedGraph), false, "schema must reject graphless fabricated operation");
assert.throws(() => validateModelIr(fabricatedGraph), /operation region reference|fabricated a program/, "semantic validator must reject graphless fabricated operation");

const promotedStatic = structuredClone(outputs.get("tflite"));
if (promotedStatic.static_runtime.projections.length) {
  promotedStatic.static_runtime.projections[0].segments[0].actual_runtime_assignment_claim = true;
  delete promotedStatic.model_ir_sha256;
  assert.throws(() => validateModelIrBodyThroughDigest(promotedStatic), /actual runtime assignment/, "static projection promotion must fail closed");
}

console.log("Common Model IR checks passed (6 serialized/graphless fixtures; schema, order, source binding, loss, static/runtime, and tamper contracts).\n");

function runGraph(file) {
  const result = spawnSync(process.execPath, ["bin/deepbom.mjs", "graph", file, "--format", "json", "--compact"], {
    cwd: root, encoding: "utf8", maxBuffer: 512 * 1024 * 1024, timeout: 120_000,
  });
  assert.equal(result.status, 0, `graph command failed for ${file}: ${result.stderr || result.stdout}`);
  return JSON.parse(result.stdout);
}

function errors(rows) { return (rows || []).map((row) => `${row.instancePath || "/"} ${row.message}`).join("; "); }
function validateModelIrBodyThroughDigest(body) {
  return validateModelIr({ ...body, model_ir_sha256: "0".repeat(64) });
}

import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { readFile } from "node:fs/promises";
import path from "node:path";

import Ajv2020 from "ajv/dist/2020.js";

import { validateArtifactEvidenceIr } from "../web/lib/artifact-ir.js";

const root = path.resolve(".");
const schema = JSON.parse(await readFile(path.join(root, "docs", "schemas", "deepbom-artifact-ir-v2.schema.json"), "utf8"));
const validateSchema = new Ajv2020({ allErrors: true, strict: true }).compile(schema);
const cases = [
  { format: "onnx", file: "web/samples/sample_cnn_float.onnx", graph: true, operators: 9, values: 16, macs: "6488384" },
  { format: "tflite", file: "web/samples/mobilenet_v2_1.0_224_quant.tflite", graph: true, operators: 65, values: 173, macs: "300775552" },
  { format: "gguf", file: "web/samples/tinymqa1m.Q4_0.gguf", graph: false },
  { format: "safetensors", file: "web/samples/nanofable-1m-fp16.safetensors", graph: false },
  { format: "coreml", file: "web/samples/MNISTClassifier.mlmodel", graph: true, operators: 14, values: 16 },
];

const outputs = new Map();
for (const entry of cases) {
  const output = runGraph(entry.file);
  const artifactIr = output.artifact_ir;
  const graphIr = output.graph_ir;
  outputs.set(entry.format, output);

  assert.equal(artifactIr.schema, "deepbom.artifact_ir.v2", `${entry.format} Artifact IR schema`);
  assert.equal(artifactIr.method_version, "2.0.0", `${entry.format} Artifact IR method version`);
  assert.deepEqual(artifactIr.hash_contract.excluded_pointers, ["/artifact_ir_sha256"], `${entry.format} self-hash exclusion contract`);
  assert.equal(validateSchema(artifactIr), true, `${entry.format} JSON Schema validation: ${formatAjvErrors(validateSchema.errors)}`);
  assert.deepEqual(validateArtifactEvidenceIr(artifactIr), artifactIr, `${entry.format} semantic and digest validation`);
  assert.equal(graphIr.artifact_ir_schema, artifactIr.schema, `${entry.format} Graph IR schema binding`);
  assert.equal(graphIr.artifact_ir_sha256, artifactIr.artifact_ir_sha256, `${entry.format} Graph IR digest binding`);
  assert.equal(artifactIr.graph.status, entry.graph ? "serialized" : "not_serialized", `${entry.format} graph applicability`);
  assert.equal(artifactIr.graph.totals.operator_count, artifactIr.graph.operators.length, `${entry.format} operator conservation`);
  assert.equal(artifactIr.graph.totals.value_count, artifactIr.graph.values.length, `${entry.format} value conservation`);
  assert.equal(artifactIr.storage_topology.totals.object_count, artifactIr.storage_topology.objects.length, `${entry.format} storage-object conservation`);
  assert(artifactIr.storage_topology.objects.every((row) => BigInt(row.serialized_byte_length?.decimal || "0") > 0n), `${entry.format} storage topology must not contain zero-byte pseudo-objects`);
  assert.equal(artifactIr.architecture_projection.totals.relationship_count, artifactIr.architecture_projection.relationships.length, `${entry.format} architecture relationship conservation`);
  assert.equal(artifactIr.graph.operators.every((row) => !("placement" in row)), true, `${entry.format} canonical operators must not embed placement projections`);

  const storageIds = new Set(artifactIr.storage_topology.objects.map((row) => row.id));
  for (const value of artifactIr.graph.values) {
    for (const storageRef of value.storage_refs) assert(storageIds.has(storageRef), `${entry.format} value storage reference ${storageRef}`);
  }
  const quantSubjects = new Set([
    ...artifactIr.graph.values.map((row) => row.id),
    ...artifactIr.storage_topology.objects.map((row) => row.id),
  ]);
  for (const record of artifactIr.quantization_contracts.records) assert(quantSubjects.has(record.subject_ref), `${entry.format} quantization subject ${record.subject_ref}`);

  if (entry.graph) {
    assert.equal(artifactIr.graph.totals.operator_count, entry.operators, `${entry.format} expected operator count`);
    assert.equal(artifactIr.graph.totals.value_count, entry.values, `${entry.format} expected value count`);
    if (entry.macs) assert.equal(artifactIr.graph.totals.assessed_macs.decimal, entry.macs, `${entry.format} expected assessed MACs`);
  } else {
    assert.equal(artifactIr.graph.scopes.length, 0, `${entry.format} graphless scope count`);
    assert.equal(artifactIr.graph.operators.length, 0, `${entry.format} graphless operator count`);
    assert.equal(artifactIr.graph.values.length, 0, `${entry.format} graphless value count`);
    assert.equal(artifactIr.architecture_projection.relationships.length, 0, `${entry.format} architecture order must not become an execution relationship`);
    assert.equal(graphIr.edges.length, 0, `${entry.format} compatibility projection must not fabricate edges`);
    assert.equal(graphIr.projection.executable_dag_claim, false, `${entry.format} compatibility projection executable-DAG boundary`);
  }
}

const onnxFirst = outputs.get("onnx");
const onnxSecond = runGraph(cases[0].file);
assert.equal(JSON.stringify(onnxFirst), JSON.stringify(onnxSecond), "Artifact IR and compatibility projection must be deterministic");

const tflite = outputs.get("tflite").artifact_ir;
assert.equal(tflite.quantization_contracts.totals.record_count, 172, "TFLite scoped quantization record count");
assert(tflite.graph.values.some((row) => row.storage_refs.length), "TFLite constants must connect canonical values to serialized storage objects");
assert(tflite.overlays.static.some((row) => row.kind === "static_placement"), "TFLite placement must be emitted as a separate static overlay");

const coreml = outputs.get("coreml").artifact_ir;
assert(coreml.storage_topology.objects.length > 0, "Core ML serialized parameters must be represented as storage objects");
assert(coreml.storage_topology.objects.length < coreml.graph.values.length, "Core ML logical graph values must not be reclassified wholesale as serialized payload objects");

const tamperedCount = structuredClone(onnxFirst.artifact_ir);
tamperedCount.graph.totals.operator_count += 1;
assert.throws(() => validateArtifactEvidenceIr(tamperedCount), /count conservation/, "tampered graph count must fail closed");

const tamperedStorageReference = structuredClone(onnxFirst.artifact_ir);
tamperedStorageReference.graph.values[0].storage_refs = ["storage:missing"];
assert.throws(() => validateArtifactEvidenceIr(tamperedStorageReference), /unknown storage object/, "unknown storage reference must fail closed");

const tamperedDigest = structuredClone(onnxFirst.artifact_ir);
tamperedDigest.artifact_ir_sha256 = "0".repeat(64);
assert.throws(() => validateArtifactEvidenceIr(tamperedDigest), /SHA-256 is invalid/, "tampered Artifact IR digest must fail closed");

console.log(`Artifact Evidence IR checks passed (${cases.length} real artifacts; schema, identity, conservation, overlay, graphless, and tamper contracts).`);

function runGraph(file) {
  const result = spawnSync(process.execPath, ["bin/deepbom.mjs", "graph", file, "--format", "json", "--compact"], {
    cwd: root,
    encoding: "utf8",
    maxBuffer: 256 * 1024 * 1024,
    timeout: 120_000,
  });
  assert.equal(result.status, 0, `graph command failed for ${file}: ${result.stderr || result.stdout}`);
  return JSON.parse(result.stdout);
}

function formatAjvErrors(errors) {
  return (errors || []).map((row) => `${row.instancePath || "/"} ${row.message}`).join("; ");
}

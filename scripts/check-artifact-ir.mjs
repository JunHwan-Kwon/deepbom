import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";
import path from "node:path";

import Ajv2020 from "ajv/dist/2020.js";

import { analyzeExecuTorchModel } from "../web/executorch.js";
import { buildGraphDiffSnapshot } from "../web/lib/artifact-diff.js";
import { buildArtifactEvidenceIr, validateArtifactEvidenceIr } from "../web/lib/artifact-ir.js";
import { getArtifactIrContext } from "../web/lib/artifact-ir-context.js";
import { decodeFixtureBase64, EXECUTORCH_ADD_PTE_BASE64 } from "./fixtures/executorch-fixtures.mjs";

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
  assert.equal(artifactIr.method_version, "2.1.0", `${entry.format} Artifact IR method version`);
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

const recursiveOnnx = runGraph("scripts/fixtures/onnx_recursive_scope.onnx");
assert.equal(validateSchema(recursiveOnnx.artifact_ir), true, `recursive ONNX JSON Schema validation: ${formatAjvErrors(validateSchema.errors)}`);
assert.deepEqual(validateArtifactEvidenceIr(recursiveOnnx.artifact_ir), recursiveOnnx.artifact_ir, "recursive ONNX semantic validation");
assert.equal(recursiveOnnx.artifact_ir.graph.totals.scope_count, 6, "recursive ONNX serialized scope count");
assert.equal(recursiveOnnx.artifact_ir.graph.totals.materialized_scope_count, 6, "recursive ONNX materialized scope count");
assert.equal(recursiveOnnx.artifact_ir.graph.totals.operator_count, 11, "recursive ONNX all-scope operator count");
assert.equal(recursiveOnnx.artifact_ir.graph.totals.value_count, 27, "recursive ONNX all-scope value count");
assert.equal(recursiveOnnx.artifact_ir.graph.totals.scope_relationship_count, 5, "recursive ONNX scope ownership count");
const nestedGraphRelationships = recursiveOnnx.artifact_ir.graph.scope_relationships
  .filter((row) => row.role.startsWith("node_attribute:"));
assert(nestedGraphRelationships.length > 0, "recursive ONNX fixture must contain operator-owned nested graphs");
assert(nestedGraphRelationships.every((row) => row.source_operator_ref), "operator-owned ONNX nested graphs must retain their serialized owner operator");
assert.equal(recursiveOnnx.graph_ir.totals.node_count, 4, "Graph IR v1 must project only the primary ONNX scope");
assert.equal(recursiveOnnx.graph_ir.projection.compatibility_status, "primary_scope_projection_only", "Graph IR v1 compatibility boundary");
assert.equal(recursiveOnnx.graph_ir.projection.omitted_materialized_scope_count, 5, "Graph IR v1 omitted nested-scope count");

const executorchBytes = decodeFixtureBase64(EXECUTORCH_ADD_PTE_BASE64);
const executorchAnalysis = analyzeExecuTorchModel(executorchBytes, "add.pte");
const executorchSha256 = createHash("sha256").update(executorchBytes).digest("hex");
const executorch = buildArtifactEvidenceIr(executorchAnalysis, { filename: "add.pte", format: "executorch", sha256: executorchSha256, size: executorchBytes.length });
assert.equal(validateSchema(executorch), true, `ExecuTorch JSON Schema validation: ${formatAjvErrors(validateSchema.errors)}`);
assert.deepEqual(validateArtifactEvidenceIr(executorch), executorch, "ExecuTorch semantic validation");
assert.equal(executorch.graph.totals.scope_count, 1, "ExecuTorch scope count");
assert.equal(executorch.graph.totals.operator_count, 1, "ExecuTorch operator count");
assert.equal(executorch.graph.totals.value_count, 3, "ExecuTorch value count");

const runtimeEvidence = {
  artifact_sha256: "e".repeat(64),
  runtime_nodes: [
    { runtime_node_ref: "fused:0", backend: "example_ep", source_subject_refs: ["operator:scope:onnx:main_graph:0", "operator:scope:onnx:main_graph:1"] },
    { runtime_node_ref: "generated:1", backend: "example_ep", source_subject_refs: [], runtime_node_kind: "generated_copy" },
  ],
};
const runtimeAnalysis = {
  format: "onnx",
  filename: "runtime-reconciliation.onnx",
  file_size: 16,
  ops: [
    { index: 0, name: "Add", inputs: [0, 1], outputs: [2], macs: 0, macs_status: "assessed" },
    { index: 1, name: "Relu", inputs: [2], outputs: [3], macs: 0, macs_status: "assessed" },
  ],
  tensors: [0, 1, 2, 3].map((index) => ({ index, name: `v${index}`, dtype: "FLOAT32", shape: [1] })),
  input_tensor_indices: [0, 1],
  output_tensor_indices: [3],
};
const runtimeBound = buildArtifactEvidenceIr(runtimeAnalysis, { filename: runtimeAnalysis.filename, format: "onnx", sha256: "e".repeat(64), size: 16 }, { runtimeEvidence });
const runtimeOverlay = runtimeBound.overlays.runtime[0];
assert.equal(validateSchema(runtimeBound), true, `runtime reconciliation JSON Schema validation: ${formatAjvErrors(validateSchema.errors)}`);
assert.equal(runtimeOverlay.summary.runtime_node_count, 2, "runtime reconciliation node count");
assert.equal(runtimeOverlay.summary.fused_runtime_node_count, 1, "runtime reconciliation fused-node count");
assert.equal(runtimeOverlay.summary.unmapped_runtime_node_count, 1, "runtime reconciliation unmapped-node count");
assert.equal(runtimeOverlay.summary.source_subject_reference_count, 2, "runtime reconciliation source-reference count");
assert.equal(runtimeOverlay.summary.name_similarity_mapping_used, false, "runtime reconciliation must not name-match");
assert.equal(runtimeOverlay.rows.length, 2, "runtime reconciliation flattened assignment conservation");
assert.throws(() => buildArtifactEvidenceIr(
  runtimeAnalysis,
  { filename: runtimeAnalysis.filename, format: "onnx", sha256: "e".repeat(64), size: 16 },
  { runtimeEvidence: { ...runtimeEvidence, artifact_sha256: "f".repeat(64) } },
), /not bound to the active artifact SHA-256/, "runtime evidence from another artifact must fail closed");

const legacyUnboundRuntime = buildArtifactEvidenceIr(runtimeAnalysis, { filename: runtimeAnalysis.filename, format: "onnx", sha256: "e".repeat(64), size: 16 }, {
  runtimeEvidence: { artifact_sha256: "e".repeat(64), rows: [{ node_name: "Add_0", provider: "example_ep" }] },
});
assert.equal(legacyUnboundRuntime.overlays.runtime.length, 0, "runtime rows without canonical subject refs must remain unreconciled");

const indexedFusion = buildArtifactEvidenceIr(runtimeAnalysis, { filename: runtimeAnalysis.filename, format: "onnx", sha256: "e".repeat(64), size: 16 }, {
  runtimeEvidence: {
    artifact_sha256: "e".repeat(64),
    assignments: [
      { runtime_node_name: "fused_add_relu", runtime_node_index: 0, provider: "example_ep", op_index: 0 },
      { runtime_node_name: "fused_add_relu", runtime_node_index: 0, provider: "example_ep", op_index: 1 },
    ],
  },
});
assert.equal(indexedFusion.overlays.runtime[0].summary.fused_runtime_node_count, 1, "artifact-bound op indices must reconcile an explicit fused runtime node");
assert.equal(indexedFusion.overlays.runtime[0].runtime_nodes[0].mapping_basis, "explicit_primary_scope_native_index_import", "native-index runtime mapping basis");
assert.equal(indexedFusion.overlays.runtime[0].summary.name_similarity_mapping_used, false, "native-index reconciliation must not use name similarity");

const runtimeContext = getArtifactIrContext(runtimeAnalysis, { filename: runtimeAnalysis.filename, format: "onnx", sha256: "e".repeat(64), size: 16 }, {
  runtimeEvidence: indexedFusion.overlays.runtime.length ? {
    artifact_sha256: "e".repeat(64),
    runtime_nodes: indexedFusion.overlays.runtime[0].runtime_nodes,
  } : null,
});
assert.equal(runtimeContext.primary_view.artifact_ir, runtimeContext.artifact_ir, "shared consumer view must retain its canonical Artifact IR binding");
assert.equal(runtimeContext.primary_view.ops[0].artifact_ir_subject_ref, "operator:scope:onnx:main_graph:0", "shared consumer op identity");
runtimeAnalysis.findings = [{ id: "EA-TEST-0001" }];
assert.equal(runtimeContext.primary_view.findings[0].id, "EA-TEST-0001", "shared consumer view must expose post-build finding updates without rebuilding the IR");
const tamperedView = structuredClone(runtimeContext.primary_view);
tamperedView.ops[0].name = "INCORRECT_NATIVE_LABEL";
const canonicalDiffSnapshot = buildGraphDiffSnapshot(tamperedView);
assert.equal(canonicalDiffSnapshot.nodes[0].name, "Add", "Artifact diff must prefer canonical IR operator identity over stale native display data");

const tamperedCount = structuredClone(onnxFirst.artifact_ir);
tamperedCount.graph.totals.operator_count += 1;
assert.throws(() => validateArtifactEvidenceIr(tamperedCount), /count conservation/, "tampered graph count must fail closed");

const tamperedStorageReference = structuredClone(onnxFirst.artifact_ir);
tamperedStorageReference.graph.values[0].storage_refs = ["storage:missing"];
assert.throws(() => validateArtifactEvidenceIr(tamperedStorageReference), /unknown storage object/, "unknown storage reference must fail closed");

const tamperedCrossScopePort = structuredClone(recursiveOnnx.artifact_ir);
const nestedOperator = tamperedCrossScopePort.graph.operators.find((row) => row.scope_ref !== tamperedCrossScopePort.graph.primary_scope_ref && row.inputs.length);
nestedOperator.inputs[0].value_ref = tamperedCrossScopePort.graph.inputs[0];
assert.throws(() => validateArtifactEvidenceIr(tamperedCrossScopePort), /crosses graph scopes/, "cross-scope operator port must fail closed");

const tamperedScopeCount = structuredClone(recursiveOnnx.artifact_ir);
const declaredScope = tamperedScopeCount.graph.scopes.find((row) => row.materialization_status === "materialized" && row.declared_operator_count != null);
declaredScope.declared_operator_count += 1;
assert.throws(() => validateArtifactEvidenceIr(tamperedScopeCount), /scope operator count/, "materialized scope declaration mismatch must fail closed");

const tamperedDigest = structuredClone(onnxFirst.artifact_ir);
tamperedDigest.artifact_ir_sha256 = "0".repeat(64);
assert.throws(() => validateArtifactEvidenceIr(tamperedDigest), /SHA-256 is invalid/, "tampered Artifact IR digest must fail closed");

console.log("Artifact Evidence IR checks passed (7 serialized artifact fixtures; schema, identity, scope conservation, explicit runtime reconciliation, graphless, and tamper contracts).");

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

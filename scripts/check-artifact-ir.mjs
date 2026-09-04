import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";
import path from "node:path";

import Ajv2020 from "ajv/dist/2020.js";

import { analyzeExecuTorchModel } from "../web/executorch.js";
import { analyzeOnnxModel } from "../web/onnx.js";
import { buildGraphDiffSnapshot } from "../web/lib/artifact-diff.js";
import { validateArtifactEvidenceIr } from "../web/lib/artifact-ir.js";
import { getArtifactIrContext, isArtifactIrConsumerView, resolveArtifactIrContext } from "../web/lib/artifact-ir-context.js";
import { buildEngineeringEvidenceDocument, buildRawDataArtifactFiles } from "../web/lib/report-evidence.js";
import { buildDeploymentContractDocuments, DEPLOYMENT_CONTRACT_FILES } from "../web/lib/report-export-contracts.js";
import { buildMlBomDocument } from "../web/lib/report-mlbom.js";
import { buildPublicCycloneDxDocuments } from "../web/lib/public-cyclonedx-export.js";
import { buildReviewState, buildSelfContainedReviewHtml } from "../web/lib/review-export.js";
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
  assert.equal(artifactIr.method_version, "2.2.0", `${entry.format} Artifact IR method version`);
  assert.equal(artifactIr.lineage_evidence.status, "not_provided", `${entry.format} absent conversion receipt status`);
  assert.equal(artifactIr.lineage_evidence.evidence_class, "NOT_ASSESSABLE", `${entry.format} absent conversion receipt evidence class`);
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

for (const format of ["gguf", "safetensors"]) {
  const analysis = {
    filename: `graphless.${format}`,
    format,
    model_sha256: format === "gguf" ? "a".repeat(64) : "b".repeat(64),
    file_size_bytes: 16,
    ops: [],
    tensors: [],
  };
  const context = getArtifactIrContext(analysis, {
    filename: analysis.filename,
    format,
    sha256: analysis.model_sha256,
    size: analysis.file_size_bytes,
  });
  assert.equal(isArtifactIrConsumerView(context.primary_view), true, `${format} graphless consumers must retain the canonical Artifact IR binding`);
  assert.equal(context.primary_view.artifact_ir, context.artifact_ir, `${format} graphless consumer Artifact IR identity`);
  assert.equal(context.primary_view.artifact_ir_primary_scope_ref, null, `${format} graphless consumer primary scope`);
  assert.equal(context.primary_view.artifact_ir_nested_scope_count, 0, `${format} graphless consumer nested scope count`);
}

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
const executorch = buildIr(executorchAnalysis, { filename: "add.pte", format: "executorch", sha256: executorchSha256, size: executorchBytes.length });
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
const repeatedInputAnalysis = structuredClone(runtimeAnalysis);
repeatedInputAnalysis.filename = "repeated-input-port.onnx";
repeatedInputAnalysis.ops[0].inputs = [0, 0];
const repeatedInputContext = getArtifactIrContext(repeatedInputAnalysis, {
  filename: repeatedInputAnalysis.filename,
  format: "onnx",
  sha256: "d".repeat(64),
  size: 16,
});
assert.equal(new Set(repeatedInputContext.graph_ir.edges.map((row) => row.id)).size, repeatedInputContext.graph_ir.edges.length,
  "Graph IR compatibility edges must retain consumer-port identity when one value feeds multiple ports of the same operator");
assert.deepEqual(repeatedInputContext.graph_ir.edges.filter((row) => row.tensor_index === 0).map((row) => row.target_port), [0, 1],
  "Graph IR compatibility edges must preserve repeated input port indices");
const runtimeBound = buildIr(runtimeAnalysis, { filename: runtimeAnalysis.filename, format: "onnx", sha256: "e".repeat(64), size: 16 }, { runtimeEvidence });
const runtimeOverlay = runtimeBound.overlays.runtime[0];
assert.equal(validateSchema(runtimeBound), true, `runtime reconciliation JSON Schema validation: ${formatAjvErrors(validateSchema.errors)}`);
assert.equal(runtimeOverlay.summary.runtime_node_count, 2, "runtime reconciliation node count");
assert.equal(runtimeOverlay.summary.fused_runtime_node_count, 1, "runtime reconciliation fused-node count");
assert.equal(runtimeOverlay.summary.unmapped_runtime_node_count, 1, "runtime reconciliation unmapped-node count");
assert.equal(runtimeOverlay.summary.source_subject_reference_count, 2, "runtime reconciliation source-reference count");
assert.equal(runtimeOverlay.summary.name_similarity_mapping_used, false, "runtime reconciliation must not name-match");
assert.equal(runtimeOverlay.rows.length, 2, "runtime reconciliation flattened assignment conservation");
assert.throws(() => buildIr(
  runtimeAnalysis,
  { filename: runtimeAnalysis.filename, format: "onnx", sha256: "e".repeat(64), size: 16 },
  { runtimeEvidence: { ...runtimeEvidence, artifact_sha256: "f".repeat(64) } },
), /not bound to the active artifact SHA-256/, "runtime evidence from another artifact must fail closed");

const legacyUnboundRuntime = buildIr(runtimeAnalysis, { filename: runtimeAnalysis.filename, format: "onnx", sha256: "e".repeat(64), size: 16 }, {
  runtimeEvidence: { artifact_sha256: "e".repeat(64), rows: [{ node_name: "Add_0", provider: "example_ep" }] },
});
assert.equal(legacyUnboundRuntime.overlays.runtime.length, 0, "runtime rows without canonical subject refs must remain unreconciled");

const indexedFusion = buildIr(runtimeAnalysis, { filename: runtimeAnalysis.filename, format: "onnx", sha256: "e".repeat(64), size: 16 }, {
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
assert.equal(isArtifactIrConsumerView(runtimeContext.primary_view), true, "serialized consumer view must be marked as Artifact IR-backed");
assert.equal(runtimeContext.primary_view.ops[0].artifact_ir_subject_ref, "operator:scope:onnx:main_graph:0", "shared consumer op identity");
const crossOutputBytes = await readFile(path.join(root, "web/samples/sample_cnn_float.onnx"));
const crossOutputSha = createHash("sha256").update(crossOutputBytes).digest("hex");
const crossOutputAnalysis = analyzeOnnxModel(new Uint8Array(crossOutputBytes), "sample_cnn_float.onnx");
crossOutputAnalysis.model_sha256 = crossOutputSha;
crossOutputAnalysis.file_size_bytes = crossOutputBytes.length;
const crossOutputRuntimeEvidence = {
  artifact_sha256: crossOutputSha,
  runtime_nodes: [{
    runtime_node_ref: "fused:cross-output:0",
    backend: "fixture_ep",
    source_subject_refs: ["operator:scope:onnx:main_graph:0", "operator:scope:onnx:main_graph:1"],
  }],
};
const crossOutputContext = getArtifactIrContext(crossOutputAnalysis, {
  filename: crossOutputAnalysis.filename,
  format: "onnx",
  sha256: crossOutputSha,
  size: crossOutputBytes.length,
}, { runtimeEvidence: crossOutputRuntimeEvidence });
const runtimeIdentity = { filename: crossOutputAnalysis.filename, format: "onnx", sha256: crossOutputSha, byte_length: crossOutputBytes.length };
const reportRuntimeEvidence = {};
const reportContext = { identity: runtimeIdentity, runtimeEvidence: reportRuntimeEvidence, artifactIrContext: crossOutputContext };
const rawEvidenceContext = { ...reportContext };
const crossOutputMlBom = buildMlBomDocument(crossOutputContext.primary_view, {
  hash: crossOutputSha,
  fileSizeBytes: crossOutputBytes.length,
  artifactIr: crossOutputContext.artifact_ir,
  timestamp: "2026-09-02T00:00:00.000Z",
});
const engineeringEvidence = buildEngineeringEvidenceDocument(crossOutputContext.primary_view, {
  reportContext,
  rawEvidenceContext,
  mlBomDocument: crossOutputMlBom,
});
const rawFiles = buildRawDataArtifactFiles(crossOutputContext.primary_view, { rawEvidenceContext });
const rawArtifactIr = JSON.parse(rawFiles.find((row) => row.name === "static/artifact_ir.json").data);
const deploymentContracts = buildDeploymentContractDocuments(crossOutputContext.primary_view, {
  hash: crossOutputSha, fileSizeBytes: crossOutputBytes.length, generatedAt: "2026-09-02T00:00:00.000Z", artifactIrContext: crossOutputContext,
});
const publicCycloneDx = buildPublicCycloneDxDocuments(crossOutputContext.primary_view, {
  hash: crossOutputSha, fileSizeBytes: crossOutputBytes.length, generatedAt: "2026-09-02T00:00:00.000Z", artifactIr: crossOutputContext.artifact_ir,
}).documents.cyclonedx_evidence;
const reviewState = buildReviewState({ analysis: crossOutputContext.primary_view, runtimeEvidence: reportRuntimeEvidence });
const reviewHtml = buildSelfContainedReviewHtml({
  analysis: crossOutputContext.primary_view,
  graphSvg: '<svg xmlns="http://www.w3.org/2000/svg" width="1" height="1"></svg>',
  reviewState,
  runtimeEvidence: reportRuntimeEvidence,
});
const embeddedReviewState = JSON.parse(reviewHtml.match(/<script id="deepbom-review-state" type="application\/json">([^<]+)<\/script>/)?.[1] || "null");
const expectedIrSha = crossOutputContext.artifact_ir.artifact_ir_sha256;
const outputIrShas = {
  ui: crossOutputContext.artifact_ir.artifact_ir_sha256,
  engineering_report: engineeringEvidence.evidence.artifact_ir.artifact_ir_sha256,
  raw_zip: rawArtifactIr.artifact_ir_sha256,
  deployment_pack: deploymentContracts.documents.artifact_ir.artifact_ir_sha256,
  public_cyclonedx: propertyValue(publicCycloneDx.properties, "deepbom:artifactIrSha256"),
  review: reviewState.artifact_ir_identity.sha256,
};
assert.deepEqual(new Set(Object.values(outputIrShas)), new Set([expectedIrSha]), `cross-output Artifact IR identity: ${JSON.stringify(outputIrShas)}`);
for (const [surface, artifactIr] of [
  ["ui", crossOutputContext.artifact_ir],
  ["engineering report", engineeringEvidence.evidence.artifact_ir],
  ["raw ZIP", rawArtifactIr],
  ["deployment pack", deploymentContracts.documents.artifact_ir],
]) {
  assert(artifactIr.overlays.runtime.length > 0, `${surface} must preserve the runtime overlay rather than reproducing an empty IR`);
  assert.equal(artifactIr.overlays.runtime[0].rows.length, 2, `${surface} runtime subject-reference conservation`);
  assertRuntimeSubjectsResolve(artifactIr, surface);
}
assert.equal(reviewState.artifact_ir_identity.runtime_overlay_count, 1, "review state runtime-overlay conservation");
assert.deepEqual(reviewState.artifact_ir_identity.runtime_subject_refs,
  ["operator:scope:onnx:main_graph:0", "operator:scope:onnx:main_graph:1"], "review state runtime subject-reference conservation");
assert.equal(embeddedReviewState?.artifact_ir_identity?.sha256, expectedIrSha, "rendered review.html Artifact IR digest conservation");
assert.equal(embeddedReviewState?.artifact_ir_identity?.runtime_overlay_count, 1, "rendered review.html runtime-overlay conservation");
assert.deepEqual(embeddedReviewState?.artifact_ir_identity?.runtime_subject_refs,
  reviewState.artifact_ir_identity.runtime_subject_refs, "rendered review.html runtime subject-reference conservation");
const packagedCycloneDx = deploymentContracts.documents.cyclonedx_evidence;
const artifactIrReference = packagedCycloneDx.metadata.component.externalReferences
  .find((row) => row.url === DEPLOYMENT_CONTRACT_FILES.artifactIr);
assert(artifactIrReference, "packaged CycloneDX must reference its Artifact IR sibling");
assert.equal(artifactIrReference.hashes?.[0]?.content,
  deploymentContracts.integrity.member_sha256[DEPLOYMENT_CONTRACT_FILES.artifactIr],
  "packaged CycloneDX Artifact IR sibling digest conservation");
assert.equal(propertyValue(packagedCycloneDx.metadata.component.properties, "deepbom:model:artifactIrLocation"),
  DEPLOYMENT_CONTRACT_FILES.artifactIr, "packaged CycloneDX Artifact IR location conservation");
assert.equal(reviewHtml.includes("<script id=\"deepbom-review-state\" type=\"application/json\">"), true,
  "review.html must expose a non-executable machine-readable review state");

const recursiveBytes = await readFile(path.join(root, "scripts/fixtures/onnx_recursive_scope.onnx"));
const recursiveSha = createHash("sha256").update(recursiveBytes).digest("hex");
const recursiveAnalysis = analyzeOnnxModel(new Uint8Array(recursiveBytes), "onnx_recursive_scope.onnx");
recursiveAnalysis.model_sha256 = recursiveSha;
recursiveAnalysis.file_size_bytes = recursiveBytes.length;
const nestedFusionRefs = ["operator:scope:onnx:main_graph:1", "operator:scope:onnx:nested:2:0"];
const recursiveContext = getArtifactIrContext(recursiveAnalysis, {
  filename: recursiveAnalysis.filename,
  format: "onnx",
  sha256: recursiveSha,
  size: recursiveBytes.length,
}, { runtimeEvidence: {
  artifact_sha256: recursiveSha,
  runtime_nodes: [{
    runtime_node_ref: "fused:nested-scope:0",
    backend: "fixture_ep",
    source_subject_refs: nestedFusionRefs,
  }],
} });
const recursiveSurfaces = materializeArtifactIrSurfaces(recursiveAnalysis, recursiveContext, recursiveSha, recursiveBytes.length);
const recursiveExpectedSha = recursiveContext.artifact_ir.artifact_ir_sha256;
assert.deepEqual(new Set(Object.values(recursiveSurfaces.irShas)), new Set([recursiveExpectedSha]),
  `nested-scope runtime-fusion Artifact IR identity: ${JSON.stringify(recursiveSurfaces.irShas)}`);
for (const [surface, artifactIr] of recursiveSurfaces.materializedIr) {
  assert.equal(artifactIr.graph.totals.scope_count, 6, `${surface} nested-scope conservation`);
  assert.equal(artifactIr.graph.totals.materialized_scope_count, 6, `${surface} materialized-scope conservation`);
  assert.equal(artifactIr.overlays.runtime.length, 1, `${surface} nested runtime-overlay conservation`);
  assert.equal(artifactIr.overlays.runtime[0].summary.fused_runtime_node_count, 1, `${surface} runtime-fusion conservation`);
  assert.deepEqual(artifactIr.overlays.runtime[0].runtime_nodes[0].source_subject_refs, nestedFusionRefs,
    `${surface} cross-scope runtime subject conservation`);
  assertRuntimeSubjectsResolve(artifactIr, surface);
}
assert.equal(recursiveSurfaces.reviewState.artifact_ir_identity.nested_scope_count, 5,
  "review state nested-scope conservation for runtime-bound IR");
assert.equal(recursiveSurfaces.reviewState.artifact_ir_identity.runtime_overlay_count, 1,
  "review state nested runtime-overlay conservation");
assert.deepEqual(recursiveSurfaces.reviewState.artifact_ir_identity.runtime_subject_refs, nestedFusionRefs,
  "review state cross-scope runtime subject conservation");
assert.equal(recursiveSurfaces.embeddedReviewState.artifact_ir_identity.nested_scope_count, 5,
  "rendered review.html nested-scope conservation");
assert.equal(recursiveSurfaces.embeddedReviewState.artifact_ir_identity.runtime_overlay_count, 1,
  "rendered review.html nested runtime-overlay conservation");
assert.deepEqual(recursiveSurfaces.embeddedReviewState.artifact_ir_identity.runtime_subject_refs, nestedFusionRefs,
  "rendered review.html cross-scope runtime subject conservation");
const recursiveArtifactIrReference = recursiveSurfaces.packagedCycloneDx.metadata.component.externalReferences
  .find((row) => row.url === DEPLOYMENT_CONTRACT_FILES.artifactIr);
assert(recursiveArtifactIrReference, "nested-scope packaged CycloneDX must reference its Artifact IR sibling");
assert.equal(recursiveArtifactIrReference.hashes?.[0]?.content,
  recursiveSurfaces.deploymentContracts.integrity.member_sha256[DEPLOYMENT_CONTRACT_FILES.artifactIr],
  "nested-scope packaged CycloneDX Artifact IR sibling digest conservation");
runtimeAnalysis.findings = [{ id: "EA-TEST-0001" }];
assert.equal(runtimeContext.primary_view.findings[0].id, "EA-TEST-0001", "shared consumer view must expose post-build finding updates without rebuilding the IR");
const tamperedView = structuredClone(runtimeContext.primary_view);
tamperedView.ops[0].name = "INCORRECT_NATIVE_LABEL";
const canonicalDiffSnapshot = buildGraphDiffSnapshot(tamperedView);
assert.equal(canonicalDiffSnapshot.nodes[0].name, "Add", "Artifact diff must prefer canonical IR operator identity over stale native display data");

const tamperedCount = structuredClone(onnxFirst.artifact_ir);
tamperedCount.graph.totals.operator_count += 1;
assert.throws(() => validateArtifactEvidenceIr(tamperedCount), /count conservation/, "tampered graph count must fail closed");

const fabricatedGraphlessTotals = structuredClone(outputs.get("gguf").artifact_ir);
fabricatedGraphlessTotals.graph.totals.operator_count = 1;
assert.equal(validateSchema(fabricatedGraphlessTotals), false, "public schema must reject non-zero graph totals for graphless containers");
const fabricatedGraphlessScope = structuredClone(outputs.get("safetensors").artifact_ir);
fabricatedGraphlessScope.graph.primary_scope_ref = "scope:invented";
assert.equal(validateSchema(fabricatedGraphlessScope), false, "public schema must reject a primary executable scope for graphless containers");

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

const runtimeRemovedWithStaleDigest = structuredClone(crossOutputContext.artifact_ir);
runtimeRemovedWithStaleDigest.overlays.runtime = [];
runtimeRemovedWithStaleDigest.completeness.runtime_overlay_count = 0;
assert.throws(() => validateArtifactEvidenceIr(runtimeRemovedWithStaleDigest), /SHA-256 is invalid/,
  "runtime removal under a stale Artifact IR digest must fail closed");

assert.equal(resolveArtifactIrContext(crossOutputAnalysis, runtimeIdentity, {
  artifactIrContext: crossOutputContext,
  runtimeEvidence: crossOutputRuntimeEvidence,
}), crossOutputContext, "matching runtime evidence may reuse the canonical Artifact IR context");
assert.throws(() => resolveArtifactIrContext(crossOutputAnalysis, runtimeIdentity, {
  artifactIrContext: crossOutputContext,
  runtimeEvidence: {
    ...crossOutputRuntimeEvidence,
    runtime_nodes: crossOutputRuntimeEvidence.runtime_nodes.map((row) => ({ ...row, backend: "different_ep" })),
  },
}), /stale for the supplied runtime evidence/, "stale Artifact IR context injection must fail closed");
assert.throws(() => resolveArtifactIrContext(crossOutputAnalysis, runtimeIdentity, {
  artifactIr: outputs.get("onnx").artifact_ir,
  runtimeEvidence: crossOutputRuntimeEvidence,
}), /runtime overlay is stale|not bound to the active artifact/, "runtime evidence must not bind to an Artifact IR without its normalized overlay");

const mismatchedArtifactIr = structuredClone(crossOutputContext.artifact_ir);
mismatchedArtifactIr.artifact.sha256 = "f".repeat(64);
delete mismatchedArtifactIr.artifact_ir_sha256;
const mismatchedBody = structuredClone(mismatchedArtifactIr);
mismatchedArtifactIr.artifact_ir_sha256 = createHash("sha256").update(canonicalJsonForTest(mismatchedBody)).digest("hex");
assert.throws(() => resolveArtifactIrContext(crossOutputAnalysis, {
  filename: crossOutputAnalysis.filename,
  format: "onnx",
  sha256: crossOutputSha,
  size: crossOutputBytes.length,
}, { artifactIr: mismatchedArtifactIr }), /not bound to the active artifact|Artifact IR/, "mismatched Artifact IR identity must fail closed");

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

function buildIr(analysis, artifact, options = {}) {
  const context = getArtifactIrContext(analysis, artifact, options);
  assert(context, "Artifact IR context must be constructible for the fixture");
  return context.artifact_ir;
}

function formatAjvErrors(errors) {
  return (errors || []).map((row) => `${row.instancePath || "/"} ${row.message}`).join("; ");
}

function propertyValue(properties, name) {
  return (properties || []).find((row) => row.name === name)?.value || null;
}

function assertRuntimeSubjectsResolve(artifactIr, surface) {
  const subjects = new Set([
    ...artifactIr.graph.operators.map((row) => row.id),
    ...artifactIr.architecture_projection.nodes.map((row) => row.id),
  ]);
  for (const overlay of artifactIr.overlays.runtime) {
    for (const row of overlay.rows) assert(subjects.has(row.subject_ref), `${surface} contains unresolved runtime subject ${row.subject_ref}`);
  }
}

function materializeArtifactIrSurfaces(analysis, artifactIrContext, sha256, byteLength) {
  const identity = { filename: analysis.filename, format: analysis.format, sha256, byte_length: byteLength };
  const reportContext = { identity, runtimeEvidence: {}, artifactIrContext };
  const rawEvidenceContext = { ...reportContext };
  const mlBomDocument = buildMlBomDocument(artifactIrContext.primary_view, {
    hash: sha256,
    fileSizeBytes: byteLength,
    artifactIr: artifactIrContext.artifact_ir,
    timestamp: "2026-09-02T00:00:00.000Z",
  });
  const engineeringEvidence = buildEngineeringEvidenceDocument(artifactIrContext.primary_view, {
    reportContext,
    rawEvidenceContext,
    mlBomDocument,
  });
  const rawFiles = buildRawDataArtifactFiles(artifactIrContext.primary_view, { rawEvidenceContext });
  const rawArtifactIr = JSON.parse(rawFiles.find((row) => row.name === "static/artifact_ir.json").data);
  const deploymentContracts = buildDeploymentContractDocuments(artifactIrContext.primary_view, {
    hash: sha256,
    fileSizeBytes: byteLength,
    generatedAt: "2026-09-02T00:00:00.000Z",
    artifactIrContext,
  });
  const publicCycloneDx = buildPublicCycloneDxDocuments(artifactIrContext.primary_view, {
    hash: sha256,
    fileSizeBytes: byteLength,
    generatedAt: "2026-09-02T00:00:00.000Z",
    artifactIr: artifactIrContext.artifact_ir,
  }).documents.cyclonedx_evidence;
  const reviewState = buildReviewState({ analysis: artifactIrContext.primary_view, runtimeEvidence: {} });
  const reviewHtml = buildSelfContainedReviewHtml({
    analysis: artifactIrContext.primary_view,
    graphSvg: '<svg xmlns="http://www.w3.org/2000/svg" width="1" height="1"></svg>',
    reviewState,
    runtimeEvidence: {},
  });
  const embeddedReviewState = JSON.parse(reviewHtml.match(/<script id="deepbom-review-state" type="application\/json">([^<]+)<\/script>/)?.[1] || "null");
  const packagedCycloneDx = deploymentContracts.documents.cyclonedx_evidence;
  return {
    deploymentContracts,
    packagedCycloneDx,
    reviewState,
    embeddedReviewState,
    materializedIr: [
      ["UI", artifactIrContext.artifact_ir],
      ["engineering report", engineeringEvidence.evidence.artifact_ir],
      ["raw ZIP", rawArtifactIr],
      ["deployment pack", deploymentContracts.documents.artifact_ir],
    ],
    irShas: {
      ui: artifactIrContext.artifact_ir.artifact_ir_sha256,
      engineering_report: engineeringEvidence.evidence.artifact_ir.artifact_ir_sha256,
      raw_zip: rawArtifactIr.artifact_ir_sha256,
      deployment_pack: deploymentContracts.documents.artifact_ir.artifact_ir_sha256,
      public_cyclonedx: propertyValue(publicCycloneDx.properties, "deepbom:artifactIrSha256"),
      review: reviewState.artifact_ir_identity.sha256,
    },
  };
}

function canonicalJsonForTest(value) {
  if (value === null || typeof value !== "object") return JSON.stringify(value);
  if (Array.isArray(value)) return `[${value.map(canonicalJsonForTest).join(",")}]`;
  return `{${Object.keys(value).sort().map((key) => `${JSON.stringify(key)}:${canonicalJsonForTest(value[key])}`).join(",")}}`;
}

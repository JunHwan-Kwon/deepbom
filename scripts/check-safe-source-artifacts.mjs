import assert from "node:assert/strict";
import { mkdir, mkdtemp, rm, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { spawnSync } from "node:child_process";

import { buildStoreZip } from "../bin/store-zip.mjs";
import { readArtifactBundle } from "../web/lib/artifact-bundle.js";
import {
  analyzeHdf5Envelope,
  analyzeLegacyPyTorchEnvelope,
  analyzeTensorFlowProtobuf,
  analyzeZipModelEnvelope,
  readHdf5EnvelopeFile,
  readZipModelEnvelopeFile,
} from "../web/lib/safe-source-artifacts.js";
import { getArtifactIrContext } from "../web/lib/artifact-ir-context.js";
import { buildModelIrVisualizationBundle } from "../web/lib/model-ir-visualization.js";
import { sha256BytesHex } from "../web/lib/sha256-sync.js";

const graphDef = message([
  field(1, node("x", "Placeholder", [], [attribute("dtype", message([varintField(6, 1)])), attribute("shape", message([field(7, tensorShape([1, 1]))]))])),
  field(1, node("w", "Const", [], [attribute("value", message([field(8, tensorProto(1, [1, 1], Uint8Array.of(0, 0, 128, 63)))]))])),
  field(1, node("mat", "MatMul", ["x", "w"])),
  field(1, node("after", "NoOp", ["^mat"])),
]);
const graphAnalysis = analyzeTensorFlowProtobuf(graphDef, "graph.pb");
assert.equal(graphAnalysis.format, "graphdef");
assert.equal(graphAnalysis.operator_count, 4);
assert.equal(graphAnalysis.tensor_count, 2);
assert.equal(graphAnalysis.graphdef.control_dependency_count, 1);
assert.equal(graphAnalysis.tensors.find((row) => row.name === "x:0").dtype, "FLOAT32");
assert.deepEqual(graphAnalysis.tensors.find((row) => row.name === "x:0").shape, [1, 1]);
assert.equal(graphAnalysis.tensors.find((row) => row.name === "w:0").serialized_payload_bytes, 4);
assert.deepEqual(graphAnalysis.input_tensor_indices, []);
assert.deepEqual(graphAnalysis.output_tensor_indices, []);
assert.match(graphAnalysis.graphdef.interpretation_boundary, /does not explicitly identify model inputs or outputs/);

const servingSignature = message([
  field(1, stringMapEntry("features", tensorInfo("x:0", 1, [1, 1]))),
  field(2, stringMapEntry("scores", tensorInfo("mat:0", 1, [1, 1]))),
  field(3, bytes("tensorflow/serving/predict")),
]);
const savedModel = message([field(1, varint(1)), field(2, message([
  field(2, graphDef),
  field(5, stringMapEntry("serving_default", servingSignature)),
]))]);
const savedAnalysis = analyzeTensorFlowProtobuf(savedModel, "saved_model.pb");
assert.equal(savedAnalysis.format, "savedmodel");
assert.equal(savedAnalysis.savedmodel.meta_graph_count, 1);
assert.equal(savedAnalysis.savedmodel.signature_defs[0].key, "serving_default");
assert.equal(savedAnalysis.savedmodel.signature_defs[0].inputs[0].tensor_name, "x:0");
assert.deepEqual(savedAnalysis.input_tensor_indices, [0]);
assert.deepEqual(savedAnalysis.output_tensor_indices, [2]);
assert.equal(savedAnalysis.savedmodel.package_closure_status, "protobuf_only_variables_assets_and_fingerprints_not_bound");

const hdf5 = new Uint8Array(512);
hdf5.set([0x89,0x48,0x44,0x46,0x0d,0x0a,0x1a,0x0a,2,8,8,0]);
hdf5.fill(0xff, 20, 28);
writeLe(hdf5, 28, 512n, 8);
writeLe(hdf5, 36, 64n, 8);
const hdfAnalysis = analyzeHdf5Envelope(hdf5, "weights.h5");
assert.equal(hdfAnalysis.safe_source_artifact.superblock_version, 2);
assert.equal(hdfAnalysis.safe_source_artifact.end_of_file_address, "512");
assert.equal(hdfAnalysis.safe_source_artifact.status, "superblock_assessed_object_graph_not_materialized");

const kerasZip = buildStoreZip([
  { name: "config.json", data: bytes(JSON.stringify({ class_name: "Sequential", config: { layers: [
    { class_name: "InputLayer", config: { name: "input", batch_shape: [null, 4], dtype: "float32" } },
    { class_name: "Dense", config: { name: "dense", dtype: "float32" }, build_config: { input_shape: [null, 4] } },
  ] } })) },
  { name: "metadata.json", data: bytes('{"keras_version":"3"}') },
  { name: "model.weights.h5", data: hdf5 },
]);
const kerasAnalysis = await analyzeZipModelEnvelope(kerasZip, "model.keras", "keras");
assert.equal(kerasAnalysis.safe_source_artifact.layer_inventory.layers[1].class_name, "Dense");
assert.equal(kerasAnalysis.keras.graph_materialized, true);
assert.equal(kerasAnalysis.operator_count, 2);
assert.equal(kerasAnalysis.safe_source_artifact.unsafe_deserialization_performed, false);

const pt2Zip = buildStoreZip([
  { name: "package/archive_format", data: bytes("pt2") },
  { name: "package/archive_version", data: bytes("0") },
  { name: "package/models/model.json", data: bytes(JSON.stringify({
    graph_module: { graph: {
      inputs: [{ as_tensor: { name: "x" } }], outputs: [{ as_tensor: { name: "y" } }],
      nodes: [{ target: "torch.ops.aten.relu.default", name: "relu", inputs: [{ name: "self", arg: { as_tensor: { name: "x" } } }], outputs: [{ as_tensor: { name: "y" } }], metadata: {} }],
      tensor_values: { x: { dtype: 7, sizes: [{ as_int: 2 }] }, y: { dtype: 7, sizes: [{ as_int: 2 }] } },
      sym_int_values: {}, sym_bool_values: {}, is_single_tensor_return: false,
    }, signature: {}, module_call_graph: [], metadata: {} },
    opset_version: { aten: 0 }, range_constraints: {}, schema_version: { major: 8, minor: 7 }, torch_version: "test",
  })) },
  { name: "package/data/sample_inputs/model.pt", data: Uint8Array.of(0x80, 0x02, 0x4e, 0x2e) },
]);
const pt2Analysis = await analyzeZipModelEnvelope(pt2Zip, "model.pt2", "pt2");
assert.equal(pt2Analysis.safe_source_artifact.archive_format, "pt2");
assert.equal(pt2Analysis.safe_source_artifact.declarative_documents.length, 1);
assert.equal(pt2Analysis.operator_count, 1);
assert.equal(pt2Analysis.tensor_count, 2);
assert.equal(pt2Analysis.pt2.graph_materialized, true);

const pickle = Uint8Array.of(0x80,0x02,0x63,0x66,0x6f,0x6f,0x0a,0x42,0x61,0x72,0x0a,0x2e);
const checkpointZip = buildStoreZip([
  { name: "archive/data.pkl", data: pickle },
  { name: "archive/data/0", data: Uint8Array.of(0, 1, 2, 3) },
]);
const checkpoint = await analyzeZipModelEnvelope(checkpointZip, "model.pt", "pytorch_checkpoint");
assert.equal(checkpoint.safe_source_artifact.pickle_members[0].pickle.literal_global_references[0].module, "foo");
assert.equal(checkpoint.safe_source_artifact.pickle_members[0].pickle.stop_opcode_observed, true);
assert.equal(checkpoint.safe_source_artifact.unsafe_deserialization_performed, false);
const legacy = await analyzeLegacyPyTorchEnvelope(pickle, "legacy.pth");
assert.equal(legacy.safe_source_artifact.pickle.literal_global_count, 1);

const digest = sha256BytesHex(graphDef);
graphAnalysis.model_sha256 = digest;
graphAnalysis.file_size = graphDef.length;
graphAnalysis.file_size_bytes = graphDef.length;
const context = getArtifactIrContext(graphAnalysis, { filename: "graph.pb", format: "graphdef", size: graphDef.length, sha256: digest });
assert.equal(context.artifact_ir.graph.status, "serialized");
assert.equal(context.model_ir.program.relationships.filter((row) => row.kind === "data_dependency").length, 2);
assert.equal(context.model_ir.program.relationships.filter((row) => row.kind === "control_dependency").length, 1);
assert.equal(context.model_ir.tensors_and_storage.storage_objects.length, 1);
assert.equal(context.model_ir.weight_bindings.bindings.length, 1);
assert.equal(context.model_ir.program.completeness.control_dependencies, "materialized");
assert.equal(context.model_ir.source_contract.native_fact_ledger.facts.control_dependencies.length, 1);
assert.equal(context.model_ir.source_contract.native_fact_ledger.artifact_ir_sha256, context.artifact_ir.artifact_ir_sha256);
const graphOperations = new Map(context.model_ir.program.operations.map((row) => [row.native_op.name, row]));
assert.equal(graphOperations.get("MatMul").dependency_order, 1);
assert.equal(graphOperations.get("NoOp").dependency_order, 2);
const graphVisualization = buildModelIrVisualizationBundle(context.model_ir, { views: ["exhaustive"] });
assert.equal(graphVisualization.manifest.conservation.expected_relationship_count, 3);
assert.equal(graphVisualization.manifest.conservation.rendered_relationship_count, 3);
assert(graphVisualization.pages[0].svg.includes("control_dependency"));

const temp = await mkdtemp(path.join(os.tmpdir(), "deepbom-safe-source-"));
try {
  const files = [
    ["graph.pb", graphDef, "graphdef"], ["saved_model.pb", savedModel, "savedmodel"], ["weights.h5", hdf5, "hdf5"],
    ["model.keras", kerasZip, "keras"], ["model.pt2", pt2Zip, "pt2"], ["model.pt", checkpointZip, "pytorch_checkpoint"],
  ];
  for (const [name, content, format] of files) {
    const target = path.join(temp, name);
    await writeFile(target, content);
    const result = spawnSync(process.execPath, ["bin/deepbom.mjs", target, "--section", "artifact_ir,model_ir", "--compact"], { encoding: "utf8", cwd: path.resolve(".") });
    assert.equal(result.status, 0, `${name}: ${result.stderr}`);
    const document = JSON.parse(result.stdout);
    assert.equal(document.artifact.format, format);
    assert.equal(document.sections.artifact_ir.schema, "deepbom.artifact_ir.v2");
    assert.equal(document.sections.model_ir.schema, "deepbom.model_ir.v1");
    if (format === "pt2") {
      assert.equal(document.sections.model_ir.program.status, "serialized");
      assert.equal(document.sections.model_ir.program.operations.length, 1);
      assert.equal(document.sections.model_ir.program.programs[0].input_value_refs.length, 1);
      assert.equal(document.sections.model_ir.program.programs[0].output_value_refs.length, 1);
    }
    if (format === "keras") {
      assert.equal(document.sections.model_ir.program.status, "serialized");
      assert.equal(document.sections.model_ir.program.operations.length, 2);
      assert.equal(document.sections.model_ir.program.relationships.filter((row) => row.kind === "data_dependency").length, 1);
    }
  }
  const savedDirectory = path.join(temp, "saved_model");
  const variablesDirectory = path.join(savedDirectory, "variables");
  await mkdir(variablesDirectory, { recursive: true });
  await writeFile(path.join(savedDirectory, "saved_model.pb"), savedModel);
  await writeFile(path.join(variablesDirectory, "variables.index"), Uint8Array.of(1, 2, 3));
  await writeFile(path.join(variablesDirectory, "variables.data-00000-of-00001"), Uint8Array.of(4, 5, 6));
  const savedResult = spawnSync(process.execPath, ["bin/deepbom.mjs", savedDirectory, "--section", "artifact_ir,model_ir", "--compact"], { encoding: "utf8", cwd: path.resolve(".") });
  assert.equal(savedResult.status, 0, savedResult.stderr);
  const savedDocument = JSON.parse(savedResult.stdout);
  assert.equal(savedDocument.artifact.format, "savedmodel");
  assert.equal(savedDocument.sections.model_ir.artifact_role, "model_package");

  const savedBundleFiles = [
    blobFile(savedModel, "saved_model/saved_model.pb"),
    blobFile(Uint8Array.of(1, 2, 3), "saved_model/variables/variables.index"),
    blobFile(Uint8Array.of(4, 5, 6), "saved_model/variables/variables.data-00000-of-00001"),
  ];
  const boundSavedModel = await readArtifactBundle(savedBundleFiles, { scanMode: "structure" });
  assert.equal(boundSavedModel.analysis.savedmodel.package_closure_status, "selected_directory_complete_hash_bound");
  assert.equal(boundSavedModel.analysis.artifact_bundle.files.length, 3);
  const hdfFile = new Blob([hdf5]); Object.defineProperty(hdfFile, "name", { value: "weights.h5" });
  const hdfRead = await readHdf5EnvelopeFile(hdfFile);
  assert.equal(hdfRead.analysis.file_size_bytes, 512);
  const zipFile = new Blob([pt2Zip]); Object.defineProperty(zipFile, "name", { value: "model.pt2" });
  const zipRead = await readZipModelEnvelopeFile(zipFile, "model.pt2", "pt2");
  assert.equal(zipRead.analysis.safe_source_artifact.file_access.full_artifact_loaded, false);
} finally {
  await rm(temp, { recursive: true, force: true });
}

const traversalZip = mutateFirstCentralName(buildStoreZip([{ name: "safe.txt", data: bytes("x") }]), "../x.txt");
await assert.rejects(() => analyzeZipModelEnvelope(traversalZip, "bad.keras", "keras"), /Unsafe ZIP member path/);
assert.throws(() => analyzeTensorFlowProtobuf(Uint8Array.of(0x0a, 0x7f), "bad.pb"), /exceeds the source byte range/);

console.log("Safe source artifact checks passed (GraphDef/SavedModel preview; HDF5/Keras/PT2/PyTorch safe envelopes; CLI parity; traversal and truncation fail closed).");

function node(name, op, inputs = [], attributes = []) { return message([field(1, bytes(name)), field(2, bytes(op)), ...inputs.map((input) => field(3, bytes(input))), ...attributes.map((value) => field(5, value))]); }
function attribute(name, value) { return message([field(1, bytes(name)), field(2, value)]); }
function tensorShape(dimensions) { return message(dimensions.map((size) => field(2, message([varintField(1, size)])))); }
function tensorProto(dtype, shape, content) { return message([varintField(1, dtype), field(2, tensorShape(shape)), field(4, content)]); }
function tensorInfo(name, dtype, shape) { return message([field(1, bytes(name)), varintField(2, dtype), field(3, tensorShape(shape))]); }
function stringMapEntry(key, value) { return message([field(1, bytes(key)), field(2, value)]); }
function field(number, payload) { return concat(varint((number << 3) | 2), varint(payload.length), payload); }
function varintField(number, value) { return concat(varint(number << 3), varint(value)); }
function message(fields) { return concat(...fields); }
function bytes(value) { return new TextEncoder().encode(value); }
function concat(...rows) { const out = new Uint8Array(rows.reduce((sum, row) => sum + row.length, 0)); let offset = 0; for (const row of rows) { out.set(row, offset); offset += row.length; } return out; }
function varint(value) { let current = BigInt(value); const out = []; do { let byte = Number(current & 0x7fn); current >>= 7n; if (current) byte |= 0x80; out.push(byte); } while (current); return Uint8Array.from(out); }
function writeLe(target, offset, value, width) { let current = BigInt(value); for (let index = 0; index < width; index += 1) { target[offset + index] = Number(current & 255n); current >>= 8n; } }
function mutateFirstCentralName(zip, name) {
  const out = new Uint8Array(zip);
  const view = new DataView(out.buffer);
  let offset = 0;
  while (offset + 4 <= out.length && view.getUint32(offset, true) !== 0x02014b50) offset += 1;
  assert(offset + 46 <= out.length);
  const length = view.getUint16(offset + 28, true);
  const replacement = bytes(name);
  assert.equal(replacement.length, length);
  out.set(replacement, offset + 46);
  return out;
}
function blobFile(content, relativePath) {
  const file = new Blob([content]);
  Object.defineProperty(file, "name", { value: relativePath.split("/").at(-1) });
  Object.defineProperty(file, "webkitRelativePath", { value: relativePath });
  return file;
}

import { createHash } from "node:crypto";
import { createServer } from "node:http";
import { copyFile, mkdir, readFile, readdir, rm, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { spawn } from "node:child_process";

import { chromium } from "playwright";

import { readArtifactBundle } from "../web/lib/artifact-bundle.js";
import { readCoreMlModelFile } from "../web/lib/coreml-metadata-adapter.js";
import { COREML_NEURAL_NETWORK_SOURCE } from "../web/lib/coreml-neural-network.js";
import { readMetadataModelFile } from "../web/lib/metadata-model-adapters.js";
import { PUBLIC_SAMPLE_MODELS, publicSampleModel } from "../web/lib/sample-models.js";
import { buildDeploymentContractDocuments } from "../web/lib/report-export-contracts.js";
import { analyzeOnnxModel } from "../web/onnx.js";
import { analyzeExecuTorchModel } from "../web/executorch.js";
import { analyze_tflite_for_target, initSync } from "../pkg/tflite_wasm_audit.js";
import { decodeFixtureBase64, EXECUTORCH_ADD_PTE_BASE64 } from "./fixtures/executorch-fixtures.mjs";
import { assertCycloneDx17 } from "./cyclonedx-17-schema.mjs";
import { launchChromium } from "./browser-launch.mjs";
import { parseStoredZip, verifyPackageBytes } from "./verify-package.mjs";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const OUTPUT_ROOT = path.join(ROOT, ".local-validation", "supported-formats");
const OUTPUT = path.join(OUTPUT_ROOT, "latest");
const FIXTURES = path.join(OUTPUT, "fixtures");
const RESULTS = path.join(OUTPUT, "results");
const EXPORTS = path.join(OUTPUT, "exports");
const LOGS = path.join(OUTPUT, "logs");
const SCREENSHOTS = path.join(OUTPUT, "screenshots");
const DELIVERABLES = path.join(OUTPUT, "deliverables");
const UI_RESULTS = path.join(OUTPUT, "ui-results");
const GENERATED_AT = new Date().toISOString();
const TARGET_ID = "android_mid_a55";
const LARGE_BROWSER_DOWNLOAD_IDS = new Set([
  "downloadEngineeringBundle",
  "downloadPublicBundle",
  "downloadRawData",
  "downloadVisualPngs",
]);

const rows = [];
const contractChecks = [];
let browserResult = { status: "not_run", cases: [], errors: [] };

await rm(OUTPUT, { recursive: true, force: true, maxRetries: 8, retryDelay: 125 });
await Promise.all([FIXTURES, RESULTS, EXPORTS, LOGS, SCREENSHOTS, DELIVERABLES, UI_RESULTS].map((directory) => mkdir(directory, { recursive: true })));

const fixturePaths = await writeFixtures();
const allCases = [
  { id: "tflite", label: ".tflite", selection: "Select model", sampleId: "tflite-mobilenet-v2-int8", fixture: "Google MobileNetV2 INT8 public trained artifact", path: fixturePaths.tflite, analyze: analyzeTflite },
  { id: "onnx", label: ".onnx", selection: "Select model", sampleId: "onnx-mnist-8", fixture: "ONNX Model Zoo MNIST-8 public trained artifact", path: fixturePaths.onnx, analyze: analyzeOnnx },
  { id: "gguf", label: ".gguf", selection: "Select model", sampleId: "gguf-tinymqa-q4", fixture: "TinyMQA 1M Q4_0 public trained GGUF artifact", path: fixturePaths.gguf, analyze: (item) => analyzeMetadata(item, "gguf") },
  { id: "safetensors", label: ".safetensors", selection: "Select model", sampleId: "safetensors-nanofable-fp16", fixture: "NanoFable-1M FP16 public trained SafeTensors artifact", path: fixturePaths.safetensors, analyze: (item) => analyzeMetadata(item, "safetensors") },
  { id: "mlmodel", label: ".mlmodel", selection: "Select model", sampleId: "coreml-mnist-classifier", fixture: "Apple MNIST public trained Core ML artifact", path: fixturePaths.mlmodel, analyze: analyzeCoreMl },
  { id: "executorch", label: ".pte", selection: "Select model", fixture: "Schema-pinned generated ExecuTorch ET12 add-program contract fixture", path: fixturePaths.executorch, analyze: analyzeExecuTorch },
  { id: "mlpackage", label: ".mlpackage", selection: "Select package", fixture: "Apple MNIST public trained Core ML artifact in a deterministic mlpackage envelope", path: fixturePaths.mlpackage, analyze: analyzePackage },
  { id: "sharded_safetensors", label: "sharded .safetensors", selection: "Select package", fixture: "NanoFable-1M FP16 tensors deterministically partitioned into two payload-preserving shards", path: fixturePaths.shardedSafetensors, analyze: analyzePackage },
];
const requestedCaseIds = new Set(String(process.env.DEEPBOM_VALIDATE_FORMATS || "")
  .split(",")
  .map((value) => value.trim())
  .filter(Boolean));
const cases = requestedCaseIds.size
  ? allCases.filter((item) => requestedCaseIds.has(item.id))
  : allCases;
if (!cases.length) throw new Error(`No supported-format cases matched DEEPBOM_VALIDATE_FORMATS=${[...requestedCaseIds].join(",")}`);

try {
  initSync({ module: await readFile(path.join(ROOT, "pkg", "tflite_wasm_audit_bg.wasm")) });
} catch (error) {
  await writeText(path.join(LOGS, "tflite-wasm-init.log"), stack(error));
}

for (const item of cases) {
  const started = performance.now();
  console.log(`[parser] ${item.id}: analyzing fixture and deterministic contract exports`);
  try {
    const result = await item.analyze(item);
    const row = {
      id: item.id,
      advertised_format: item.label,
      selection_control: item.selection,
      fixture_provenance: item.fixture,
      status: "pass",
      duration_ms: rounded(performance.now() - started),
      ...result.summary,
    };
    await writeJson(path.join(RESULTS, `${item.id}.analysis.json`), result.analysis);
    await writeJson(path.join(RESULTS, `${item.id}.result.json`), row);
    await writeExportDocuments(item.id, result.analysis, result.subjectHash, result.subjectBytes);
    rows.push(row);
    console.log(`[parser] ${item.id}: pass (${row.duration_ms} ms)`);
  } catch (error) {
    const row = {
      id: item.id,
      advertised_format: item.label,
      selection_control: item.selection,
      fixture_provenance: item.fixture,
      status: "fail",
      duration_ms: rounded(performance.now() - started),
      error: error?.message || String(error),
    };
    rows.push(row);
    await writeJson(path.join(RESULTS, `${item.id}.result.json`), row);
    await writeText(path.join(LOGS, `${item.id}.error.log`), stack(error));
    console.log(`[parser] ${item.id}: fail - ${row.error}`);
  }
}

for (const script of [
  "scripts/check-public-samples.mjs",
  "scripts/check-model-file-contract.mjs",
  "scripts/check-metadata-model-adapters.mjs",
  "scripts/check-artifact-bundle.mjs",
  "scripts/check-export-contract-documents.mjs",
]) {
  const result = await runContractCheck(script);
  contractChecks.push(result);
  await writeText(path.join(LOGS, `${path.basename(script, ".mjs")}.log`), result.output);
}

try {
  browserResult = await validateBrowserSelection(cases);
} catch (error) {
  browserResult = { status: "fail", cases: [], errors: [error?.message || String(error)] };
  await writeText(path.join(LOGS, "browser-selection.error.log"), stack(error));
}
await writeJson(path.join(RESULTS, "browser-selection.json"), browserResult);

const failed = rows.filter((row) => row.status !== "pass");
const failedContracts = contractChecks.filter((item) => item.status !== "pass");
const browserFailures = browserResult.cases?.filter((item) => item.status !== "pass") || [];
const overall = failed.length || failedContracts.length || browserResult.status !== "pass" || browserFailures.length ? "fail" : "pass";
const repositoryIdentity = await repositoryContentIdentity();
const matrix = {
  schema: "deepbom.supported_format_validation.v2",
  generated_at: GENERATED_AT,
  overall_status: overall,
  environment: {
    node: process.version,
    platform: process.platform,
    architecture: process.arch,
    git_commit: (await runCommand("git", ["rev-parse", "HEAD"])).stdout.trim(),
    git_worktree_dirty: Boolean((await runCommand("git", ["status", "--porcelain"])).stdout.trim()),
    repository_content_sha256: repositoryIdentity.sha256,
    repository_content_file_count: repositoryIdentity.fileCount,
    repository_content_hash_basis: repositoryIdentity.hashBasis,
    target_profile: TARGET_ID,
  },
  coverage: {
    advertised_paths: cases.length,
    parser_and_export_passes: rows.filter((row) => row.status === "pass").length,
    browser_selection_passes: browserResult.cases?.filter((row) => row.status === "pass").length || 0,
    contract_check_passes: contractChecks.filter((row) => row.status === "pass").length,
    actual_provided_downloads: browserResult.cases?.reduce((sum, row) => sum + Number(row.provided_download_count || 0), 0) || 0,
    unavailable_download_states: browserResult.cases?.reduce((sum, row) => sum + Number(row.unavailable_download_count || 0), 0) || 0,
    captured_ui_surfaces: browserResult.cases?.reduce((sum, row) => sum + Number(row.ui_surface_count || 0), 0) || 0,
    skipped_ui_surface_states: browserResult.cases?.reduce((sum, row) => sum + Number(row.ui_surface_skipped_count || 0), 0) || 0,
  },
  formats: rows,
  browser_selection: browserResult,
  contract_checks: contractChecks.map(({ output, ...item }) => item),
};
await writeJson(path.join(OUTPUT, "validation-matrix.json"), matrix);
await writeJson(path.join(OUTPUT, "artifact-catalog.json"), await buildArtifactCatalog(cases));
await writeText(path.join(OUTPUT, "README.md"), summaryMarkdown(matrix));
await writeManifest();
await writeJson(path.join(OUTPUT_ROOT, "latest.json"), {
  schema: matrix.schema,
  generated_at: GENERATED_AT,
  overall_status: overall,
  result_directory: "latest",
  manifest: "latest/sha256-manifest.json",
});

console.log(`Supported-format validation ${overall}: ${rows.filter((row) => row.status === "pass").length}/${cases.length} parser/export, ${matrix.coverage.browser_selection_passes}/${cases.length} browser paths, ${matrix.coverage.contract_check_passes}/${contractChecks.length} contract checks.`);
console.log(`Results: ${path.relative(ROOT, OUTPUT)}`);
if (overall !== "pass") process.exitCode = 1;

async function analyzeTflite(item) {
  const bytes = new Uint8Array(await readFile(item.path));
  const hash = sha256(bytes);
  const analysis = analyze_tflite_for_target(bytes, path.basename(item.path), TARGET_ID);
  analysis.model_sha256 = hash;
  analysis.file_size_bytes = bytes.byteLength;
  assert(analysis.format === "tflite", "TFLite analyzer returned the wrong format");
  assert(analysis.target_profile?.id === TARGET_ID, "TFLite target profile was not bound");
  assert(Number(analysis.total_ops) > 0 && Number(analysis.tensor_count) > 0, "TFLite graph inventory is empty");
  return {
    analysis,
    subjectHash: hash,
    subjectBytes: bytes.byteLength,
    summary: graphSummary(analysis, hash, bytes.byteLength),
  };
}

async function analyzeOnnx(item) {
  const bytes = new Uint8Array(await readFile(item.path));
  const hash = sha256(bytes);
  const analysis = analyzeOnnxModel(bytes, path.basename(item.path), { id: TARGET_ID, label: "Validation target" });
  analysis.model_sha256 = hash;
  analysis.file_size_bytes = bytes.byteLength;
  assert(analysis.format === "onnx", "ONNX analyzer returned the wrong format");
  assert(Number(analysis.operator_count) > 0 && Number(analysis.tensor_count) > 0, "ONNX graph inventory is empty");
  assert(Number(analysis.size_breakdown?.constant_bytes) > 0, "ONNX initializer payload was not accounted");
  return {
    analysis,
    subjectHash: hash,
    subjectBytes: bytes.byteLength,
    summary: graphSummary(analysis, hash, bytes.byteLength),
  };
}

async function analyzeExecuTorch(item) {
  const bytes = new Uint8Array(await readFile(item.path));
  const hash = sha256(bytes);
  const analysis = analyzeExecuTorchModel(bytes, path.basename(item.path));
  analysis.model_sha256 = hash;
  analysis.file_size_bytes = bytes.byteLength;
  assert(analysis.format === "executorch" && analysis.executorch_container === "pte", "ExecuTorch analyzer returned the wrong format/container");
  assert(analysis.operator_count === 1 && analysis.tensor_count === 3, "ExecuTorch ET12 graph inventory is not conserved");
  assert(analysis.mac_assessment?.status === "assessed_source_bound_portable_kernel_signatures"
    && analysis.mac_assessment?.complete === true
    && analysis.total_macs === 0
    && analysis.total_macs_decimal === "0",
  "ExecuTorch source-bound elementwise ADD must emit an exact zero nominal tensor-contraction MAC total");
  return {
    analysis,
    subjectHash: hash,
    subjectBytes: bytes.byteLength,
    summary: graphSummary(analysis, hash, bytes.byteLength),
  };
}

async function analyzeMetadata(item, format) {
  const file = await browserFile(item.path);
  const hash = sha256(new Uint8Array(await file.arrayBuffer()));
  const parsed = await readMetadataModelFile(file, format);
  const analysis = parsed.analysis;
  analysis.model_sha256 = hash;
  analysis.file_size_bytes = file.size;
  assert(analysis.format === format, `${format} adapter returned the wrong format`);
  assert(Number(analysis.tensor_count) > 0, `${format} tensor inventory is empty`);
  if (format === "gguf") {
    assert(analysis.gguf?.version === 3, "GGUF v3 fixture version was not decoded");
    assert(analysis.gguf?.architecture === "llama", "GGUF architecture metadata was not decoded");
  } else {
    assert(analysis.safetensors?.payload_coverage_status === "complete_without_gaps_or_overlaps", "SafeTensors payload conservation failed");
  }
  return {
    analysis,
    subjectHash: hash,
    subjectBytes: file.size,
    summary: metadataSummary(analysis, hash, file.size),
  };
}

async function analyzeCoreMl(item) {
  const file = await browserFile(item.path);
  const bytes = new Uint8Array(await file.arrayBuffer());
  const hash = sha256(bytes);
  const parsed = await readCoreMlModelFile(file);
  const analysis = parsed.analysis;
  analysis.model_sha256 = hash;
  analysis.file_size_bytes = file.size;
  assert(analysis.format === "coreml", "Core ML adapter returned the wrong format");
  assert(analysis.coreml?.model_type === "neuralNetworkClassifier", "Core ML model type was not decoded");
  assert(analysis.inputs?.length === 1 && analysis.outputs?.length === 2, "Core ML interface contract is incomplete");
  assert(analysis.metadata_presence?.metadata_author === "Apple, Inc.", "Core ML public-model provenance was not decoded");
  assert(analysis.operator_count === 14 && analysis.ops?.length === 14, "Core ML legacy NeuralNetwork DAG layer count is not conserved");
  assert(analysis.ops?.[0]?.name === "CONVOLUTION" && analysis.ops?.at(-1)?.name === "SOFTMAX", "Core ML layer oneof decoding did not preserve graph order");
  assert(analysis.quantization_status?.weight_parameter_count === 10, "Core ML WeightParams cardinality is not exact");
  assert(analysis.quantization_status?.quantized_weight_parameter_count === 0 && analysis.quantization_status?.fp32_weight_parameter_count === 10, "Core ML weight storage classification is incorrect");
  assert(analysis.quantization_status?.scanned_layer_count === 14 && analysis.quantization_status?.layer_count === 14, "Core ML layer weight assessment denominator is incomplete");
  assert(JSON.stringify(analysis.tensors?.[analysis.input_tensor_indices?.[0]]?.shape) === JSON.stringify([1, 1, 1, 28, 28]), "Core ML default rank-5 image input mapping is incorrect");
  assert(analysis.coreml?.neural_network?.preprocessing?.length === 1, "Core ML serialized preprocessing cardinality is incorrect");
  assert(analysis.coreml.neural_network.preprocessing[0].feature_name === "image" && analysis.coreml.neural_network.preprocessing[0].kind === "image_scaler", "Core ML preprocessing is not bound to the named image input");
  assert(analysis.coreml.neural_network.preprocessing[0].serialized_values.channel_scale === Math.fround(1 / 255), "Core ML serialized FP32 image-scaler value is incorrect");
  assert(analysis.coreml?.source_basis?.neural_network_proto_sha256 === COREML_NEURAL_NETWORK_SOURCE.neural_network_proto_sha256, "Core ML DAG decoding is not bound to the pinned NeuralNetwork.proto digest");
  return {
    analysis,
    subjectHash: hash,
    subjectBytes: file.size,
    summary: metadataSummary(analysis, hash, file.size),
  };
}

async function analyzePackage(item) {
  const files = await directoryFiles(item.path);
  const parsed = await readArtifactBundle(files);
  const analysis = parsed.analysis;
  assert(/^[a-f0-9]{64}$/.test(analysis.model_sha256 || ""), "Package canonical bundle digest is missing");
  assert(analysis.artifact_bundle?.files?.length === files.length, "Package file ledger is incomplete");
  if (item.id === "mlpackage") {
    assert(analysis.artifact_bundle?.kind === "coreml_mlpackage", "Core ML package kind was not identified");
    assert(analysis.artifact_bundle.files.some((file) => file.path.endsWith(".mlmodel")), "Core ML package root model was not manifest-bound");
    assert(analysis.coreml?.model_type === "neuralNetworkClassifier", "Packaged Apple MNIST model type was not decoded");
  } else {
    assert(analysis.safetensors?.index_binding_status === "complete_bidirectional", "SafeTensors shard index binding is incomplete");
    assert(analysis.tensor_count > 2, "SafeTensors shard tensor conservation failed");
    assert(analysis.tensors.every((tensor, index) => tensor.index === index), "SafeTensors shard aggregate tensor indices are not unique and contiguous");
    assert(analysis.safetensors.tensor_count === analysis.tensor_count, "SafeTensors aggregate metadata tensor count is inconsistent");
    assert(analysis.safetensors.payload_byte_length > 1_000_000, "SafeTensors aggregate payload byte count is inconsistent");
  }
  return {
    analysis,
    subjectHash: analysis.model_sha256,
    subjectBytes: analysis.file_size_bytes,
    summary: {
      detected_format: analysis.format,
      filename: analysis.filename,
      byte_length: analysis.file_size_bytes,
      sha256: analysis.model_sha256,
      hash_basis: analysis.artifact_bundle.hash_basis,
      package_kind: analysis.artifact_bundle.kind,
      package_file_count: analysis.artifact_bundle.files.length,
      tensor_count: assessedCount(analysis.tensor_count, 0),
      binding_status: analysis.safetensors?.index_binding_status || "manifest_resolved",
      ...(assessmentStatus(analysis, assessedCount(analysis.tensor_count, 0)) ? { inventory_assessment_status: assessmentStatus(analysis, assessedCount(analysis.tensor_count, 0)) } : {}),
    },
  };
}

async function writeExportDocuments(id, analysis, hash, fileSizeBytes) {
  const exportSet = buildDeploymentContractDocuments(analysis, {
    generatedAt: GENERATED_AT,
    hash,
    fileSizeBytes,
  });
  assertCycloneDx17(exportSet.documents.cyclonedx_evidence, `${id} CycloneDX evidence`);
  assertCycloneDx17(exportSet.documents.observed_formulation, `${id} observed formulation`);
  assert(exportSet.subject.sha256 === hash, `${id} export subject SHA-256 does not match the analyzed artifact`);
  assert(exportSet.subject.byte_length === fileSizeBytes, `${id} export subject byte length does not match the analyzed artifact`);
  assert(exportSet.subject.schema_or_opset === expectedArtifactSchema(analysis), `${id} export subject format/schema identity is incorrect`);
  const envelope = exportSet.documents.artifact_evidence_envelope;
  assert(envelope.identity.sha256 === hash && envelope.identity.byte_length === fileSizeBytes, `${id} canonical evidence identity diverges from the export subject`);
  const expectedOperatorCount = ["gguf", "safetensors"].includes(analysis.format) ? null : (analysis.operator_count ?? analysis.ops?.length ?? null);
  assert(envelope.graph.operator_count === expectedOperatorCount, `${id} exported operator count diverges from parser output`);
  assert(envelope.graph.tensor_count === (analysis.tensor_count ?? analysis.tensors?.length ?? null), `${id} exported tensor count diverges from parser output`);
  assert(envelope.graph.total_macs === (analysis.total_macs ?? null), `${id} exported MAC total diverges from parser output or collapses null to zero`);
  const runtimeFamily = {
    tflite: "TensorFlow Lite / LiteRT",
    onnx: "ONNX Runtime",
    gguf: "GGUF-compatible runtime (unbound)",
    safetensors: "SafeTensors-compatible loader (unbound)",
    coreml: "Core ML",
    executorch: "ExecuTorch runtime (build unbound)",
  }[analysis.format];
  assert(exportSet.documents.runtime_requirement_manifest.necessary_runtime_floor.runtime === runtimeFamily, `${id} runtime-requirement export names the wrong runtime family`);
  if (analysis.format === "gguf") {
    assert(envelope.format_extensions.gguf.declared_tensor_byte_length === analysis.gguf.declared_tensor_byte_length, `${id} GGUF tensor payload bytes were lost from canonical evidence`);
  }
  if (analysis.format === "safetensors") {
    assert(envelope.format_extensions.safetensors.payload_byte_length === analysis.safetensors.payload_byte_length, `${id} SafeTensors payload bytes were lost from canonical evidence`);
  }
  if (analysis.format === "coreml") {
    assert(envelope.format_extensions.coreml.description?.predicted_feature_name === analysis.coreml.description?.predicted_feature_name, `${id} Core ML predicted-feature binding was lost from canonical evidence`);
    assert(envelope.format_extensions.coreml.description?.predicted_probabilities_name === analysis.coreml.description?.predicted_probabilities_name, `${id} Core ML probability binding was lost from canonical evidence`);
  }
  if (analysis.format === "executorch") {
    assert(envelope.format_extensions.executorch?.container === analysis.executorch_container, `${id} ExecuTorch container identity was lost from canonical evidence`);
    assert(envelope.format_extensions.executorch?.program?.source?.commit === analysis.executorch_program?.source?.commit, `${id} ExecuTorch pinned schema provenance was lost from canonical evidence`);
    assert(envelope.format_extensions.executorch?.planned_memory?.planned_non_const_memory_bytes_decimal === analysis.tensor_liveness?.planned_non_const_memory_bytes_decimal, `${id} ExecuTorch planned-memory evidence was lost from canonical evidence`);
  }
  const directory = path.join(EXPORTS, id);
  await mkdir(directory, { recursive: true });
  await writeJson(path.join(directory, "export-set.json"), exportSet);
  for (const [name, document] of Object.entries(exportSet.documents)) {
    await writeJson(path.join(directory, `${name}.json`), document);
  }
}

function expectedArtifactSchema(analysis) {
  const format = String(analysis?.format || "").toLowerCase();
  if (format === "tflite") return `TFLite schema ${analysis.version || "unknown"}`;
  if (format === "onnx") return `IR ${analysis.onnx_ir_version || "unknown"} / opset ${(analysis.opsets || []).map((item) => `${item.domain || "ai.onnx"}:${item.version || "unknown"}`).join(" / ") || "unknown"}`;
  if (format === "gguf") return `GGUF v${analysis.gguf?.version || "unknown"}`;
  if (format === "safetensors") return "SafeTensors format (unversioned)";
  if (format === "coreml") return `Core ML specification ${analysis.coreml?.specification_version || "unknown"}`;
  if (format === "executorch") return `ExecuTorch ${String(analysis.executorch_container || "artifact").toUpperCase()} schema ${analysis.version ?? "unknown"}`;
  return `${format || "unknown"} format; version not declared`;
}

async function writeFixtures() {
  const samples = {
    tflite: publicSampleModel("tflite-mobilenet-v2-int8"),
    onnx: publicSampleModel("onnx-mnist-8"),
    gguf: publicSampleModel("gguf-tinymqa-q4"),
    safetensors: publicSampleModel("safetensors-nanofable-fp16"),
    mlmodel: publicSampleModel("coreml-mnist-classifier"),
  };
  const copied = {};
  for (const [id, sample] of Object.entries(samples)) {
    assert(sample, `Public sample declaration is missing for ${id}`);
    const source = path.join(ROOT, "web", sample.path);
    const bytes = new Uint8Array(await readFile(source));
    assert(bytes.byteLength === sample.byteLength && sha256(bytes) === sample.sha256, `${id} public sample identity does not match its pinned declaration`);
    copied[id] = path.join(FIXTURES, sample.filename);
    await copyFile(source, copied[id]);
  }

  const mlpackage = path.join(FIXTURES, "Fixture.mlpackage");
  await mkdir(path.join(mlpackage, "Data", "com.apple.CoreML"), { recursive: true });
  await writeJson(path.join(mlpackage, "Manifest.json"), {
    fileFormatVersion: "1.0.0",
    rootModelIdentifier: "model-id",
    itemInfoEntries: {
      "model-id": { path: "com.apple.CoreML/MNISTClassifier.mlmodel", name: "MNISTClassifier.mlmodel", author: "com.apple.CoreML", description: "Apple public MNIST classifier" },
    },
  });
  await copyFile(copied.mlmodel, path.join(mlpackage, "Data", "com.apple.CoreML", "MNISTClassifier.mlmodel"));

  const shardedSafetensors = path.join(FIXTURES, "Sharded");
  await writeShardedSafeTensors(copied.safetensors, shardedSafetensors, samples.safetensors.sha256);
  const executorch = path.join(FIXTURES, "add.pte");
  await writeFile(executorch, decodeFixtureBase64(EXECUTORCH_ADD_PTE_BASE64));
  await writeJson(path.join(FIXTURES, "source-artifacts.json"), Object.fromEntries(Object.entries(samples).map(([id, sample]) => [id, sample])));
  return { ...copied, executorch, mlpackage, shardedSafetensors };
}

async function writeShardedSafeTensors(sourcePath, directory, sourceSha256) {
  const source = new Uint8Array(await readFile(sourcePath));
  const headerLength = Number(new DataView(source.buffer, source.byteOffset, 8).getBigUint64(0, true));
  const payloadOffset = 8 + headerLength;
  const header = JSON.parse(new TextDecoder().decode(source.subarray(8, payloadOffset)).trim());
  const tensors = Object.entries(header).filter(([name]) => name !== "__metadata__");
  assert(tensors.length > 2, "Source SafeTensors model has too few tensors to shard");
  const groups = [tensors.filter((_, index) => index % 2 === 0), tensors.filter((_, index) => index % 2 === 1)];
  const weightMap = {};
  let conservedPayloadBytes = 0;
  await mkdir(directory, { recursive: true });
  for (const [groupIndex, group] of groups.entries()) {
    const filename = `model-${String(groupIndex + 1).padStart(5, "0")}-of-00002.safetensors`;
    const shardHeader = { __metadata__: { ...(header.__metadata__ || {}), deepbom_source_sha256: sourceSha256 } };
    const payloads = [];
    let offset = 0;
    for (const [name, descriptor] of group) {
      const [start, end] = descriptor.data_offsets;
      const payload = source.slice(payloadOffset + start, payloadOffset + end);
      shardHeader[name] = { ...descriptor, data_offsets: [offset, offset + payload.byteLength] };
      payloads.push(payload);
      offset += payload.byteLength;
      weightMap[name] = filename;
    }
    conservedPayloadBytes += offset;
    await writeFile(path.join(directory, filename), encodeSafeTensors(shardHeader, payloads));
  }
  assert(conservedPayloadBytes === source.byteLength - payloadOffset, "Sharded SafeTensors payload bytes were not conserved");
  await writeJson(path.join(directory, "model.safetensors.index.json"), {
    metadata: { total_size: conservedPayloadBytes, source_sha256: sourceSha256, derivation: "tensor_boundary_partition_with_exact_payload_byte_conservation" },
    weight_map: weightMap,
  });
}

function encodeSafeTensors(header, payloads) {
  const encoded = new TextEncoder().encode(JSON.stringify(header));
  const paddedLength = Math.ceil(encoded.byteLength / 8) * 8;
  const prefix = new Uint8Array(8 + paddedLength);
  new DataView(prefix.buffer).setBigUint64(0, BigInt(paddedLength), true);
  prefix.fill(0x20, 8);
  prefix.set(encoded, 8);
  return concatBytes(prefix, ...payloads);
}

async function validateBrowserSelection(items) {
  const server = createStaticServer(ROOT);
  const browserErrors = [];
  let browser;
  try {
    await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
    browser = await launchChromium(chromium);
    const page = await browser.newPage({ viewport: { width: 1440, height: 1000 }, deviceScaleFactor: 1 });
    await installLocalValidationApi(page);
    page.on("pageerror", (error) => {
      const detail = `page: ${error.message}`;
      browserErrors.push(detail);
      console.error(`[browser-console] ${detail}`);
    });
    page.on("console", async (message) => {
      if (message.type() === "error" && !/Failed to load resource/i.test(message.text())) {
        const values = await Promise.all(message.args().map((handle) => handle.evaluate((value) => value instanceof Error ? value.stack : String(value)).catch(() => "[unreadable console argument]")));
        const detail = `console: ${values.join(" ") || message.text()}`;
        browserErrors.push(detail);
        console.error(`[browser-console] ${detail}`);
      }
    });
    await page.goto(`http://127.0.0.1:${server.address().port}/web/`, { waitUntil: "domcontentloaded" });
    await page.waitForFunction(() => document.querySelector("#status")?.textContent?.includes("Ready"), null, { timeout: 180_000 });
    if (await page.locator("#agreementBackdrop").isVisible()) {
      await page.locator("#privacyAgree").check();
      await page.locator("#acceptAgreement").click();
    }
    const labels = await page.locator(".file-button-formats small").allTextContents();
    assert(labels.some((value) => value.includes(".tflite") && value.includes(".mlmodel")), "Select model extension disclosure is incomplete");
    assert(labels.some((value) => value.includes(".mlpackage") && value.includes("sharded .safetensors")), "Select package extension disclosure is incomplete");
    const sampleIds = await page.locator("#sampleModelSelect option").evaluateAll((options) => options.map((option) => option.value));
    const expectedSampleIds = PUBLIC_SAMPLE_MODELS.map((sample) => sample.id);
    assert(sampleIds.length === expectedSampleIds.length
      && new Set(sampleIds).size === sampleIds.length
      && expectedSampleIds.every((sampleId) => sampleIds.includes(sampleId))
      && items.filter((item) => item.sampleId).every((item) => sampleIds.includes(item.sampleId)), "Public sample menu is incomplete or duplicated");

    const caseResults = [];
    for (const item of [...items.slice(1), items[0]]) {
      const started = performance.now();
      const browserErrorOffset = browserErrors.length;
      console.log(`[browser] ${item.id}: selecting artifact and running the actual UI audit`);
      try {
        if (item.sampleId) {
          await page.locator("#sampleModelSelect").selectOption(item.sampleId);
          await page.locator("#trySampleModel").click();
        } else {
          const input = item.selection === "Select package" ? "#artifactBundleInput" : "#fileInput";
          await page.locator(input).setInputFiles(item.path);
        }
        const expectedName = item.id === "sharded_safetensors"
          ? "Sharded/model.safetensors.index.json"
          : path.basename(item.path);
        await page.waitForFunction((name) => document.querySelector("#selectedModelName")?.textContent === name, expectedName, { timeout: 30_000 });
        if (!item.sampleId) {
          await page.waitForFunction(() => {
            const button = document.querySelector("#runAudit");
            return /audit/i.test(button?.textContent || "") && !button.disabled;
          }, null, { timeout: 30_000 });
          await page.locator("#runAudit").click();
        }
        await page.waitForFunction(() => {
          const value = document.querySelector("#analysisPlanStatus")?.textContent || "";
          return value.includes("audit run complete") || value.startsWith("Audit failed:");
        }, null, { timeout: item.id === "tflite" ? 300_000 : 90_000 });
        const terminalStatus = await page.locator("#analysisPlanStatus").textContent();
        assert(terminalStatus.includes("audit run complete"), `${item.id} ${terminalStatus}`);
        await page.waitForFunction(() => document.querySelector("#analysisEstimate")?.textContent?.includes("measured"), null, { timeout: 10_000 });
        await page.evaluate(() => scrollTo(0, 0));
        const state = await page.evaluate(() => ({
          selected_model: document.querySelector("#selectedModelName")?.textContent || "",
          selected_meta: document.querySelector("#selectedModelMeta")?.textContent || "",
          audit_status: document.querySelector("#analysisPlanStatus")?.textContent || "",
          global_status: document.querySelector("#status")?.textContent || "",
          measured_time: document.querySelector("#analysisEstimate")?.textContent || "",
          progress: document.querySelector("#auditProgress")?.getAttribute("aria-valuenow") || "",
          model_format: document.body.dataset.modelFormat || "",
          viewport_overflow_px: Math.max(0, document.documentElement.scrollWidth - innerWidth),
          artifact_binding_bytes: document.querySelector(".artifact-identity-binding strong")?.textContent || "",
          metric_labels: [...document.querySelectorAll(".artifact-metric-grid .metric > span")].map((node) => node.textContent || ""),
          overview_text: document.querySelector("#summary")?.textContent || "",
          placement_hidden: document.querySelector("#executionPlacementPanel")?.hidden ?? true,
          placement_format: document.querySelector("#executionPlacementPanel")?.dataset.placementFormat || "",
          placement_state: document.querySelector("#executionPlacementPanel")?.dataset.placementState || "",
          placement_level_count: document.querySelectorAll("#executionPlacementPanel .execution-placement-level").length,
          placement_topology_count: document.querySelectorAll("#executionPlacementPanel .execution-placement-topology").length,
          placement_relation_count: document.querySelectorAll("#executionPlacementPanel .execution-placement-relation").length,
          placement_external_relation: [...document.querySelectorAll("#executionPlacementPanel .execution-placement-relation")].findIndex((node) => node.classList.contains("external")),
          placement_text: document.querySelector("#executionPlacementPanel")?.textContent || "",
        }));
        assert(state.audit_status.includes("audit run complete") && state.global_status.includes("audit run complete"), `${item.id} did not reach rendered audit completion`);
        assert(state.progress === "100", `${item.id} progress did not reach 100`);
        assert(state.viewport_overflow_px <= 1, `${item.id} viewer overflows the desktop viewport`);
        assert(state.artifact_binding_bytes !== "0 B", `${item.id} rendered a false zero artifact size`);
        assert(!state.placement_hidden, `${item.id} execution-placement evidence is hidden after audit`);
        assert(state.placement_format === state.model_format, `${item.id} execution-placement format binding diverges from the active artifact`);
        assert(state.placement_state !== "invalid", `${item.id} execution-placement evidence was rejected by the UI`);
        assert(state.placement_level_count === 4, `${item.id} execution-placement evidence ladder is incomplete`);
        assert(state.placement_topology_count === 1 && state.placement_relation_count === 3,
          `${item.id} execution-placement topology is incomplete`);
        assert(state.placement_external_relation === 2, `${item.id} runtime evidence boundary is misplaced`);
        assert(state.placement_text.includes("Claim progression, not a physical-routing observation"),
          `${item.id} execution-placement topology omits its interpretation boundary`);
        assert(/Artifact observed|Artifact observed/.test(state.placement_text), `${item.id} execution-placement evidence omits the artifact-observed level`);
        await validateExecutionPlacementResponsive(page, item.id);
        if (!["tflite", "onnx"].includes(item.id)) {
          assert(!/FlatBuffer schema|Converter evidence|TFLite Signatures/.test(state.overview_text), `${item.id} rendered TFLite-specific evidence labels`);
          const expectedLabel = item.id === "mlmodel" || item.id === "mlpackage" ? "Core ML evidence"
            : item.id === "executorch" ? "ExecuTorch evidence" : "Container evidence";
          assert(state.metric_labels.includes(expectedLabel), `${item.id} did not render ${expectedLabel}`);
        }
        if (item.id === "sharded_safetensors") {
          assert(state.overview_text.includes("SafeTensors / 2 shards"), "Sharded SafeTensors aggregate shard count is not visible");
          const expectedTensorCount = Number(rows.find((row) => row.id === item.id)?.tensor_count || 0);
          assert(expectedTensorCount > 0 && state.overview_text.includes(`Container tensor ${expectedTensorCount}`), "Sharded SafeTensors tensors are misclassified in the Overview");
        }
        if (["mlmodel", "mlpackage"].includes(item.id)) {
          assert(state.overview_text.includes("QuantizationNo quantized WeightParams"), "Core ML decoded WeightParams classification is not visible");
          assert(state.overview_text.includes("0/10 decoded WeightParams") && state.overview_text.includes("Weight field coverage14/14"), "Core ML exact WeightParams and field-scan denominators are not visible");
          assert(!state.overview_text.includes("tensors carry quantization parameters"), "Core ML WeightParams were misrepresented as graph-tensor affine metadata");
          assert(state.overview_text.includes("classLabel") && state.overview_text.includes("labelProbabilities"), `${item.id} omitted protobuf-declared prediction bindings from the Overview`);
        }
        if (item.id === "tflite") assert(state.overview_text.includes("300,775,552"), "TFLite Overview diverged from the independently verified MAC total");
        if (item.id === "onnx") assert(state.overview_text.includes("786,560"), "ONNX Overview diverged from the independently verified MAC total");
        if (item.id === "gguf") {
          assert(state.overview_text.includes("Block-quantized tensors") && state.overview_text.includes("30"), "GGUF block-quantized tensor count is not visible in the Overview");
          assert(state.overview_text.includes("507,392") && state.overview_text.includes("complete_without_gaps_or_overlaps"), "GGUF exact tensor payload conservation is not visible in the Overview");
          assert(state.overview_text.includes("Largest Tensor Payloads"), "GGUF exact tensor byte ranges are not linked to the largest-payload viewer");
        }
        if (["safetensors", "sharded_safetensors"].includes(item.id)) {
          assert(state.overview_text.includes("2,754,816") && state.overview_text.includes("complete_without_gaps_or_overlaps"), `${item.id} exact payload conservation is not visible in the Overview`);
          assert(state.overview_text.includes("Largest Tensor Payloads"), `${item.id} exact tensor byte ranges are not linked to the largest-payload viewer`);
        }
        delete state.overview_text;
        delete state.placement_text;
        await page.screenshot({ path: path.join(SCREENSHOTS, `${item.id}.png`), fullPage: false });
        if (item.id === "tflite") await validateResponsiveNavigationContracts(page);
        console.log(`[browser] ${item.id}: audit complete; capturing every reachable UI surface`);
        const uiCapture = await captureUiResults(page, item);
        console.log(`[browser] ${item.id}: captured ${uiCapture.captured} UI surface(s), ${uiCapture.skipped} unavailable state(s); exercising downloads`);
        const downloadCapture = await captureProvidedDownloads(page, item);
        caseResults.push({
          id: item.id,
          status: "pass",
          duration_ms: rounded(performance.now() - started),
          ui_surface_count: uiCapture.captured,
          ui_surface_skipped_count: uiCapture.skipped,
          provided_download_count: downloadCapture.downloaded,
          unavailable_download_count: downloadCapture.unavailable,
          ...state,
        });
        console.log(`[browser] ${item.id}: pass; ${downloadCapture.downloaded} actual download(s), ${downloadCapture.unavailable} unavailable state(s)`);
      } catch (error) {
        caseResults.push({
          id: item.id,
          status: "fail",
          duration_ms: rounded(performance.now() - started),
          error: error?.message || String(error),
          browser_errors: browserErrors.slice(browserErrorOffset),
        });
        await page.screenshot({ path: path.join(SCREENSHOTS, `${item.id}-failure.png`), fullPage: false }).catch(() => {});
        console.log(`[browser] ${item.id}: fail - ${error?.message || String(error)}`);
      }
    }
    return {
      status: caseResults.every((item) => item.status === "pass") && !browserErrors.length ? "pass" : "fail",
      extension_disclosures: labels,
      cases: caseResults,
      errors: browserErrors,
    };
  } finally {
    if (browser) await browser.close();
    await new Promise((resolve) => server.close(resolve));
  }
}

async function installLocalValidationApi(page) {
  const user = {
    id: "local-validation-account",
    name: "Local Validation Account",
    email: "local-validation@deepbom.invalid",
    role: "user",
    provider: "email",
    email_verified: true,
    access_profile: "medical_ai",
    access_status: "active",
  };
  const allowed = {
    report: true,
    export: true,
    raw_export: true,
    regulatory_report: true,
    deepbom: false,
    perturbation: false,
    runtime_basin: false,
    deployment_sensitivity: false,
  };
  await page.route("**/api/**", async (route) => {
    const request = route.request();
    const pathname = new URL(request.url()).pathname;
    const fulfill = (body, status = 200) => route.fulfill({
      status,
      contentType: "application/json",
      body: JSON.stringify(body),
    });
    if (pathname === "/api/auth/config") return fulfill({ enabled: true, password: true, google: false });
    if (pathname === "/api/auth/me") return fulfill({ user });
    if (pathname === "/api/access/status" || pathname === "/api/access/check") {
      return fulfill({ status: "active", access_profile: "medical_ai", user, allowed });
    }
    if (pathname === "/api/report/sign") {
      const body = request.postDataJSON?.() || JSON.parse(request.postData() || "{}");
      return fulfill(localValidationAttestation(body));
    }
    if (pathname === "/api/reports" && request.method() === "GET") return fulfill({ reports: [] });
    return fulfill({ error: "local_validation_route_not_implemented", path: pathname }, 404);
  });
}

function localValidationAttestation(body = {}) {
  const kid = "deepbom-local-validation-hs256-20260804";
  const packageMembers = Array.isArray(body.package_members) ? body.package_members : [];
  const signedPayload = {
    schema: "deepbom.attestation_payload.v2.2",
    attestation_type: "local_validation_test_digest_registration",
    issuer: "DEEPBOM LOCAL VALIDATION - NOT PRODUCTION SIGNED",
    origin: "http://127.0.0.1",
    scope: body.scope || "local_validation",
    signature_version: "local-validation-v1",
    issued_at: GENERATED_AT,
    canonicalization: "RFC8785-JCS",
    signing_key: { kid, alg: "HS256" },
    attestation_scope: {
      attests: ["local browser export path executed against deterministic fixtures"],
      does_not_attest: ["production account authentication", "production signing", "analytical correctness"],
    },
    subject: {
      model_sha256: body.model_sha256 || "",
      artifact_id: body.artifact_id || "",
      target_id: body.target_id || "",
      target_label: body.target_label || "",
      target_profile_sha256: body.target_profile_sha256 || "",
    },
    package: {
      package_hash_sha256: body.package_hash_sha256 || "",
      package_hash_method: body.package_hash_method || "",
      unsigned_package_members: packageMembers,
      attestation_member: body.attestation_member || "attestation.json",
      attestation_member_excluded_from_package_hash: true,
      member_order: "lexicographic_utf8",
      path_normalization: "relative_posix_nfc",
      duplicate_member_policy: "reject_after_nfc_and_case_fold",
      undeclared_member_policy: "reject",
    },
    privacy: {
      model_uploaded: false,
      report_uploaded: false,
      account_identity_embedded: false,
      local_validation_only: true,
    },
  };
  return {
    schema: "deepbom.attestation.v2.2",
    signed_payload: signedPayload,
    signature: {
      alg: "HS256",
      kid,
      verification: "test-only placeholder; no production signing secret was used",
      value: "LOCAL_VALIDATION_NOT_A_PRODUCTION_SIGNATURE",
    },
    verification_status: {
      status: "local_validation_test_digest_registered",
      package_hash_sha256: body.package_hash_sha256 || "",
      member_digest_verification: "offline-capable",
      signature_verification: "not production signed",
      signing_key_id: kid,
    },
    verification_note: "This attestation exists only to exercise and verify the complete local export path. It is not a production signature.",
  };
}

async function captureProvidedDownloads(page, item) {
  const directory = path.join(DELIVERABLES, item.id);
  const downloadsDirectory = path.join(directory, "downloads");
  const extractedDirectory = path.join(directory, "extracted");
  await Promise.all([downloadsDirectory, extractedDirectory].map((value) => mkdir(value, { recursive: true })));
  const candidates = await page.locator('button[id^="download"]').evaluateAll((buttons) => buttons.map((button) => ({
    id: button.id,
    label: button.textContent?.trim() || button.id,
    disabled: Boolean(button.disabled),
    hidden: Boolean(button.hidden),
    ariaDisabled: button.getAttribute("aria-disabled") === "true",
    title: button.title || "",
  })));
  const excluded = new Map([
    ["downloadComparison", "requires two separately persisted audit snapshots"],
    ["downloadDeepBom", "requires a completed controlled analysis run"],
  ]);
  const records = [];
  const usedNames = new Set();
  for (const candidate of candidates) {
    const exclusion = excluded.get(candidate.id);
    if (exclusion || candidate.disabled || candidate.ariaDisabled) {
      records.push({
        ...candidate,
        status: "unavailable",
        reason: exclusion || candidate.title || "button disabled for this artifact/result state",
      });
      continue;
    }
    try {
      const { download, attempts } = await waitForProvidedDownload(page, item, candidate);
      const failure = await download.failure();
      if (failure) throw new Error(failure);
      let filename = safeFilename(download.suggestedFilename() || `${candidate.id}.bin`);
      if (usedNames.has(filename.toLocaleLowerCase("en-US"))) filename = `${candidate.id}--${filename}`;
      usedNames.add(filename.toLocaleLowerCase("en-US"));
      const destination = path.join(downloadsDirectory, filename);
      await download.saveAs(destination);
      const bytes = new Uint8Array(await readFile(destination));
      const record = {
        ...candidate,
        status: "downloaded",
        filename,
        byte_length: bytes.byteLength,
        sha256: sha256(bytes),
        capture_attempts: attempts,
      };
      if (filename.toLowerCase().endsWith(".zip")) {
        record.zip = await extractDownloadedZip(bytes, path.join(extractedDirectory, filename.replace(/\.zip$/i, "")));
      }
      records.push(record);
      await page.waitForFunction((id) => !document.getElementById(id)?.disabled, candidate.id, { timeout: 30_000 }).catch(() => {});
    } catch (error) {
      const message = error?.message || String(error);
      console.error(`[browser] ${item.id}: download ${candidate.id} failed - ${message}`);
      records.push({ ...candidate, status: "failed", error: message });
    }
  }
  const summary = {
    schema: "deepbom.local_provided_download_capture.v1",
    generated_at: GENERATED_AT,
    format_case: item.id,
    capture_basis: "actual browser button click and Playwright download event",
    authorization_fixture: {
      account: "verified local validation member",
      report: true,
      raw_export: true,
      regulatory_report: true,
      research_modules: false,
      production_authentication_exercised: false,
      production_attestation_exercised: false,
    },
    downloaded: records.filter((record) => record.status === "downloaded").length,
    unavailable: records.filter((record) => record.status === "unavailable").length,
    failed: records.filter((record) => record.status === "failed").length,
    records,
  };
  await writeJson(path.join(directory, "provided-downloads.json"), summary);
  const failures = records
    .filter((record) => record.status === "failed")
    .map((record) => `${record.id}: ${record.error}`);
  assert(summary.failed === 0, `${item.id} has ${summary.failed} failed provided download(s): ${failures.join("; ")}`);
  await assertProvidedDownloadLinkage(item.id, downloadsDirectory, extractedDirectory, records);
  return summary;
}

async function waitForProvidedDownload(page, item, candidate) {
  const large = LARGE_BROWSER_DOWNLOAD_IDS.has(candidate.id);
  const maxAttempts = large ? 1 : 2;
  const debugTimeout = Number(process.env.DEEPBOM_DOWNLOAD_TIMEOUT_MS || 0);
  const timeout = Number.isFinite(debugTimeout) && debugTimeout > 0
    ? debugTimeout
    : large ? 180_000 : 30_000;
  let lastError;
  for (let attempt = 1; attempt <= maxAttempts; attempt += 1) {
    try {
      const [download] = await Promise.all([
        page.waitForEvent("download", { timeout }),
        page.locator(`#${candidate.id}`).evaluate((button) => button.click()),
      ]);
      return { download, attempts: attempt };
    } catch (error) {
      lastError = error;
      if (attempt >= maxAttempts) break;
      const state = await page.evaluate((id) => ({
        auditStatus: document.querySelector("#analysisPlanStatus")?.textContent || "",
        selectedModel: document.querySelector("#selectedModelName")?.textContent || "",
        buttonPresent: Boolean(document.getElementById(id)),
        buttonDisabled: Boolean(document.getElementById(id)?.disabled),
      }), candidate.id);
      if (!state.auditStatus.includes("audit run complete") || !state.buttonPresent || state.buttonDisabled) {
        throw new Error(`${candidate.id} lost its completed-audit binding before retry: ${JSON.stringify(state)}; ${error?.message || error}`);
      }
      console.warn(`[browser] ${item.id}: retrying lightweight download ${candidate.id} after attempt ${attempt} produced no event`);
      await page.waitForTimeout(250);
    }
  }
  throw lastError;
}

async function assertProvidedDownloadLinkage(id, downloadsDirectory, extractedDirectory, records) {
  console.log(`[browser] ${id}: verifying downloaded artifact linkage`);
  const expected = JSON.parse(await readFile(path.join(RESULTS, `${id}.analysis.json`), "utf8"));
  const runtime = records.find((record) => record.id === "downloadRuntimeRequirements" && record.status === "downloaded");
  assert(runtime, `${id} runtime-requirement manifest download was not captured`);
  const runtimeDocument = JSON.parse(await readFile(path.join(downloadsDirectory, runtime.filename), "utf8"));
  assert(runtimeDocument.subject.sha256 === expected.model_sha256, `${id} downloaded runtime manifest is not artifact-hash-bound`);
  assert(runtimeDocument.subject.byte_length === expected.file_size_bytes, `${id} downloaded runtime manifest byte length diverges from analysis`);
  assert(runtimeDocument.subject.schema_or_opset === expectedArtifactSchema(expected), `${id} downloaded runtime manifest names the wrong artifact schema`);

  const pack = records.find((record) => record.id === "downloadContractPack" && record.status === "downloaded");
  assert(pack?.zip, `${id} Deployment Contract Pack download was not captured and verified`);
  const packRoot = path.join(extractedDirectory, pack.filename.replace(/\.zip$/i, ""));
  const envelope = JSON.parse(await readFile(path.join(packRoot, "deepbom_artifact_evidence_envelope.json"), "utf8"));
  assert(envelope.identity.sha256 === expected.model_sha256 && envelope.identity.byte_length === expected.file_size_bytes, `${id} downloaded contract-pack envelope diverges from analysis identity`);
  assert(envelope.graph.total_macs === (expected.total_macs ?? null), `${id} downloaded contract-pack envelope diverges from analysis MAC state`);
  console.log(`[browser] ${id}: contract-pack linkage verified`);

  const report = records.find((record) => record.id === "downloadMarkdown" && record.status === "downloaded");
  assert(report, `${id} Engineering Report download was not captured`);
  const reportText = await readFile(path.join(downloadsDirectory, report.filename), "utf8");
  assert(reportText.includes(expected.model_sha256), `${id} Engineering Report is not bound to the analyzed artifact SHA-256`);
  if (expected.total_macs != null) assert(reportText.includes(Number(expected.total_macs).toLocaleString("en-US")), `${id} Engineering Report omits the exact analyzed MAC total`);
  if (expected.gguf?.declared_tensor_byte_length != null) assert(reportText.includes(Number(expected.gguf.declared_tensor_byte_length).toLocaleString("en-US")), `${id} Engineering Report omits exact GGUF tensor payload bytes`);
  if (expected.safetensors?.payload_byte_length != null) assert(reportText.includes(Number(expected.safetensors.payload_byte_length).toLocaleString("en-US")), `${id} Engineering Report omits exact SafeTensors payload bytes`);
  if (expected.coreml?.description?.predicted_feature_name) assert(reportText.includes(expected.coreml.description.predicted_feature_name), `${id} Engineering Report omits the Core ML predicted-feature binding`);
  if (expected.coreml?.description?.predicted_probabilities_name) assert(reportText.includes(expected.coreml.description.predicted_probabilities_name), `${id} Engineering Report omits the Core ML probability binding`);
  console.log(`[browser] ${id}: Engineering Report linkage verified`);

  const raw = records.find((record) => record.id === "downloadRawData" && record.status === "downloaded");
  assert(raw?.zip, `${id} Raw Data ZIP download was not captured and verified`);
  const extracted = path.join(extractedDirectory, raw.filename.replace(/\.zip$/i, ""), "static", "static_analysis.json");
  const actual = JSON.parse(await readFile(extracted, "utf8"));
  const paths = [
    ["format"], ["model_sha256"], ["operator_count"], ["tensor_count"], ["total_macs"],
    ["size_breakdown", "constant_bytes"], ["tensor_liveness", "peak_bytes"],
    ["tensor_arena_plan", "combined_arena_bytes"], ["gguf", "declared_tensor_byte_length"],
    ["gguf", "payload_coverage_status"], ["safetensors", "payload_byte_length"],
    ["safetensors", "payload_coverage_status"], ["coreml", "description", "predicted_feature_name"],
    ["coreml", "description", "predicted_probabilities_name"],
  ];
  for (const pathParts of paths) {
    let expectedValue = expected;
    let actualValue = actual;
    for (const part of pathParts) {
      expectedValue = expectedValue?.[part];
      actualValue = actualValue?.[part];
    }
    if (expectedValue === undefined) continue;
    assert(Object.is(actualValue, expectedValue), `${id} Raw Data ZIP diverges at /${pathParts.join("/")}: expected ${JSON.stringify(expectedValue)}, got ${JSON.stringify(actualValue)}`);
  }
  console.log(`[browser] ${id}: Raw Data linkage verified`);
}

async function extractDownloadedZip(bytes, directory) {
  const members = parseStoredZip(bytes);
  await mkdir(directory, { recursive: true });
  const rows = [];
  for (const [name, memberBytes] of members) {
    const destination = path.resolve(directory, ...name.split("/"));
    assert(destination.startsWith(`${path.resolve(directory)}${path.sep}`), `ZIP member escaped extraction root: ${name}`);
    await mkdir(path.dirname(destination), { recursive: true });
    await writeFile(destination, memberBytes);
    rows.push({ name, byte_length: memberBytes.byteLength, sha256: sha256(memberBytes) });
  }
  let packageVerification = { status: "not_applicable", reason: "archive has no attestation.json member" };
  if (members.has("attestation.json")) {
    const verified = verifyPackageBytes(bytes);
    packageVerification = {
      status: "pass",
      package_hash_sha256: verified.package_hash_sha256,
      member_count: verified.member_count,
      signature_scope: "local_validation_test_only",
    };
  }
  const index = {
    schema: "deepbom.local_zip_contents.v1",
    member_count: rows.length,
    package_verification: packageVerification,
    members: rows,
  };
  await writeJson(path.join(directory, "_contents.json"), index);
  return index;
}

async function captureUiResults(page, item) {
  const directory = path.join(UI_RESULTS, item.id);
  await mkdir(directory, { recursive: true });
  const records = [];
  const workspaceButtons = await page.locator("[data-workflow-step]").evaluateAll((buttons) => buttons.map((button) => ({
    id: button.dataset.workflowStep || "",
    hidden: Boolean(button.hidden),
    disabled: Boolean(button.disabled),
    ariaDisabled: button.getAttribute("aria-disabled") === "true",
    module: button.dataset.workflowModule || "",
    label: button.textContent?.replace(/\s+/g, " ").trim() || "",
  })));
  for (const workspace of workspaceButtons) {
    if (!workspace.id) continue;
    const moduleLocked = workspace.module
      ? await page.locator(`[data-module-tab="${workspace.module}"]`).getAttribute("aria-disabled").catch(() => "true") === "true"
      : false;
    if (workspace.hidden || workspace.disabled || workspace.ariaDisabled || moduleLocked) {
      records.push({ type: "workspace", id: workspace.id, status: "skipped", reason: workspace.hidden ? "hidden" : moduleLocked ? "authorization_required" : "disabled", label: workspace.label });
      continue;
    }
    await clickAndSettle(page, `[data-workflow-step="${workspace.id}"]`);
    if (workspace.id === "audit") {
      const auditTabs = await page.locator("[data-audit-tab]").evaluateAll((buttons) => buttons.map((button) => ({ id: button.dataset.auditTab || "", hidden: Boolean(button.hidden), disabled: Boolean(button.disabled) })));
      for (const tab of auditTabs) {
        if (!tab.id || tab.hidden || tab.disabled) {
          records.push({ type: "audit_tab", id: tab.id, status: "skipped", reason: tab.hidden ? "format_hidden" : "disabled" });
          continue;
        }
        await clickAndSettle(page, `[data-audit-tab="${tab.id}"]`);
        records.push(await captureCurrentUiSurface(page, directory, `audit-${tab.id}`, "audit_tab", tab.id));
      }
      continue;
    }
    if (workspace.id === "graph") {
      const explorerTabs = await page.locator("[data-explorer-tab]").evaluateAll((buttons) => buttons.map((button) => ({ id: button.dataset.explorerTab || "", hidden: Boolean(button.hidden), disabled: Boolean(button.disabled) })));
      for (const tab of explorerTabs) {
        if (!tab.id || tab.hidden || tab.disabled) {
          records.push({ type: "explorer_tab", id: tab.id, status: "skipped", reason: tab.hidden ? "format_hidden" : "disabled" });
          continue;
        }
        await clickAndSettle(page, `[data-explorer-tab="${tab.id}"]`);
        records.push(await captureCurrentUiSurface(page, directory, `explorer-${tab.id}`, "explorer_tab", tab.id));
      }
      continue;
    }
    if (workspace.id === "output") {
      const moduleTabs = await page.locator("[data-module-tab]").evaluateAll((buttons) => buttons.map((button) => ({
        id: button.dataset.moduleTab || "",
        hidden: Boolean(button.hidden),
        disabled: Boolean(button.disabled),
        ariaDisabled: button.getAttribute("aria-disabled") === "true",
        label: button.textContent?.replace(/\s+/g, " ").trim() || "",
      })));
      for (const tab of moduleTabs) {
        if (!tab.id || tab.hidden || tab.disabled || tab.ariaDisabled) {
          records.push({ type: "output_tab", id: tab.id, status: "skipped", reason: tab.hidden ? "hidden" : tab.ariaDisabled ? "authorization_required" : "disabled", label: tab.label });
          continue;
        }
        await clickAndSettle(page, `[data-module-tab="${tab.id}"]`);
        records.push(await captureCurrentUiSurface(page, directory, `output-${tab.id}`, "output_tab", tab.id));
      }
      continue;
    }
    records.push(await captureCurrentUiSurface(page, directory, `workspace-${workspace.id}`, "workspace", workspace.id));
  }
  const index = {
    schema: "deepbom.local_ui_surface_capture.v1",
    generated_at: GENERATED_AT,
    format_case: item.id,
    viewport: { width: 1440, height: 1000, device_scale_factor: 1 },
    capture_basis: "actual UI tab clicks followed by rendered DOM/text and viewport screenshot",
    captured: records.filter((record) => record.status === "captured").length,
    skipped: records.filter((record) => record.status === "skipped").length,
    records,
  };
  await writeJson(path.join(directory, "ui-surface-index.json"), index);
  return index;
}

async function clickAndSettle(page, selector) {
  await page.locator(selector).evaluate((element) => element.click());
  await page.evaluate(() => new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve))));
  await page.waitForTimeout(40);
}

async function validateExecutionPlacementResponsive(page, id) {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.evaluate(() => new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve))));
  const geometry = await page.evaluate(() => {
    const root = document.querySelector("#executionPlacementPanel");
    const rect = root?.getBoundingClientRect();
    const levels = [...(root?.querySelectorAll(".execution-placement-level") || [])]
      .map((node) => node.getBoundingClientRect());
    const portfolios = [...(root?.querySelectorAll(".execution-placement-portfolio") || [])]
      .map((node) => node.getBoundingClientRect());
    return {
      hidden: root?.hidden ?? true,
      viewport_overflow_px: Math.max(0, document.documentElement.scrollWidth - innerWidth),
      local_overflow_px: root ? Math.max(0, root.scrollWidth - root.clientWidth) : 1,
      root_left: rect?.left ?? -1,
      root_right: rect?.right ?? innerWidth + 1,
      child_overflow: [...levels, ...portfolios].some((child) => child.left < rect.left - 1 || child.right > rect.right + 1),
    };
  });
  assert(!geometry.hidden, `${id} execution-placement panel is hidden on mobile`);
  assert(geometry.viewport_overflow_px <= 1 && geometry.local_overflow_px <= 1, `${id} execution-placement panel overflows the mobile viewport`);
  assert(geometry.root_left >= -1 && geometry.root_right <= 391 && !geometry.child_overflow, `${id} execution-placement cards escape their mobile container`);
  await page.setViewportSize({ width: 1440, height: 1000 });
  await page.evaluate(() => new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve))));
}

async function validateResponsiveNavigationContracts(page) {
  await clickAndSettle(page, '[data-workflow-step="output"]');
  const outputState = await page.evaluate(() => {
    const active = document.querySelector("[data-module-tab].active");
    return {
      id: active?.dataset.moduleTab || "",
      aria_disabled: active?.getAttribute("aria-disabled") === "true",
    };
  });
  assert(outputState.id && !outputState.aria_disabled, "Output navigation left an authorization-required module active");

  await page.setViewportSize({ width: 390, height: 844 });
  await clickAndSettle(page, '[data-workflow-step="graph"]');
  await clickAndSettle(page, '[data-explorer-tab="node"]');
  const mobileState = await page.evaluate(() => ({
    document_overflow_px: Math.max(0, document.documentElement.scrollWidth - innerWidth),
    segment_columns: getComputedStyle(document.querySelector(".nv-segments")).gridTemplateColumns,
  }));
  assert(mobileState.document_overflow_px <= 1, `TFLite Explorer overflows the 390 px viewport by ${mobileState.document_overflow_px} px`);
  assert(mobileState.segment_columns.split(" ").length === 2, "TFLite Explorer mobile overlays are not arranged in two bounded columns");
  await page.setViewportSize({ width: 1440, height: 1000 });
  await clickAndSettle(page, '[data-workflow-step="audit"]');
  await clickAndSettle(page, '[data-audit-tab="overview"]');
}

async function captureCurrentUiSurface(page, directory, name, type, id) {
  const safeName = safeFilename(name);
  const state = await page.evaluate(() => {
    const selectors = [
      "#modelPlan", "#auditWorkbench", "#summary", "#insightDashboard", "#perfVisuals", "#tables", "#diagramSection",
      "#findingsPanel", "#graphExplorer", "#redesignPanel", "#inferencePanel", "#outputModuleSelector", "#moduleRunConsole", "#actions",
    ];
    const visible = (element) => Boolean(element && !element.hidden && element.getClientRects().length > 0 && getComputedStyle(element).visibility !== "hidden");
    const nodes = selectors.map((selector) => document.querySelector(selector)).filter(visible);
    return {
      workspace: document.body.dataset.workspace || "",
      active_audit_tab: document.querySelector("[data-audit-tab].active")?.dataset.auditTab || "",
      active_explorer_tab: document.querySelector("[data-explorer-tab].active")?.dataset.explorerTab || "",
      active_module_tab: document.querySelector("[data-module-tab].active")?.dataset.moduleTab || "",
      viewport_overflow_px: Math.max(0, document.documentElement.scrollWidth - innerWidth),
      document_height_px: document.documentElement.scrollHeight,
      visible_root_ids: nodes.map((node) => node.id),
      text: nodes.map((node) => node.innerText || "").join("\n\n--- UI ROOT ---\n\n"),
      html: nodes.map((node) => node.outerHTML || "").join("\n\n<!-- UI ROOT -->\n\n"),
    };
  });
  const preferredSelector = type === "output_tab"
    ? `[data-module-panel="${id}"]`
    : type === "explorer_tab"
      ? "#graphExplorer"
      : type === "audit_tab"
        ? id === "roofline" ? "#tables" : id === "stage" ? "#diagramSection" : id === "overview" ? "#summary" : "#perfVisuals"
        : ({ input: "#modelPlan", findings: "#findingsPanel", redesign: "#redesignPanel", runtime: "#inferencePanel" })[id] || "#moduleRunConsole";
  const preferred = page.locator(preferredSelector).first();
  if (await preferred.count() && await preferred.isVisible().catch(() => false)) await preferred.scrollIntoViewIfNeeded();
  else await page.evaluate(() => scrollTo(0, 0));
  await page.screenshot({ path: path.join(directory, `${safeName}.png`), fullPage: false });
  await writeText(path.join(directory, `${safeName}.txt`), state.text);
  await writeText(path.join(directory, `${safeName}.html`), state.html);
  return {
    type,
    id,
    status: "captured",
    screenshot: `${safeName}.png`,
    text: `${safeName}.txt`,
    dom: `${safeName}.html`,
    workspace: state.workspace,
    active_audit_tab: state.active_audit_tab,
    active_explorer_tab: state.active_explorer_tab,
    active_module_tab: state.active_module_tab,
    visible_root_ids: state.visible_root_ids,
    viewport_overflow_px: state.viewport_overflow_px,
    document_height_px: state.document_height_px,
  };
}

function safeFilename(value) {
  const safe = String(value || "artifact")
    .replace(/[<>:"/\\|?*\u0000-\u001f]/g, "_")
    .replace(/[. ]+$/g, "")
    .slice(0, 180);
  return safe || "artifact";
}

async function runContractCheck(script) {
  const started = performance.now();
  const result = await runCommand(process.execPath, [script]);
  return {
    script,
    status: result.code === 0 ? "pass" : "fail",
    exit_code: result.code,
    duration_ms: rounded(performance.now() - started),
    output: `${result.stdout}${result.stderr ? `\n${result.stderr}` : ""}`.trim(),
  };
}

async function runCommand(command, args) {
  return new Promise((resolve) => {
    const child = spawn(command, args, { cwd: ROOT, windowsHide: true });
    let stdout = "";
    let stderr = "";
    child.stdout.on("data", (chunk) => { stdout += chunk; });
    child.stderr.on("data", (chunk) => { stderr += chunk; });
    child.on("error", (error) => resolve({ code: -1, stdout, stderr: `${stderr}\n${stack(error)}` }));
    child.on("close", (code) => resolve({ code: code ?? -1, stdout, stderr }));
  });
}

async function repositoryContentIdentity() {
  const listing = await runCommand(
    "git",
    ["ls-files", "-z", "--cached", "--others", "--exclude-standard"],
  );
  if (listing.code !== 0) {
    throw new Error(`Cannot enumerate repository content: ${listing.stderr || listing.stdout}`);
  }
  const files = listing.stdout.split("\0").filter(Boolean).sort();
  const digest = createHash("sha256");
  for (const relative of files) {
    let bytes;
    try {
      bytes = await readFile(path.join(ROOT, relative));
    } catch (error) {
      if (error?.code !== "ENOENT") throw error;
      digest.update(`${relative}\0-1\0missing\n`);
      continue;
    }
    digest.update(`${relative}\0${bytes.byteLength}\0${sha256(bytes)}\n`);
  }
  return {
    sha256: digest.digest("hex"),
    fileCount: files.length,
    hashBasis: "sha256(sorted records: git ls-files --cached --others --exclude-standard path + NUL + byte_length + NUL + file_sha256 + LF; deleted tracked files use -1/missing)",
  };
}

// A null count is the analyzer stating that it did not decode that inventory.
// Collapsing it to 0 would publish "not assessed" as a measured zero, so the
// null is preserved and reported alongside the analyzer's own reason.
function assessedCount(value, fallback = 0) {
  if (value === null) return null;
  if (value === undefined) return Number(fallback ?? 0);
  return Number(value);
}

function assessmentStatus(analysis, ...values) {
  if (!values.some((value) => value === null)) return undefined;
  return analysis.mac_assessment?.status || analysis.tensor_inventory?.status || "not_assessed";
}

function graphSummary(analysis, hash, bytes) {
  const operatorCount = assessedCount(analysis.operator_count, analysis.ops?.length ?? 0);
  const tensorCount = assessedCount(analysis.tensor_count, analysis.tensors?.length ?? 0);
  const totalMacs = assessedCount(analysis.total_macs, 0);
  const status = assessmentStatus(analysis, operatorCount, tensorCount, totalMacs);
  return {
    detected_format: analysis.format,
    filename: analysis.filename,
    byte_length: bytes,
    sha256: hash,
    hash_basis: "artifact_file_bytes_sha256",
    operator_count: operatorCount,
    tensor_count: tensorCount,
    input_count: Number(analysis.inputs?.length || 0),
    output_count: Number(analysis.outputs?.length || 0),
    total_macs: totalMacs,
    ...(status ? { inventory_assessment_status: status } : {}),
  };
}

function metadataSummary(analysis, hash, bytes) {
  const tensorCount = assessedCount(analysis.tensor_count, 0);
  const status = assessmentStatus(analysis, tensorCount);
  return {
    detected_format: analysis.format,
    filename: analysis.filename,
    byte_length: bytes,
    sha256: hash,
    hash_basis: "artifact_file_bytes_sha256",
    tensor_count: tensorCount,
    input_count: Number(analysis.inputs?.length || 0),
    output_count: Number(analysis.outputs?.length || 0),
    parser_schema: analysis.metadata_presence?.schema || analysis.tensor_inventory?.schema || null,
    ...(status ? { inventory_assessment_status: status } : {}),
  };
}

function concatBytes(...chunks) {
  const result = new Uint8Array(chunks.reduce((sum, chunk) => sum + chunk.length, 0));
  let offset = 0;
  for (const chunk of chunks) { result.set(chunk, offset); offset += chunk.length; }
  return result;
}

async function browserFile(filePath, relativePath = "") {
  const bytes = await readFile(filePath);
  const value = new File([bytes], path.basename(filePath));
  if (relativePath) Object.defineProperty(value, "webkitRelativePath", { value: relativePath.replaceAll("\\", "/") });
  return value;
}

async function directoryFiles(directory) {
  const paths = await recursiveFiles(directory);
  const rootName = path.basename(directory);
  return Promise.all(paths.map((filePath) => browserFile(filePath, `${rootName}/${path.relative(directory, filePath).replaceAll("\\", "/")}`)));
}

async function recursiveFiles(directory) {
  const entries = await readdir(directory, { withFileTypes: true });
  const rows = [];
  for (const entry of entries.sort((a, b) => a.name.localeCompare(b.name))) {
    const item = path.join(directory, entry.name);
    if (entry.isDirectory()) rows.push(...await recursiveFiles(item));
    else if (entry.isFile()) rows.push(item);
  }
  return rows;
}

async function writeManifest() {
  const files = (await recursiveFiles(OUTPUT))
    .filter((file) => path.basename(file) !== "sha256-manifest.json")
    .sort((a, b) => a.localeCompare(b));
  const manifest = [];
  for (const file of files) {
    const bytes = await readFile(file);
    manifest.push({ path: path.relative(OUTPUT, file).replaceAll("\\", "/"), byte_length: bytes.byteLength, sha256: sha256(bytes) });
  }
  await writeJson(path.join(OUTPUT, "sha256-manifest.json"), {
    schema: "deepbom.local_validation_manifest.v1",
    generated_at: GENERATED_AT,
    hash_method: "SHA-256 over exact file bytes",
    file_count: manifest.length,
    files: manifest,
  });
}

async function buildArtifactCatalog(items) {
  const formats = [];
  for (const item of items) {
    const providedPath = path.join(DELIVERABLES, item.id, "provided-downloads.json");
    const uiPath = path.join(UI_RESULTS, item.id, "ui-surface-index.json");
    const provided = await readJsonIfPresent(providedPath);
    const ui = await readJsonIfPresent(uiPath);
    formats.push({
      id: item.id,
      advertised_format: item.label,
      analysis: `results/${item.id}.analysis.json`,
      deterministic_contract_exports: `exports/${item.id}/`,
      provided_download_index: provided ? `deliverables/${item.id}/provided-downloads.json` : null,
      actual_downloads: (provided?.records || []).filter((record) => record.status === "downloaded").map((record) => ({
        control_id: record.id,
        label: record.label,
        path: `deliverables/${item.id}/downloads/${record.filename}`,
        byte_length: record.byte_length,
        sha256: record.sha256,
        extracted_zip_directory: record.zip ? `deliverables/${item.id}/extracted/${record.filename.replace(/\.zip$/i, "")}/` : null,
        zip_member_count: record.zip?.member_count || 0,
      })),
      unavailable_downloads: (provided?.records || []).filter((record) => record.status === "unavailable").map((record) => ({
        control_id: record.id,
        label: record.label,
        reason: record.reason,
      })),
      ui_surface_index: ui ? `ui-results/${item.id}/ui-surface-index.json` : null,
      captured_ui_surfaces: (ui?.records || []).filter((record) => record.status === "captured").map((record) => ({
        type: record.type,
        id: record.id,
        screenshot: `ui-results/${item.id}/${record.screenshot}`,
        text: `ui-results/${item.id}/${record.text}`,
        dom: `ui-results/${item.id}/${record.dom}`,
        viewport_overflow_px: record.viewport_overflow_px,
      })),
      unavailable_ui_surfaces: (ui?.records || []).filter((record) => record.status === "skipped").map((record) => ({
        type: record.type,
        id: record.id,
        reason: record.reason,
      })),
    });
  }
  return {
    schema: "deepbom.local_validation_artifact_catalog.v1",
    generated_at: GENERATED_AT,
    capture_statement: "Downloads were produced by clicking the actual enabled UI controls. UI evidence was captured after clicking each reachable workspace and subtab. No report or viewer payload was hand-authored for this catalog.",
    integrity_manifest: "sha256-manifest.json",
    limitations: [
      "Authentication and package attestation use an explicitly marked local validation fixture, not production identity or signing secrets.",
      "Authorization-scoped research results and comparison exports are not fabricated; their unavailable UI states and reasons are recorded.",
      "Fixtures prove the bounded paths listed here, not every producer-specific or large-model variant.",
    ],
    formats,
  };
}

async function readJsonIfPresent(file) {
  try {
    return JSON.parse(await readFile(file, "utf8"));
  } catch (error) {
    if (error?.code === "ENOENT") return null;
    throw error;
  }
}

function summaryMarkdown(matrix) {
  const lines = [
    "# Supported Format Validation",
    "",
    `- Generated: ${matrix.generated_at}`,
    `- Overall: ${matrix.overall_status.toUpperCase()}`,
    `- Git commit: ${matrix.environment.git_commit}${matrix.environment.git_worktree_dirty ? " (working tree dirty)" : ""}`,
    `- Repository content SHA-256: ${matrix.environment.repository_content_sha256}`,
    `- Runtime: ${matrix.environment.node} / ${matrix.environment.platform} ${matrix.environment.architecture}`,
    `- Target profile: ${matrix.environment.target_profile}`,
    `- Actual UI downloads captured: ${matrix.coverage.actual_provided_downloads}`,
    `- Rendered UI surfaces captured: ${matrix.coverage.captured_ui_surfaces}`,
    "",
    "| Advertised path | Control | Fixture provenance | Parser + export | Browser selection | SHA-256 |",
    "| --- | --- | --- | --- | --- | --- |",
  ];
  for (const item of matrix.formats) {
    const browser = matrix.browser_selection.cases?.find((row) => row.id === item.id);
    lines.push(`| ${item.advertised_format} | ${item.selection_control} | ${item.fixture_provenance} | ${item.status} | ${browser?.status || "not_run"} | ${item.sha256 || "n/a"} |`);
  }
  lines.push(
    "",
    "`deliverables/<format>/downloads/` contains the exact bytes produced by the real enabled download buttons. ZIP files are expanded under `deliverables/<format>/extracted/` without replacing the original archive.",
    "`ui-results/<format>/` contains a viewport PNG, rendered text, and rendered DOM for every reachable workspace/subtab. `ui-surface-index.json` records inaccessible or format-hidden states instead of fabricating results.",
    "`artifact-catalog.json` links the analyses, actual downloads, extracted ZIP members, UI captures, and unavailable states. Parser-level deployment-contract documents remain under `exports/` for independent comparison.",
    "The fixture and every result byte are covered by `sha256-manifest.json`. This directory is local-only and ignored by Git.",
    "Local validation uses an explicitly non-production authentication and attestation fixture. Authorization-scoped research results and comparison exports are recorded as unavailable and are not synthesized.",
    "This matrix proves the advertised selection, bounded parsing, export, and viewer paths for the listed fixtures. It is not a claim that every producer-specific or large-model variant has corpus coverage.",
    "",
  );
  return lines.join("\n");
}

function createStaticServer(root) {
  return createServer(async (request, response) => {
    try {
      const url = new URL(request.url, "http://127.0.0.1");
      if (url.pathname.startsWith("/api/")) return send(response, 404, "application/json", '{"error":"not_found"}');
      const relative = url.pathname === "/web/" ? "web/index.html" : decodeURIComponent(url.pathname).replace(/^\/+/, "");
      const file = path.resolve(root, relative);
      if (!file.startsWith(`${root}${path.sep}`)) return send(response, 403, "text/plain", "forbidden");
      send(response, 200, mimeType(file), await readFile(file));
    } catch (error) {
      send(response, 404, "text/plain", `not found: ${error?.code || error?.message || "unknown"}`);
    }
  });
}

function send(response, status, type, body) {
  response.writeHead(status, { "content-type": type, "cache-control": "no-store" });
  response.end(body);
}

function mimeType(file) {
  return ({
    ".css": "text/css; charset=utf-8",
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".mjs": "text/javascript; charset=utf-8",
    ".json": "application/json",
    ".wasm": "application/wasm",
  })[path.extname(file).toLowerCase()] || "application/octet-stream";
}

function sha256(bytes) {
  return createHash("sha256").update(bytes).digest("hex");
}

function assert(value, message) {
  if (!value) throw new Error(message);
}

function rounded(value) {
  return Number(Number(value || 0).toFixed(3));
}

function stack(error) {
  return String(error?.stack || error?.message || error);
}

async function writeJson(file, value) {
  await writeFile(file, `${JSON.stringify(value, null, 2)}\n`);
}

async function writeText(file, value) {
  await writeFile(file, String(value || ""));
}

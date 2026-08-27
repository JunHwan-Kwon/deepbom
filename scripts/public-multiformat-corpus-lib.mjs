import AdmZip from "adm-zip";
import { createHash } from "node:crypto";
import { createReadStream, createWriteStream } from "node:fs";
import { mkdir, readFile, rename, rm } from "node:fs/promises";
import path from "node:path";
import { Readable } from "node:stream";
import { pipeline } from "node:stream/promises";

import { readArtifactBundle } from "../web/lib/artifact-bundle.js";
import { readCoreMlModelFile } from "../web/lib/coreml-metadata-adapter.js";
import { readMetadataModelFile } from "../web/lib/metadata-model-adapters.js";
import { analyzeOnnxModel } from "../web/onnx.js";
import { canonicalJson } from "../web/lib/report-utils.js";
import { deepBomCorpusCacheDir, fileIdentity, huggingFaceRevisionUrl } from "./hash-bound-corpus-cache.mjs";

export const PUBLIC_MULTIFORMAT_CORPUS_PATH = "corpus/public-multiformat-corpus.v1.json";
export const PUBLIC_MULTIFORMAT_CORPUS_SCHEMA = "deepbom.public_multiformat_corpus.v1";
export const PUBLIC_MULTIFORMAT_RECEIPT_SCHEMA = "deepbom.public_multiformat_artifact_receipt.v1";
export const CYCLONEDX_OBSERVATION_SCHEMA = "deepbom.cyclonedx_generalization_observation.v1";
const FORMATS = new Set(["onnx", "gguf", "safetensors", "coreml"]);
const DOWNLOAD_ATTEMPTS = 3;
const DOWNLOAD_TIMEOUT_MS = 300_000;

export async function readPublicMultiformatCorpus(filename = PUBLIC_MULTIFORMAT_CORPUS_PATH) {
  return validatePublicMultiformatCorpus(JSON.parse(await readFile(filename, "utf8")));
}

export function validatePublicMultiformatCorpus(manifest) {
  if (manifest?.schema !== PUBLIC_MULTIFORMAT_CORPUS_SCHEMA || !Array.isArray(manifest.artifacts)) {
    throw new Error(`Unsupported public multiformat corpus schema: ${manifest?.schema || "missing"}.`);
  }
  const ids = new Set();
  const pathRecords = new Set();
  for (const artifact of manifest.artifacts) {
    if (!/^[a-z0-9][a-z0-9-]+$/.test(String(artifact.id || "")) || ids.has(artifact.id)) throw new Error(`Invalid or duplicate artifact id ${artifact.id || "missing"}.`);
    if (!FORMATS.has(artifact.format)) throw new Error(`${artifact.id}: unsupported format ${artifact.format || "missing"}.`);
    validateSource(artifact);
    if (!Array.isArray(artifact.files) || !artifact.files.length) throw new Error(`${artifact.id}: source files are missing.`);
    const paths = new Set();
    for (const file of artifact.files) {
      if (!safeRelativePath(file.path) || paths.has(file.path)) throw new Error(`${artifact.id}: unsafe or duplicate path ${file.path || "missing"}.`);
      if (!Number.isSafeInteger(file.size_bytes) || file.size_bytes < 1 || !isSha256(file.sha256)) throw new Error(`${artifact.id}/${file.path}: byte identity is invalid.`);
      if (artifact.source.kind === "apple_developer_asset" && (!file.source_url || new URL(file.source_url).hostname !== "ml-assets.apple.com")) {
        throw new Error(`${artifact.id}/${file.path}: Apple source URL is invalid.`);
      }
      paths.add(file.path);
      pathRecords.add(`${artifact.id}\0${file.path}`);
    }
    ids.add(artifact.id);
  }
  if (manifest.summary?.path_record_count !== manifest.artifacts.length) throw new Error("Corpus path-record count does not conserve artifacts.");
  const uniquePrimary = new Set(manifest.artifacts.map(primaryArtifactSha256));
  if (manifest.summary?.unique_primary_artifact_count !== uniquePrimary.size) throw new Error("Corpus unique-primary count does not conserve SHA-256 identities.");
  if (pathRecords.size < manifest.artifacts.length) throw new Error("Corpus path/file inventory is unexpectedly sparse.");
  const sourceFiles = manifest.artifacts.flatMap((artifact) => artifact.files);
  if (manifest.summary?.source_file_count !== sourceFiles.length) throw new Error("Corpus source-file count does not conserve artifact files.");
  if (manifest.summary?.total_declared_download_bytes !== sourceFiles.reduce((sum, file) => sum + file.size_bytes, 0)) throw new Error("Corpus declared byte total does not conserve source files.");
  const boundOnnxSidecars = manifest.artifacts.filter((artifact) => artifact.format === "onnx" && artifact.files.some((file) => file.role === "external_data")).length;
  if (manifest.summary?.bound_onnx_sidecar_record_count !== boundOnnxSidecars) throw new Error("Corpus ONNX sidecar count does not conserve source records.");
  return manifest;
}

function validateSource(artifact) {
  const source = artifact.source || {};
  if (source.kind === "huggingface_repository") {
    if (!/^[A-Za-z0-9_.-]+\/[A-Za-z0-9_.-]+$/.test(String(source.repository || "")) || !/^[0-9a-f]{40}$/.test(String(source.revision || ""))) {
      throw new Error(`${artifact.id}: Hugging Face source is not revision-bound.`);
    }
  } else if (source.kind === "apple_developer_asset") {
    if (source.publisher !== "Apple" || source.revision !== null || source.repository !== null) throw new Error(`${artifact.id}: Apple no-revision boundary is not explicit.`);
  } else throw new Error(`${artifact.id}: unsupported source kind ${source.kind || "missing"}.`);
}

export function publicMultiformatCacheDir() {
  return deepBomCorpusCacheDir("public-multiformat-corpus-v1");
}

export function publicArtifactFilePath(cacheDir, artifact, file) {
  return path.join(cacheDir, artifact.format, artifact.id, ...file.path.split("/"));
}

export async function ensurePublicArtifactFiles(cacheDir, artifact, { offline = false } = {}) {
  const rows = [];
  for (const file of artifact.files) {
    const filename = publicArtifactFilePath(cacheDir, artifact, file);
    const observed = await fileIdentity(filename);
    if (observed?.size === file.size_bytes && observed.sha256 === file.sha256) {
      rows.push({ ...file, filename, downloaded: false });
      continue;
    }
    if (offline) throw new Error(`${artifact.id}/${file.path}: verified cache entry is unavailable in offline mode.`);
    const url = artifact.source.kind === "huggingface_repository" ? huggingFaceRevisionUrl({ repository: artifact.source.repository, revision: artifact.source.revision }, file) : file.source_url;
    await mkdir(path.dirname(filename), { recursive: true });
    const partial = `${filename}.partial`;
    await rm(partial, { force: true });
    await downloadPinnedFile(url, partial, artifact, file);
    await rm(filename, { force: true });
    await rename(partial, filename);
    rows.push({ ...file, filename, downloaded: true });
  }
  return rows;
}

async function downloadPinnedFile(url, partial, artifact, file) {
  let lastError = null;
  for (let attempt = 1; attempt <= DOWNLOAD_ATTEMPTS; attempt += 1) {
    await rm(partial, { force: true });
    try {
      const response = await fetch(url, {
        redirect: "follow",
        headers: { "User-Agent": "DeepBOM-public-corpus/1.0" },
        signal: AbortSignal.timeout(DOWNLOAD_TIMEOUT_MS),
      });
      if (!response.ok || !response.body) throw new Error(`HTTP ${response.status}`);
      validateDownloadHost(artifact, response.url);
      const declaredLength = Number(response.headers.get("content-length"));
      if (Number.isSafeInteger(declaredLength) && declaredLength > 0 && declaredLength !== file.size_bytes) {
        await response.body.cancel();
        throw new Error(`server length ${declaredLength} differs from pinned ${file.size_bytes}`);
      }
      await pipeline(Readable.fromWeb(response.body), createWriteStream(partial, { flags: "wx" }));
      const downloaded = await fileIdentity(partial);
      if (downloaded?.size !== file.size_bytes || downloaded.sha256 !== file.sha256) throw new Error("downloaded bytes do not match the pinned identity");
      return;
    } catch (error) {
      lastError = error;
      await rm(partial, { force: true });
      if (attempt < DOWNLOAD_ATTEMPTS) await new Promise((resolve) => setTimeout(resolve, attempt * 1_000));
    }
  }
  throw new Error(`${artifact.id}/${file.path}: download failed after ${DOWNLOAD_ATTEMPTS} attempts (${lastError?.message || lastError}).`);
}

function validateDownloadHost(artifact, finalUrl) {
  const host = new URL(finalUrl).hostname.toLowerCase();
  if (artifact.source.kind === "apple_developer_asset") {
    if (host !== "ml-assets.apple.com") throw new Error(`${artifact.id}: Apple asset redirected to unexpected host ${host}.`);
    return;
  }
  if (!(host === "huggingface.co" || host.endsWith(".huggingface.co") || host.endsWith(".xethub.hf.co") || host.endsWith(".hf.co"))) {
    throw new Error(`${artifact.id}: Hugging Face asset redirected to unexpected host ${host}.`);
  }
}

export async function analyzePublicMultiformatArtifact(cacheDir, artifact, { offline = false } = {}) {
  const cached = await ensurePublicArtifactFiles(cacheDir, artifact, { offline });
  const analysis = await analyzeCachedArtifact(artifact, cached);
  const serialized = JSON.stringify(analysis);
  const observation = buildCycloneDxGeneralizationObservation(artifact, analysis);
  const body = {
    artifact_id: artifact.id,
    format: artifact.format,
    source: artifact.source,
    stratum: artifact.stratum,
    source_files: artifact.files.map(({ source_url: _sourceUrl, ...file }) => file),
    primary_artifact_sha256: primaryArtifactSha256(artifact),
    analysis_sha256: sha256(serialized),
    analysis_summary: compactAnalysisSummary(analysis),
    cyclonedx_observation: observation,
  };
  return {
    receipt: { schema: PUBLIC_MULTIFORMAT_RECEIPT_SCHEMA, ...body, receipt_sha256: sha256(canonicalJson(body)) },
    analysis,
  };
}

async function analyzeCachedArtifact(artifact, cached) {
  if (artifact.format === "onnx") {
    const primary = cached.find((row) => row.role === "model") || cached.find((row) => row.path.toLowerCase().endsWith(".onnx")) || cached[0];
    const bytes = new Uint8Array(await readFile(primary.filename));
    const modelDirectory = path.posix.dirname(primary.path);
    const externalDataFiles = [];
    for (const source of cached.filter((row) => row !== primary)) {
      const sidecarBytes = new Uint8Array(await readFile(source.filename));
      externalDataFiles.push({
        path: path.posix.relative(modelDirectory, source.path),
        bytes: sidecarBytes,
        sha256: source.sha256,
        sha1: createHash("sha1").update(sidecarBytes).digest("hex"),
      });
    }
    return analyzeOnnxModel(bytes, path.basename(primary.filename), null, { externalDataFiles });
  }
  if (artifact.format === "gguf") {
    const bytes = await readFile(cached[0].filename);
    return (await readMetadataModelFile(new File([bytes], path.basename(cached[0].filename)), "gguf")).analysis;
  }
  if (artifact.format === "safetensors") {
    return (await readArtifactBundle(await browserFiles(artifact, cached))).analysis;
  }
  const source = cached[0];
  if (source.path.toLowerCase().endsWith(".zip")) return (await readArtifactBundle(coreMlZipFiles(source.filename))).analysis;
  const bytes = await readFile(source.filename);
  return (await readCoreMlModelFile(new File([bytes], path.basename(source.filename)))).analysis;
}

async function browserFiles(artifact, cached) {
  const rows = [];
  for (const source of cached) {
    const bytes = await readFile(source.filename);
    const file = new File([bytes], path.basename(source.path));
    Object.defineProperty(file, "webkitRelativePath", { value: `${artifact.id}/${source.path}` });
    rows.push(file);
  }
  return rows;
}

function coreMlZipFiles(filename) {
  const zip = new AdmZip(filename);
  const entries = zip.getEntries().filter((entry) => !entry.isDirectory);
  if (!entries.length || entries.length > 20_000) throw new Error(`Core ML package ZIP entry count ${entries.length} is invalid.`);
  return entries.map((entry) => {
    const relative = safeRelativePath(entry.entryName) ? entry.entryName : null;
    if (!relative) throw new Error(`Core ML package ZIP contains unsafe path ${entry.entryName}.`);
    const file = new File([entry.getData()], path.posix.basename(relative));
    Object.defineProperty(file, "webkitRelativePath", { value: relative });
    return file;
  });
}

export function compactAnalysisSummary(analysis) {
  const storage = analysis.tensor_storage_summary || {};
  const contractAssessment = serializedContractAssessment(analysis);
  const graph = serializedGraphAssessment(analysis);
  return {
    status: contractAssessment.status,
    format: analysis.format,
    file_size_bytes: Number(analysis.file_size_bytes ?? analysis.file_size ?? 0),
    operator_count: nullableInteger(analysis.operator_count),
    tensor_count: nullableInteger(analysis.tensor_count),
    input_count: (analysis.inputs || []).length,
    output_count: (analysis.outputs || []).length,
    graph_presence: graph.status,
    quantization_classification: analysis.quantization_status?.classification || "not_emitted",
    quantization_assessment_status: analysis.quantization_status?.assessment_status || "assessed_from_serialized_contract",
    storage_encoding_count: Number(storage.encoding_count || 0),
    storage_effective_bits_per_element: storage.effective_bits_per_element ?? null,
    serialized_contract_status: contractAssessment.status,
    serialized_contract_issue_count: contractAssessment.issue_count,
    serialized_contract_issue_codes: contractAssessment.issues.map((row) => row.code),
    weight_integrity_status: analysis.weight_integrity?.status || analysis.tensor_numerical_integrity?.status || "not_emitted",
    external_data_status: analysis.onnx_external_data?.status || "not_applicable",
    model_type: analysis.coreml?.model_type || analysis.safetensors?.hf_architecture_contract?.model_type || analysis.gguf?.architecture || null,
    opset_imports: (analysis.opsets || []).map((row) => ({ domain: row.domain || "ai.onnx", version: row.version })),
  };
}

export function buildCycloneDxGeneralizationObservation(artifact, analysis) {
  const parameters = [...parameterRows("input", analysis.inputs), ...parameterRows("output", analysis.outputs)];
  const representationProfiles = quantizationRepresentationProfiles(analysis);
  const quantizationProfiles = representationProfiles.filter((row) => row.category === "quantization");
  const axes = [...new Set(quantizationProfiles.flatMap((row) => row.axes || []).filter(Number.isInteger))].sort((a, b) => a - b);
  const schemes = [...new Set(quantizationProfiles.map((row) => row.scheme).filter(Boolean))].sort();
  const granularities = [...new Set(quantizationProfiles.flatMap((row) => row.granularities || []).filter(Boolean))].sort();
  const singleton = singletonQuantizationAssessment(analysis, quantizationProfiles, axes, granularities);
  const contractAssessment = serializedContractAssessment(analysis);
  const graph = serializedGraphAssessment(analysis);
  const representationsUnavailable = representationProfiles.length === 0 && graph.status === "serialized_graph_payload_not_decoded";
  const fieldEvidence = [
    field("artifact.hash.sha256", "OBSERVED", primaryArtifactSha256(artifact), "downloaded bytes"),
    field("artifact.format", "OBSERVED", artifact.format, "container signature/parser"),
    field("artifact.source.revision", artifact.source.revision ? "EXTERNAL" : "UNAVAILABLE", artifact.source.revision, artifact.source.revision ? "manifest-bound immutable repository revision; not encoded in artifact bytes" : "publisher URL exposes no source revision"),
    field("artifact.source.license", artifact.source.license_metadata ? "EXTERNAL" : "UNAVAILABLE", artifact.source.license_metadata, "repository/catalog metadata; not inferred from bytes"),
    field("model.graph", graph.evidence_class, graph.value, graph.basis),
    field("model.parameters.external", parameters.length ? "OBSERVED" : "NOT_APPLICABLE", parameters.length || null, parameters.length ? "serialized model interface" : "external interface is not serialized in this container"),
    field("model.precision_or_quantization.representations", representationProfiles.length ? "OBSERVED_DERIVED" : representationsUnavailable ? "UNAVAILABLE" : "NOT_APPLICABLE", representationProfiles.length || null,
      representationsUnavailable ? "serialized model payload was not decoded within the bounded inspection limit" : "serialized dtype, operator, and storage metadata"),
    field("artifact.serialized_contract.validity", "OBSERVED_DERIVED", contractAssessment, "format parser invariants and source-pinned field/cardinality rules"),
    field("deployment.assignment", "RUNTIME_REQUIRED", null, "actual backend/device assignment is not established by static bytes"),
    field("task.accuracy", "OUT_OF_SCOPE", null, "artifact inspection does not execute a labeled validation set"),
  ];
  return {
    schema: CYCLONEDX_OBSERVATION_SCHEMA,
    artifact_id: artifact.id,
    artifact_sha256: primaryArtifactSha256(artifact),
    format: artifact.format,
    evidence_boundary: {
      observed: "Artifact identity, serialized structure, tensor/storage contracts, and external parameter contracts where the format carries them.",
      not_established: "Observed runtime placement, device latency, task accuracy, clinical validity, and release readiness.",
    },
    field_evidence: fieldEvidence,
    serialized_contract_assessment: contractAssessment,
    external_parameters: parameters,
    model_representations: {
      representation_profiles: representationProfiles,
      representation_count: representationProfiles.length,
      quantization_representation_count: quantizationProfiles.length,
      precision_or_storage_representation_count: representationProfiles.length - quantizationProfiles.length,
    },
    model_quantization: {
      classification: analysis.quantization_status?.classification || "not_emitted",
      representation_profiles: quantizationProfiles,
      scheme_set: schemes,
      granularity_set: granularities,
      axis_set: axes,
      exact_numerical_contract_representable_by_one_flat_object: singleton.value,
      singleton_assessment_status: singleton.status,
      singleton_limitation: singleton.reason,
    },
    schema_pressure: schemaPressure(artifact.format, parameters, quantizationProfiles, singleton),
  };
}

function serializedContractAssessment(analysis) {
  const issues = [];
  const gguf = analysis.gguf || {};
  for (const [code, count] of [
    ["GGUF_INVALID_TENSOR_BLOCK_CARDINALITY", gguf.invalid_tensor_cardinality_count],
    ["GGUF_UNSUPPORTED_TENSOR_ENCODING", gguf.unsupported_ggml_type_count],
    ["GGUF_OVERLAPPING_TENSOR_RANGE", gguf.overlapping_tensor_range_count],
    ["GGUF_TENSOR_RANGE_OUT_OF_BOUNDS", gguf.out_of_bounds_tensor_range_count],
    ["GGUF_INVALID_METADATA_KEY", gguf.invalid_metadata_key_count],
  ]) if (Number(count || 0) > 0) issues.push({ code, count: Number(count) });
  const preprocessing = analysis.coreml?.preprocessing_binding;
  if (Number(preprocessing?.unbound_entry_count || 0) > 0) issues.push({
    code: "COREML_PREPROCESSING_FEATURE_NAME_MISSING",
    count: Number(preprocessing.unbound_entry_count),
  });
  return {
    status: issues.length ? "partial_serialized_contract_issue_observed" : "assessed_no_checked_contract_issue",
    evidence_class: "OBSERVED_DERIVED",
    issue_count: issues.reduce((sum, row) => sum + row.count, 0),
    issues,
    boundary: "Only implemented format invariants are assessed; a clean result is not a proof of semantic correctness, runtime compatibility, or task quality.",
  };
}

function serializedGraphAssessment(analysis) {
  const operatorCount = (analysis.ops || []).length;
  if (operatorCount > 0) return {
    status: "serialized_graph_present",
    evidence_class: "OBSERVED",
    value: operatorCount,
    basis: "serialized graph",
  };
  if (analysis.format === "onnx") return {
    status: "serialized_graph_present_empty",
    evidence_class: "OBSERVED",
    value: 0,
    basis: "serialized ONNX GraphProto contains no decoded nodes",
  };
  if (analysis.format === "coreml" && analysis.quantization_status?.classification === "coreml_payload_not_decoded") return {
    status: "serialized_graph_payload_not_decoded",
    evidence_class: "UNAVAILABLE",
    value: null,
    basis: "Core ML model type is serialized, but its payload exceeded the bounded decoder limit",
  };
  return {
    status: "not_serialized_in_container",
    evidence_class: "NOT_APPLICABLE",
    value: null,
    basis: "weight container has no serialized execution graph",
  };
}

function parameterRows(direction, rows = []) {
  return rows.map((row, index) => {
    const scales = numericArray(row.interface_scale_values?.length ? row.interface_scale_values : row.scale_sample);
    const zeroPoints = numericArray(row.interface_zero_point_values?.length ? row.interface_zero_point_values : row.zero_point_sample);
    const declared = String(row.quantization_parameterization || "").toLowerCase();
    const granularity = declared.includes("axis") || declared.includes("channel") || scales.length > 1 ? "per-axis" : scales.length ? "per-tensor" : null;
    const dataType = row.dtype || row.feature_type || "UNKNOWN";
    const featureType = String(row.feature_type || "").toLowerCase();
    const nonTensorFeature = ["dictionary", "sequence", "string"].includes(featureType);
    const serializedShape = Array.isArray(row.shape) ? row.shape : null;
    const missingCoreMlArrayShape = featureType === "multi_array" && serializedShape?.length === 0 && !row.constraints?.flexibility;
    const shape = nonTensorFeature || missingCoreMlArrayShape ? null : serializedShape;
    const shapeStatus = nonTensorFeature ? "not_applicable_non_tensor_feature"
      : missingCoreMlArrayShape ? "unavailable_not_declared"
      : !shape ? "unavailable"
        : shape.length === 0 ? "rank_zero"
          : shape.some((value) => value === -1 || typeof value === "string") ? "dynamic_or_symbolic" : "static";
    return {
      direction,
      index,
      name: row.name || `${direction}_${index}`,
      data_type: dataType,
      shape,
      shape_status: shapeStatus,
      shape_signature: Array.isArray(row.shape_signature) && row.shape_signature.length ? row.shape_signature : null,
      quantization: granularity ? {
        evidence_class: "OBSERVED_DERIVED",
        scheme: "affine",
        granularity,
        scale: scales,
        zero_point: zeroPoints,
        axis: granularity === "per-axis" && Number.isInteger(row.quantized_dimension) ? row.quantized_dimension : null,
        axis_source: granularity === "per-axis" ? row.quantization_axis_source || "serialized_or_operator_binding" : "not_applicable",
      } : null,
      preprocessing_contract: direction === "output"
        ? { evidence_class: "NOT_APPLICABLE", value: null }
        : row.coreml_preprocessing
          ? { evidence_class: "OBSERVED", value: row.coreml_preprocessing }
          : { evidence_class: "UNAVAILABLE", value: null },
    };
  });
}

function quantizationRepresentationProfiles(analysis) {
  const storage = analysis.tensor_storage_summary || {};
  if (analysis.format === "gguf") return (storage.encodings || []).map((row) => {
    const tensors = (analysis.tensors || []).filter((tensor) => tensor.dtype === row.dtype);
    const blockSizes = [...new Set(tensors.map((tensor) => Number(tensor.block_elements)).filter((value) => Number.isSafeInteger(value) && value > 1))].sort((a, b) => a - b);
    const blockEncoded = blockSizes.length > 0;
    return {
      category: blockEncoded ? "quantization" : "precision_or_storage",
      scope: "stored_weights",
      scheme: blockEncoded ? "ggml_block_encoding" : "ggml_scalar_dtype_storage",
      encoding: row.dtype,
      tensor_count: row.tensor_count,
      effective_bits_per_element: row.effective_bits_per_element,
      unknown_byte_length_tensor_count: row.unknown_byte_length_tensor_count || 0,
      granularities: [blockEncoded ? "block" : "scalar_storage"],
      group_sizes: blockSizes,
      axes: [],
    };
  });
  if (analysis.format === "safetensors") {
    const profiles = (storage.encodings || []).map((row) => ({
      category: "precision_or_storage", scope: "stored_tensors", scheme: "safetensors_dtype_storage",
      encoding: row.dtype, tensor_count: row.tensor_count, effective_bits_per_element: row.effective_bits_per_element,
      granularities: [], axes: [],
    }));
    const quant = analysis.safetensors?.quantization_contract;
    if (quant?.status === "assessed") profiles.push({
      category: "quantization",
      scope: "stored_weight_modules",
      scheme: quant.method,
      encoding: `${quant.bits}-bit codes in ${quant.pack_word_bits}-bit words`,
      module_count: quant.module_count,
      granularities: ["per-group"],
      group_sizes: [quant.group_size],
      axes: [quant.logical_group_axis],
    });
    return profiles;
  }
  if (analysis.format === "coreml") {
    const quant = analysis.quantization_status || {};
    const profiles = [];
    const quantizedCount = Number(quant.quantized_weight_parameter_count || 0);
    const perAxisCount = Number(quant.per_axis_quantized_weight_parameter_count || 0);
    if (quantizedCount) profiles.push({
      category: "quantization", scope: "stored_weights", scheme: "coreml_weightparams_quantized_storage",
      encoding: "quantized_weight", parameter_count: quantizedCount,
      granularities: [...(perAxisCount ? ["per-axis"] : []), ...(quantizedCount > perAxisCount ? ["unspecified_or_per-tensor"] : [])], axes: [],
    });
    for (const [encoding, count] of [["float16_weight", quant.fp16_weight_parameter_count], ["float32_weight", quant.fp32_weight_parameter_count]]) {
      if (Number(count || 0) > 0) profiles.push({ category: "precision_or_storage", scope: "stored_weights", scheme: "coreml_float_weight_storage", encoding, parameter_count: Number(count), granularities: [], axes: [] });
    }
    const transformCount = (quant.op_state_counts || []).filter((row) => row.state === "serialized_quantization_transform").reduce((sum, row) => sum + Number(row.count || 0), 0);
    if (transformCount) profiles.push({ category: "quantization", scope: "serialized_mil_graph", scheme: "coreml_serialized_quantization_transform", encoding: "mil_operation", operation_count: transformCount, granularities: ["operation-scoped"], axes: [] });
    const dtypeCounts = new Map();
    for (const tensor of analysis.tensors || []) {
      const dtype = String(tensor.dtype || "UNKNOWN");
      if (["UNKNOWN", "NON_TENSOR"].includes(dtype)) continue;
      dtypeCounts.set(dtype, (dtypeCounts.get(dtype) || 0) + 1);
    }
    for (const [encoding, tensorCount] of [...dtypeCounts].sort(([left], [right]) => left.localeCompare(right))) profiles.push({
      category: "precision_or_storage", scope: "serialized_graph_typed_values", scheme: "coreml_typed_value_precision",
      encoding, tensor_count: tensorCount, granularities: [], axes: [],
    });
    return profiles;
  }
  const quant = analysis.quantization_status || {};
  const affine = (analysis.tensors || []).filter((row) => Number(row.quant_scales || row.interface_scale_values?.length || 0) > 0);
  const profiles = [];
  if ((analysis.tensor_types || []).length) profiles.push({
    category: "precision_or_storage", scope: "graph_and_initializers", scheme: "onnx_tensor_dtype_storage",
    encodings: analysis.tensor_types.map((row) => ({ name: row.dtype || row.name, tensor_count: row.count })), granularities: [], axes: [],
  });
  for (const tensor of affine) {
    const perAxis = String(tensor.quantization_parameterization || "").toLowerCase().includes("axis") || Number(tensor.quant_scales || 0) > 1;
    profiles.push({
      category: "quantization", scope: `tensor:${tensor.name || tensor.index}`, scheme: "affine",
      encoding: tensor.dtype || "UNKNOWN", tensor_index: tensor.index,
      granularities: [perAxis ? "per-axis" : "per-tensor"],
      axes: perAxis && Number.isInteger(tensor.quantized_dimension) ? [tensor.quantized_dimension] : [],
      scale_count: Number(tensor.quant_scales || tensor.interface_scale_values?.length || 0),
      zero_point_count: Number(tensor.quant_zero_points || tensor.interface_zero_point_values?.length || 0),
    });
  }
  const serializedQuantOps = Number(quant.quantize_ops || 0) + Number(quant.dequantize_ops || 0)
    + Number(quant.qlinear_ops || quant.qlinear_compute_ops || 0) + Number(quant.quantized_compute_ops || 0)
    + (quant.op_state_counts || []).filter((row) => row.name === "quant_signal_only").reduce((sum, row) => sum + Number(row.count || 0), 0);
  if (serializedQuantOps && !affine.length) profiles.push({
    category: "quantization", scope: "serialized_graph", scheme: "onnx_integer_operator_or_scale_graph",
    encoding: "operator_scoped", granularities: ["operation-scoped"], axes: [],
    quantize_linear_ops: Number(quant.quantize_ops || 0), dequantize_linear_ops: Number(quant.dequantize_ops || 0),
    quantized_compute_ops: Number(quant.quantized_compute_ops || 0), serialized_quantization_signal_count: serializedQuantOps,
  });
  return profiles;
}

function schemaPressure(format, parameters, profiles, singleton) {
  const rows = [
    {
      id: "separate-model-and-parameter-quantization",
      status: parameters.length ? "required_for_lossless_external_contracts" : "not_applicable_no_serialized_external_parameters",
      rationale: "Model-wide storage/precision summaries and named external input/output affine contracts have different scope and cardinality.",
    },
    {
      id: "repeatable-model-quantization-representations",
      status: singleton.value === true ? "optional_for_this_artifact"
        : singleton.value === false ? "required_for_lossless_model_level_description" : "not_applicable_or_not_assessed",
      rationale: "A singleton scheme/granularity/bits/axis object cannot losslessly carry mixed storage encodings, axes, or representation families.",
    },
    {
      id: "format-neutral-axis-contract",
      status: profiles.some((row) => row.axes?.length) ? "axis_values_observed" : "not_observed_in_this_artifact",
      rationale: "Axis requires a declared tensor scope and rank-aware normalization policy; model-level storage summaries may not have one tensor shape.",
    },
    {
      id: "evidence-class-and-provenance",
      status: "required",
      rationale: "Serialized facts, deterministic derivations, external declarations, runtime observations, and unavailable fields must not collapse into one truth class.",
    },
    {
      id: "runtime-binding-separation",
      status: "required",
      rationale: `${format} static bytes do not establish actual device/backend assignment or measured latency.`,
    },
  ];
  return rows;
}

function singletonQuantizationAssessment(analysis, profiles, axes, granularities) {
  if (analysis.format === "safetensors") return {
    value: null,
    status: "not_applicable_storage_dtype_without_quantization_mapping",
    reason: "SafeTensors declares storage dtypes but no executable quantization mapping; dtype storage must not be mislabeled as an affine numerical contract.",
  };
  if (analysis.format === "gguf") return {
    value: profiles.length ? false : null,
    status: profiles.length ? "lossy_block_scales_and_encodings_are_tensor_or_block_scoped" : "not_applicable_no_block_quantized_storage",
    reason: profiles.length ? `${profiles.length} observed GGML block-encoding representation(s) carry block-local numerical mappings; one model-level affine tuple is not their exact contract.` : null,
  };
  if (analysis.format === "coreml") {
    if (analysis.quantization_status?.classification === "coreml_payload_not_decoded") return {
      value: null,
      status: "not_assessed_serialized_payload_not_decoded",
      reason: "A Core ML executable payload is serialized, but its numerical representation was not decoded within the bounded inspection limit.",
    };
    const quantized = profiles.length > 0;
    return quantized ? {
      value: false,
      status: "lossy_tensor_scoped_weight_or_transform_contracts",
      reason: "Core ML weight encodings and serialized compression transforms are tensor-scoped; one model-level flat quantization object is only a summary.",
    } : {
      value: null,
      status: "not_applicable_no_serialized_quantization_mapping",
      reason: null,
    };
  }
  const affine = (analysis.tensors || []).filter((row) => Number(row.quant_scales || row.interface_scale_values?.length || 0) > 0);
  const quant = analysis.quantization_status || {};
  const serializedQuantOps = Number(quant.quantize_ops || 0) + Number(quant.dequantize_ops || 0)
    + Number(quant.qlinear_ops || quant.qlinear_compute_ops || 0) + Number(quant.quantized_compute_ops || 0)
    + (quant.op_state_counts || []).filter((row) => row.name === "quant_signal_only").reduce((sum, row) => sum + Number(row.count || 0), 0);
  if (!affine.length && serializedQuantOps === 0) return { value: null, status: "not_applicable_no_serialized_quantization_mapping", reason: null };
  if (!affine.length) return {
    value: null,
    status: "not_assessed_quantized_graph_without_extracted_tensor_mapping",
    reason: "Serialized quantized operators exist, but no complete tensor-scoped affine mapping was extracted for a losslessness decision.",
  };
  const affineProfiles = profiles.filter((row) => row.scheme === "affine");
  const lossless = affine.length === 1 && affineProfiles.length === 1 && axes.length <= 1 && granularities.length <= 1;
  return {
    value: lossless,
    status: lossless ? "lossless_single_tensor_scoped_mapping" : "lossy_multiple_tensor_scoped_mappings",
    reason: lossless ? null : axes.length > 1
      ? `Multiple serialized axes (${axes.join(", ")}) occur across tensor-scoped contracts; one model-level axis is lossy.`
      : `${affine.length} tensor-scoped affine mappings cannot be represented losslessly as one model-level flat object.`,
  };
}

function field(pathValue, evidenceClass, value, basis) { return { path: pathValue, evidence_class: evidenceClass, value, basis }; }
function numericArray(value) { return Array.isArray(value) ? value.filter(Number.isFinite) : []; }
function nullableInteger(value) { return Number.isSafeInteger(value) ? value : null; }
function sha256(value) { return createHash("sha256").update(value).digest("hex"); }
function isSha256(value) { return /^[0-9a-f]{64}$/.test(String(value || "")); }
function safeRelativePath(value) {
  const normalized = String(value || "").replaceAll("\\", "/");
  return Boolean(normalized) && !normalized.startsWith("/") && !/^[A-Za-z]:/.test(normalized) && normalized.split("/").every((part) => part && part !== "." && part !== "..");
}
export function primaryArtifactSha256(artifact) {
  const primary = artifact.files.find((file) => /\.(?:onnx|gguf|safetensors|mlmodel|zip)$/i.test(file.path));
  return (primary || artifact.files[0]).sha256;
}

import { readCoreMlModelFile, refreshCoreMlDerivedEvidence } from "./coreml-metadata-adapter.js";
import { scanCoreMlBlobFile } from "./coreml-blob.js";
import { sha256FileHex } from "./hash.js";
import { buildTensorStorageSummary, parseStrictJson, readMetadataModelFile } from "./metadata-model-adapters.js";
import { buildHfSafeTensorsContract } from "./hf-safetensors-contract.js";
import { buildSafeTensorsQuantizationContract } from "./safetensors-quantization-contract.js";
import { buildOnDeviceLlmContract } from "./on-device-llm-contract.js";
import { safeTensorsNumericalSourceEvidence } from "./tensor-numerical-integrity.js";
import { sha256TextHex } from "./sha256-sync.js";

const MAX_BUNDLE_FILES = 20_000;
const MAX_MANIFEST_BYTES = 16 * 1024 * 1024;

function selectedPath(file) {
  return String(file?.webkitRelativePath || file?.name || "").replaceAll("\\", "/").replace(/^\.\//, "");
}

function safePath(path, label = "bundle path") {
  const value = String(path || "").replaceAll("\\", "/");
  if (!value || value.startsWith("/") || /^[A-Za-z]:/.test(value) || value.includes("\0")) throw new Error(`${label} is not a safe relative path`);
  const parts = value.split("/");
  if (parts.some((part) => !part || part === "." || part === "..")) throw new Error(`${label} contains an empty or traversal segment`);
  return parts.join("/");
}

function bundleEntries(files) {
  const rows = [...(files || [])];
  if (!rows.length) throw new Error("No package files were selected");
  if (rows.length > MAX_BUNDLE_FILES) throw new Error(`Package selection exceeds ${MAX_BUNDLE_FILES} files`);
  const exact = new Map();
  const folded = new Map();
  for (const file of rows) {
    const path = safePath(selectedPath(file));
    const lower = path.toLowerCase();
    if (exact.has(path)) throw new Error(`Package selection repeats ${path}`);
    if (folded.has(lower)) throw new Error(`Package selection contains a cross-platform case collision: ${folded.get(lower)} and ${path}`);
    exact.set(path, file);
    folded.set(lower, path);
  }
  return exact;
}

async function jsonFile(file, label) {
  if (!file || file.size > MAX_MANIFEST_BYTES) throw new Error(`${label} exceeds ${MAX_MANIFEST_BYTES} bytes`);
  let text;
  try { text = new TextDecoder("utf-8", { fatal: true }).decode(await file.arrayBuffer()); }
  catch { throw new Error(`${label} is not valid UTF-8`); }
  return parseStrictJson(text, label);
}

async function textFile(file, label) {
  if (!file || file.size > MAX_MANIFEST_BYTES) throw new Error(`${label} exceeds ${MAX_MANIFEST_BYTES} bytes`);
  try { return new TextDecoder("utf-8", { fatal: true }).decode(await file.arrayBuffer()); }
  catch { throw new Error(`${label} is not valid UTF-8`); }
}

function coreMlPlan(entries) {
  const manifests = [...entries].filter(([path]) => path.endsWith("/Manifest.json") || path === "Manifest.json");
  if (manifests.length !== 1) return null;
  const [manifestPath, manifestFile] = manifests[0];
  const root = manifestPath.slice(0, -"Manifest.json".length);
  return { kind: "coreml_package", format: "coreml", entries, manifestPath, manifestFile, root };
}

function safeTensorsPlan(entries) {
  const indexes = [...entries].filter(([path]) => path.endsWith(".safetensors.index.json"));
  if (indexes.length > 1) throw new Error("Package selection contains multiple SafeTensors shard indexes");
  if (indexes.length === 1) {
    const [manifestPath, manifestFile] = indexes[0];
    const slash = manifestPath.lastIndexOf("/");
    return { kind: "safetensors_shards", format: "safetensors", entries, manifestPath, manifestFile, root: slash < 0 ? "" : manifestPath.slice(0, slash + 1) };
  }
  const tensors = [...entries].filter(([path]) => path.toLowerCase().endsWith(".safetensors"));
  if (!tensors.length) return null;
  if (tensors.length !== 1) throw new Error("Multiple SafeTensors files require exactly one .safetensors.index.json binding manifest");
  const [tensorPath, tensorFile] = tensors[0];
  const slash = tensorPath.lastIndexOf("/");
  return {
    kind: "safetensors_single_repository",
    format: "safetensors",
    entries,
    manifestPath: null,
    manifestFile: null,
    tensorPath,
    tensorFile,
    root: slash < 0 ? "" : tensorPath.slice(0, slash + 1),
  };
}

async function resolveCoreMlPlan(plan) {
  const manifest = await jsonFile(plan.manifestFile, "Core ML package Manifest.json");
  if (manifest?.fileFormatVersion !== "1.0.0") throw new Error(`Unsupported Core ML package fileFormatVersion ${manifest?.fileFormatVersion ?? "missing"}`);
  if (!manifest.itemInfoEntries || typeof manifest.itemInfoEntries !== "object" || Array.isArray(manifest.itemInfoEntries)) throw new Error("Core ML package itemInfoEntries must be an object");
  const identifiers = Object.keys(manifest.itemInfoEntries);
  if (!identifiers.length || identifiers.length > MAX_BUNDLE_FILES) throw new Error("Core ML package itemInfoEntries count is invalid");
  if (typeof manifest.rootModelIdentifier !== "string" || !Object.hasOwn(manifest.itemInfoEntries, manifest.rootModelIdentifier)) throw new Error("Core ML package rootModelIdentifier is missing or unresolved");
  const items = identifiers.map((identifier) => {
    const item = manifest.itemInfoEntries[identifier];
    if (!item || typeof item !== "object" || Array.isArray(item)) throw new Error(`Core ML package item ${identifier} is invalid`);
    for (const key of ["path", "name", "author", "description"]) if (typeof item[key] !== "string") throw new Error(`Core ML package item ${identifier} is missing string field ${key}`);
    const relative = safePath(item.path, `Core ML package item ${identifier} path`);
    const path = `${plan.root}Data/${relative}`;
    const files = plan.entries.has(path)
      ? [{ path, file: plan.entries.get(path) }]
      : [...plan.entries].filter(([candidate]) => candidate.startsWith(`${path}/`)).map(([candidate, file]) => ({ path: candidate, file }));
    if (!files.length) throw new Error(`Core ML package item ${identifier} is missing ${path}`);
    return { identifier, path, files, ...item };
  });
  const rootItem = items.find((item) => item.identifier === manifest.rootModelIdentifier);
  const rootModels = rootItem.files.filter((item) => item.path.toLowerCase().endsWith(".mlmodel"));
  if (rootModels.length !== 1) throw new Error("Core ML package root item does not resolve to exactly one .mlmodel file");
  return { ...plan, manifest, items, rootItem, rootFile: rootModels[0].file, rootModelPath: rootModels[0].path, displayName: plan.root.replace(/\/$/, "") || "model.mlpackage" };
}

async function resolveSafeTensorsPlan(plan) {
  let manifest = null;
  let tensorNames = null;
  let shards;
  if (plan.kind === "safetensors_shards") {
    manifest = await jsonFile(plan.manifestFile, "SafeTensors shard index");
    if (!manifest?.weight_map || typeof manifest.weight_map !== "object" || Array.isArray(manifest.weight_map)) throw new Error("SafeTensors shard index weight_map must be an object");
    tensorNames = Object.keys(manifest.weight_map);
    if (!tensorNames.length) throw new Error("SafeTensors shard index weight_map is empty");
    const shardPaths = [...new Set(tensorNames.map((name) => {
      const value = manifest.weight_map[name];
      if (typeof value !== "string" || !value.toLowerCase().endsWith(".safetensors")) throw new Error(`SafeTensors weight_map entry ${name} has an invalid shard path`);
      return `${plan.root}${safePath(value, `SafeTensors weight_map entry ${name}`)}`;
    }))].sort();
    shards = shardPaths.map((path) => {
      const file = plan.entries.get(path);
      if (!file) throw new Error(`SafeTensors shard index references missing file ${path}`);
      return { path, file };
    });
  } else shards = [{ path: plan.tensorPath, file: plan.tensorFile }];
  const configPath = `${plan.root}config.json`;
  const configFile = plan.entries.get(configPath) || null;
  const config = configFile ? await jsonFile(configFile, "Hugging Face config.json") : null;
  const sidecarDefinitions = [
    ["tokenizer_config", "tokenizer_config.json", "tokenizer_config", "Hugging Face tokenizer_config.json", "json"],
    ["generation_config", "generation_config.json", "generation_config", "Hugging Face generation_config.json", "json"],
    ["special_token_map", "special_tokens_map.json", "special_token_map", "Hugging Face special_tokens_map.json", "json"],
    ["chat_template", "chat_template.jinja", "chat_template", "Hugging Face chat_template.jinja", "text"],
    ["tokenizer_definition", "tokenizer.json", "tokenizer_definition", "Hugging Face tokenizer.json", "identity"],
    ["tokenizer_model", "tokenizer.model", "tokenizer_definition", "SentencePiece tokenizer.model", "identity"],
    ["deployment_declaration", "deepbom.deployment.json", "deployment_declaration", "DEEPBOM on-device LLM deployment declaration", "json"],
    ["runtime_manifest", "deepbom.runtime.json", "llm_runtime_manifest", "DEEPBOM on-device LLM runtime manifest", "json"],
    ["static_memory_profile", "deepbom.memory-profile.json", "llm_static_memory_profile", "DEEPBOM static LLM memory profile", "json"],
    ["tensorrt_llm_engine_config", "tensorrt_llm_engine_config.json", "tensorrt_llm_engine_config", "TensorRT-LLM engine config", "json"],
    ["tensorrt_llm_binding", "deepbom.tensorrt-llm.json", "tensorrt_llm_binding", "DEEPBOM TensorRT-LLM artifact binding", "json"],
    ["quant_config", "quant_config.json", "quantization_config", "AWQ quant_config.json", "json"],
    ["quantize_config", "quantize_config.json", "quantization_config", "GPTQ quantize_config.json", "json"],
    ["quantization_config", "quantization_config.json", "quantization_config", "quantization_config.json", "json"],
  ];
  const llmSidecars = {
    ...(configFile ? { architecture_config: { path: configPath, file: configFile, role: "architecture_config", document: config } } : {}),
  };
  for (const [key, filename, role, label, kind] of sidecarDefinitions) {
    const path = `${plan.root}${filename}`;
    const file = plan.entries.get(path) || null;
    if (!file) continue;
    const document = kind === "json" ? await jsonFile(file, label) : kind === "text" ? await textFile(file, label) : null;
    llmSidecars[key] = { path, file, role, document };
  }
  return {
    ...plan,
    manifest,
    tensorNames,
    shards,
    configPath: configFile ? configPath : null,
    configFile,
    config,
    llmSidecars,
    rootFile: plan.manifestFile || plan.tensorFile,
    displayName: plan.manifestPath || plan.tensorPath,
  };
}

export async function inspectArtifactBundle(files) {
  const entries = bundleEntries(files);
  const plan = coreMlPlan(entries) || safeTensorsPlan(entries);
  if (!plan) throw new Error("Selection is neither a Core ML package nor a SafeTensors shard set");
  return plan.kind === "coreml_package" ? resolveCoreMlPlan(plan) : resolveSafeTensorsPlan(plan);
}

async function hashRecords(rows, onProgress) {
  const result = [];
  for (let index = 0; index < rows.length; index += 1) {
    const row = rows[index];
    onProgress?.({ index, count: rows.length, phase: "hashing", path: row.path });
    result.push({ ...row, byte_length: row.file.size, sha256: await sha256FileHex(row.file) });
  }
  return result;
}

function bundleDigest(records) {
  const rows = records.map(({ path, byte_length, sha256, role, required }) => ({ path, byte_length, sha256, role, required })).sort((a, b) => a.path.localeCompare(b.path));
  return sha256TextHex(JSON.stringify({ schema: "deepbom.artifact_bundle_digest.v1", files: rows }));
}

const SAFETENSORS_MODEL_SOURCE_ROLES = Object.freeze(new Set([
  "shard_index",
  "architecture_config",
  "quantization_config",
  "tensor_shard",
]));

function safeTensorsModelSourceDigest(records) {
  const sourceRecords = records.filter((row) => SAFETENSORS_MODEL_SOURCE_ROLES.has(row.role));
  if (!sourceRecords.length || !sourceRecords.some((row) => row.role === "tensor_shard")) {
    throw new Error("SafeTensors model-source digest requires at least one tensor shard.");
  }
  return {
    sha256: bundleDigest(sourceRecords),
    roles: [...SAFETENSORS_MODEL_SOURCE_ROLES],
    file_count: sourceRecords.length,
  };
}

function unboundBundleRows(plan, covered, role) {
  return [...plan.entries]
    .filter(([path]) => !covered.has(path))
    .map(([path, file]) => ({ path, file, role, required: false }));
}

function bundleFileEvidence(row, verifiedStatus) {
  return {
    path: row.path,
    byte_length: row.byte_length,
    sha256: row.sha256,
    role: row.role,
    required: Boolean(row.required),
    evidence_class: "OBSERVED",
    verification_status: row.required ? verifiedStatus : "selected_and_hash_verified_not_manifest_bound",
  };
}

function resolveCoreMlBlobPath(plan, fileName) {
  const value = String(fileName || "").replaceAll("\\", "/");
  const prefix = "@model_path/";
  if (!value.startsWith(prefix)) throw new Error(`Core ML MIL blob path ${value || "<empty>"} is not rooted at @model_path`);
  const suffix = safePath(value.slice(prefix.length), "Core ML MIL blob path");
  const slash = plan.rootModelPath.lastIndexOf("/");
  const modelDirectory = slash < 0 ? "" : plan.rootModelPath.slice(0, slash + 1);
  const exact = `${modelDirectory}${suffix}`;
  if (plan.entries.has(exact)) return exact;
  const candidates = plan.rootItem.files.map((item) => item.path).filter((path) => path === suffix || path.endsWith(`/${suffix}`));
  if (candidates.length !== 1) throw new Error(`Core ML MIL blob path ${value} does not resolve uniquely inside the root model package item`);
  return candidates[0];
}

async function bindCoreMlProgramBlobs(plan, analysis, onProgress) {
  const references = Array.isArray(analysis.coreml_blob_references) ? analysis.coreml_blob_references : [];
  if (!references.length) return;
  const groups = new Map();
  for (const reference of references) {
    const path = resolveCoreMlBlobPath(plan, reference.file_name);
    if (!groups.has(path)) groups.set(path, []);
    const tensor = analysis.tensors?.[reference.tensor_index];
    groups.get(path).push({
      ...reference,
      tensor_name: tensor?.name || `tensor_${reference.tensor_index}`,
      dtype: tensor?.dtype || null,
      shape: tensor?.shape || [],
    });
  }
  const scans = [];
  for (const [path, bindings] of groups) {
    const file = plan.entries.get(path);
    if (!file) throw new Error(`Core ML package is missing resolved blob file ${path}`);
    scans.push({ path, ...(await scanCoreMlBlobFile(file, bindings, { onProgress })) });
  }
  const records = scans.flatMap((scan) => scan.records.map((record) => ({ ...record, file_path: scan.path })));
  const parameterByTensor = new Map();
  for (const record of records) {
    for (const tensorIndex of record.tensor_indices) {
      const tensor = analysis.tensors[tensorIndex];
      tensor.numerical_integrity = { ...record.numerical_integrity, file_path: record.file_path, metadata_offset: record.metadata_offset };
      tensor.storage_status = "coreml_mil_blob_v2_assessed";
      parameterByTensor.set(tensorIndex, {
        tensor_index: tensorIndex,
        tensor_name: tensor.name,
        role: tensor.role,
        storage: "mil_blob_v2",
        dtype: tensor.dtype,
        shape: tensor.shape,
        byte_length: record.size_in_bytes,
        value_count: record.value_count,
        file_path: record.file_path,
        metadata_offset: record.metadata_offset,
        numerical_integrity: tensor.numerical_integrity,
      });
    }
  }
  const previous = analysis.weight_integrity?.parameters || [];
  const immediate = previous.filter((row) => row.storage === "mil_immediate");
  const assessedImmediate = immediate.filter((row) => row.numerical_integrity?.status?.startsWith("assessed"));
  const blobParameters = [...parameterByTensor.values()];
  const parameters = [...immediate, ...blobParameters];
  const payloadBytes = records.reduce((sum, record) => sum + Number(record.size_in_bytes || 0), 0);
  const nonfinite = records.reduce((sum, record) => sum + Number(record.numerical_integrity.nonfinite_count || 0), 0)
    + assessedImmediate.reduce((sum, row) => sum + Number(row.numerical_integrity.nonfinite_count || 0), 0);
  analysis.coreml_blob_integrity = {
    schema: "deepbom.coreml.package_blob_integrity.v1",
    status: "assessed",
    file_count: scans.length,
    declared_blob_count: scans.reduce((sum, scan) => sum + scan.declared_blob_count, 0),
    referenced_blob_count: scans.reduce((sum, scan) => sum + scan.referenced_blob_count, 0),
    unreferenced_blob_count: scans.reduce((sum, scan) => sum + scan.unreferenced_blob_count, 0),
    payload_bytes: payloadBytes,
    decoded_value_count: scans.reduce((sum, scan) => sum + scan.decoded_value_count, 0),
    nonfinite_value_count: scans.reduce((sum, scan) => sum + scan.nonfinite_value_count, 0),
    all_zero_blob_count: scans.reduce((sum, scan) => sum + scan.all_zero_blob_count, 0),
    files: scans,
  };
  analysis.weight_integrity = {
    schema: "deepbom.coreml.mlprogram_weight_integrity.v1",
    status: parameters.length === assessedImmediate.length + blobParameters.length ? "assessed" : "partial",
    parameter_count: parameters.length,
    assessed_parameter_count: assessedImmediate.length + blobParameters.length,
    payload_bytes: payloadBytes,
    assessed_payload_bytes: payloadBytes,
    payload_byte_conservation: true,
    nonfinite_value_count: nonfinite,
    all_zero_parameter_count: parameters.filter((row) => row.numerical_integrity?.all_zero).length,
    blob_reference_count: references.length,
    unique_blob_count: records.length,
    parameters,
  };
}

async function analyzeCoreMlPackage(plan, onProgress) {
  const requiredRows = [
    { path: plan.manifestPath, file: plan.manifestFile, role: "package_manifest", required: true },
    ...plan.items.flatMap((item) => item.files.map((physical) => ({
      ...physical,
      role: item.identifier === plan.manifest.rootModelIdentifier && physical.file === plan.rootFile ? "root_model" : /weight/i.test(`${item.name} ${item.description} ${physical.path}`) ? "weights" : "package_item",
      required: true,
      package_identifier: item.identifier,
    }))),
  ];
  const covered = new Set(requiredRows.map((row) => row.path));
  const rows = [...requiredRows, ...unboundBundleRows(plan, covered, "unreferenced_package_file")];
  const records = await hashRecords(rows, onProgress);
  const parsed = await readCoreMlModelFile(plan.rootFile);
  await bindCoreMlProgramBlobs(plan, parsed.analysis, onProgress);
  const digest = bundleDigest(records);
  const total = records.reduce((sum, row) => sum + row.byte_length, 0);
  const analysis = parsed.analysis;
  analysis.filename = plan.displayName;
  analysis.file_size = total;
  analysis.file_size_bytes = total;
  analysis.model_sha256 = digest;
  analysis.artifact_bundle = {
    schema: "deepbom.artifact_bundle.coreml.v1",
    kind: "coreml_mlpackage",
    hash_basis: "sha256_of_canonical_path_size_digest_role_manifest",
    bundle_sha256: digest,
    root_model_identifier: plan.manifest.rootModelIdentifier,
    files: records.map((row) => bundleFileEvidence(row, "manifest_resolved_and_hash_verified")),
  };
  analysis.format_extensions.coreml.package = { file_format_version: plan.manifest.fileFormatVersion, item_count: plan.items.length, root_model_identifier: plan.manifest.rootModelIdentifier };
  refreshCoreMlDerivedEvidence(analysis);
  return { analysis, retainedBytes: parsed.retainedBytes, payloadLoaded: false, rootFile: plan.rootFile };
}

async function analyzeSafeTensorsShards(plan, onProgress, scanMode = "full") {
  const sidecarRows = Object.values(plan.llmSidecars || {}).filter((row) => row.role !== "architecture_config").map((row) => ({
    path: row.path, file: row.file, role: row.role, required: true,
  }));
  const requiredRows = [
    ...(plan.manifestFile ? [{ path: plan.manifestPath, file: plan.manifestFile, role: "shard_index", required: true }] : []),
    ...(plan.configFile ? [{ path: plan.configPath, file: plan.configFile, role: "architecture_config", required: true }] : []),
    ...sidecarRows,
    ...plan.shards.map((item) => ({ ...item, role: "tensor_shard", required: true })),
  ];
  const covered = new Set(requiredRows.map((row) => row.path));
  const rows = [...requiredRows, ...unboundBundleRows(plan, covered, "repository_supporting_file")];
  const records = await hashRecords(rows, onProgress);
  const tensors = [];
  const seen = new Set();
  const shardAnalyses = [];
  for (let index = 0; index < plan.shards.length; index += 1) {
    const shard = plan.shards[index];
    onProgress?.({ index, count: plan.shards.length, phase: "parsing", path: shard.path });
    const parsed = await readMetadataModelFile(shard.file, "safetensors", {
      onProgress: (progress) => onProgress?.({ ...progress, phase: "tensor_payload", shard_index: index, shard_count: plan.shards.length, path: shard.path }),
      scanMode,
    });
    shardAnalyses.push(parsed.analysis);
    for (const tensor of parsed.analysis.tensors) {
      if (seen.has(tensor.name)) throw new Error(`SafeTensors shard set repeats tensor ${tensor.name}`);
      seen.add(tensor.name);
      if (plan.manifest) {
        const declared = plan.manifest.weight_map[tensor.name];
        const expected = declared == null ? null : `${plan.root}${safePath(declared)}`;
        if (expected !== shard.path) throw new Error(`SafeTensors tensor ${tensor.name} is not bound to its actual shard`);
      }
      const tensorIndex = tensors.length;
      tensors.push({
        ...tensor,
        index: tensorIndex,
        shard_path: shard.path,
        numerical_integrity: tensor.numerical_integrity
          ? { ...tensor.numerical_integrity, tensor_index: tensorIndex, shard_path: shard.path }
          : null,
      });
    }
  }
  for (const name of plan.tensorNames || []) if (!seen.has(name)) throw new Error(`SafeTensors shard index names missing tensor ${name}`);
  const digest = bundleDigest(records);
  const modelSource = safeTensorsModelSourceDigest(records);
  const total = records.reduce((sum, row) => sum + row.byte_length, 0);
  const analysis = {
    ...shardAnalyses[0],
    filename: plan.displayName,
    file_size: total,
    file_size_bytes: total,
    model_sha256: digest,
    tensor_count: tensors.length,
    tensors,
    tensor_inventory: {
      status: "assessed",
      tensor_count: tensors.length,
      total_declared_tensor_bytes: tensors.reduce((sum, tensor) => sum + Number(tensor.byte_length || 0), 0),
      tensors,
    },
  };
  const numericalRecords = tensors.map((tensor) => tensor.numerical_integrity).filter(Boolean);
  const assessedNumericalRecords = numericalRecords.filter((record) => record.status === "assessed_full_payload");
  const unassessedNumericalRecords = numericalRecords.filter((record) => record.status !== "assessed_full_payload");
  const sumNumerical = (key) => assessedNumericalRecords.reduce((sum, record) => sum + Number(record[key] || 0), 0);
  const declaredTensorBytes = analysis.tensor_inventory.total_declared_tensor_bytes;
  const assessedTensorBytes = sumNumerical("byte_length");
  const unassessedTensorBytes = unassessedNumericalRecords.reduce((sum, record) => sum + Number(record.byte_length || 0), 0);
  analysis.tensor_numerical_integrity = {
    schema: "deepbom.tensor_numerical_integrity.v1",
    status: numericalRecords.some((record) => record.status !== "assessed_full_payload")
      ? assessedNumericalRecords.length ? "partial" : "not_assessed" : "assessed",
    evidence_class: "OBSERVED/DERIVED",
    scan_scope: plan.manifest ? "full_declared_tensor_payloads_across_manifest_bound_shards" : "full_declared_tensor_payloads_in_bundle_bound_single_file",
    tensor_count: tensors.length,
    assessed_tensor_count: assessedNumericalRecords.length,
    unassessed_tensor_count: tensors.length - assessedNumericalRecords.length,
    declared_tensor_bytes: declaredTensorBytes,
    assessed_tensor_bytes: assessedTensorBytes,
    unassessed_tensor_bytes: unassessedTensorBytes,
    byte_conservation_status: numericalRecords.length === tensors.length && assessedTensorBytes + unassessedTensorBytes === declaredTensorBytes ? "complete" : "invalid",
    decoded_value_count: sumNumerical("value_count"),
    nonfinite_value_count: sumNumerical("nan_value_count") + sumNumerical("positive_infinity_value_count") + sumNumerical("negative_infinity_value_count"),
    exact_zero_value_count: sumNumerical("zero_value_count"),
    all_zero_tensor_count: assessedNumericalRecords.filter((record) => record.all_zero).length,
    constant_tensor_count: assessedNumericalRecords.filter((record) => record.constant_finite).length,
    invalid_encoding_value_count: sumNumerical("invalid_encoding_value_count"),
    nonfinite_scale_block_count: sumNumerical("nonfinite_scale_block_count"),
    tensor_records: numericalRecords,
    shard_count: plan.shards.length,
    source_file_bytes: plan.shards.reduce((sum, item) => sum + Number(item.file.size || 0), 0),
    decoder_source: safeTensorsNumericalSourceEvidence(),
    limitations: unassessedNumericalRecords.map((record) => ({ tensor_name: record.tensor_name, dtype: record.dtype, reason: record.reason })),
  };
  analysis.tensor_inventory.numerical_integrity_status = analysis.tensor_numerical_integrity.status;
  analysis.tensor_inventory.assessed_payload_bytes = analysis.tensor_numerical_integrity.assessed_tensor_bytes;
  analysis.tensor_inventory.unassessed_payload_bytes = analysis.tensor_numerical_integrity.unassessed_tensor_bytes;
  analysis.tensor_inventory.decoded_value_count = analysis.tensor_numerical_integrity.decoded_value_count;
  analysis.tensor_storage_summary = buildTensorStorageSummary("safetensors", tensors, analysis.tensor_numerical_integrity);
  analysis.artifact_bundle = {
    schema: "deepbom.artifact_bundle.safetensors.v1",
    kind: plan.manifest ? "safetensors_sharded_repository" : "safetensors_single_file_repository",
    hash_basis: "sha256_of_canonical_path_size_digest_role_manifest",
    bundle_sha256: digest,
    model_source_sha256: modelSource.sha256,
    model_source_hash_basis: "deepbom.artifact_bundle_digest.v1 over shard_index, architecture_config, quantization_config, and tensor_shard roles only",
    model_source_roles: modelSource.roles,
    model_source_file_count: modelSource.file_count,
    files: records.map((row) => bundleFileEvidence(row, plan.manifest ? "index_resolved_and_hash_verified" : "repository_selected_and_hash_verified")),
  };
  analysis.safetensors = {
    ...analysis.safetensors,
    header_byte_length: shardAnalyses.reduce((sum, item) => sum + Number(item.safetensors?.header_byte_length || 0), 0),
    tensor_count: tensors.length,
    payload_byte_length: shardAnalyses.reduce((sum, item) => sum + Number(item.safetensors?.payload_byte_length || 0), 0),
    sharded: Boolean(plan.manifest),
    shard_count: plan.shards.length,
    index_tensor_count: plan.tensorNames?.length || 0,
    index_binding_status: plan.manifest ? "complete_bidirectional" : "not_required_single_file",
    hf_config_path: plan.configPath,
    hf_config_status: plan.config ? "parsed_and_bundle_bound" : "not_selected",
    hf_architecture_contract: plan.config ? buildHfSafeTensorsContract(plan.config, tensors) : {
      schema: "deepbom.hf_safetensors_architecture_contract.v1",
      status: "not_assessed_config_not_selected",
      evidence_class: "NOT_ASSESSED",
      reason: "Select config.json with the SafeTensors repository to derive a registered architecture contract.",
    },
    quantization_contract: buildSafeTensorsQuantizationContract(plan.config, tensors, { sidecars: plan.llmSidecars }),
  };
  const sidecars = Object.fromEntries(Object.entries(plan.llmSidecars || {}).map(([key, sidecar]) => {
    const record = records.find((row) => row.path === sidecar.path);
    return [key, {
      role: sidecar.role,
      path: sidecar.path,
      byte_length: record?.byte_length ?? sidecar.file.size,
      sha256: record?.sha256 ?? null,
      document: sidecar.document,
    }];
  }));
  analysis.on_device_llm = buildOnDeviceLlmContract(analysis, { sidecars });
  analysis.format_extensions.safetensors = analysis.safetensors;
  return { analysis, retainedBytes: new Uint8Array(), payloadLoaded: false, rootFile: plan.rootFile };
}

export async function readArtifactBundle(files, { onProgress, scanMode = "full" } = {}) {
  if (!["structure", "integrity", "full"].includes(scanMode)) throw new Error(`Unsupported artifact bundle scan mode ${scanMode}.`);
  const plan = await inspectArtifactBundle(files);
  return plan.kind === "coreml_package" ? analyzeCoreMlPackage(plan, onProgress) : analyzeSafeTensorsShards(plan, onProgress, scanMode);
}

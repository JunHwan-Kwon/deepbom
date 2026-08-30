import { createHash } from "node:crypto";
import { readFile, stat } from "node:fs/promises";
import path from "node:path";

import { parseStrictJson } from "../web/lib/metadata-model-adapters.js";
import { defaultArtifactCacheDir, parseArtifactSource, resolveArtifactSource } from "./remote-artifact-resolver.mjs";

const MAX_INDEX_BYTES = 16 * 1024 * 1024;
const MAX_COMPANIONS = 20_000;

export async function resolveHuggingFaceSafeTensorsClosure(spec, primary, options = {}) {
  const source = parseArtifactSource(spec, options.expectedSha256 || "");
  if (source.kind !== "huggingface" || !source.file.toLowerCase().endsWith(".safetensors.index.json")) return primary;
  const metadata = await stat(primary.path);
  if (!metadata.isFile() || metadata.size > MAX_INDEX_BYTES) throw new Error(`SafeTensors shard index exceeds ${MAX_INDEX_BYTES} bytes.`);
  const manifest = parseStrictJson(await readFile(primary.path, "utf8"), "remote SafeTensors shard index");
  if (!manifest?.weight_map || typeof manifest.weight_map !== "object" || Array.isArray(manifest.weight_map)) {
    throw new Error("Remote SafeTensors shard index weight_map must be an object.");
  }
  const tensorNames = Object.keys(manifest.weight_map);
  if (!tensorNames.length || tensorNames.length > MAX_COMPANIONS * 1_000) throw new Error("Remote SafeTensors weight_map cardinality is invalid.");
  const shardPaths = [...new Set(tensorNames.map((name) => {
    const value = manifest.weight_map[name];
    if (typeof value !== "string" || !value.toLowerCase().endsWith(".safetensors")) {
      throw new Error(`Remote SafeTensors weight_map entry ${name} has an invalid shard path.`);
    }
    return safeRelativePath(value, `SafeTensors weight_map entry ${name}`);
  }))].sort();
  if (!shardPaths.length || shardPaths.length > MAX_COMPANIONS) throw new Error("Remote SafeTensors shard count is invalid.");
  return resolveHuggingFaceClosure(source, primary, shardPaths.map((relativePath) => ({ relativePath, role: "shard" })), options);
}

export async function resolveHuggingFaceOnnxExternalDataClosure(spec, primary, locations, options = {}) {
  const source = parseArtifactSource(spec, options.expectedSha256 || "");
  const rows = [...new Set((locations || []).map((value) => safeRelativePath(value, "ONNX external_data location")))].sort();
  if (source.kind !== "huggingface" || !rows.length) return primary;
  if (rows.length > MAX_COMPANIONS) throw new Error(`ONNX external_data exceeds ${MAX_COMPANIONS} files.`);
  return resolveHuggingFaceClosure(source, primary, rows.map((relativePath) => ({ relativePath, role: "sidecar" })), options);
}

async function resolveHuggingFaceClosure(source, primary, companions, options) {
  if (!primary?.acquisition || primary.acquisition.source?.canonical_locator !== source.canonical_locator
    || primary.acquisition.source?.immutability?.kind !== "repository_commit"
    || primary.acquisition.source?.immutability?.value !== source.revision) {
    throw new Error("Hugging Face closure primary acquisition does not match the requested repository, commit, and path.");
  }
  const cacheDir = options.cacheDir || defaultArtifactCacheDir();
  const modelDirectory = path.posix.dirname(source.file) === "." ? "" : path.posix.dirname(source.file);
  const members = [{
    role: "primary",
    path: source.file,
    resolved_path: primary.path,
    acquisition: primary.acquisition,
  }];
  for (const companion of companions) {
    const repositoryPath = modelDirectory ? `${modelDirectory}/${companion.relativePath}` : companion.relativePath;
    const locator = `hf://${source.repository}@${source.revision}/${repositoryPath}`;
    const resolved = await resolveArtifactSource(locator, {
      cacheDir,
      offline: options.offline,
      maximumBytes: options.maximumBytes,
      fetchImpl: options.fetchImpl,
      progress: options.progress,
      environment: options.environment,
    });
    members.push({ role: companion.role, path: repositoryPath, resolved_path: resolved.path, acquisition: resolved.acquisition });
    members[members.length - 1].model_relative_path = companion.relativePath;
  }
  const identity = createHash("sha256").update(JSON.stringify(members.map((row) => ({
    role: row.role,
    path: row.path,
    sha256: row.acquisition.file.sha256,
    byte_length: row.acquisition.file.byte_length.decimal,
  })))).digest("hex");
  const kind = source.file.toLowerCase().endsWith(".safetensors.index.json")
    ? "huggingface_safetensors_shards" : "huggingface_onnx_external_data";
  return {
    path: primary.path,
    acquisition: primary.acquisition,
    virtual_bundle_members: kind === "huggingface_safetensors_shards" ? members.map((row) => ({
      role: row.role,
      path: row.path,
      resolved_path: row.resolved_path,
    })) : null,
    external_data_members: kind === "huggingface_onnx_external_data" ? members.filter((row) => row.role === "sidecar").map((row) => ({
      role: row.role,
      path: row.path,
      model_relative_path: row.model_relative_path,
      resolved_path: row.resolved_path,
      sha256: row.acquisition.file.sha256,
      byte_length: row.acquisition.file.byte_length,
    })) : null,
    closure: {
      schema: "deepbom.remote_artifact_closure.v1",
      evidence_class: "OBSERVED_ACQUISITION",
      kind,
      repository: source.repository,
      revision: source.revision,
      members: members.map((row) => ({
        role: row.role,
        path: row.path,
        sha256: row.acquisition.file.sha256,
        byte_length: row.acquisition.file.byte_length,
      })),
      closure_sha256: identity,
      materialization: "content_addressed_member_map_no_copy",
      model_root: null,
      remote_code_execution: "forbidden",
    },
  };
}

function safeRelativePath(value, label) {
  const normalized = String(value || "").replaceAll("\\", "/");
  if (!normalized || normalized.length > 2048 || normalized.startsWith("/") || /^[A-Za-z]:/.test(normalized)
    || normalized.split("/").some((part) => !part || part === "." || part === "..")) {
    throw new Error(`${label} is not a safe relative path.`);
  }
  return normalized;
}

import { readFile } from "node:fs/promises";

import {
  ensureHfFile,
  hfCorpusCacheDir,
  readHfCorpus,
} from "./huggingface-community-corpus-lib.mjs";

export const GOOGLE_MODERN_COMPARATOR_PATH = "corpus/google_legacy/paired-modern.v1.json";
export const GOOGLE_MODERN_COMPARATOR_SCHEMA = "deepbom.paired_modern_comparators.v1";

export async function readGoogleModernComparators(filename = GOOGLE_MODERN_COMPARATOR_PATH) {
  const manifest = JSON.parse(await readFile(filename, "utf8"));
  if (manifest?.schema !== GOOGLE_MODERN_COMPARATOR_SCHEMA
    || !Array.isArray(manifest.artifacts)
    || manifest.artifact_count !== manifest.artifacts.length) {
    throw new Error("Modern comparator manifest is invalid.");
  }
  const ids = new Set();
  for (const artifact of manifest.artifacts) {
    if (!/^[a-z0-9][a-z0-9.-]+$/.test(String(artifact.id || "")) || ids.has(artifact.id)) {
      throw new Error(`Invalid or duplicate modern comparator id: ${artifact.id || "(missing)"}.`);
    }
    if (!/^[0-9a-f]{40}$/.test(String(artifact.revision || ""))
      || !/^[0-9a-f]{40}$/.test(String(artifact.model_card_blob_sha1 || ""))
      || !/^[0-9a-f]{64}$/.test(String(artifact.sha256 || ""))
      || !Number.isSafeInteger(artifact.size_bytes)
      || artifact.size_bytes < 1
      || !String(artifact.path || "").endsWith(".tflite")) {
      throw new Error(`${artifact.id}: identity metadata is invalid.`);
    }
    ids.add(artifact.id);
  }
  return manifest;
}

export async function loadResolvedGoogleModernComparators(
  manifest,
  catalogFilename = manifest.source_catalog,
) {
  const catalog = await readHfCorpus(catalogFilename);
  return manifest.artifacts.map((artifact) => {
    const repository = catalog.repositories.find((row) => row.id === artifact.repository_id);
    if (!repository || repository.revision !== artifact.revision) {
      throw new Error(`${artifact.id}: repository revision is absent from the pinned catalog.`);
    }
    const modelCard = repository.files.find((file) => file.path === "README.md");
    if (modelCard?.blob_id !== artifact.model_card_blob_sha1) {
      throw new Error(`${artifact.id}: model-card blob identity changed.`);
    }
    const file = repository.files.find((candidate) => candidate.path === artifact.path);
    if (!file || file.size_bytes !== artifact.size_bytes || file.lfs_sha256 !== artifact.sha256) {
      throw new Error(`${artifact.id}: model file identity differs from the pinned catalog.`);
    }
    return { artifact, repository, file };
  });
}

export async function ensureGoogleModernComparator(
  resolved,
  cacheDir = hfCorpusCacheDir(),
  options = {},
) {
  return ensureHfFile(resolved.repository, resolved.file, cacheDir, options);
}

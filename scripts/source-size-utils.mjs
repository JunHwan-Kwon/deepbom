import { existsSync } from "node:fs";
import { readFile } from "node:fs/promises";
import path from "node:path";
import { normalizePath } from "./path-utils.mjs";
import { privateModuleSourcePrefixes } from "./private-wasm-modules.mjs";
import { stripRustTests } from "./rust-source-utils.mjs";
import { collectFileSizes } from "./size-utils.mjs";
export { formatBytes, kibToBytes } from "./size-utils.mjs";

export const DEFAULT_IGNORED_DIRS = new Set([
  ".git",
  ".local-validation",
  ".astro",
  ".wrangler",
  "dist",
  "node_modules",
  "node_modules.incomplete",
  "node_modules.exfat-incomplete",
  "pkg",
  "reference",
  "target",
  "reports",
]);
export const DEFAULT_IGNORED_FILES = new Set(["package-lock.json"]);
export const DEFAULT_IGNORED_FILE_SUFFIXES = [".local.md"];
export const BUDGET_SOURCE_EXTENSIONS = new Set([
  ".js",
  ".mjs",
  ".ts",
  ".tsx",
  ".rs",
  ".html",
  ".css",
  ".json",
  ".jsonc",
  ".md",
  ".yml",
  ".yaml",
]);

const RUNTIME_SOURCE_PREFIXES = [
  "web/",
  "worker/",
  "src/",
  "protected/deepbom_wasm/src/",
];
const PRIVATE_SOURCE_PREFIXES = privateModuleSourcePrefixes();
const NATIVE_TOOLING_SOURCE_PREFIXES = ["native/"];
const PRIVATE_TEST_SOURCE_SUFFIXES = [
  "/tests.rs",
];
const DOCS_SOURCE_PREFIXES = [
  "docs/",
  "docs-site/",
];
const DOCS_SOURCE_FILES = new Set([
  "README.md",
]);
const CORPUS_EVIDENCE_SOURCE_EXTENSIONS = new Set([".json", ".jsonc", ".yml", ".yaml"]);
const GENERATED_RUNTIME_DATA_PATHS = new Set([
  "protected/deepbom_wasm/src/ort_rulepack_generated.rs",
  "protected/deepbom_wasm/src/tflite_delegate_rulepack_generated.rs",
  "src/xnnpack_rulepack_generated.rs",
  "web/lib/gguf-backend-contract.generated.js",
  "web/lib/gguf-codebooks.generated.js",
  "web/lib/executorch-operator-signatures.generated.js",
]);
const VERIFICATION_SOURCE_PREFIXES = [
  "scripts/check-",
  "scripts/validate-",
  "scripts/verify-",
];

export async function collectSourceFiles({
  roots = ["."],
  extensions = BUDGET_SOURCE_EXTENSIONS,
  ignoredDirs = DEFAULT_IGNORED_DIRS,
  cwd = ".",
} = {}) {
  const rootDir = path.resolve(cwd);
  const files = [];
  for (const root of roots) {
    const fullRoot = path.resolve(rootDir, root);
    if (!existsSync(fullRoot)) continue;
    files.push(...(await collectFileSizes(fullRoot, {
      relativeRoot: rootDir,
      extensions,
      ignoredDirs,
    })).filter((file) => !isIgnoredSourceFile(file.path)));
  }
  return files.sort((a, b) => a.path.localeCompare(b.path));
}

export async function addRuntimeSourceBytes(files) {
  return Promise.all(files.map(async (file) => {
    const source = canonicalSourceText(await readFile(file.path, "utf8"));
    const bytes = Buffer.byteLength(source, "utf8");
    const runtimeSource = isRuntimeSourcePath(file.path);
    return {
      ...file,
      bytes,
      runtimeBytes: runtimeSource ? runtimeBytesForSource(file, source, bytes) : 0,
      runtimeSource,
      privateSource: isPrivateSourcePath(file.path),
      privateTestSource: isPrivateTestSourcePath(file.path),
      nativeToolingSource: isNativeToolingSourcePath(file.path),
      docsSource: isDocsSourcePath(file.path),
      corpusEvidenceSource: isCorpusEvidenceSourcePath(file.path),
      generatedRuntimeData: isGeneratedRuntimeDataPath(file.path),
      verificationSource: isVerificationSourcePath(file.path),
    };
  }));
}

export function canonicalSourceText(source) {
  return String(source).replace(/\r\n?/g, "\n");
}

export function sourceTotals(files) {
  const sourceBytes = files.reduce((sum, file) => sum + file.bytes, 0);
  const runtimeBytes = files.reduce((sum, file) => sum + (file.runtimeBytes || 0), 0);
  const privateBytes = files
    .filter((file) => file.privateSource)
    .reduce((sum, file) => sum + file.bytes, 0);
  const privateTestBytes = files
    .filter((file) => file.privateTestSource)
    .reduce((sum, file) => sum + file.bytes, 0);
  const privateRuntimeBytes = Math.max(0, privateBytes - privateTestBytes);
  const docsBytes = files
    .filter((file) => file.docsSource)
    .reduce((sum, file) => sum + file.bytes, 0);
  const nativeToolingBytes = files
    .filter((file) => file.nativeToolingSource)
    .reduce((sum, file) => sum + file.bytes, 0);
  const corpusEvidenceBytes = files
    .filter((file) => file.corpusEvidenceSource)
    .reduce((sum, file) => sum + file.bytes, 0);
  const generatedRuntimeDataBytes = files
    .filter((file) => file.generatedRuntimeData)
    .reduce((sum, file) => sum + (file.runtimeBytes || 0), 0);
  const handwrittenRuntimeBytes = Math.max(0, runtimeBytes - generatedRuntimeDataBytes);
  const publicSourceBytes = Math.max(0, sourceBytes - privateBytes);
  const devBytes = Math.max(0, sourceBytes - runtimeBytes - privateBytes - docsBytes - nativeToolingBytes - corpusEvidenceBytes);
  const verificationBytes = files
    .filter((file) => file.verificationSource && !file.runtimeSource && !file.privateSource)
    .reduce((sum, file) => sum + file.bytes, 0);
  const devToolingBytes = Math.max(0, devBytes - verificationBytes);
  return {
    sourceBytes,
    publicSourceBytes,
    publicCodeBytes: runtimeBytes,
    runtimeBytes,
    handwrittenRuntimeBytes,
    generatedRuntimeDataBytes,
    privateBytes,
    privateRuntimeBytes,
    privateTestBytes,
    docsBytes,
    nativeToolingBytes,
    corpusEvidenceBytes,
    devBytes,
    verificationBytes,
    devToolingBytes,
  };
}

export function isRuntimeSourcePath(relativePath) {
  const normalized = normalizePath(relativePath.trim());
  return RUNTIME_SOURCE_PREFIXES.some((prefix) => normalized.startsWith(prefix));
}

export function isPrivateSourcePath(relativePath) {
  const normalized = normalizePath(relativePath.trim());
  return PRIVATE_SOURCE_PREFIXES.some((prefix) => normalized.startsWith(prefix));
}

export function isNativeToolingSourcePath(relativePath) {
  const normalized = normalizePath(relativePath.trim());
  return NATIVE_TOOLING_SOURCE_PREFIXES.some((prefix) => normalized.startsWith(prefix));
}

export function isPrivateTestSourcePath(relativePath) {
  const normalized = normalizePath(relativePath.trim());
  return isPrivateSourcePath(normalized)
    && PRIVATE_TEST_SOURCE_SUFFIXES.some((suffix) => normalized.endsWith(suffix));
}

export function isDocsSourcePath(relativePath) {
  const normalized = normalizePath(relativePath.trim());
  return DOCS_SOURCE_FILES.has(normalized)
    || DOCS_SOURCE_PREFIXES.some((prefix) => normalized.startsWith(prefix));
}

export function isCorpusEvidenceSourcePath(relativePath) {
  const normalized = normalizePath(relativePath.trim());
  return normalized.startsWith("corpus/") && CORPUS_EVIDENCE_SOURCE_EXTENSIONS.has(path.extname(normalized).toLowerCase());
}

export function isGeneratedRuntimeDataPath(relativePath) {
  return GENERATED_RUNTIME_DATA_PATHS.has(normalizePath(relativePath.trim()));
}

export function isVerificationSourcePath(relativePath) {
  const normalized = normalizePath(relativePath.trim());
  return VERIFICATION_SOURCE_PREFIXES.some((prefix) => normalized.startsWith(prefix));
}

export function isIgnoredSourceFile(relativePath) {
  const normalized = normalizePath(relativePath.trim());
  return DEFAULT_IGNORED_FILES.has(normalized)
    || normalized.endsWith("/package-lock.json")
    || DEFAULT_IGNORED_FILE_SUFFIXES.some((suffix) => normalized.endsWith(suffix));
}

function runtimeBytesForSource(file, source, bytes) {
  return file.path.endsWith(".rs") ? Buffer.byteLength(stripRustTests(source), "utf8") : bytes;
}

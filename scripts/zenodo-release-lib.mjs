import { execFileSync } from "node:child_process";
import { createHash } from "node:crypto";
import { mkdirSync, readFileSync, readdirSync, statSync, writeFileSync } from "node:fs";
import path from "node:path";
import AdmZip from "adm-zip";

export const ZENODO_SCHEMA = "deepbom.zenodo_release_manifest.v1";

const SOFTWARE_RECORD_FILES = new Set([
  ".zenodo.json",
  "CHANGELOG.md",
  "CITATION.cff",
  "docs/REPRODUCIBILITY.md",
  "docs/SOFTWARE_RECORD.md",
]);

export const PROTECTED_PREFIXES = [
  ".local-validation/",
  ".wrangler/",
  "dist/",
  "docs-site/",
  "docs/private/",
  "node_modules/",
  "pro/",
  "reports/",
  "scripts/__pycache__/",
  "target/",
  "web/protected/",
];

export const PROTECTED_FILES = new Set([
  ".wrangler-deploy.json",
  "LOCAL_PRIVATE_ROADMAP.local.md",
  "test_smoke.tflite",
  "web/lib/report-bundle.js",
  "web/lib/report-conformance.js",
  "web/lib/report-conformance-onnx-tfidf.js",
  "web/lib/report-conformance-onnx-tree.js",
  "web/lib/report-engineering-entry.js",
  "web/lib/report-engineering.js",
  "web/lib/report-evidence.js",
  "web/lib/report-export-contracts.js",
  "web/lib/report-findings.js",
  "web/lib/report-integrity.js",
  "web/lib/report-metadata.js",
  "web/lib/report-mlbom.js",
  "web/lib/report-mlbom-compat.js",
  "web/lib/report-quantization-contracts.js",
  "web/lib/report-raw-entry.js",
  "web/lib/report-regulatory-entry.js",
  "web/lib/report-regulatory.js",
  "web/lib/report-sections.js",
  "web/samples/mobilenet_v1_025_224_float.tflite",
  "web/samples/sample_cnn_float.onnx",
]);

const VALIDATION_FORMATS = [
  "tflite",
  "onnx",
  "gguf",
  "safetensors",
  "mlmodel",
  "mlpackage",
  "sharded_safetensors",
];

const SENSITIVE_TEXT_PATTERNS = [
  [/(?:C:\\Users\\|F:\\consistency\\)/i, "local absolute filesystem path"],
  [/-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----/, "private key material"],
  [/(?:oauth_token|refresh_token)\s*=\s*["'][^"']{12,}/i, "stored OAuth credential"],
  [/CLOUDFLARE_API_TOKEN\s*=\s*["'][A-Za-z0-9_-]{20,}/, "Cloudflare API token assignment"],
];

const NONPUBLIC_ARTIFACT_PATH = /(?:^|\/)(?:src|web|scripts|pkg|native|worker|pro)\/|\.(?:js|mjs|cjs|ts|tsx|rs|wasm|py|ps1|sh|dll|dylib|exe|so|tflite|onnx|gguf|safetensors|mlmodel)$/i;

function normalizePath(value) {
  const normalized = String(value || "").replaceAll("\\", "/").replace(/^\.\//, "");
  if (!normalized || normalized.startsWith("/") || normalized.split("/").includes("..")) {
    throw new Error(`Unsafe release path: ${value}`);
  }
  return normalized;
}

function git(args) {
  return execFileSync("git", args, { encoding: "utf8" }).trim();
}

export function repositoryState() {
  const status = execFileSync("git", ["status", "--porcelain", "--untracked-files=all"], { encoding: "utf8" }).trimEnd();
  const tags = git(["tag", "--points-at", "HEAD"]).split(/\r?\n/).filter(Boolean);
  return {
    commit: git(["rev-parse", "HEAD"]),
    commit_date: git(["show", "-s", "--format=%cI", "HEAD"]),
    dirty: Boolean(status),
    dirty_paths: status ? status.split(/\r?\n/).map((line) => line.slice(3).trim()).filter(Boolean) : [],
    tags,
  };
}

export function softwareVersion() {
  const packageJson = JSON.parse(readFileSync("package.json", "utf8"));
  const cargo = readFileSync("Cargo.toml", "utf8");
  const cargoVersion = cargo.match(/^version\s*=\s*"([^"]+)"/m)?.[1];
  const versions = new Set([packageJson.version, cargoVersion]);
  if (versions.size !== 1 || versions.has(undefined)) {
    throw new Error(`Current package versions disagree: ${JSON.stringify([...versions])}`);
  }
  return packageJson.version;
}

export function citationRecordVersion() {
  const citation = readFileSync("CITATION.cff", "utf8");
  const zenodo = JSON.parse(readFileSync(".zenodo.json", "utf8"));
  const citationVersion = citation.match(/^version:\s*["']?([^\s"']+)/m)?.[1];
  const versions = new Set([citationVersion, zenodo.version]);
  if (versions.size !== 1 || versions.has(undefined)) {
    throw new Error(`Published citation-record versions disagree: ${JSON.stringify([...versions])}`);
  }
  return citationVersion;
}

export function analyzerIdentity() {
  const source = readFileSync("web/lib/app-config.js", "utf8");
  const analyzerBuild = source.match(/ANALYZER_VERSION\s*=\s*"([^"]+)"/)?.[1];
  const semanticVersion = source.match(/ANALYZER_SEMANTIC_VERSION\s*=\s*"([^"]+)"/)?.[1];
  const rulepackVersion = source.match(/RULEPACK_VERSION\s*=\s*"([^"]+)"/)?.[1];
  if (!analyzerBuild || !semanticVersion || !rulepackVersion) {
    throw new Error("Analyzer or rulepack identity is missing from web/lib/app-config.js.");
  }
  if (semanticVersion !== softwareVersion()) {
    throw new Error(`Analyzer semantic version ${semanticVersion} does not match release ${softwareVersion()}.`);
  }
  return { analyzer_build_id: analyzerBuild, semantic_version: semanticVersion, rulepack_version: rulepackVersion };
}

export function isPublicSoftwarePath(value) {
  const file = normalizePath(value);
  return SOFTWARE_RECORD_FILES.has(file);
}

function gitKnownFiles() {
  const output = execFileSync("git", ["ls-files", "-co", "--exclude-standard", "-z"]);
  return output.toString("utf8").split("\0").filter(Boolean).map(normalizePath);
}

function sha256(bytes) {
  return createHash("sha256").update(bytes).digest("hex");
}

function entry(file, bytes) {
  return { path: normalizePath(file), bytes: Buffer.from(bytes), sha256: sha256(bytes) };
}

function assertNoSensitiveText(item, context = item.path) {
  if (item.bytes.length > 128 * 1024 * 1024) return;
  if (/\.(?:wasm|png|jpe?g|gif|bin|tflite|onnx|gguf|safetensors|mlmodel|gz)$/i.test(item.path)) return;
  const text = item.bytes.toString("utf8");
  for (const [pattern, label] of SENSITIVE_TEXT_PATTERNS) {
    if (pattern.test(text)) throw new Error(`${context} contains ${label}.`);
  }
}

function assertNoSensitiveZip(item) {
  if (!item.path.toLowerCase().endsWith(".zip")) return;
  const zip = new AdmZip(item.bytes);
  for (const nested of zip.getEntries()) {
    if (nested.isDirectory) continue;
    const nestedPath = normalizePath(nested.entryName);
    if (NONPUBLIC_ARTIFACT_PATH.test(nestedPath)) {
      throw new Error(`${item.path}!/${nestedPath} contains non-public implementation, executable, or model content.`);
    }
    const nestedItem = entry(nestedPath, nested.getData());
    assertNoSensitiveText(nestedItem, `${item.path}!/${nestedPath}`);
  }
}

export function collectSoftwareEntries() {
  const knownFiles = new Set(gitKnownFiles());
  const files = [...SOFTWARE_RECORD_FILES].sort();
  for (const required of files) {
    if (!knownFiles.has(required)) throw new Error(`Required citation-record file is missing: ${required}`);
  }
  const entries = files.map((file) => entry(file, readFileSync(file)));
  for (const item of entries) assertNoSensitiveText(item);
  return entries;
}

function validationPathAllowed(relative) {
  const file = normalizePath(relative);
  if (["README.md", "artifact-catalog.json", "validation-matrix.json"].includes(file)) return true;
  if (file.startsWith("results/") || file.startsWith("ui-results/")) return true;
  return VALIDATION_FORMATS.some((format) => file === `deliverables/${format}/downloads/deepbom_public_artifact_bundle.zip`);
}

export function collectValidationEntries(validationRoot = ".local-validation/supported-formats/latest") {
  const root = path.resolve(validationRoot);
  const matrix = JSON.parse(readFileSync(path.join(root, "validation-matrix.json"), "utf8"));
  const catalog = JSON.parse(readFileSync(path.join(root, "artifact-catalog.json"), "utf8"));
  if (matrix.schema !== "deepbom.supported_format_validation.v2" || matrix.overall_status !== "pass") {
    throw new Error("Supported-format validation matrix is not a passing v2 record.");
  }
  if (!catalog || typeof catalog !== "object") throw new Error("Validation artifact catalog is invalid.");

  const candidates = [];
  const walk = (directory) => {
    for (const dirent of readdirSync(directory, { withFileTypes: true })) {
      const absolute = path.join(directory, dirent.name);
      if (dirent.isDirectory()) walk(absolute);
      else candidates.push(absolute);
    }
  };
  walk(root);

  const entries = candidates
    .map((absolute) => ({ absolute, relative: normalizePath(path.relative(root, absolute)) }))
    .filter(({ relative }) => validationPathAllowed(relative))
    .sort((left, right) => left.relative.localeCompare(right.relative))
    .map(({ absolute, relative }) => entry(relative, readFileSync(absolute)));

  for (const format of VALIDATION_FORMATS) {
    const required = `deliverables/${format}/downloads/deepbom_public_artifact_bundle.zip`;
    if (!entries.some((item) => item.path === required)) throw new Error(`Validation bundle is missing: ${required}`);
  }
  for (const required of ["README.md", "artifact-catalog.json", "validation-matrix.json"]) {
    if (!entries.some((item) => item.path === required)) throw new Error(`Validation record is missing: ${required}`);
  }
  for (const item of entries) {
    if (NONPUBLIC_ARTIFACT_PATH.test(item.path)) {
      throw new Error(`Validation package contains non-public implementation, executable, or model content: ${item.path}`);
    }
    assertNoSensitiveText(item);
    assertNoSensitiveZip(item);
  }
  return { entries, matrix };
}

function selectedTreeHash(entries) {
  const canonical = entries
    .slice()
    .sort((left, right) => left.path.localeCompare(right.path))
    .map((item) => `${item.path}\0${item.bytes.length}\0${item.sha256}\n`)
    .join("");
  return sha256(Buffer.from(canonical));
}

function releaseDate() {
  const citation = readFileSync("CITATION.cff", "utf8");
  const value = citation.match(/^date-released:\s*["']?([^\s"']+)/m)?.[1];
  if (!/^\d{4}-\d{2}-\d{2}$/.test(value || "")) throw new Error("CITATION.cff date-released is invalid.");
  return value;
}

export function publicationBlockers(kind, state, extra = {}) {
  const blockers = [];
  if (state.dirty) blockers.push("worktree_dirty");
  if (kind === "validation" && extra.validation_git_worktree_dirty) {
    blockers.push("validation_worktree_dirty");
  }
  if (kind === "validation" && extra.validation_git_commit !== state.commit) {
    blockers.push("validation_commit_mismatch");
  }
  return blockers;
}

function manifestEntry(kind, entries, state, extra = {}) {
  const identity = analyzerIdentity();
  const blockers = publicationBlockers(kind, state, extra);
  const manifest = {
    schema: ZENODO_SCHEMA,
    package_kind: kind,
    title: kind === "software"
      ? "DEEPBOM Software Citation Record"
      : "DEEPBOM Supported-Format Validation Evidence",
    version: softwareVersion(),
    release_date: releaseDate(),
    generated_at: extra.validation_generated_at || state.commit_date,
    timestamp_basis: extra.validation_generated_at ? "validation_matrix_generated_at" : "git_commit_date",
    git_commit: state.commit,
    git_commit_date: state.commit_date,
    git_tag: state.tags.includes(`v${softwareVersion()}`) ? `v${softwareVersion()}` : null,
    git_worktree_dirty: state.dirty,
    publication_ready: blockers.length === 0,
    publication_blockers: blockers,
    selected_payload_file_count: entries.length,
    selected_payload_bytes: entries.reduce((sum, item) => sum + item.bytes.length, 0),
    selected_tree_sha256: selectedTreeHash(entries),
    analyzer_identity: identity,
    doi: process.env.ZENODO_DOI || null,
    scope: kind === "software"
      ? "Restricted non-code citation and release-provenance dossier for the hosted software release."
      : "Hash-bound parser results, browser UI captures, and public engineering bundles for advertised format paths.",
    exclusions: kind === "software"
      ? ["source code", "JavaScript application modules", "WebAssembly and native binaries", "model artifacts", "credentials", "protected report logic"]
      : ["source model duplication", "local logs", "temporary fixtures", "pre-fix backups", "expanded duplicate ZIP members"],
    ...extra,
  };
  return entry("ZENODO_RELEASE_MANIFEST.json", Buffer.from(`${JSON.stringify(manifest, null, 2)}\n`));
}

function checksumEntry(entries) {
  const text = entries
    .slice()
    .sort((left, right) => left.path.localeCompare(right.path))
    .map((item) => `${item.sha256}  ${item.bytes.length}  ${item.path}`)
    .join("\n");
  return entry("SHA256SUMS", Buffer.from(`${text}\n`));
}

function zipTimestamp(dateText) {
  return new Date(`${dateText}T00:00:00.000Z`);
}

function writeVerifiedZip(outputPath, rootName, entries, dateText) {
  const zip = new AdmZip();
  const expected = new Map();
  for (const item of entries.slice().sort((left, right) => left.path.localeCompare(right.path))) {
    const archivePath = `${rootName}/${item.path}`;
    zip.addFile(archivePath, item.bytes, "", 0o100644 << 16);
    const added = zip.getEntry(archivePath);
    added.header.time = zipTimestamp(dateText);
    expected.set(archivePath, item.sha256);
  }
  mkdirSync(path.dirname(outputPath), { recursive: true });
  zip.writeZip(outputPath);

  const reopened = new AdmZip(outputPath);
  const actualFiles = reopened.getEntries().filter((item) => !item.isDirectory);
  if (actualFiles.length !== expected.size) throw new Error("Zenodo ZIP member count changed after serialization.");
  for (const item of actualFiles) {
    const expectedHash = expected.get(item.entryName);
    if (!expectedHash || sha256(item.getData()) !== expectedHash) {
      throw new Error(`Zenodo ZIP verification failed for ${item.entryName}.`);
    }
  }
}

export function buildZenodoArchive({
  kind = "software",
  allowDirty = false,
  outputDirectory = ".local-validation/zenodo-citation",
  validationRoot = ".local-validation/supported-formats/latest",
} = {}) {
  if (!new Set(["software", "validation"]).has(kind)) throw new Error(`Unknown Zenodo package kind: ${kind}`);
  if (citationRecordVersion() !== softwareVersion()) {
    throw new Error(`Zenodo metadata is still pinned to published record ${citationRecordVersion()}; prepare and reserve a version-specific DOI for ${softwareVersion()} before building a new archive.`);
  }
  const state = repositoryState();
  if (state.dirty && !allowDirty) {
    throw new Error(`Final Zenodo package requires a clean worktree. Dirty paths: ${state.dirty_paths.join(", ")}`);
  }
  const expectedTag = `v${softwareVersion()}`;
  if (!allowDirty && !state.tags.includes(expectedTag)) {
    throw new Error(`Final Zenodo package requires tag ${expectedTag} on commit ${state.commit}.`);
  }

  let payload;
  let extra = {};
  if (kind === "software") {
    payload = collectSoftwareEntries();
  } else {
    const collected = collectValidationEntries(validationRoot);
    payload = collected.entries;
    extra = {
      validation_generated_at: collected.matrix.generated_at,
      validation_git_commit: collected.matrix.environment?.git_commit || null,
      validation_git_worktree_dirty: Boolean(collected.matrix.environment?.git_worktree_dirty),
      validation_repository_content_sha256: collected.matrix.environment?.repository_content_sha256 || null,
      validation_overall_status: collected.matrix.overall_status,
    };
    const datasetMetadata = JSON.parse(readFileSync("docs/zenodo-validation-metadata.json", "utf8"));
    if (datasetMetadata.version !== softwareVersion() || datasetMetadata.upload_type !== "dataset") {
      throw new Error("Zenodo validation-dataset metadata does not match the software release.");
    }
    payload.push(entry("ZENODO_DATASET_METADATA.json", Buffer.from(`${JSON.stringify(datasetMetadata, null, 2)}\n`)));
    if (!allowDirty && (extra.validation_git_worktree_dirty || extra.validation_git_commit !== state.commit)) {
      throw new Error("Final validation package must come from the same clean Git commit as the software release.");
    }
  }

  const releaseManifest = manifestEntry(kind, payload, state, extra);
  const checksums = checksumEntry([...payload, releaseManifest]);
  const entries = [...payload, releaseManifest, checksums];
  const version = softwareVersion();
  const validationUnready = kind === "validation"
    && (extra.validation_git_worktree_dirty || extra.validation_git_commit !== state.commit);
  const draft = state.dirty || validationUnready;
  const blockers = publicationBlockers(kind, state, extra);
  const baseName = kind === "software" ? `deepbom-v${version}-citation-record` : `deepbom-v${version}-validation-evidence`;
  const archiveName = `${baseName}${draft ? "-DRAFT-dirty" : ""}.zip`;
  const rootName = `${baseName}${draft ? "-DRAFT-dirty" : ""}`;
  const outputPath = path.resolve(outputDirectory, archiveName);
  writeVerifiedZip(outputPath, rootName, entries, releaseDate());
  const archiveBytes = readFileSync(outputPath);
  const archiveSha256 = sha256(archiveBytes);
  writeFileSync(`${outputPath}.sha256`, `${archiveSha256}  ${archiveName}\n`);
  const summary = {
    schema: "deepbom.zenodo_package_summary.v1",
    kind,
    publication_ready: !draft,
    publication_blockers: blockers,
    archive: outputPath,
    archive_bytes: statSync(outputPath).size,
    archive_sha256: archiveSha256,
    archive_member_count: entries.length,
    git_commit: state.commit,
    git_tag: state.tags.includes(expectedTag) ? expectedTag : null,
    git_worktree_dirty: state.dirty,
  };
  writeFileSync(`${outputPath}.json`, `${JSON.stringify(summary, null, 2)}\n`);
  return summary;
}

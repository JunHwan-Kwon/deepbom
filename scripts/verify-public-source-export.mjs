import { createHash } from "node:crypto";
import { existsSync } from "node:fs";
import { readFile, readdir } from "node:fs/promises";
import path from "node:path";
import process from "node:process";

const root = process.cwd();
const defaultExportRoot = existsSync(path.join(root, "PUBLIC_SOURCE_MANIFEST.json"))
  ? root : path.join(root, ".local-validation", "public-source");
const exportRoot = path.resolve(process.argv[2] || defaultExportRoot);
const manifestPath = path.join(exportRoot, "PUBLIC_SOURCE_MANIFEST.json");
const manifest = JSON.parse(await readFile(manifestPath, "utf8"));
assert(manifest.schema === "deepbom.public_source_export.v1", "Unexpected public source manifest schema.");
assert(manifest.license === "Apache-2.0", "Public source manifest license drifted.");
assert(/^[a-f0-9]{40}$/.test(manifest.source_commit), "Public source manifest commit is invalid.");
assert(["clean", "dirty"].includes(manifest.source_state), "Public source manifest source state is invalid.");

const declared = manifest.files || [];
assert(manifest.file_count === declared.length, "Public source manifest file count does not conserve declared records.");
const declaredPaths = declared.map((record) => record.path);
assert(new Set(declaredPaths).size === declaredPaths.length, "Public source manifest contains duplicate paths.");
assert(JSON.stringify(declaredPaths) === JSON.stringify([...declaredPaths].sort()), "Public source manifest paths are not sorted.");

const actualPaths = (await collectFiles(exportRoot))
  .map((file) => normalize(path.relative(exportRoot, file)))
  .filter((file) => file !== "PUBLIC_SOURCE_MANIFEST.json")
  .sort();
assertEqual(actualPaths, declaredPaths, "public source member ledger");

for (const record of declared) {
  const file = path.join(exportRoot, record.path);
  const bytes = await readFile(file);
  assert(bytes.byteLength === record.byte_length, `Public source byte length drifted for ${record.path}.`);
  assert(sha256(bytes) === record.sha256, `Public source SHA-256 drifted for ${record.path}.`);
  assert(!isForbidden(record.path), `Forbidden path entered public source export: ${record.path}`);
}

const publicLicense = await readFile(path.join(exportRoot, "channels", "LICENSE"));
const rootLicense = await readFile(path.join(exportRoot, "LICENSE"));
assert(sha256(publicLicense) === sha256(rootLicense), "Public source root and channel licenses differ.");
assert(sha256(await readFile(path.join(exportRoot, "README.md"))) === sha256(await readFile(path.join(exportRoot, "docs", "PUBLIC_README.md"))), "Public root README differs from its reviewed public source.");

const packageDocument = JSON.parse(await readFile(path.join(exportRoot, "package.json"), "utf8"));
const lockDocument = JSON.parse(await readFile(path.join(exportRoot, "package-lock.json"), "utf8"));
const wasmPackageDocument = JSON.parse(await readFile(path.join(exportRoot, "pkg", "package.json"), "utf8"));
assert(packageDocument.private === true, "Public source root must retain the accidental-publication guard.");
assert(packageDocument.license === "Apache-2.0", "Public source package metadata license drifted.");
assert(packageDocument.homepage === "https://deepbom.org", "Public source homepage metadata drifted.");
assert(packageDocument.repository?.url === "git+https://github.com/JunHwan-Kwon/deepbom.git", "Public source repository metadata drifted.");
assert(packageDocument.bugs?.url === "https://github.com/JunHwan-Kwon/deepbom/issues", "Public source issue-tracker metadata drifted.");
assert(lockDocument.packages?.[""]?.license === "Apache-2.0", "Public source lockfile root license drifted.");
assert(wasmPackageDocument.license === "Apache-2.0", "Public WASM package metadata license drifted.");
assert(sha256(await readFile(path.join(exportRoot, "pkg", "LICENSE"))) === sha256(publicLicense), "Public WASM package license differs from the root Apache-2.0 license.");
for (const privateScript of [
  "generate:ort-rulepack",
  "generate:tflite-delegate-rulepack",
  "generate:xnnpack-delegate-rulepack",
  "verify:tflite-delegate-source-pins",
]) assert(!(privateScript in (packageDocument.scripts || {})), `Private package script entered public source: ${privateScript}`);

for (const required of [
  ".github/workflows/public-quality.yml",
  ".github/workflows/release-channels.yml",
  "docs/CLI_REFERENCE.md",
  "docs/PUBLIC_PRIVATE_BOUNDARY.md",
  "scripts/check-artifact-ir-import-boundary.mjs",
  "config/artifact-ir-import-policy.v1.json",
  "scripts/check-artifact-ir-consumers.mjs",
  "config/artifact-ir-consumer-policy.v1.json",
  "scripts/check-evidence-ui-contract.mjs",
  "scripts/check-ir-stabilization-ui.mjs",
  "scripts/run-release-validation.mjs",
  "scripts/check-public-package-boundary.mjs",
  "scripts/generate-cli-docs.mjs",
  "web/lib/artifact-ir-context.js",
  "web/lib/artifact-ir-runtime.js",
  "web/lib/artifact-ir/internal/architecture.js",
  "web/lib/artifact-ir/internal/constants.js",
  "web/lib/artifact-ir/internal/graph.js",
  "web/lib/artifact-ir/internal/identity.js",
  "web/lib/artifact-ir/internal/overlays.js",
  "web/lib/artifact-ir/internal/quantization.js",
  "web/lib/artifact-ir/internal/shared.js",
  "web/lib/artifact-ir/internal/storage.js",
  "web/lib/artifact-ir/internal/validation.js",
  "web/lib/evidence-applicability.js",
  "web/lib/evidence-class.js",
  "web/lib/evidence-visual-contract.js",
]) assert(actualPaths.includes(required), `Required public source member is missing: ${required}`);
assert(packageDocument.scripts?.["generate:cli-docs"] === "node scripts/generate-cli-docs.mjs",
  "Public source CLI documentation generator script drifted.");
assert(packageDocument.scripts?.["check:cli-docs"] === "node scripts/generate-cli-docs.mjs --check",
  "Public source CLI documentation check script drifted.");

console.log(`Public source export verification passed (${declared.length} files; exact member, byte-length, SHA-256, license, workflow, package-script, and private-path contracts checked).`);

async function collectFiles(directory, rootDirectory = directory) {
  const files = [];
  for (const entry of await readdir(directory, { withFileTypes: true })) {
    const file = path.join(directory, entry.name);
    const relative = normalize(path.relative(rootDirectory, file));
    if (entry.isDirectory() && isLocalOnlyPath(relative)) continue;
    if (entry.isDirectory()) files.push(...await collectFiles(file, rootDirectory));
    else if (entry.isFile()) files.push(file);
  }
  return files;
}

function isLocalOnlyPath(file) {
  return /^(?:\.git|node_modules|\.local-validation)(?:\/|$)/.test(file);
}

function isForbidden(file) {
  return file === ".github/workflows/pages.yml"
    || file === ".github/workflows/quality.yml"
    || file === "wrangler.jsonc"
    || file === "LICENSE.private"
    || file.startsWith("protected/")
    || file.startsWith("web/protected/")
    || file.startsWith("docs/private/")
    || file.startsWith("migrations/")
    || file.startsWith("worker/")
    || file.endsWith(".local.md")
    || file.endsWith(".map")
    || /^scripts\/generate-(?:ort-rulepack|tflite-delegate-rulepack|xnnpack-delegate-rulepack)\.mjs$/.test(file);
}

function sha256(bytes) {
  return createHash("sha256").update(bytes).digest("hex");
}

function normalize(value) {
  return String(value).replaceAll("\\", "/").normalize("NFC");
}

function assertEqual(actual, expected, label) {
  assert(JSON.stringify(actual) === JSON.stringify(expected), `${label} mismatch:\nexpected ${JSON.stringify(expected)}\nobserved ${JSON.stringify(actual)}`);
}

function assert(condition, message) {
  if (!condition) throw new Error(message);
}

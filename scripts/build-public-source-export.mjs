import { createHash } from "node:crypto";
import { execFileSync } from "node:child_process";
import { existsSync } from "node:fs";
import { copyFile, lstat, mkdir, readFile, readdir, rm, writeFile } from "node:fs/promises";
import path from "node:path";
import process from "node:process";

const root = process.cwd();
const allowlistPath = path.join(root, "config", "public-source-files.v1.txt");
const defaultOutput = path.join(root, ".local-validation", "public-source");
const output = path.resolve(argumentValue("--output") || defaultOutput);
const forbiddenPrefixes = [
  "protected/",
  "web/protected/",
  "docs/private/",
  "migrations/",
  "worker/",
];
const forbiddenFiles = new Set([
  "LICENSE",
  ".github/workflows/pages.yml",
  ".github/workflows/quality.yml",
  "wrangler.jsonc",
  "scripts/generate-ort-rulepack.mjs",
  "scripts/generate-tflite-delegate-rulepack.mjs",
  "scripts/generate-xnnpack-delegate-rulepack.mjs",
]);
const forbiddenSuffixes = [".local.md", ".map"];

if (process.argv.includes("--refresh")) await refreshAllowlist();
const allowed = await readAllowlist();
await verifyAllowlist(allowed);
if (process.argv.includes("--export")) await exportPublicSource(allowed);
else console.log(`Public source allowlist passed (${allowed.length} exact files; private source roots and generators excluded).`);

async function refreshAllowlist() {
  if (!process.argv.includes("--acknowledge-public-file-review")) {
    throw new Error("Refreshing the public allowlist requires --acknowledge-public-file-review after reviewing every added path.");
  }
  const discovered = gitLines(["ls-files", "--cached", "--others", "--exclude-standard"])
    .map(normalize)
    .filter((file) => file && !isForbidden(file))
    .sort();
  const candidates = [];
  for (const file of discovered) {
    const record = await fileRecordOrNull(path.join(root, file));
    if (record?.isFile()) candidates.push(file);
  }
  await mkdir(path.dirname(allowlistPath), { recursive: true });
  await writeFile(allowlistPath, `${candidates.join("\n")}\n`);
  console.log(`Refreshed public source allowlist with ${candidates.length} reviewed candidates.`);
}

async function readAllowlist() {
  const source = await readFile(allowlistPath, "utf8");
  return source.split(/\r?\n/).map((row) => row.trim()).filter((row) => row && !row.startsWith("#"));
}

async function verifyAllowlist(files) {
  const sorted = [...files].sort();
  assert(JSON.stringify(files) === JSON.stringify(sorted), "Public source allowlist must be sorted.");
  assert(new Set(files).size === files.length, "Public source allowlist contains duplicate paths.");
  for (const file of files) {
    assert(file === normalize(file), `Public source path is not normalized: ${file}`);
    assert(!path.isAbsolute(file) && file !== ".." && !file.startsWith("../"), `Public source path escapes the repository: ${file}`);
    assert(!isForbidden(file), `Private path entered the public allowlist: ${file}`);
    const record = await fileRecordOrNull(path.join(root, file));
    assert(record, `Public source member is missing: ${file}`);
    assert(record.isFile(), `Public source member is missing or not a file: ${file}`);
  }
  for (const required of [
    "bin/deepbom.mjs",
    "channels/LICENSE",
    "docs/CLI_REFERENCE.md",
    "docs/PUBLIC_PRIVATE_BOUNDARY.md",
    "scripts/build-channel-artifacts.mjs",
    "scripts/build-public-source-export.mjs",
    "scripts/check-public-package-boundary.mjs",
    "scripts/generate-cli-docs.mjs",
    "src/lib.rs",
    "web/lib/artifact-ir-context.js",
  ]) assert(files.includes(required), `Public source allowlist is missing required member ${required}.`);
  assert(!files.some((file) => forbiddenPrefixes.some((prefix) => file.startsWith(prefix))), "Protected source prefix entered public allowlist.");
}

async function exportPublicSource(files) {
  const allowedRoot = path.join(root, ".local-validation") + path.sep;
  assert(output.startsWith(allowedRoot), `Public source output must remain under ${allowedRoot}`);
  await rm(output, { recursive: true, force: true });
  for (const file of files) {
    const source = path.join(root, file);
    const destination = path.join(output, file);
    await mkdir(path.dirname(destination), { recursive: true });
    await copyFile(source, destination);
  }
  await copyFile(path.join(root, "channels", "LICENSE"), path.join(output, "LICENSE"));
  await copyFile(path.join(root, "docs", "PUBLIC_README.md"), path.join(output, "README.md"));
  await rewritePublicPackageMetadata();
  writePublicBuildMetadata();
  const records = [];
  for (const file of await collectFiles(output)) {
    records.push(await fileRecord(file, normalize(path.relative(output, file))));
  }
  records.sort((a, b) => a.path < b.path ? -1 : a.path > b.path ? 1 : 0);
  const manifest = {
    schema: "deepbom.public_source_export.v1",
    source_commit: gitCapture(["rev-parse", "HEAD"]),
    source_state: gitCapture(["status", "--porcelain=v1", "--untracked-files=all", "--", ".", ":!.local-validation"]) ? "dirty" : "clean",
    license: "Apache-2.0",
    file_count: records.length,
    files: records,
    excluded_classes: ["protected source", "protected generated modules", "rulepack generators", "hosted backend and deployment bindings", "private roadmap", "unreleased private research"],
  };
  await writeFile(path.join(output, "PUBLIC_SOURCE_MANIFEST.json"), `${JSON.stringify(manifest, null, 2)}\n`);
  await assertExportHasNoForbiddenPaths(output);
  console.log(`Built clean public source export at ${output} (${records.length} files plus manifest).`);
}

function writePublicBuildMetadata() {
  execFileSync(process.execPath, ["scripts/write-build-metadata.mjs", "--public-distribution"], {
    cwd: output,
    encoding: "utf8",
    stdio: ["ignore", "pipe", "pipe"],
  });
  assert(existsSync(path.join(output, "web", "lib", "build-metadata.js")), "Public build metadata was not generated.");
}

async function rewritePublicPackageMetadata() {
  const publicLicense = path.join(root, "channels", "LICENSE");
  const packagePath = path.join(output, "package.json");
  const packageDocument = JSON.parse(await readFile(packagePath, "utf8"));
  packageDocument.license = "Apache-2.0";
  for (const name of [
    "generate:ort-rulepack",
    "generate:tflite-delegate-rulepack",
    "generate:xnnpack-delegate-rulepack",
    "verify:tflite-delegate-source-pins",
  ]) delete packageDocument.scripts?.[name];
  await writeFile(packagePath, `${JSON.stringify(packageDocument, null, 2)}\n`);

  const lockPath = path.join(output, "package-lock.json");
  const lockDocument = JSON.parse(await readFile(lockPath, "utf8"));
  if (lockDocument.packages?.[""]) lockDocument.packages[""].license = "Apache-2.0";
  await writeFile(lockPath, `${JSON.stringify(lockDocument, null, 2)}\n`);

  const wasmPackagePath = path.join(output, "pkg", "package.json");
  const wasmPackageDocument = JSON.parse(await readFile(wasmPackagePath, "utf8"));
  wasmPackageDocument.license = "Apache-2.0";
  await writeFile(wasmPackagePath, `${JSON.stringify(wasmPackageDocument, null, 2)}\n`);
  await copyFile(publicLicense, path.join(output, "pkg", "LICENSE"));
}

async function assertExportHasNoForbiddenPaths(directory) {
  const files = await collectFiles(directory);
  for (const absolute of files) {
    const file = normalize(path.relative(directory, absolute));
    if (file === "PUBLIC_SOURCE_MANIFEST.json" || file === "LICENSE") continue;
    assert(!isForbidden(file), `Forbidden path was copied to public export: ${file}`);
  }
}

async function collectFiles(directory) {
  const files = [];
  for (const entry of await readdir(directory, { withFileTypes: true })) {
    const file = path.join(directory, entry.name);
    if (entry.isDirectory()) files.push(...await collectFiles(file));
    else if (entry.isFile()) files.push(file);
  }
  return files;
}

async function fileRecord(file, displayPath) {
  const bytes = await readFile(file);
  return { path: normalize(displayPath), byte_length: bytes.byteLength, sha256: createHash("sha256").update(bytes).digest("hex") };
}

async function fileRecordOrNull(file) {
  try {
    return await lstat(file);
  } catch (error) {
    if (error?.code === "ENOENT") return null;
    throw error;
  }
}

function isForbidden(file) {
  return forbiddenFiles.has(file)
    || forbiddenPrefixes.some((prefix) => file.startsWith(prefix))
    || forbiddenSuffixes.some((suffix) => file.endsWith(suffix));
}

function gitLines(args) {
  const value = execFileSync("git", args, { cwd: root, encoding: "utf8" }).trim();
  return value ? value.split(/\r?\n/) : [];
}

function gitCapture(args) {
  return execFileSync("git", args, { cwd: root, encoding: "utf8" }).trim();
}

function argumentValue(flag) {
  const index = process.argv.indexOf(flag);
  return index >= 0 ? process.argv[index + 1] : "";
}

function normalize(value) {
  return String(value).replaceAll("\\", "/").normalize("NFC");
}

function assert(condition, message) {
  if (!condition) throw new Error(message);
}

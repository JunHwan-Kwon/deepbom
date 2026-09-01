import { readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";
import { readVersionContract } from "./version-contract.mjs";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const args = process.argv.slice(2);
const check = args.includes("--check");
const validateIndex = args.indexOf("--validate-release");
const releaseIndex = args.indexOf("--release");
const releaseVersion = validateIndex >= 0 ? requiredValue(validateIndex) : releaseIndex >= 0 ? requiredValue(releaseIndex) : "";
const contract = await readVersionContract(root, { releaseVersion });

if (validateIndex >= 0) {
  console.log(`Release ${contract.baseVersion} matches release/version.json.`);
  process.exit(0);
}

const updates = new Map();
await updateJson("package.json", (document) => ({ ...document, version: contract.npmVersion }));
await updateJson("pkg/package.json", (document) => ({ ...document, version: contract.npmVersion }));
await updateJson("package-lock.json", (document) => {
  document.version = contract.npmVersion;
  if (document.packages?.[""]) document.packages[""].version = contract.npmVersion;
  return document;
});
await updateTomlVersion("Cargo.toml", contract.cargoVersion);
await updateCargoLock("Cargo.lock", "tflite_wasm_audit", contract.cargoVersion);
await updateTomlVersion("channels/cargo/Cargo.toml", contract.cargoVersion);
await updateCargoLock("channels/cargo/Cargo.lock", "deepbom", contract.cargoVersion);
await updateTomlVersion("channels/python/pyproject.toml", contract.pythonVersion);
await updateRegex("channels/python/src/deepbom/__init__.py", /^__version__\s*=\s*"[^"]+"/m, `__version__ = "${contract.pythonVersion}"`);
await updateRegex("bin/deepbom.mjs", /const VERSION = typeof __DEEPBOM_RELEASE_VERSION__ === "string" \? __DEEPBOM_RELEASE_VERSION__ : "[^"]+";/,
  `const VERSION = typeof __DEEPBOM_RELEASE_VERSION__ === "string" ? __DEEPBOM_RELEASE_VERSION__ : "${contract.displayVersion}";`);
await updateRegex("web/lib/app-config.js", /^export const ANALYZER_SEMANTIC_VERSION\s*=\s*"[^"]+";/m,
  `export const ANALYZER_SEMANTIC_VERSION = "${contract.displayVersion}";`);
await updateRegex("web/index.html", /<meta name="deepbom-release" content="[^"]+"\s*\/>/,
  `<meta name="deepbom-release" content="${contract.displayVersion}" />`);
await updateRegex("web/index.html", /"softwareVersion":\s*"[^"]+"/,
  `"softwareVersion": "${contract.displayVersion}"`);
await updateRegex("examples/expected-output/gpu-partition-probe.human.txt", /^DEEPBOM\s+\S+\s+deployment-artifact audit/m,
  `DEEPBOM ${contract.displayVersion} deployment-artifact audit`);

const drift = [];
for (const [relativePath, next] of updates) {
  const current = await readFile(path.join(root, relativePath), "utf8");
  if (current === next) continue;
  drift.push(relativePath);
  if (!check) await writeFile(path.join(root, relativePath), next, "utf8");
}
if (check && drift.length) throw new Error(`Version drift from release/version.json: ${drift.join(", ")}`);
console.log(`${check ? "Verified" : "Synchronized"} ${contract.displayVersion} (${contract.channel}); ${drift.length} file(s) ${check ? "differ" : "updated"}.`);

function requiredValue(index) {
  const value = args[index + 1];
  if (!value || value.startsWith("--")) throw new Error(`${args[index]} requires an exact version.`);
  return value;
}

async function updateJson(relativePath, transform) {
  const document = JSON.parse(await readFile(path.join(root, relativePath), "utf8"));
  updates.set(relativePath, `${JSON.stringify(transform(document), null, 2)}\n`);
}

async function updateTomlVersion(relativePath, version) {
  await updateRegex(relativePath, /^version\s*=\s*"[^"]+"/m, `version = "${version}"`);
}

async function updateCargoLock(relativePath, packageName, version) {
  const source = await readFile(path.join(root, relativePath), "utf8");
  const escaped = packageName.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  const pattern = new RegExp(`(\\[\\[package\\]\\]\\r?\\nname = "${escaped}"\\r?\\nversion = ")[^"]+(")`);
  if (!pattern.test(source)) throw new Error(`${relativePath} does not contain package ${packageName}.`);
  updates.set(relativePath, source.replace(pattern, `$1${version}$2`));
}

async function updateRegex(relativePath, pattern, replacement) {
  const source = updates.get(relativePath) ?? await readFile(path.join(root, relativePath), "utf8");
  if (!pattern.test(source)) throw new Error(`${relativePath} does not match the version contract pattern.`);
  updates.set(relativePath, source.replace(pattern, replacement));
}

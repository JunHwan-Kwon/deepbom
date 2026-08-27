import { execFileSync } from "node:child_process";
import { appendFileSync, existsSync, readFileSync } from "node:fs";
import { resolve } from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";
import { normalizePath } from "./path-utils.mjs";
import { privateModuleSourcePrefixes } from "./private-wasm-modules.mjs";
import { stripRustTests } from "./rust-source-utils.mjs";

export { stripRustTests } from "./rust-source-utils.mjs";

const ALWAYS_DEPLOY_PREFIXES = [
  "web/",
  "worker/",
  "pkg/",
];

const ALWAYS_DEPLOY_FILES = new Set([
  "Cargo.toml",
  "Cargo.lock",
  "package-lock.json",
  "wrangler.jsonc",
  "scripts/build-pages.mjs",
  "scripts/build-release.mjs",
  "scripts/release-generated-artifacts.mjs",
  "scripts/write-build-metadata.mjs",
  "scripts/write-cloudflare-deploy-config.mjs",
  "scripts/harden-wasm-binary.mjs",
  "scripts/wasm-binary-hardening.mjs",
  "scripts/patch-litert-int8.mjs",
  "scripts/sw-utils.mjs",
  "protected/deepbom_wasm/Cargo.toml",
  "protected/deepbom_wasm/Cargo.lock",
]);

const CHECK_ONLY_PREFIXES = [
  "docs/",
  ".github/workflows/",
];

const CHECK_ONLY_FILES = new Set([
  ".gitignore",
  ".github/workflows/pages.yml",
  "README.md",
  "scripts/check-all.mjs",
  "scripts/check-deploy.mjs",
  "scripts/check-browser-import-contract.mjs",
  "scripts/check-ci-deploy-contract.mjs",
  "scripts/check-dist-assets.mjs",
  "scripts/check-dist-budget.mjs",
  "scripts/check-dom-contract.mjs",
  "scripts/check-export-artifact-contract.mjs",
  "scripts/check-git-privacy.mjs",
  "scripts/check-html-entrypoints.mjs",
  "scripts/check-importmap-runtime-cache.mjs",
  "scripts/check-regulatory-bundle-contract.mjs",
  "scripts/check-rust.mjs",
  "scripts/check-size-utils.mjs",
  "scripts/check-source-budget.mjs",
  "scripts/check-sw-assets.mjs",
  "scripts/check-sw-version.mjs",
  "scripts/check-web-imports.mjs",
  "scripts/check-worker-config.mjs",
  "scripts/check-workflow-contract.mjs",
  "scripts/check-js.mjs",
  "scripts/detect-deployable-changes.mjs",
  "scripts/report-dist-size.mjs",
  "scripts/report-source-size.mjs",
  "scripts/run-utils.mjs",
  "scripts/verify-local.mjs",
  "scripts/visualizer-parity.mjs",
]);

const PACKAGE_BUILD_SCRIPT_KEYS = new Set([
  "postinstall",
  "patch:litert",
  "build:wasm",
  "build:deepbom",
  "build:worker",
  "build:pages",
]);

const PRIVATE_OPTIONAL_WASM_PREFIXES = privateModuleSourcePrefixes();

export function classifyChangeSet(files, readers = {}) {
  const reasons = [];
  const rustCheckReasons = [];
  const privateWasmCheckReasons = [];
  const normalized = files.map(normalizeChangedPath).filter(Boolean);

  for (const file of normalized) {
    const reason = deployReasonForFile(file, readers);
    if (reason) {
      reasons.push(`${file}: ${reason}`);
    }
    const rustCheckReason = rustCheckReasonForFile(file, readers);
    if (rustCheckReason) {
      rustCheckReasons.push(`${file}: ${rustCheckReason}`);
    }
    const privateWasmCheckReason = privateWasmCheckReasonForFile(file, readers);
    if (privateWasmCheckReason) {
      privateWasmCheckReasons.push(`${file}: ${privateWasmCheckReason}`);
    }
  }

  return {
    changed: reasons.length > 0,
    reason: reasons.length > 0 ? "deployable-path" : "check-or-test-only",
    details: reasons,
    rustCheck: rustCheckReasons.length > 0,
    rustCheckDetails: rustCheckReasons,
    privateWasmCheck: privateWasmCheckReasons.length > 0,
    privateWasmCheckDetails: privateWasmCheckReasons,
  };
}

export function deployReasonForFile(file, readers = {}) {
  if (file === "pkg/README.md") return "";
  if (ALWAYS_DEPLOY_FILES.has(file)) return "deploy asset/build input";
  if (ALWAYS_DEPLOY_PREFIXES.some((prefix) => file.startsWith(prefix))) {
    return "deploy asset/runtime path";
  }
  if (file === "package.json") {
    return packageJsonDeployReason(readers.readBefore?.(file), readers.readAfter?.(file));
  }
  if (isPublicRustRuntimeSource(file)) {
    return rustSourceDeployReason(readers.readBefore?.(file), readers.readAfter?.(file));
  }
  if (isPrivateOptionalWasmPath(file)) return "";
  if (CHECK_ONLY_FILES.has(file) || CHECK_ONLY_PREFIXES.some((prefix) => file.startsWith(prefix))) {
    return "";
  }
  if (file.startsWith("scripts/")) return "";
  return "unclassified path";
}

export function packageJsonDeployReason(beforeText = "{}", afterText = "{}") {
  let before;
  let after;
  try {
    before = JSON.parse(beforeText || "{}");
    after = JSON.parse(afterText || "{}");
  } catch {
    return "package.json parse changed";
  }

  const beforeScripts = before.scripts || {};
  const afterScripts = after.scripts || {};
  const beforeRest = { ...before, scripts: undefined };
  const afterRest = { ...after, scripts: undefined };
  if (stableJson(beforeRest) !== stableJson(afterRest)) {
    return "package metadata/dependencies changed";
  }

  const scriptKeys = new Set([...Object.keys(beforeScripts), ...Object.keys(afterScripts)]);
  for (const key of scriptKeys) {
    if (beforeScripts[key] === afterScripts[key]) continue;
    if (PACKAGE_BUILD_SCRIPT_KEYS.has(key)) {
      return `build script changed (${key})`;
    }
  }
  return "";
}

export function rustSourceDeployReason(beforeText = "", afterText = "") {
  if (stripRustTests(beforeText) === stripRustTests(afterText)) {
    return "";
  }
  return "rust runtime source changed";
}

export function rustCheckReasonForFile(file, readers = {}) {
  if (
    file === "Cargo.toml"
    || file === "Cargo.lock"
    || file === "build.rs"
    || file === "scripts/check-rust.mjs"
  ) {
    return "rust quality input changed";
  }
  if (file === "package.json") {
    return packageJsonRustCheckReason(readers.readBefore?.(file), readers.readAfter?.(file));
  }
  if (file.startsWith("native/") && (file.endsWith("Cargo.toml")
    || file.endsWith("Cargo.lock")
    || file.endsWith(".cc")
    || file.endsWith(".h")
    || file.endsWith("CMakeLists.txt"))) {
    return "native quality input changed";
  }
  if (isRustSource(file)) {
    const before = readers.readBefore?.(file) || "";
    const after = readers.readAfter?.(file) || "";
    return before === after ? "" : "rust source/test changed";
  }
  if (file.startsWith("protected/") && file.includes("/Cargo.")) {
    return "rust crate manifest changed";
  }
  return "";
}

export function packageJsonRustCheckReason(beforeText = "{}", afterText = "{}") {
  let before;
  let after;
  try {
    before = JSON.parse(beforeText || "{}");
    after = JSON.parse(afterText || "{}");
  } catch {
    return "package.json parse changed";
  }
  const rustScriptKeys = [
    "build:wasm",
    "build:deepbom",
    "build:target-inversion",
    "build:native-collector",
    "build:native-probe",
    "check:rust",
  ];
  for (const key of rustScriptKeys) {
    if (before.scripts?.[key] !== after.scripts?.[key]) {
      return `Rust-related package script changed (${key})`;
    }
  }
  for (const key of ["dependencies", "devDependencies", "optionalDependencies", "overrides"]) {
    if (stableJson(before[key] || {}) !== stableJson(after[key] || {})) {
      return `package ${key} changed`;
    }
  }
  return "";
}

export function privateWasmCheckReasonForFile(file, readers = {}) {
  if (
    file === "scripts/check-private-wasm-build.mjs"
    || file === "scripts/check-private-wasm-contract.mjs"
    || file === "scripts/private-wasm-modules.mjs"
  ) {
    return "private optional WASM validation input changed";
  }
  if (file === "package.json") {
    return packageJsonPrivateWasmCheckReason(readers.readBefore?.(file), readers.readAfter?.(file));
  }
  if (isPrivateOptionalWasmPath(file)) {
    return "private optional WASM module changed";
  }
  return "";
}

export function packageJsonPrivateWasmCheckReason(beforeText = "{}", afterText = "{}") {
  let before;
  let after;
  try {
    before = JSON.parse(beforeText || "{}");
    after = JSON.parse(afterText || "{}");
  } catch {
    return "package.json parse changed";
  }
  for (const key of ["build:target-inversion", "check:private-wasm-build"]) {
    if (before.scripts?.[key] !== after.scripts?.[key]) {
      return `private WASM package script changed (${key})`;
    }
  }
  for (const key of ["dependencies", "devDependencies", "optionalDependencies", "overrides"]) {
    if (stableJson(before[key] || {}) !== stableJson(after[key] || {})) {
      return `package ${key} changed`;
    }
  }
  return "";
}

function isPrivateOptionalWasmPath(file) {
  return PRIVATE_OPTIONAL_WASM_PREFIXES.some((prefix) => file.startsWith(prefix));
}

function isPublicRustRuntimeSource(file) {
  return (file.startsWith("src/") || file.startsWith("protected/deepbom_wasm/src/")) && file.endsWith(".rs");
}

function isRustSource(file) {
  return (file.startsWith("src/") || file.startsWith("protected/") || file.startsWith("native/")) && file.endsWith(".rs");
}

function stableJson(value) {
  if (Array.isArray(value)) {
    return `[${value.map(stableJson).join(",")}]`;
  }
  if (value && typeof value === "object") {
    return `{${Object.keys(value)
      .filter((key) => value[key] !== undefined)
      .sort()
      .map((key) => `${JSON.stringify(key)}:${stableJson(value[key])}`)
      .join(",")}}`;
  }
  return JSON.stringify(value);
}

function gitShow(commit, file) {
  try {
    return execFileSync("git", ["show", `${commit}:${file}`], {
      encoding: "utf8",
      stdio: ["ignore", "pipe", "ignore"],
    });
  } catch {
    return "";
  }
}

function gitChangedFiles(before, after) {
  if (after === "--working-tree") return gitWorkingTreeChangedFiles(before);
  return execFileSync("git", ["diff", "--name-only", before, after], { encoding: "utf8" })
    .split(/\r?\n/)
    .map(normalizeChangedPath)
    .filter(Boolean);
}

function gitWorkingTreeChangedFiles(base) {
  const changed = execFileSync("git", ["diff", "--name-only", base, "--"], { encoding: "utf8" });
  const untracked = execFileSync("git", ["ls-files", "--others", "--exclude-standard"], { encoding: "utf8" });
  return [...new Set(`${changed}\n${untracked}`.split(/\r?\n/).map(normalizeChangedPath).filter(Boolean))];
}

function readWorkingTreeFile(file) {
  try {
    return existsSync(file) ? readFileSync(file, "utf8") : "";
  } catch {
    return "";
  }
}

function normalizeChangedPath(file) {
  return normalizePath(file.trim());
}

function emitGitHubOutput(decision) {
  const lines = [
    `changed=${decision.changed ? "true" : "false"}`,
    `reason=${decision.reason}`,
    `rust_check=${decision.rustCheck ? "true" : "false"}`,
    `private_wasm_check=${decision.privateWasmCheck ? "true" : "false"}`,
  ];
  if (process.env.GITHUB_OUTPUT) {
    appendFileSync(process.env.GITHUB_OUTPUT, `${lines.join("\n")}\n`);
  } else {
    console.log(lines.join("\n"));
  }
}

function main() {
  const { before, after, workingTree } = parseDeployArgs(process.argv.slice(2));
  if (!before || !after) {
    console.error("Usage: node scripts/detect-deployable-changes.mjs <before> <after>|<base> --working-tree|--working-tree [base]");
    process.exit(2);
  }
  const files = gitChangedFiles(before, after);
  console.log(files.join("\n"));
  const decision = classifyChangeSet(files, {
    readBefore: (file) => gitShow(before, file),
    readAfter: (file) => workingTree ? readWorkingTreeFile(file) : gitShow(after, file),
  });
  for (const detail of decision.details) {
    console.log(`Deployable: ${detail}`);
  }
  for (const detail of decision.rustCheckDetails) {
    console.log(`Rust check: ${detail}`);
  }
  for (const detail of decision.privateWasmCheckDetails) {
    console.log(`Private WASM check: ${detail}`);
  }
  if (!decision.changed) {
    console.log("No deployable asset changes detected; remote WASM build and Cloudflare deploy will be skipped.");
  }
  emitGitHubOutput(decision);
}

export function parseDeployArgs(args) {
  if (args.includes("--working-tree")) {
    return {
      before: args.find((arg) => arg !== "--working-tree") || "HEAD",
      after: "--working-tree",
      workingTree: true,
    };
  }
  const [before, after] = args;
  return { before, after, workingTree: false };
}

if (process.argv[1] && resolve(fileURLToPath(import.meta.url)) === resolve(process.argv[1])) {
  main();
}

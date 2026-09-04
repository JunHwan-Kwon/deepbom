import { execFileSync } from "node:child_process";
import { existsSync, readFileSync, statSync } from "node:fs";
import path from "node:path";
import { normalizePath } from "./path-utils.mjs";
import {
  PRIVATE_WASM_MODULES,
  privateModuleIgnoredPrefixes,
  privateModuleIgnoredRoots,
} from "./private-wasm-modules.mjs";

const privatePrefixes = [
  "dist/",
  "reports/",
  ".wrangler/",
  "docs/private/",
  ...privateModuleIgnoredPrefixes(),
];

const privateSuffixes = [
  ".local.md",
];

const modelArtifactExtensions = new Set([
  ".ckpt",
  ".gguf",
  ".h5",
  ".keras",
  ".litertlm",
  ".mlmodel",
  ".mlpackage",
  ".npy",
  ".npz",
  ".onnx",
  ".pb",
  ".pte",
  ".pt",
  ".pth",
  ".safetensors",
  ".task",
  ".tflite",
]);

const sanctionedModelFixtures = new Map([
  ["corpus/coreml-legacy-quantization-corpus/per-output-channel-linear-int4.mlmodel", "scripts/build-coreml-legacy-quantization-corpus.mjs"],
  ["corpus/onnx-extension-contract-corpus/deepbom-static-per-axis-qdq.onnx", "scripts/build-onnx-extension-contract-corpus.mjs"],
  ["corpus/onnx-extension-contract-corpus/onnx-dequantize-linear-runtime-per-axis.onnx", "scripts/build-onnx-extension-contract-corpus.mjs"],
  ["corpus/onnx-extension-contract-corpus/onnx-quantize-linear-runtime-per-axis.onnx", "scripts/build-onnx-extension-contract-corpus.mjs"],
  ["corpus/onnx-extension-contract-corpus/onnxruntime-extensions-custom-op-test.onnx", "scripts/build-onnx-extension-contract-corpus.mjs"],
  ["scripts/fixtures/onnx_dynamic_conv.onnx", "scripts/generate-onnx-dynamic-shape-fixtures.py"],
  ["scripts/fixtures/onnx_recursive_scope.onnx", "scripts/generate-onnx-recursive-scope-fixture.py"],
  ["scripts/fixtures/onnx_zero_dim_identity.onnx", "scripts/generate-onnx-dynamic-shape-fixtures.py"],
]);

const tracked = gitLines(["ls-files"]);
const staged = gitLines(["diff", "--cached", "--name-only", "--diff-filter=ACMRT"]);
const checked = unique([...tracked, ...staged]).map(normalizePath);
const blocked = checked.filter(isPrivateArtifact);
const packageJson = JSON.parse(readFileSync("package.json", "utf8"));
const gitignore = readFileSync(".gitignore", "utf8");
const protectedSourcePresent = existsSync("protected/deepbom_wasm/src/lib.rs");

if (blocked.length) {
  throw new Error(`Private/generated artifacts must not be tracked or staged:\n${blocked.map((file) => `  - ${file}`).join("\n")}`);
}

assertSyntheticFixtureContract();
assertPrivateBuildContract();
assertPublicSurfaceHasNoPrivateRoadmapTerms();
assertProtectedSelectorImplementationBoundary();
assertProtectedOrtImplementationBoundary();

console.log(`Git privacy guard passed (${checked.length} tracked/staged paths checked; ${protectedSourcePresent ? "private monorepo" : "clean public source"} boundary).`);

function gitLines(args) {
  const output = execFileSync("git", args, { encoding: "utf8" }).trim();
  return output ? output.split(/\r?\n/) : [];
}

function unique(values) {
  return [...new Set(values)];
}

function isPrivateArtifact(file) {
  if (privatePrefixes.some((prefix) => file.startsWith(prefix))) return true;
  if (privateSuffixes.some((suffix) => file.endsWith(suffix))) return true;
  if (isEnvFile(file)) return true;
  // web/samples/ holds public upstream reference models (e.g. MobileNet,
  // Apache-2.0) shipped as try-it demos — the one sanctioned model-artifact
  // location. User/research models must never be placed there.
  if (file.startsWith("web/samples/")) return false;
  if (sanctionedModelFixtures.has(file)) return false;
  return modelArtifactExtensions.has(path.posix.extname(file).toLowerCase());
}

function assertSyntheticFixtureContract() {
  for (const [fixture, generator] of sanctionedModelFixtures) {
    if (!checked.includes(fixture) || !checked.includes(generator)) {
      throw new Error(`Sanctioned model fixture and generator must be tracked together: ${fixture} / ${generator}`);
    }
    const size = statSync(fixture).size;
    if (size < 1 || size > 4096) {
      throw new Error(`Sanctioned model fixture must remain non-empty and <= 4 KiB: ${fixture} (${size} bytes)`);
    }
  }
}

function isEnvFile(file) {
  const basename = path.posix.basename(file);
  return basename === ".env" || basename.startsWith(".env.");
}

function assertPrivateBuildContract() {
  const scripts = packageJson.scripts || {};
  const publicBuild = scripts["build:worker"] || "";
  for (const moduleSpec of PRIVATE_WASM_MODULES) {
    const buildCommand = scripts[moduleSpec.buildScript] || "";
    const sourceRoot = normalizePath(moduleSpec.sourceRoot);
    if (!buildCommand.includes(sourceRoot) || !buildCommand.includes("--out-dir pkg")) {
      throw new Error(`${moduleSpec.buildScript} must build only into ${sourceRoot}/pkg.`);
    }
    if (publicBuild.includes(moduleSpec.buildScript)) {
      throw new Error(`build:worker must not include private optional WASM module build script: ${moduleSpec.buildScript}.`);
    }
  }
  for (const ignored of ["pkg/README.md", ...privateModuleIgnoredRoots()]) {
    if (!gitignore.includes(ignored)) {
      throw new Error(`.gitignore must keep private/generated output ignored: ${ignored}`);
    }
  }
}

function assertPublicSurfaceHasNoPrivateRoadmapTerms() {
  const publicFiles = checked.filter((file) => isPublicSurfaceFile(file) && existsSync(file));
  const leaked = [];
  for (const file of publicFiles) {
    const text = readFileSync(file, "utf8");
    for (const term of privateRoadmapTerms()) {
      if (term.pattern.test(text)) {
        leaked.push(`${file}: ${term.label}`);
      }
    }
  }
  if (leaked.length) {
    throw new Error(`Public docs/UI must not expose private research roadmap terms:\n${leaked.map((item) => `  - ${item}`).join("\n")}`);
  }
}

function assertProtectedSelectorImplementationBoundary() {
  const protectedOwner = "protected/deepbom_wasm/src/xnnpack_selector.rs";
  const ownershipPatterns = protectedSelectorOwnershipPatterns();
  if (protectedSourcePresent) {
    if (!checked.includes(protectedOwner)) {
      throw new Error(`Protected XNNPACK selector owner is not tracked: ${protectedOwner}`);
    }
    const protectedSource = readFileSync(protectedOwner, "utf8");
    const incompleteOwner = ownershipPatterns
      .filter(({ pattern }) => !pattern.test(protectedSource))
      .map(({ label }) => label);
    if (incompleteOwner.length) {
      throw new Error(`Protected XNNPACK selector owner is incomplete: ${incompleteOwner.join(", ")}`);
    }
  }

  const publicRustSources = checked.filter((file) => file.startsWith("src/") && file.endsWith(".rs"));
  const leaked = [];
  for (const file of publicRustSources) {
    const source = readFileSync(file, "utf8");
    for (const { label, pattern } of ownershipPatterns) {
      if (pattern.test(source)) leaked.push(`${file}: ${label}`);
    }
  }
  if (leaked.length) {
    throw new Error(
      `Protected XNNPACK selector implementation must have one source owner:\n${leaked.map((item) => `  - ${item}`).join("\n")}`,
    );
  }
}

function protectedSelectorOwnershipPatterns() {
  return [
    { label: "pinned selector commit constant", pattern: /const\s+XNNPACK_SOURCE_COMMIT\b/ },
    { label: "GEMM source-file hash constant", pattern: /const\s+XNNPACK_GEMM_CONFIG_SHA256\b/ },
    { label: "DWCONV source-file hash constant", pattern: /const\s+XNNPACK_DWCONV_CONFIG_SHA256\b/ },
    { label: "AArch64 candidate enumerator", pattern: /fn\s+a55_kernel_candidates\s*\(/ },
    { label: "WASM SIMD candidate enumerator", pattern: /fn\s+wasm_kernel_candidates\s*\(/ },
    { label: "pinned XNNPACK configuration source path", pattern: /src\/configs\/(?:gemm|dwconv)-config\.c/ },
  ];
}

function assertProtectedOrtImplementationBoundary() {
  const owners = [
    "protected/deepbom_wasm/src/ort_rulepack.rs",
    "protected/deepbom_wasm/src/ort_rulepack_generated.rs",
  ];
  const patterns = [
    { label: "ORT compatibility evidence schema", pattern: /deepbom\.ort_source_compatibility\.v1\.15/ },
    { label: "ORT EP portability frontier schema", pattern: /deepbom\.ort_ep_portability_frontier\.v2/ },
    { label: "pinned ORT source commit", pattern: /ORT_RULEPACK_SOURCE_COMMIT/ },
    { label: "generated EP rule table", pattern: /ORT_EP_RULES/ },
    { label: "generated release floor table", pattern: /ORT_RELEASES/ },
    { label: "CPU operator source hash", pattern: /23ecda7c40bacc71f5362ec01f994f50eab809324587d3e8a2cac46ff707652c/ },
    { label: "compatibility matrix source hash", pattern: /b79ae7035295a6009879b4d8f882ff4a9402726e53533b9d3787fe50e0a42664/ },
    { label: "ONNX operator schema history", pattern: /ORT_OP_SCHEMA_HISTORIES/ },
    { label: "ONNX-ML schema source hash", pattern: /fa3d663df091a0cadc85d902a12d84d7465bfec8cf7433861f82b99f921278a4/ },
    { label: "ORT contrib schema source hash", pattern: /a00b931b8df0db12e03c3346ef9d1abc84200156e1910acaa40dd622711978c9/ },
  ];
  if (protectedSourcePresent) {
    for (const owner of owners) {
      if (!existsSync(owner)) throw new Error(`Protected ORT compatibility owner is missing: ${owner}`);
    }
    const protectedSource = owners.map((owner) => readFileSync(owner, "utf8")).join("\n");
    const incomplete = patterns.filter(({ pattern }) => !pattern.test(protectedSource)).map(({ label }) => label);
    if (incomplete.length) throw new Error(`Protected ORT compatibility owners are incomplete: ${incomplete.join(", ")}`);
  }

  const publicRustSources = checked.filter((file) => file.startsWith("src/") && file.endsWith(".rs"));
  const leaked = [];
  for (const file of publicRustSources) {
    const source = readFileSync(file, "utf8");
    for (const { label, pattern } of patterns) {
      if (pattern.test(source)) leaked.push(`${file}: ${label}`);
    }
  }
  if (leaked.length) {
    throw new Error(`Protected ORT compatibility implementation leaked into public Rust:\n${leaked.map((item) => `  - ${item}`).join("\n")}`);
  }
}

function isPublicSurfaceFile(file) {
  return file === "README.md"
    || file === "docs/PROJECT_STATUS.md"
    || file.startsWith("web/")
    || file.startsWith("worker/");
}

function privateRoadmapTerms() {
  return [
    { label: "Haar-like target search", pattern: /Haar-like/i },
    { label: "fake-weight reconstruction", pattern: /fake[- ]weight/i },
    { label: "synthetic reconstruction", pattern: /synthetic model reconstruction/i },
    { label: "surrogate model reconstruction", pattern: /surrogate model/i },
    { label: "inverse fusion", pattern: /inverse fusion/i },
    { label: "gradient restoration", pattern: /gradient restoration/i },
    { label: "target inversion", pattern: /target[ _-]inversion/i },
  ];
}

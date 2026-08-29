import { existsSync, readdirSync, readFileSync } from "node:fs";
import path from "node:path";
import { pathToFileURL } from "node:url";
import {
  parseAppAssets,
  parseRuntimeCacheableSuffixes,
} from "./sw-utils.mjs";
import { privateModuleDistLeakNeedles } from "./private-wasm-modules.mjs";
import { inspectWasmFile } from "./wasm-binary-hardening.mjs";
import { PUBLIC_SAMPLE_MODELS } from "../web/lib/sample-models.js";

const distRoot = "dist";
const distSwPath = path.join(distRoot, "web", "sw.js");
const distBuildMetadataPath = path.join(distRoot, "web", "lib", "build-metadata.js");
if (!existsSync(distBuildMetadataPath)) {
  throw new Error("dist/web/lib/build-metadata.js is missing. Run `npm run build:worker` before checking dist assets.");
}
const { ANALYZER_BUNDLE_CONTENT_SHA256 } = await import(`${pathToFileURL(path.resolve(distBuildMetadataPath)).href}?dist-asset-check=${Date.now()}`);
const blockedDistExtensions = new Set([
  ".ckpt",
  ".gguf",
  ".h5",
  ".keras",
  ".mlmodel",
  ".npy",
  ".npz",
  ".onnx",
  ".pb",
  ".pt",
  ".pth",
  ".safetensors",
  ".tflite",
]);
const privateDistLeakNeedles = privateModuleDistLeakNeedles();
const publicAnalyzerWasm = path.join(distRoot, "pkg", "tflite_wasm_audit_bg.wasm");
const protectedDeepBomRoot = path.join(distRoot, "web", "protected", "deepbom", "pkg");
const protectedDeepBomWasm = path.join(protectedDeepBomRoot, "deepbom_wasm_bg.wasm");
const protectedDeepBomJs = path.join(protectedDeepBomRoot, "deepbom_wasm.js");
const protectedDeepBomSourceTypes = path.join("web", "protected", "deepbom", "pkg", "deepbom_wasm.d.ts");
const protectedDeploymentMetadata = [
  path.join(protectedDeepBomRoot, "deepbom_wasm.d.ts"),
  path.join(protectedDeepBomRoot, "deepbom_wasm_bg.wasm.d.ts"),
  path.join(protectedDeepBomRoot, "package.json"),
];
const deploymentExcludedFloatSample = path.join(distRoot, "web", "samples", "mobilenet_v1_025_224_float.tflite");
const deploymentExcludedSyntheticOnnx = path.join(distRoot, "web", "samples", "sample_cnn_float.onnx");
const deploymentHardeningManifest = path.join(distRoot, "deployment-hardening.json");
const publicSampleModelPaths = new Set(PUBLIC_SAMPLE_MODELS
  .map((sample) => sample.path)
  .filter((samplePath) => samplePath.startsWith("samples/"))
  .map((samplePath) => `dist/web/${samplePath}`));
const publicAnalyzerNeedles = [
  "deepbom.deployment_frontier.v1.6",
  "deepbom.deployment_delta.v1.1",
  "deepbom.delegation_repair.v1.3",
  "se_global_pool_keepdims",
  "2026-07-29.1",
  "2026-08-13.1",
  "deepbom.channel_vitality.v1",
  "deepbom.rounding_equivalence.v1",
  "deepbom.numerical_abi_propagation.v1.1",
  "2026-07-28.1",
  "exact_zero_kernel_slice_count",
  "min_threshold_eligible_grid_utilization",
  "bias_int32_utilization_sample",
];
const protectedSelectorNeedles = [
  "deepbom.xnnpack_selector_evidence.v2",
  "SOURCE_ENUMERATED_CANDIDATE_SET",
  "L6088-L6112",
  "545f0fd6bab43186819199cf846a1f1cbb5eaa2b1df9b70e5e43431185f2b7f6",
  "F32 GEMM WASM relaxed SIMD",
  "deepbom.ort_source_compatibility.v1.15",
  "deepbom.ort_ep_portability_frontier.v2",
  "SOURCE_ARTIFACT_CONDITION_DEFINITE_FAIL",
  "23ecda7c40bacc71f5362ec01f994f50eab809324587d3e8a2cac46ff707652c",
  "d1c94c1b4b890350a5ff8cc8bf24bd062b09b7a0689293afb1fdc1f7e987b479",
  "fa3d663df091a0cadc85d902a12d84d7465bfec8cf7433861f82b99f921278a4",
  "a00b931b8df0db12e03c3346ef9d1abc84200156e1910acaa40dd622711978c9",
  "b79ae7035295a6009879b4d8f882ff4a9402726e53533b9d3787fe50e0a42664",
  "ort_dependency_manifest",
];
const protectedImplementationLeakNeedles = [
  "L6088-L6112",
  "F32 GEMM WASM relaxed SIMD",
  "const RELEASES = Object.freeze",
  "NEW_RELEASE_FLOOR_SOURCES",
  "ASSIGNMENT_CAPTURE_SOURCES",
  "SCHEMA_SOURCES = Object.freeze(new Map",
  '["0.1", 8, 1, 3]',
  "ort_dependency_manifest",
];

if (!existsSync(distSwPath)) {
  throw new Error("dist/web/sw.js is missing. Run `npm run build:worker` before checking dist assets.");
}

const distSwSource = readFileSync(distSwPath, "utf8");
const expectedBuildCacheSuffix = ANALYZER_BUNDLE_CONTENT_SHA256.slice(0, 16);
if (!expectedBuildCacheSuffix || !new RegExp(`const CACHE_NAME = "[^"]+-${expectedBuildCacheSuffix}";`).test(distSwSource)) {
  throw new Error("dist service-worker cache name is not bound to the analyzer build-content SHA-256.");
}
const appAssets = parseAppAssets(distSwSource);
const runtimeSuffixes = parseRuntimeCacheableSuffixes(distSwSource);

const expected = [
  ...appAssets.map((asset) => [asset, distPathForAppAsset(asset)]),
  ...runtimeSuffixes.map((suffix) => [suffix, distPathForAbsoluteSuffix(suffix)]),
].filter(([, filePath]) => filePath);

const missing = expected.filter(([, filePath]) => !existsSync(filePath));
if (missing.length) {
  throw new Error(`dist is missing service-worker cacheable assets: ${missing.map(([asset]) => asset).join(", ")}`);
}

const unresolvedNodeModules = [
  path.join(distRoot, "web", "app.js"),
  path.join(distRoot, "web", "onnx.js"),
  path.join(distRoot, "web", "sw.js"),
  path.join(distRoot, "web", "lib", "runtime-module-loader.js"),
].filter((filePath) => existsSync(filePath) && readFileSync(filePath, "utf8").includes("node_modules"));

if (unresolvedNodeModules.length) {
  throw new Error(`dist still references node_modules paths: ${unresolvedNodeModules.join(", ")}`);
}

if (existsSync(deploymentExcludedFloatSample)) {
  throw new Error("dist must exclude the internal FLOAT32 MobileNet fixture.");
}
if (existsSync(deploymentExcludedSyntheticOnnx)) throw new Error("dist must exclude the synthetic ONNX regression fixture.");
checkDeploymentHardening();

const medicalShellPath = path.join(distRoot, "medical.html");
if (!existsSync(medicalShellPath)) {
  throw new Error("dist/medical.html is missing. Build should generate the /medical static shell from web/index.html.");
}
const medicalShell = readFileSync(medicalShellPath, "utf8");
if (!medicalShell.includes('<base href="/web/" />')) {
  throw new Error("dist/medical.html must include <base href=\"/web/\" /> so shared web assets resolve from /medical.");
}
if (!medicalShell.includes("./bootstrap.js")) {
  throw new Error("dist/medical.html must reuse the shared web/bootstrap.js entrypoint.");
}
if (medicalShell.includes("Regulatory Bundle ZIP")) {
  throw new Error("dist/medical.html should not duplicate medical/regulatory UI; app-surface.js injects it at runtime.");
}

const blockedDistPaths = collectDistPaths(distRoot).filter(isBlockedDistPath);
if (blockedDistPaths.length) {
  throw new Error(`dist contains private/generated local artifacts:\n${blockedDistPaths.map((file) => `  - ${file}`).join("\n")}`);
}

checkProtectedSelectorBoundary();

console.log(`Dist service-worker asset check passed (${expected.length} cacheable assets, ${blockedDistPaths.length} private artifacts).`);

function checkProtectedSelectorBoundary() {
  for (const filePath of [publicAnalyzerWasm, protectedDeepBomWasm, protectedDeepBomJs, protectedDeepBomSourceTypes]) {
    if (!existsSync(filePath)) throw new Error(`Selector protection check is missing build artifact: ${filePath}`);
  }
  for (const filePath of [publicAnalyzerWasm, protectedDeepBomWasm]) {
    const inspection = inspectWasmFile(filePath);
    if (inspection.customSections.length || inspection.sensitivePaths.length) {
      throw new Error(`Deploy WASM is not hardened: ${filePath}; custom=${inspection.customSections.join(",") || "none"}; internal_paths=${inspection.sensitivePaths.length}.`);
    }
  }
  const leakedMetadata = protectedDeploymentMetadata.filter((filePath) => existsSync(filePath));
  if (leakedMetadata.length) throw new Error(`Protected runtime dist includes non-runtime package metadata: ${leakedMetadata.join(", ")}`);
  if (readFileSync(protectedDeepBomJs, "utf8").includes("@ts-self-types")) {
    throw new Error("Protected runtime JS still advertises a deployment-excluded TypeScript contract.");
  }
  const publicBytes = readFileSync(publicAnalyzerWasm).toString("utf8");
  const protectedBytes = readFileSync(protectedDeepBomWasm).toString("utf8");
  const missingPublic = publicAnalyzerNeedles.filter((needle) => !publicBytes.includes(needle));
  if (missingPublic.length) {
    throw new Error(`Public analyzer WASM is missing public analysis evidence: ${missingPublic.join(", ")}`);
  }
  const leaked = protectedSelectorNeedles.filter((needle) => publicBytes.includes(needle));
  if (leaked.length) {
    throw new Error(`Public analyzer WASM contains protected selector evidence: ${leaked.join(", ")}`);
  }
  const missingProtected = protectedSelectorNeedles.filter((needle) => !protectedBytes.includes(needle));
  if (missingProtected.length) {
    throw new Error(`Protected DEEPBOM WASM is missing selector evidence: ${missingProtected.join(", ")}`);
  }
  const publicDistFiles = collectDistPaths(distRoot)
    .filter((filePath) => !filePath.startsWith(`${protectedDeepBomRoot}${path.sep}`));
  const implementationLeaks = [];
  for (const filePath of publicDistFiles) {
    const bytes = readFileSync(filePath).toString("utf8");
    for (const needle of protectedImplementationLeakNeedles) {
      if (bytes.includes(needle)) implementationLeaks.push(`${needle} in ${filePath}`);
    }
  }
  if (implementationLeaks.length) {
    throw new Error(`Public dist contains protected rulepack implementation fingerprints: ${implementationLeaks.join(", ")}`);
  }
  const typeDefs = readFileSync(protectedDeepBomSourceTypes, "utf8");
  if (!typeDefs.includes("analyze_deepbom")
    || typeDefs.includes("analyze_xnnpack_selector")
    || typeDefs.includes("analyze_ort_compatibility")) {
    throw new Error("Protected DEEPBOM TypeScript definitions must expose only the aggregate business analysis API.");
  }
}

function checkDeploymentHardening() {
  if (!existsSync(deploymentHardeningManifest)) {
    throw new Error("dist/deployment-hardening.json is missing.");
  }
  const manifest = JSON.parse(readFileSync(deploymentHardeningManifest, "utf8"));
  if (manifest.schema !== "deepbom.deployment_hardening.v1"
    || manifest.source_maps !== "forbidden"
    || !(manifest.javascript_files > 0)
    || !(manifest.css_files > 0)
    || !(manifest.output_bytes < manifest.source_bytes)) {
    throw new Error(`Deployment hardening manifest is incomplete: ${JSON.stringify(manifest)}`);
  }
  const deployedFiles = collectDistPaths(distRoot);
  const sourceMaps = deployedFiles.filter((filePath) => filePath.toLowerCase().endsWith(".map"));
  if (sourceMaps.length) {
    throw new Error(`dist contains forbidden source maps: ${sourceMaps.join(", ")}`);
  }
  const mapReferences = deployedFiles
    .filter((filePath) => [".js", ".css", ".html"].includes(path.extname(filePath).toLowerCase()))
    .filter((filePath) => /sourceMappingURL\s*=/.test(readFileSync(filePath, "utf8")));
  if (mapReferences.length) {
    throw new Error(`dist contains forbidden source-map references: ${mapReferences.join(", ")}`);
  }
  const sourceApp = readFileSync(path.join("web", "app.js"), "utf8");
  const deployedApp = readFileSync(path.join(distRoot, "web", "app.js"), "utf8");
  if (!(deployedApp.length < sourceApp.length * 0.8)) {
    throw new Error(`dist/web/app.js was not materially minified (${deployedApp.length}/${sourceApp.length} bytes).`);
  }
}

function distPathForAppAsset(asset) {
  if (asset === "./") return path.join(distRoot, "web", "index.html");
  if (asset.startsWith("./")) return path.join(distRoot, "web", asset.slice(2));
  if (asset.startsWith("../")) return path.join(distRoot, asset.slice(3));
  return "";
}

function distPathForAbsoluteSuffix(suffix) {
  if (!suffix.startsWith("/")) return "";
  return path.join(distRoot, suffix.slice(1));
}

function collectDistPaths(root) {
  const paths = [];
  for (const entry of readdirSync(root, { withFileTypes: true })) {
    const filePath = path.join(root, entry.name);
    if (entry.isDirectory()) {
      paths.push(...collectDistPaths(filePath));
    } else if (entry.isFile()) {
      paths.push(filePath);
    }
  }
  return paths;
}

function isBlockedDistPath(filePath) {
  const normalized = filePath.replaceAll(path.sep, "/");
  if (normalized.toLowerCase().endsWith(".map")) return true;
  if (normalized.startsWith("dist/web/samples/") && blockedDistExtensions.has(path.extname(normalized).toLowerCase())) {
    return !publicSampleModelPaths.has(normalized);
  }
  return privateDistLeakNeedles.some((needle) => normalized.includes(needle))
    || normalized.includes("/reports/")
    || normalized.includes("/docs/private/")
    || normalized.endsWith(".local.md")
    || normalized.includes("LOCAL_PRIVATE_ROADMAP")
    || blockedDistExtensions.has(path.extname(normalized).toLowerCase());
}

import { parseOrtReducedOperatorConfig } from "./ort-reduced-operator-config.js";
import { canonicalJson } from "./report-utils.js";
import { sha256TextHex } from "./sha256-sync.js";

export const ORT_BUILD_ATTESTATION_SCHEMA = "deepbom.ort_node_source_build_attestation.v1";
export const ORT_BUILD_SOURCE = Object.freeze({
  repository: "microsoft/onnxruntime",
  commit: "8c546c37b43caaca1fa25db430dab94b901cf277",
  runtime_version: "1.26.0",
  files: Object.freeze({
    build_driver: Object.freeze({ path: "tools/ci_build/build.py", sha256: "7484ca5cacc4893844785ec055cfd65f5a2e007c02ef1b2b7d3c6bfd9d4c8101" }),
    node_build_driver: Object.freeze({ path: "js/node/script/build.ts", sha256: "aa02bf6d0afabdb7230a40fb3101b13b263210dd4684741f9fdcc7d96ac74f5e" }),
    node_cmake: Object.freeze({ path: "js/node/CMakeLists.txt", sha256: "866bd62302178793772cfedfa96bfeb37b886a4b0681d65a12182eecfbabe78c" }),
    node_manifest: Object.freeze({ path: "js/node/package.json", sha256: "b70472b57184b136aa118a39d3ba00641490467f2558074b94155a9c7df9e5fe" }),
  }),
});

const SHA256 = /^[a-f0-9]{64}$/;
const SAFE_PATH = /^(?![A-Za-z]:)(?!\/)(?!.*(?:^|\/)\.\.(?:\/|$))[A-Za-z0-9._+@()\/-]+$/;

export function validateOrtBuildAttestation(value) {
  if (!value || value.schema !== ORT_BUILD_ATTESTATION_SCHEMA
    || value.evidence_class !== "REPRODUCIBLE_SOURCE_BUILD_ATTESTATION") {
    throw new Error(`ORT source-build attestation must use ${ORT_BUILD_ATTESTATION_SCHEMA}.`);
  }
  const source = {
    repository: requiredText(value.source?.repository, "ORT repository"),
    commit: requiredCommit(value.source?.commit, "ORT source commit"),
    pristine_before_build: value.source?.pristine_before_build === true,
    submodule_status_sha256: requiredSha(value.source?.submodule_status_sha256, "ORT submodule status SHA-256"),
    post_build_diff_sha256: requiredSha(value.source?.post_build_diff_sha256, "ORT post-build diff SHA-256"),
    pinned_files: validatePinnedFiles(value.source?.pinned_files),
  };
  if (source.repository !== ORT_BUILD_SOURCE.repository || source.commit !== ORT_BUILD_SOURCE.commit || !source.pristine_before_build) {
    throw new Error("ORT source-build attestation does not bind a pristine checkout of the pinned source commit.");
  }
  const reducedOperatorConfig = validateReducedConfig(value.reduced_operator_config);
  const build = {
    configuration: requiredText(value.build?.configuration, "ORT build configuration"),
    build_arguments: boundedStrings(value.build?.build_arguments, "ORT build arguments", 128, 4096),
    build_stdout_sha256: requiredSha(value.build?.build_stdout_sha256, "ORT build stdout SHA-256"),
    build_stderr_sha256: requiredSha(value.build?.build_stderr_sha256, "ORT build stderr SHA-256"),
    cmake_cache_sha256: requiredSha(value.build?.cmake_cache_sha256, "ORT CMake cache SHA-256"),
    node_build_arguments: boundedStrings(value.build?.node_build_arguments, "ORT Node build arguments", 128, 4096),
    node_install_stdout_sha256: requiredSha(value.build?.node_install_stdout_sha256, "ORT Node install stdout SHA-256"),
    node_install_stderr_sha256: requiredSha(value.build?.node_install_stderr_sha256, "ORT Node install stderr SHA-256"),
    node_build_stdout_sha256: requiredSha(value.build?.node_build_stdout_sha256, "ORT Node build stdout SHA-256"),
    node_build_stderr_sha256: requiredSha(value.build?.node_build_stderr_sha256, "ORT Node build stderr SHA-256"),
  };
  if (build.configuration !== "Release"
    || !build.build_arguments.includes("--build_shared_lib")
    || !build.build_arguments.includes("--skip_tests")) {
    throw new Error("ORT source-build attestation must identify the canonical Release shared-library build.");
  }
  const includeConfigIndices = build.build_arguments
    .map((argument, index) => argument === "--include_ops_by_config" ? index : -1)
    .filter((index) => index >= 0);
  const typeReductionEnabled = build.build_arguments.includes("--enable_reduced_operator_type_support");
  if (reducedOperatorConfig) {
    if (includeConfigIndices.length !== 1 || !build.build_arguments[includeConfigIndices[0] + 1]) {
      throw new Error("Binary-attested reduced-operator config requires --include_ops_by_config in the observed build arguments.");
    }
    const observedConfigPath = build.build_arguments[includeConfigIndices[0] + 1].replaceAll("\\", "/");
    if (observedConfigPath.split("/").pop() !== reducedOperatorConfig.source_name) {
      throw new Error("ORT reduced-operator config source name does not match the observed build input path.");
    }
    if (typeReductionEnabled !== reducedOperatorConfig.type_reduction_enabled) {
      throw new Error("ORT reduced-operator type-reduction declaration does not match the observed build arguments.");
    }
  } else if (includeConfigIndices.length || typeReductionEnabled) {
    throw new Error("ORT reduced-build arguments require an embedded reduced-operator build input.");
  }
  const runtimePackage = validateRuntimePackage(value.runtime_package);
  const normalized = {
    schema: ORT_BUILD_ATTESTATION_SCHEMA,
    evidence_class: "REPRODUCIBLE_SOURCE_BUILD_ATTESTATION",
    source,
    reduced_operator_config: reducedOperatorConfig,
    build,
    runtime_package: runtimePackage,
    boundary: boundedOptional(value.boundary, 8192),
  };
  const attestationSha256 = sha256TextHex(canonicalJson(normalized));
  if (requiredSha(value.attestation_sha256, "ORT build attestation SHA-256") !== attestationSha256) {
    throw new Error("ORT source-build attestation canonical SHA-256 does not reconstruct.");
  }
  return { ...normalized, attestation_sha256: attestationSha256 };
}

function validatePinnedFiles(value) {
  if (!Array.isArray(value) || value.length !== Object.keys(ORT_BUILD_SOURCE.files).length) throw new Error("ORT pinned build-file inventory is incomplete.");
  const expected = Object.values(ORT_BUILD_SOURCE.files).sort((left, right) => left.path.localeCompare(right.path));
  const rows = value.map((row) => ({ path: safePath(row?.path, "ORT pinned build file"), sha256: requiredSha(row?.sha256, "ORT pinned build-file SHA-256") }))
    .sort((left, right) => left.path.localeCompare(right.path));
  if (canonicalJson(rows) !== canonicalJson(expected)) throw new Error("ORT pinned build-file inventory differs from the source contract.");
  return rows;
}

function validateReducedConfig(value) {
  if (value == null) return null;
  if (value.schema !== "deepbom.ort_reduced_operator_build_input.v1"
    || value.binary_binding_status !== "BOUND_AS_OBSERVED_BUILD_INPUT"
    || !value.source_name || /[\\/]/.test(value.source_name)
    || typeof value.source_text !== "string") {
    throw new Error("ORT reduced-operator build input is invalid.");
  }
  const sourceSha256 = sha256TextHex(value.source_text);
  const normalizedConfig = parseOrtReducedOperatorConfig(value.source_text);
  const normalizedSha256 = sha256TextHex(canonicalJson(normalizedConfig));
  if (requiredSha(value.source_sha256, "ORT reduced config source SHA-256") !== sourceSha256
    || requiredSha(value.normalized_sha256, "ORT reduced config normalized SHA-256") !== normalizedSha256
    || canonicalJson(value.normalized_config) !== canonicalJson(normalizedConfig)) {
    throw new Error("ORT reduced-operator build input does not reconstruct from its embedded source text.");
  }
  return {
    schema: "deepbom.ort_reduced_operator_build_input.v1",
    source_name: value.source_name,
    source_text: value.source_text,
    source_sha256: sourceSha256,
    normalized_sha256: normalizedSha256,
    normalized_config: normalizedConfig,
    type_reduction_enabled: value.type_reduction_enabled === true,
    binary_binding_status: "BOUND_AS_OBSERVED_BUILD_INPUT",
  };
}

function validateRuntimePackage(value) {
  if (!value || value.package_name !== "onnxruntime-node" || value.version !== ORT_BUILD_SOURCE.runtime_version) {
    throw new Error("ORT attested runtime package identity is invalid.");
  }
  const binaryInventory = validateBinaryInventory(value.binary_inventory);
  const binaryInventorySha256 = sha256TextHex(canonicalJson(binaryInventory));
  const primaryBinaryPath = safePath(value.primary_binary_path, "ORT primary binary path");
  const primaryBinarySha256 = requiredSha(value.primary_binary_sha256, "ORT primary binary SHA-256");
  if (requiredSha(value.binary_inventory_sha256, "ORT binary inventory SHA-256") !== binaryInventorySha256
    || !binaryInventory.some((row) => row.path === primaryBinaryPath && row.sha256 === primaryBinarySha256)) {
    throw new Error("ORT attested primary binary does not bind to the binary inventory.");
  }
  const packageManifestSha256 = requiredSha(value.package_manifest_sha256, "ORT package manifest SHA-256");
  if (packageManifestSha256 !== ORT_BUILD_SOURCE.files.node_manifest.sha256) {
    throw new Error("ORT runtime package manifest differs from the pinned source manifest.");
  }
  return {
    package_name: "onnxruntime-node",
    version: ORT_BUILD_SOURCE.runtime_version,
    package_manifest_sha256: packageManifestSha256,
    platform: requiredText(value.platform, "ORT runtime platform"),
    arch: requiredText(value.arch, "ORT runtime architecture"),
    node_napi: value.node_napi === "napi-v6" ? "napi-v6" : (() => { throw new Error("ORT runtime N-API contract must be napi-v6."); })(),
    binary_inventory: binaryInventory,
    binary_inventory_sha256: binaryInventorySha256,
    primary_binary_path: primaryBinaryPath,
    primary_binary_sha256: primaryBinarySha256,
  };
}

function validateBinaryInventory(value) {
  if (!Array.isArray(value) || !value.length || value.length > 256) throw new Error("ORT attested binary inventory is empty or oversized.");
  const rows = value.map((row) => ({
    path: safePath(row?.path, "ORT binary path"),
    byte_length: positiveInteger(row?.byte_length, "ORT binary byte length"),
    sha256: requiredSha(row?.sha256, "ORT binary SHA-256"),
  }));
  if (new Set(rows.map((row) => row.path)).size !== rows.length
    || canonicalJson([...rows].sort((left, right) => left.path.localeCompare(right.path))) !== canonicalJson(rows)) {
    throw new Error("ORT attested binary inventory must be unique and canonically sorted.");
  }
  return rows;
}

function requiredSha(value, label) { const text = String(value || "").toLowerCase(); if (!SHA256.test(text)) throw new Error(`${label} is required.`); return text; }
function requiredCommit(value, label) { const text = String(value || "").toLowerCase(); if (!/^[a-f0-9]{40,64}$/.test(text)) throw new Error(`${label} is required.`); return text; }
function requiredText(value, label) { const text = String(value || "").trim(); if (!text || text.length > 8192) throw new Error(`${label} is required or oversized.`); return text; }
function safePath(value, label) { const text = String(value || "").replaceAll("\\", "/"); if (!SAFE_PATH.test(text)) throw new Error(`${label} is unsafe.`); return text; }
function positiveInteger(value, label) { const number = Number(value); if (!Number.isSafeInteger(number) || number < 1) throw new Error(`${label} must be a positive integer.`); return number; }
function boundedOptional(value, limit) { if (value == null || value === "") return null; const text = String(value); if (text.length > limit) throw new Error("ORT build-attestation text is oversized."); return text; }
function boundedStrings(value, label, maxItems, maxLength) { if (!Array.isArray(value) || value.length > maxItems) throw new Error(`${label} is missing or oversized.`); return value.map((item) => { const text = String(item); if (!text || text.length > maxLength) throw new Error(`${label} contains an empty or oversized value.`); return text; }); }

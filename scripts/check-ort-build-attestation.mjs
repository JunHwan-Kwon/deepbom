import assert from "node:assert/strict";

import { ORT_BUILD_ATTESTATION_SCHEMA, ORT_BUILD_SOURCE, validateOrtBuildAttestation } from "../web/lib/ort-build-attestation.js";
import { parseOrtReducedOperatorConfig } from "../web/lib/ort-reduced-operator-config.js";
import { canonicalJson } from "../web/lib/report-utils.js";
import { sha256TextHex } from "../web/lib/sha256-sync.js";

const configText = "ai.onnx;13;Add,Conv\n";
const normalizedConfig = parseOrtReducedOperatorConfig(configText);
const inventory = [
  { path: "bin/napi-v6/test/x64/libonnxruntime.so", byte_length: 1024, sha256: "1".repeat(64) },
  { path: "bin/napi-v6/test/x64/onnxruntime_binding.node", byte_length: 512, sha256: "2".repeat(64) },
];
const body = {
  schema: ORT_BUILD_ATTESTATION_SCHEMA,
  evidence_class: "REPRODUCIBLE_SOURCE_BUILD_ATTESTATION",
  source: {
    repository: ORT_BUILD_SOURCE.repository,
    commit: ORT_BUILD_SOURCE.commit,
    pristine_before_build: true,
    submodule_status_sha256: "3".repeat(64),
    post_build_diff_sha256: "4".repeat(64),
    pinned_files: Object.values(ORT_BUILD_SOURCE.files).map((row) => ({ ...row })).sort((left, right) => left.path.localeCompare(right.path)),
  },
  reduced_operator_config: {
    schema: "deepbom.ort_reduced_operator_build_input.v1",
    source_name: "required_operators.config",
    source_text: configText,
    source_sha256: sha256TextHex(configText),
    normalized_sha256: sha256TextHex(canonicalJson(normalizedConfig)),
    normalized_config: normalizedConfig,
    type_reduction_enabled: true,
    binary_binding_status: "BOUND_AS_OBSERVED_BUILD_INPUT",
  },
  build: {
    configuration: "Release",
    build_arguments: ["--config", "Release", "--build_shared_lib", "--parallel", "--skip_tests", "--include_ops_by_config", "required_operators.config", "--enable_reduced_operator_type_support"],
    build_stdout_sha256: "5".repeat(64),
    build_stderr_sha256: "6".repeat(64),
    cmake_cache_sha256: "7".repeat(64),
    node_build_arguments: ["run", "build", "--", "--rebuild", "--config=Release"],
    node_install_stdout_sha256: "8".repeat(64),
    node_install_stderr_sha256: "9".repeat(64),
    node_build_stdout_sha256: "a".repeat(64),
    node_build_stderr_sha256: "b".repeat(64),
  },
  runtime_package: {
    package_name: "onnxruntime-node",
    version: ORT_BUILD_SOURCE.runtime_version,
    package_manifest_sha256: ORT_BUILD_SOURCE.files.node_manifest.sha256,
    platform: "test",
    arch: "x64",
    node_napi: "napi-v6",
    binary_inventory: inventory,
    binary_inventory_sha256: sha256TextHex(canonicalJson(inventory)),
    primary_binary_path: inventory[0].path,
    primary_binary_sha256: inventory[0].sha256,
  },
  boundary: "Synthetic attestation fixture.",
};
const source = { ...body, attestation_sha256: sha256TextHex(canonicalJson(body)) };
const parsed = validateOrtBuildAttestation(source);
assert.equal(parsed.reduced_operator_config.binary_binding_status, "BOUND_AS_OBSERVED_BUILD_INPUT");
assert.equal(parsed.runtime_package.binary_inventory_sha256, source.runtime_package.binary_inventory_sha256);
assert.equal(parsed.attestation_sha256, source.attestation_sha256);

assert.throws(() => validateOrtBuildAttestation({ ...source, source: { ...source.source, commit: "0".repeat(40) } }), /pinned source commit/);
assert.throws(() => validateOrtBuildAttestation({ ...source, reduced_operator_config: { ...source.reduced_operator_config, source_text: "ai.onnx;13;Mul\n" } }), /does not reconstruct/);
assert.throws(() => validateOrtBuildAttestation({ ...source, runtime_package: { ...source.runtime_package, primary_binary_sha256: "f".repeat(64) } }), /does not bind/);
assert.throws(() => validateOrtBuildAttestation({ ...source, build: { ...source.build, build_arguments: source.build.build_arguments.filter((item) => item !== "--include_ops_by_config") } }), /requires --include_ops_by_config/);
assert.throws(() => validateOrtBuildAttestation({ ...source, build: { ...source.build, build_arguments: source.build.build_arguments.map((item) => item === "required_operators.config" ? "different.config" : item) } }), /source name does not match/);
assert.throws(() => validateOrtBuildAttestation({ ...source, reduced_operator_config: { ...source.reduced_operator_config, type_reduction_enabled: false } }), /type-reduction declaration does not match/);
assert.throws(() => validateOrtBuildAttestation({ ...source, runtime_package: { ...source.runtime_package, package_manifest_sha256: "c".repeat(64) } }), /differs from the pinned source manifest/);
const fullBuild = {
  ...source,
  reduced_operator_config: null,
  build: {
    ...source.build,
    build_arguments: source.build.build_arguments.filter((item, index, items) => {
      if (item === "--include_ops_by_config" || item === "--enable_reduced_operator_type_support") return false;
      return items[index - 1] !== "--include_ops_by_config";
    }),
  },
};
delete fullBuild.attestation_sha256;
fullBuild.attestation_sha256 = sha256TextHex(canonicalJson(fullBuild));
assert.equal(validateOrtBuildAttestation(fullBuild).reduced_operator_config, null);
assert.throws(() => validateOrtBuildAttestation({ ...fullBuild, build: { ...fullBuild.build, build_arguments: [...fullBuild.build.build_arguments, "--enable_reduced_operator_type_support"] } }), /require an embedded reduced-operator build input/);

console.log("ORT source-build attestation checks passed (source/build-script/config/binary inventory binding and tamper rejection). ");

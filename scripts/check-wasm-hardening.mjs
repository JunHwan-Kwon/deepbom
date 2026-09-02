import { existsSync, readFileSync } from "node:fs";
import { inspectWasmFile } from "./wasm-binary-hardening.mjs";

const protectedCargoPath = "protected/deepbom_wasm/Cargo.toml";
const protectedWasmPath = "web/protected/deepbom/pkg/deepbom_wasm_bg.wasm";
const protectedSourcePresent = existsSync(protectedCargoPath);
const protectedWasmPresent = existsSync(protectedWasmPath);
const artifacts = [
  ["public analyzer", "pkg/tflite_wasm_audit_bg.wasm"],
  ...(protectedSourcePresent && protectedWasmPresent
    ? [["protected DEEPBOM", protectedWasmPath]]
    : []),
];

const failures = [];
if (protectedSourcePresent !== protectedWasmPresent) {
  failures.push(`protected DEEPBOM: source/artifact boundary mismatch (${protectedCargoPath}=${protectedSourcePresent}, ${protectedWasmPath}=${protectedWasmPresent})`);
}
const expectedPublicAnalyzerExports = [
  "analyze_tflite",
  "analyze_tflite_for_target",
  "compute_activation_haar",
  "compute_delegation_repair",
  "compute_deployment_delta",
  "compute_deployment_frontier",
  "compute_input_influence",
  "compute_kernel_haar_decomposition",
  "compute_model_tomography",
  "compute_output_influence",
  "compute_quick_low_norm_stat",
  "compute_static_runtime_calibration",
  "compute_weight_histogram",
  "explore_tflite_redesign_pareto",
  "landscape_directions",
  "landscape_tomography",
  "layer_landscape_grid",
  "preprocess_rgba_to_float32",
  "preprocess_rgba_to_int8",
  "preprocess_rgba_to_uint8",
  "project_tflite_redesign",
  "runtime_guard",
  "synthetic_landscape_grid",
  "target_profiles",
].sort();
const requiredPublicEvidenceContracts = [
  "deepbom.delegation_repair.v1.3",
  "se_global_pool_keepdims",
  "2026-08-13.1",
  "deepbom.numerical_abi_propagation.v1.1",
  "2026-07-28.1",
  "exact_zero_kernel_slice_count",
  "min_threshold_eligible_grid_utilization",
  "bias_int32_utilization_sample",
];
for (const [label, filePath] of artifacts) {
  if (!existsSync(filePath)) {
    failures.push(`${label}: missing ${filePath}`);
    continue;
  }
  const inspection = inspectWasmFile(filePath);
  if (inspection.customSections.length) failures.push(`${label}: custom sections ${inspection.customSections.join(", ")}`);
  if (inspection.sensitivePaths.length) {
    failures.push(`${label}: internal paths ${inspection.sensitivePaths.map((item) => item.value).join(", ")}`);
  }
  if (label === "public analyzer") {
    const publicExports = inspection.exports
      .map((item) => item.name)
      .filter((name) => name !== "memory" && !name.startsWith("__"))
      .sort();
    if (JSON.stringify(publicExports) !== JSON.stringify(expectedPublicAnalyzerExports)) {
      failures.push(`${label}: unreviewed export surface ${publicExports.join(", ")}`);
    }
    const binaryText = readFileSync(filePath).toString("latin1");
    const missingContracts = requiredPublicEvidenceContracts.filter((contract) => !binaryText.includes(contract));
    if (missingContracts.length) {
      failures.push(`${label}: stale binary missing current evidence contracts ${missingContracts.join(", ")}`);
    }
  }
  if (label === "protected DEEPBOM") {
    const publicExports = inspection.exports
      .map((item) => item.name)
      .filter((name) => name !== "memory" && !name.startsWith("__"));
    const expected = ["analyze_deepbom", "deepbom_version"];
    if (JSON.stringify(publicExports.sort()) !== JSON.stringify(expected.sort())) {
      failures.push(`${label}: protected analysis export surface ${publicExports.join(", ")}`);
    }
  }
}

const buildSource = readFileSync("scripts/build-wasm.mjs", "utf8");
for (const filePath of artifacts.slice(0, 2).map(([, filePath]) => filePath)) {
  if (!buildSource.includes(filePath)) failures.push(`build-wasm.mjs does not harden ${filePath}`);
}
for (const cargoPath of protectedSourcePresent ? [protectedCargoPath] : []) {
  const source = readFileSync(cargoPath, "utf8");
  if (/wasm-opt\s*=\s*false/.test(source) || !source.includes('"-Oz"') || !source.includes('"--strip-producers"')) {
    failures.push(`${cargoPath}: optimized stripped release profile is not enabled`);
  }
}

if (failures.length) throw new Error(`WASM hardening check failed:\n${failures.map((item) => `  - ${item}`).join("\n")}`);
const protectedSummary = protectedSourcePresent ? "protected analysis exports 2" : "protected integration omitted at the public boundary";
console.log(`WASM hardening check passed (${artifacts.length} binaries; custom sections 0; internal source paths 0; public analyzer exports ${expectedPublicAnalyzerExports.length}; ${protectedSummary}).`);
